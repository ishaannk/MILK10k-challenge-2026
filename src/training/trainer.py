"""The training loop: mixed precision, cosine LR, EMA, early stopping, logging.

Responsibilities are deliberately narrow. ``Trainer`` owns the epoch loop and the
bookkeeping around it (AMP, gradient accumulation, checkpointing, TensorBoard/W&B);
it does not decide the architecture, the loss, or the data. Those arrive fully
constructed, which is what makes swapping Stage 1 -> 2 -> 3 a config change.

One choice worth calling out: **validation tunes per-class thresholds every epoch**
and early stopping monitors the tuned macro-F1 by default. Selecting on the raw
0.5-threshold macro-F1 would pick whichever epoch happened to be best-calibrated
rather than the epoch with the best-ranking model, and since we ship tuned
thresholds anyway (see ``validation/thresholds.py``) the tuned score is the honest
model-selection signal.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..constants import CLASSES
from ..utils.checkpoint import EarlyStopping, save_checkpoint
from ..utils.config import Config
from ..utils.logging_utils import ExperimentLogger, get_logger
from ..validation.metrics import (
    compute_metrics,
    flatten_metrics,
    format_metrics_table,
    plot_confusion_matrix,
)
from ..validation.thresholds import evaluate_threshold_strategies, optimize_thresholds
from .ema import ModelEMA
from .losses import build_loss, cutmix_batch, mixup_batch
from .optim import build_optimizer, build_scheduler, current_lrs

LOGGER = get_logger(__name__)


class Trainer:
    """Train one model on one train/valid split.

    Parameters
    ----------
    model:
        A :class:`~src.models.lesion_net.LesionNet` (or anything with
        ``forward_batch`` and ``param_groups``).
    cfg:
        The resolved experiment config.
    train_loader, valid_loader:
        Already-built loaders. ``valid_loader`` may be ``None`` for a
        train-on-everything final run.
    output_dir:
        Where checkpoints, metric JSON and the resolved config are written.
    class_counts:
        Per-class training counts, used to build loss weights.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        train_loader: DataLoader,
        valid_loader: DataLoader | None = None,
        output_dir: str | Path = "checkpoints/run",
        class_counts: np.ndarray | None = None,
        experiment_logger: ExperimentLogger | None = None,
        fold: int | None = None,
    ):
        self.cfg = cfg
        self.fold = fold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        # channels_last measurably speeds up convolutional backbones under AMP.
        if bool(cfg.train.get("channels_last", True)):
            self.model = self.model.to(memory_format=torch.channels_last)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.criterion = build_loss(cfg, class_counts).to(self.device)

        self.accum_steps = max(1, int(cfg.train.get("accum_steps", 1)))
        self.steps_per_epoch = max(1, len(train_loader) // self.accum_steps)
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler, self.sched_interval = build_scheduler(self.optimizer, cfg, self.steps_per_epoch)

        # --- mixed precision. bf16 needs no loss scaling and is numerically safer;
        # fp16 is kept for older cards. `amp=false` runs in fp32.
        amp_cfg = str(cfg.train.get("amp", "bf16")).lower()
        self.amp_enabled = amp_cfg in ("fp16", "bf16", "true", "1")
        self.amp_dtype = torch.bfloat16 if amp_cfg == "bf16" else torch.float16
        if self.amp_dtype is torch.bfloat16 and self.amp_enabled and not torch.cuda.is_bf16_supported():
            LOGGER.warning("bf16 unsupported on this device; falling back to fp16")
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.amp_dtype is torch.float16)

        self.grad_clip = float(cfg.train.get("grad_clip", 0.0))

        # --- EMA weights
        self.ema: ModelEMA | None = None
        if bool(cfg.train.get("ema", True)):
            self.ema = ModelEMA(
                self.model,
                decay=float(cfg.train.get("ema_decay", 0.999)),
                warmup_steps=int(cfg.train.get("ema_warmup", 100)),
                device=str(self.device),
            )

        # --- mixup / cutmix
        self.mixup_alpha = float(cfg.get("augment", {}).get("mixup_alpha", 0.0))
        self.cutmix_alpha = float(cfg.get("augment", {}).get("cutmix_alpha", 0.0))
        self.mix_prob = float(cfg.get("augment", {}).get("mix_prob", 0.5))
        # Stop mixing near the end so the model finishes on clean, correctly
        # calibrated targets -- which matters because we threshold probabilities.
        self.mix_off_epochs = int(cfg.get("augment", {}).get("mix_off_last_epochs", 3))

        self.monitor = str(cfg.train.get("monitor", "macro_f1_tuned"))
        self.early_stopping = EarlyStopping(
            patience=int(cfg.train.get("patience", 10)),
            mode=str(cfg.train.get("monitor_mode", "max")),
            min_delta=float(cfg.train.get("min_delta", 1e-4)),
        )

        self.exp_logger = experiment_logger
        self.epochs = int(cfg.train.epochs)
        self.global_step = 0
        self.best_thresholds: np.ndarray | None = None
        self.best_metrics: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

        tag = "" if fold is None else f" [fold {fold}]"
        LOGGER.info(
            "Trainer ready%s | device=%s amp=%s epochs=%d steps/epoch=%d accum=%d",
            tag,
            self.device,
            amp_cfg,
            self.epochs,
            self.steps_per_epoch,
            self.accum_steps,
        )

    # ------------------------------------------------------------------ utils
    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = dict(batch)
        for key in ("derm", "clin", "meta", "target"):
            if key in out:
                tensor = out[key].to(self.device, non_blocking=True)
                if key in ("derm", "clin") and bool(self.cfg.train.get("channels_last", True)):
                    tensor = tensor.to(memory_format=torch.channels_last)
                out[key] = tensor
        return out

    def _maybe_mix(self, batch: dict[str, Any], epoch: int) -> dict[str, Any]:
        """Apply mixup or cutmix to the image tensors and soften the targets."""
        if epoch > self.epochs - self.mix_off_epochs:
            return batch
        if self.mixup_alpha <= 0 and self.cutmix_alpha <= 0:
            return batch
        if np.random.rand() > self.mix_prob:
            return batch

        images = {k: batch[k] for k in ("derm", "clin") if k in batch}
        if not images:
            return batch

        use_cutmix = self.cutmix_alpha > 0 and (self.mixup_alpha <= 0 or np.random.rand() < 0.5)
        fn = cutmix_batch if use_cutmix else mixup_batch
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        mixed_images, mixed_targets = fn(images, batch["target"], alpha=alpha)

        out = dict(batch)
        out.update(mixed_images)
        out["target"] = mixed_targets
        return out

    # ------------------------------------------------------------- train step
    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        running_loss, seen_batches = 0.0, 0
        log_every = int(self.cfg.train.get("log_every", 50))
        start = time.time()

        self.optimizer.zero_grad(set_to_none=True)

        for step, raw_batch in enumerate(self.train_loader):
            batch = self._to_device(raw_batch)
            batch = self._maybe_mix(batch, epoch)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                logits = self.model.forward_batch(batch)
                loss = self.criterion(logits, batch["target"])
                # Scale so that accumulated gradients match a single large batch.
                loss_for_backward = loss / self.accum_steps

            self.scaler.scale(loss_for_backward).backward()

            is_step_boundary = (step + 1) % self.accum_steps == 0
            if is_step_boundary:
                if self.grad_clip > 0:
                    # Unscale before clipping, otherwise the clip norm is meaningless.
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update(self.model)
                if self.scheduler is not None and self.sched_interval == "step":
                    self.scheduler.step()
                self.global_step += 1

            running_loss += float(loss.detach())
            seen_batches += 1

            if log_every and step % log_every == 0:
                LOGGER.info(
                    "  epoch %d | step %d/%d | loss %.4f | lr %.2e",
                    epoch,
                    step,
                    len(self.train_loader),
                    running_loss / max(1, seen_batches),
                    self.optimizer.param_groups[0]["lr"],
                )
                if self.exp_logger is not None:
                    self.exp_logger.log_scalars(
                        {"loss_step": running_loss / max(1, seen_batches), **current_lrs(self.optimizer)},
                        step=self.global_step,
                        prefix="train",
                    )

        return {
            "loss": running_loss / max(1, seen_batches),
            "epoch_time_s": time.time() - start,
            **{k.replace("lr/", "lr_"): v for k, v in current_lrs(self.optimizer).items()},
        }

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_loader(self, loader: DataLoader, use_ema: bool = True) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
        """Run the model over a loader. Returns ``(probs, targets_or_None, lesion_ids)``."""
        model = self.ema.module if (use_ema and self.ema is not None) else self.model
        model.eval()

        all_probs, all_targets, all_ids = [], [], []
        for raw_batch in loader:
            batch = self._to_device(raw_batch)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                logits = model.forward_batch(batch)
            all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            if "target" in batch:
                all_targets.append(batch["target"].cpu().numpy())
            all_ids.extend(raw_batch["lesion_id"])

        probs = np.concatenate(all_probs, axis=0)
        targets = np.concatenate(all_targets, axis=0) if all_targets else None
        return probs, targets, all_ids

    def validate(self, epoch: int) -> dict[str, Any]:
        """Score the validation split, tuning thresholds on it."""
        probs, targets, _ = self.predict_loader(self.valid_loader, use_ema=True)

        thresholds = optimize_thresholds(
            targets,
            probs,
            plateau_tolerance=float(self.cfg.get("threshold", {}).get("plateau_tolerance", 0.01)),
            min_threshold=float(self.cfg.get("threshold", {}).get("min", 0.02)),
            max_threshold=float(self.cfg.get("threshold", {}).get("max", 0.95)),
            max_predict_multiple=float(self.cfg.get("threshold", {}).get("max_predict_multiple", 4.0)),
            verbose=False,
        )
        metrics = compute_metrics(targets, probs, thresholds=thresholds)
        metrics["strategies"] = evaluate_threshold_strategies(targets, probs, thresholds)
        metrics["_probs"] = probs      # kept for OOF collection; stripped before JSON
        metrics["_targets"] = targets
        return metrics

    # -------------------------------------------------------------------- fit
    def fit(self) -> dict[str, Any]:
        """Run the full training loop. Returns the best epoch's metric bundle."""
        best_path = self.output_dir / "best.pt"
        last_path = self.output_dir / "last.pt"
        oof_probs: np.ndarray | None = None
        oof_targets: np.ndarray | None = None

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)

            if self.valid_loader is None:
                LOGGER.info("epoch %d | train loss %.4f (no validation split)", epoch, train_metrics["loss"])
                self._save(last_path, epoch, {}, )
                continue

            val_metrics = self.validate(epoch)
            probs = val_metrics.pop("_probs")
            targets = val_metrics.pop("_targets")

            monitored = float(val_metrics.get(self.monitor, val_metrics["macro_f1"]))
            improved = self.early_stopping.step(monitored, epoch)

            LOGGER.info(
                "epoch %d/%d | loss %.4f | f1@0.5 %.4f | f1_tuned %.4f | f1_argmax %.4f | AUC %.4f | %.0fs%s",
                epoch,
                self.epochs,
                train_metrics["loss"],
                val_metrics["macro_f1"],
                val_metrics.get("macro_f1_tuned", float("nan")),
                val_metrics.get("macro_f1_argmax", float("nan")),
                val_metrics["macro_auc"],
                train_metrics["epoch_time_s"],
                "  <-- best" if improved else "",
            )

            # --- logging
            if self.exp_logger is not None:
                self.exp_logger.log_scalars(
                    {k: v for k, v in train_metrics.items() if isinstance(v, float)}, epoch, prefix="train"
                )
                self.exp_logger.log_scalars(flatten_metrics(val_metrics), epoch, prefix="val")
                self.exp_logger.log_scalars(val_metrics.get("strategies", {}), epoch, prefix="strategy")

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    **{k: v for k, v in val_metrics.items() if isinstance(v, (int, float))},
                }
            )

            # --- checkpoints
            if improved:
                self.best_thresholds = np.asarray(val_metrics["thresholds"], dtype=np.float64)
                self.best_metrics = val_metrics
                oof_probs, oof_targets = probs, targets
                self._save(best_path, epoch, val_metrics)
                LOGGER.info("\n%s", format_metrics_table(val_metrics, tuned=True))
                if self.exp_logger is not None:
                    figure = plot_confusion_matrix(
                        np.array(val_metrics["confusion_matrix"]),
                        CLASSES,
                        title=f"Confusion matrix (epoch {epoch})",
                    )
                    self.exp_logger.log_figure("val/confusion_matrix", figure, epoch)
                    import matplotlib.pyplot as plt

                    plt.close(figure)

            self._save(last_path, epoch, val_metrics)

            if self.scheduler is not None and self.sched_interval == "epoch_metric":
                self.scheduler.step(monitored)
            elif self.scheduler is not None and self.sched_interval == "epoch":
                self.scheduler.step()

            if self.early_stopping.should_stop:
                LOGGER.info("Early stopping at epoch %d | %s", epoch, self.early_stopping.status)
                break

        # --- persist artefacts for downstream analysis and ensembling
        if oof_probs is not None:
            # Saved from the *best* epoch, so cross-fold threshold tuning and
            # ensembling downstream see consistent, model-selected predictions.
            np.save(self.output_dir / "oof_probs.npy", oof_probs)
            np.save(self.output_dir / "oof_targets.npy", oof_targets)
            with open(self.output_dir / "oof_lesion_ids.json", "w") as fh:
                json.dump([str(x) for x in self.valid_loader.dataset.lesion_ids], fh)
        self._write_summary()
        return self.best_metrics

    # ---------------------------------------------------------------- helpers
    def _save(self, path: Path, epoch: int, metrics: dict[str, Any]) -> None:
        save_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            config=self.cfg.to_dict(),
            thresholds=np.asarray(metrics["thresholds"]) if metrics.get("thresholds") is not None else None,
            ema_state=self.ema.state_dict() if self.ema is not None else None,
        )

    def _write_summary(self) -> None:
        """Write ``metrics.json`` and ``history.json`` next to the checkpoints."""
        clean_best = {k: v for k, v in self.best_metrics.items() if not k.startswith("_")}
        payload = {
            "fold": self.fold,
            "monitor": self.monitor,
            "best_epoch": self.early_stopping.best_epoch,
            "best_value": self.early_stopping.best,
            "best_metrics": clean_best,
        }
        with open(self.output_dir / "metrics.json", "w") as fh:
            json.dump(payload, fh, indent=2, default=float)
        with open(self.output_dir / "history.json", "w") as fh:
            json.dump(self.history, fh, indent=2, default=float)
        self.cfg.save(self.output_dir / "config.yaml")
