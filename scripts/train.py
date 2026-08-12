#!/usr/bin/env python
"""Train a MILK10k model on one fold, several folds, or full cross-validation.

Examples
--------
Single fold (fast iteration)::

    python scripts/train.py --config configs/stage1_derm.yaml --fold 0

Full 5-fold cross-validation, then pooled out-of-fold threshold tuning::

    python scripts/train.py --config configs/stage3_meta.yaml --folds all

Ad-hoc override without editing YAML::

    python scripts/train.py --config configs/stage1_derm.yaml --fold 0 \
        --set train.epochs=10 data.image_size=224

Why pooled OOF matters
----------------------
When more than one fold is trained, the out-of-fold predictions are concatenated
and thresholds are tuned **once on the pooled set**. Per-fold tuning fits each
threshold on ~10 positives for the rare classes and overfits badly; pooling gives
the full ~50 and a threshold that transfers. The pooled thresholds and the
strategy comparison are written to ``oof_summary.json``, and that is what
``predict.py`` consumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

from src.constants import CLASSES
from src.datasets.milk10k import build_dataloader, build_datasets, make_balanced_sampler
from src.models.lesion_net import build_model
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logging_utils import ExperimentLogger, get_logger, setup_logging
from src.utils.seed import seed_everything
from src.validation.metrics import compute_metrics, format_metrics_table
from src.validation.thresholds import (
    evaluate_threshold_strategies,
    nested_threshold_estimate,
    optimize_thresholds,
)

LOGGER = get_logger("train")


def parse_folds(spec: str, n_folds: int) -> list[int]:
    """Turn ``"all"`` / ``"0"`` / ``"0,2,4"`` into a list of fold indices."""
    if spec.strip().lower() == "all":
        return list(range(n_folds))
    return [int(part) for part in spec.split(",") if part.strip() != ""]


def _load_pretrained(model, checkpoint_path: str, use_ema: bool = True) -> None:
    """Initialise ``model`` from another run's checkpoint, keeping what matches.

    Loads the EMA weights when present (they are the better model). Any parameter
    whose shape disagrees is dropped and reported, so a pretrained dermoscopy-only
    tower can seed a model with extra streams without hand-editing state dicts.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if (use_ema and ckpt.get("ema")) else ckpt["model"]

    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(state) - set(compatible))

    model.load_state_dict(compatible, strict=False)
    LOGGER.info(
        "Initialised from %s (%s weights): loaded %d/%d tensors",
        checkpoint_path,
        "EMA" if (use_ema and ckpt.get("ema")) else "raw",
        len(compatible),
        len(own),
    )
    if skipped:
        LOGGER.info("  skipped %d incompatible/absent tensor(s), e.g. %s", len(skipped), skipped[:5])
    if not compatible:
        raise RuntimeError(f"No compatible weights found in {checkpoint_path} -- wrong architecture?")


def train_one_fold(cfg, train_frame, folds_frame, fold: int, run_dir: Path) -> dict:
    """Train a single fold and return its best-epoch metrics."""
    fold_ids = folds_frame.loc[folds_frame["fold"] == fold, "lesion_id"]
    is_valid = train_frame["lesion_id"].isin(set(fold_ids))

    train_split = train_frame.loc[~is_valid].reset_index(drop=True)
    valid_split = train_frame.loc[is_valid].reset_index(drop=True)
    LOGGER.info("Fold %d | train=%d valid=%d", fold, len(train_split), len(valid_split))

    train_ds, valid_ds, processor = build_datasets(cfg, train_split, valid_split)

    # Persist the fitted metadata encoder: inference must reproduce the exact same
    # feature layout and standardisation statistics.
    fold_dir = run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if processor is not None:
        processor.save(fold_dir / "meta_processor.json")

    sampler = None
    if str(cfg.train.get("sampler", "balanced")).lower() == "balanced":
        sampler = make_balanced_sampler(
            train_ds.labels,
            power=float(cfg.train.get("sampler_power", 0.5)),
            seed=int(cfg.seed) + fold,
        )
        LOGGER.info("Balanced sampler enabled (power=%.2f)", float(cfg.train.get("sampler_power", 0.5)))

    train_loader = build_dataloader(train_ds, cfg, train=True, sampler=sampler)
    valid_loader = build_dataloader(valid_ds, cfg, train=False)

    model = build_model(cfg, meta_dim=processor.dim if processor is not None else 0)

    # Optional transfer from a previous run (e.g. external-data pretraining).
    # strict=False so a shape-mismatched head or an absent metadata tower is skipped
    # rather than fatal -- that is exactly the case when pretraining was
    # dermoscopy-only and the fine-tune adds streams.
    init_ckpt = cfg.model.get("init_checkpoint")
    if init_ckpt:
        _load_pretrained(model, init_ckpt, use_ema=bool(cfg.model.get("init_from_ema", True)))

    exp_logger = ExperimentLogger(
        log_dir=Path(cfg.paths.log_dir) / "tb" / f"{cfg.experiment_name}_fold{fold}",
        run_name=f"{cfg.experiment_name}_fold{fold}",
        use_tensorboard=bool(cfg.logging.get("tensorboard", True)),
        use_wandb=bool(cfg.logging.get("wandb", False)),
        wandb_project=str(cfg.logging.get("wandb_project", "milk10k")),
        wandb_entity=cfg.logging.get("wandb_entity"),
        config=cfg.to_dict(),
    )

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=fold_dir,
        class_counts=train_ds.class_counts(),
        experiment_logger=exp_logger,
        fold=fold,
    )
    try:
        best = trainer.fit()
    finally:
        exp_logger.close()

    return {k: v for k, v in best.items() if not k.startswith("_")}


def pool_oof(run_dir: Path, folds: list[int]) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Concatenate saved per-fold OOF predictions into one pooled set."""
    probs, targets, ids = [], [], []
    for fold in folds:
        fold_dir = run_dir / f"fold{fold}"
        prob_path, target_path = fold_dir / "oof_probs.npy", fold_dir / "oof_targets.npy"
        if not (prob_path.exists() and target_path.exists()):
            LOGGER.warning("Fold %d has no OOF predictions; excluded from pooling", fold)
            continue
        probs.append(np.load(prob_path))
        targets.append(np.load(target_path))
        id_path = fold_dir / "oof_lesion_ids.json"
        ids.extend(json.loads(id_path.read_text()) if id_path.exists() else [])
    if not probs:
        return None
    return np.concatenate(probs), np.concatenate(targets), ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None, help="train a single fold")
    parser.add_argument("--folds", default=None, help='"all" or a comma-separated list, e.g. 0,1,2')
    parser.add_argument("--name", default=None, help="override experiment_name (and thus the output dir)")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.name:
        cfg.experiment_name = args.name
    if "experiment_name" not in cfg:
        cfg.experiment_name = Path(args.config).stem

    run_dir = Path(cfg.paths.checkpoint_dir) / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(Path(cfg.paths.log_dir) / f"{cfg.experiment_name}.log")

    seed_everything(int(cfg.seed), deterministic=bool(cfg.deterministic), strict=bool(cfg.strict_determinism))

    LOGGER.info("=" * 78)
    LOGGER.info("Experiment: %s", cfg.experiment_name)
    LOGGER.info("Config: %s", args.config)
    LOGGER.info(
        "Model: %s | clinical=%s metadata=%s fusion=%s | image_size=%d",
        cfg.model.backbone,
        cfg.model.get("use_clinical", False),
        cfg.model.get("use_metadata", False),
        cfg.model.get("fusion", "concat"),
        cfg.data.image_size,
    )
    if torch.cuda.is_available():
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))
    LOGGER.info("=" * 78)
    cfg.save(run_dir / "config.yaml")

    # Both are configurable so external-data pretraining reuses this whole path.
    train_csv = cfg.data.get("train_lesions_csv") or (Path(cfg.paths.processed_dir) / "train_lesions.csv")
    train_frame = pd.read_csv(train_csv)
    folds_frame = pd.read_csv(cfg.data.folds_csv)
    LOGGER.info("Training table: %s (%d lesions) | folds: %s", train_csv, len(train_frame), cfg.data.folds_csv)

    if args.folds is not None:
        fold_list = parse_folds(args.folds, int(cfg.data.n_folds))
    elif args.fold is not None:
        fold_list = [args.fold]
    else:
        fold_list = [0]
    LOGGER.info("Training folds: %s", fold_list)

    results: dict[int, dict] = {}
    for fold in fold_list:
        # Re-seed per fold so a fold's result does not depend on which folds ran
        # before it -- important for reproducing a single fold in isolation.
        seed_everything(int(cfg.seed) + fold, deterministic=bool(cfg.deterministic))
        results[fold] = train_one_fold(cfg, train_frame, folds_frame, fold, run_dir)
        LOGGER.info(
            "Fold %d done | f1@0.5=%.4f f1_tuned=%.4f AUC=%.4f",
            fold,
            results[fold].get("macro_f1", float("nan")),
            results[fold].get("macro_f1_tuned", float("nan")),
            results[fold].get("macro_auc", float("nan")),
        )

    # ---------------------------------------------------------------- summary
    summary: dict = {
        "experiment_name": cfg.experiment_name,
        "config": args.config,
        "folds": fold_list,
        "per_fold": {
            str(f): {k: v for k, v in m.items() if isinstance(v, (int, float))} for f, m in results.items()
        },
    }

    for metric in ("macro_f1", "macro_f1_tuned", "macro_f1_argmax", "macro_auc", "balanced_accuracy"):
        values = [m[metric] for m in results.values() if metric in m and np.isfinite(m[metric])]
        if values:
            summary[f"cv_{metric}_mean"] = float(np.mean(values))
            summary[f"cv_{metric}_std"] = float(np.std(values))
            LOGGER.info("CV %-18s %.4f +/- %.4f", metric, np.mean(values), np.std(values))

    # ------------------------------------------------- pooled OOF calibration
    pooled = pool_oof(run_dir, fold_list)
    if pooled is not None:
        probs, targets, ids = pooled
        LOGGER.info("Pooled OOF: %d lesions across %d fold(s)", len(probs), len(fold_list))

        thresholds = optimize_thresholds(
            targets,
            probs,
            plateau_tolerance=float(cfg.get("threshold", {}).get("plateau_tolerance", 0.01)),
            min_threshold=float(cfg.get("threshold", {}).get("min", 0.02)),
            max_threshold=float(cfg.get("threshold", {}).get("max", 0.95)),
            max_predict_multiple=float(cfg.get("threshold", {}).get("max_predict_multiple", 4.0)),
            verbose=True,
        )
        metrics = compute_metrics(targets, probs, thresholds=thresholds)
        strategies = evaluate_threshold_strategies(targets, probs, thresholds)

        LOGGER.info("Pooled OOF per-class metrics at tuned thresholds:\n%s", format_metrics_table(metrics, tuned=True))
        LOGGER.info("Submission strategy comparison (pooled OOF macro-F1):")
        for name, value in sorted(strategies.items(), key=lambda kv: -kv[1]):
            LOGGER.info("  %-18s %.4f", name, value)
        best_strategy = max(strategies, key=strategies.get)
        LOGGER.info("Recommended strategy: %s (%.4f)", best_strategy, strategies[best_strategy])

        # The strategy scores above fit thresholds on the same lesions they score, so
        # they are optimistic. This is the unbiased version, and it is the number to
        # compare against a leaderboard.
        honest = nested_threshold_estimate(targets, probs, holdout_size=479)
        LOGGER.info(
            "HONEST estimate (thresholds fitted on %d, scored on a disjoint 479 -- test-set size): "
            "%.4f +/- %.4f  [p5-p95: %.4f - %.4f]",
            len(probs) - int(honest["holdout_size"]),
            honest["honest_macro_f1_mean"],
            honest["honest_macro_f1_std"],
            honest["honest_macro_f1_p5"],
            honest["honest_macro_f1_p95"],
        )
        LOGGER.info(
            "  (pooled-OOF 'tuned' score %.4f is biased upward by ~%.4f; the submission "
            "averages all folds with TTA, which should recover part of the gap)",
            strategies.get("tuned", float("nan")),
            strategies.get("tuned", 0.0) - honest["honest_macro_f1_mean"],
        )

        np.save(run_dir / "oof_probs.npy", probs)
        np.save(run_dir / "oof_targets.npy", targets)
        np.save(run_dir / "oof_thresholds.npy", thresholds)
        with open(run_dir / "oof_summary.json", "w") as fh:
            json.dump(
                {
                    "n_lesions": int(len(probs)),
                    "thresholds": thresholds.tolist(),
                    "class_names": CLASSES,
                    "strategies": strategies,
                    "recommended_strategy": best_strategy,
                    "honest_estimate": honest,
                    "metrics": {k: v for k, v in metrics.items() if not isinstance(v, dict)},
                    "per_class_tuned": metrics.get("per_class_tuned", {}),
                    "lesion_ids": ids,
                },
                fh,
                indent=2,
                default=float,
            )
        summary["oof_strategies"] = strategies
        summary["oof_recommended_strategy"] = best_strategy
        summary["oof_thresholds"] = thresholds.tolist()

    with open(run_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    LOGGER.info("Run artefacts written to %s", run_dir)


if __name__ == "__main__":
    main()
