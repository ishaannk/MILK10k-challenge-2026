"""Console logging plus a thin TensorBoard / Weights & Biases facade.

The ``ExperimentLogger`` lets training code call ``log_scalars`` once and stay
agnostic about which backends happen to be enabled for this run.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger to write to stdout and, optionally, a file.

    Idempotent: calling it twice will not duplicate handlers, which matters when
    scripts import each other.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # These libraries are chatty at INFO and drown out our own progress lines.
    for noisy in ("PIL", "matplotlib", "urllib3", "timm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class ExperimentLogger:
    """Fan out metrics to TensorBoard and/or W&B, degrading gracefully.

    Both backends are optional. If TensorBoard or ``wandb`` is not installed, or
    W&B is disabled in the config, the corresponding calls become no-ops rather
    than errors: a missing logger should never take down a training run.
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "milk10k",
        wandb_entity: str | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.logger = get_logger(self.__class__.__name__)
        self.writer = None
        self.wandb = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                self.logger.info("TensorBoard logging to %s", self.log_dir)
            except Exception as exc:  # pragma: no cover - environment dependent
                self.logger.warning("TensorBoard unavailable (%s); continuing without it", exc)

        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    dir=str(self.log_dir),
                    config=dict(config or {}),
                    reinit=True,
                )
                self.wandb = wandb
                self.logger.info("W&B run initialised: %s", run_name)
            except Exception as exc:  # pragma: no cover - environment dependent
                self.logger.warning("W&B unavailable (%s); continuing without it", exc)

    # -- scalars ------------------------------------------------------------
    def log_scalars(self, metrics: Mapping[str, float], step: int, prefix: str = "") -> None:
        """Log a flat ``{name: value}`` mapping under an optional ``prefix/``."""
        payload = {}
        for key, value in metrics.items():
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                continue
            name = f"{prefix}/{key}" if prefix else key
            payload[name] = float(value)
            if self.writer is not None:
                self.writer.add_scalar(name, float(value), step)
        if self.wandb is not None and payload:
            self.wandb.log(payload, step=step)

    def log_figure(self, tag: str, figure, step: int) -> None:
        """Log a matplotlib figure (confusion matrix, PR curves, ...)."""
        if self.writer is not None:
            self.writer.add_figure(tag, figure, step)
        if self.wandb is not None:
            self.wandb.log({tag: self.wandb.Image(figure)}, step=step)

    def log_hparams(self, hparams: Mapping[str, Any], metrics: Mapping[str, float]) -> None:
        """Record the hyper-parameter/metric pair that TensorBoard's HParams tab uses."""
        if self.writer is None:
            return
        # TensorBoard only accepts scalar-ish hparam values.
        clean = {
            k: (v if isinstance(v, (int, float, str, bool)) else str(v))
            for k, v in hparams.items()
        }
        try:
            self.writer.add_hparams(clean, dict(metrics))
        except Exception as exc:  # pragma: no cover
            self.logger.debug("add_hparams failed: %s", exc)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()
