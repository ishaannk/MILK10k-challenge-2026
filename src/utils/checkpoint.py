"""Checkpoint saving/loading and early stopping.

A checkpoint carries everything needed to *resume* (model, optimiser, scheduler,
AMP scaler, epoch, RNG state) as well as everything needed to *reproduce an
evaluation* (the config and the tuned per-class thresholds). Keeping thresholds
inside the checkpoint means inference never has to guess how the model was
calibrated.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .logging_utils import get_logger

LOGGER = get_logger(__name__)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    config: dict | None = None,
    thresholds: np.ndarray | None = None,
    ema_state: dict | None = None,
    save_rng: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config,
        "thresholds": None if thresholds is None else np.asarray(thresholds).tolist(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler"] = scaler.state_dict()
    if ema_state is not None:
        payload["ema"] = ema_state
    if save_rng:
        payload["rng"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    # Write to a temp file then move, so an interrupted save cannot corrupt the
    # previous best checkpoint.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str = "cpu",
    strict: bool = True,
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring training state in place."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        state = ckpt["model"]
        # Tolerate checkpoints saved from DataParallel/DDP wrappers.
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing:
            LOGGER.warning("Missing keys when loading %s: %s", path, missing)
        if unexpected:
            LOGGER.warning("Unexpected keys when loading %s: %s", path, unexpected)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    if restore_rng and ckpt.get("rng"):
        rng = ckpt["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

    return ckpt


class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of epochs without improvement to tolerate before stopping.
    mode:
        ``"max"`` for metrics like F1/AUC, ``"min"`` for losses.
    min_delta:
        Improvements smaller than this are treated as noise, not progress.
    """

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Register the epoch's metric. Returns ``True`` if it is a new best."""
        if self._is_better(value):
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

    @property
    def status(self) -> str:
        best = "n/a" if self.best is None else f"{self.best:.4f}"
        return f"best={best} @epoch {self.best_epoch} (no improvement for {self.counter}/{self.patience})"
