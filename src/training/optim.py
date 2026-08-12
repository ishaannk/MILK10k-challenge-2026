"""Optimiser and learning-rate schedule construction.

The schedule is stepped **per iteration**, not per epoch. Epochs here are short
(5,240 lesions), so an epoch-granular cosine would only have a couple of dozen
distinct learning rates; per-iteration stepping gives a smooth decay and makes
the warmup actually do its job.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from ..utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_optimizer(model: torch.nn.Module, cfg) -> Optimizer:
    """Build AdamW over discriminative parameter groups.

    AdamW (decoupled weight decay) is the right default for fine-tuning modern
    pretrained backbones: ConvNeXt and ViT are both trained with it upstream, so
    keeping the optimiser family consistent avoids surprises.
    """
    optim_cfg = cfg.optim
    backbone_lr = float(optim_cfg.get("backbone_lr", optim_cfg.get("lr", 1e-4)))
    head_lr = float(optim_cfg.get("head_lr", backbone_lr * 10))
    weight_decay = float(optim_cfg.get("weight_decay", 0.05))

    if hasattr(model, "param_groups"):
        groups = model.param_groups(backbone_lr, head_lr, weight_decay)
    else:  # pragma: no cover - fallback for plain nn.Module
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": backbone_lr}]

    name = str(optim_cfg.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"Only 'adamw' is supported, got {name!r}")

    optimizer = AdamW(
        groups,
        lr=backbone_lr,
        betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
        eps=float(optim_cfg.get("eps", 1e-8)),
        weight_decay=weight_decay,
    )
    LOGGER.info(
        "AdamW | backbone_lr=%.2e head_lr=%.2e wd=%.3f | groups=%s",
        backbone_lr,
        head_lr,
        weight_decay,
        [(g.get("name", "?"), len(g["params"])) for g in groups],
    )
    return optimizer


def build_scheduler(optimizer: Optimizer, cfg, steps_per_epoch: int) -> tuple[Any, str]:
    """Build the LR schedule.

    Returns ``(scheduler, interval)`` where ``interval`` is ``"step"`` or
    ``"epoch"`` so the trainer knows when to call ``.step()``.

    ``cosine`` (the default) decays from the base LR to ``min_lr_ratio * base_lr``
    over the whole run, after a short linear warmup. Warmup matters when the head
    is randomly initialised at 10x the backbone LR: without it, the first few
    steps can push large gradients back into the pretrained encoder.
    """
    sched_cfg = cfg.get("scheduler", {})
    name = str(sched_cfg.get("name", "cosine")).lower()
    epochs = int(cfg.train.epochs)
    total_steps = max(1, epochs * steps_per_epoch)

    warmup_epochs = float(sched_cfg.get("warmup_epochs", 1.0))
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))

    if name == "none":
        return None, "epoch"

    if name == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup; +1 avoids a exactly-zero LR on the first step.
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        LOGGER.info(
            "Cosine schedule | total_steps=%d warmup_steps=%d min_lr_ratio=%.3f",
            total_steps,
            warmup_steps,
            min_lr_ratio,
        )
        return LambdaLR(optimizer, lr_lambda), "step"

    if name == "cosine_restarts":
        first_cycle = int(sched_cfg.get("first_cycle_epochs", max(1, epochs // 3))) * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, first_cycle),
            T_mult=int(sched_cfg.get("cycle_mult", 2)),
            eta_min=float(sched_cfg.get("eta_min", 1e-7)),
        )
        return scheduler, "step"

    if name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(sched_cfg.get("factor", 0.3)),
            patience=int(sched_cfg.get("patience", 3)),
            min_lr=float(sched_cfg.get("min_lr", 1e-7)),
        )
        return scheduler, "epoch_metric"

    raise ValueError(f"Unknown scheduler: {name!r}")


def current_lrs(optimizer: Optimizer) -> dict[str, float]:
    """Snapshot of each group's LR, for logging."""
    out = {}
    for i, group in enumerate(optimizer.param_groups):
        out[f"lr/{group.get('name', f'group{i}')}"] = float(group["lr"])
    return out
