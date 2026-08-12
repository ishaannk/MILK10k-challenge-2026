"""Losses and label-level regularisation for the 11-way MILK10k task.

The challenge frames the task as **11 independent probabilities per lesion**, and
scores macro-F1 after thresholding each at 0.5. That framing is what motivates
``BCEWithLogitsLoss`` over cross-entropy even though the ground truth happens to
be one-hot: BCE optimises each class's own decision boundary rather than a
competition between classes, which is exactly what a per-class threshold metric
rewards. It also leaves room to predict two classes for one lesion, which — under
macro-F1 with very rare classes — is often the score-maximising move.

Class imbalance is severe (BCC 2,522 lesions vs. MAL_OTH 9), so this module
provides positive-class reweighting with several sane schemes, plus focal
weighting for the hard/rare tail.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_pos_weight(
    class_counts: np.ndarray,
    scheme: str = "sqrt_inverse",
    power: float = 0.5,
    clip: float = 20.0,
    beta: float = 0.9999,
) -> torch.Tensor | None:
    """Per-class positive weight for ``BCEWithLogitsLoss``.

    Schemes
    -------
    ``none``
        No reweighting; rely on the balanced sampler instead.
    ``balanced``
        Textbook ``(N - n_c) / n_c``. Correct in principle, but for MAL_OTH that
        is ~580, which makes the loss surface violent and tends to produce a
        model that screams MAL_OTH at everything. Included for completeness.
    ``sqrt_inverse`` (default)
        ``((N - n_c) / n_c) ** power`` with ``power=0.5``. A damped version of
        ``balanced``: it lifts the rare classes without letting nine lesions
        dominate the gradient.
    ``effective``
        Class-balanced weighting of Cui et al. (2019) using an effective sample
        count ``(1 - beta^n) / (1 - beta)``.

    ``clip`` bounds the result, which is the practical safeguard that keeps the
    rarest class from destabilising training.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    counts = np.where(counts <= 0, 1.0, counts)
    total = counts.sum()

    if scheme == "none":
        return None
    if scheme == "balanced":
        weights = (total - counts) / counts
    elif scheme == "sqrt_inverse":
        weights = ((total - counts) / counts) ** float(power)
    elif scheme == "effective":
        effective = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        weights = effective.max() / effective
    else:
        raise ValueError(f"Unknown pos_weight scheme: {scheme!r}")

    weights = np.clip(weights, 1.0, float(clip))
    LOGGER.info("pos_weight (%s): %s", scheme, np.round(weights, 2).tolist())
    return torch.tensor(weights, dtype=torch.float32)


def smooth_targets(targets: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    """Label smoothing for multi-label targets.

    Pulls positives to ``1 - eps + eps/C`` and negatives to ``eps/C``. Prevents the
    logits from running away, which both calibrates probabilities (important,
    because we threshold them) and mildly regularises a small dataset.
    """
    if epsilon <= 0:
        return targets
    num_classes = targets.size(-1)
    return targets * (1.0 - epsilon) + epsilon / num_classes


class BCEWithLogitsLossWrapper(nn.Module):
    """``BCEWithLogitsLoss`` with label smoothing and optional per-class weights."""

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        class_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None, persistent=False)
        self.register_buffer("class_weight", class_weight if class_weight is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = smooth_targets(targets.float(), self.label_smoothing)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        if self.class_weight is not None:
            loss = loss * self.class_weight
        return loss.mean()


class FocalBCELoss(nn.Module):
    """Focal loss on top of BCE, for when the tail classes need extra help.

    Down-weights already-confident predictions by ``(1 - p_t) ** gamma`` so the
    gradient concentrates on hard examples. On a set where 48% of lesions are BCC,
    a lot of the loss budget is otherwise spent on easy BCCs. ``gamma=2`` is the
    usual starting point; ``gamma=0`` recovers plain (weighted) BCE.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = 0.25,
        pos_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = smooth_targets(targets.float(), self.label_smoothing)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        # p_t is the probability assigned to the true label of each cell.
        p_t = probs * targets + (1 - probs) * (1 - targets)
        modulation = (1.0 - p_t).clamp_min(1e-6) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            modulation = modulation * alpha_t
        return (bce * modulation).mean()


def build_loss(cfg, class_counts: np.ndarray | None = None) -> nn.Module:
    """Loss factory driven by the ``loss`` section of a config."""
    loss_cfg = cfg.get("loss", {})
    name = str(loss_cfg.get("name", "bce")).lower()

    pos_weight = None
    if class_counts is not None:
        pos_weight = build_pos_weight(
            class_counts,
            scheme=str(loss_cfg.get("pos_weight", "sqrt_inverse")),
            power=float(loss_cfg.get("pos_weight_power", 0.5)),
            clip=float(loss_cfg.get("pos_weight_clip", 20.0)),
            beta=float(loss_cfg.get("pos_weight_beta", 0.9999)),
        )

    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))

    if name == "bce":
        return BCEWithLogitsLossWrapper(pos_weight=pos_weight, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalBCELoss(
            gamma=float(loss_cfg.get("gamma", 2.0)),
            alpha=loss_cfg.get("alpha", 0.25),
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Unknown loss name: {name!r} (expected bce/focal)")


# ---------------------------------------------------------------------------
# Mixup / CutMix
# ---------------------------------------------------------------------------
def mixup_batch(
    images: dict[str, torch.Tensor],
    targets: torch.Tensor,
    alpha: float = 0.4,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Mixup for paired images with soft BCE targets.

    Both images of a lesion are mixed against the **same** partner lesion with the
    **same** lambda, so the pair stays coherent: mixing the dermoscopy of lesion A
    with the clinical photo of lesion B would teach the fusion module nonsense.

    Soft targets are a natural fit for BCE, and mixup is one of the few
    regularisers that reliably helps at this dataset size.
    """
    if alpha <= 0:
        return images, targets

    batch_size = targets.size(0)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(batch_size, generator=generator, device=targets.device)

    mixed = {key: lam * value + (1.0 - lam) * value[perm] for key, value in images.items()}
    mixed_targets = lam * targets + (1.0 - lam) * targets[perm]
    return mixed, mixed_targets


def cutmix_batch(
    images: dict[str, torch.Tensor],
    targets: torch.Tensor,
    alpha: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """CutMix for paired images, pasting the same relative box into both views."""
    if alpha <= 0:
        return images, targets

    batch_size = targets.size(0)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(batch_size, generator=generator, device=targets.device)

    out = {}
    actual_lam = lam
    for key, value in images.items():
        _, _, height, width = value.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
        cy, cx = np.random.randint(height), np.random.randint(width)
        y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, height)
        x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, width)

        mixed = value.clone()
        mixed[:, :, y1:y2, x1:x2] = value[perm][:, :, y1:y2, x1:x2]
        out[key] = mixed
        # Recompute lambda from the box actually pasted (clipping changes the area).
        actual_lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))

    return out, actual_lam * targets + (1.0 - actual_lam) * targets[perm]
