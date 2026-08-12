"""Test-time augmentation.

Only *label-preserving* transforms belong in TTA, and for skin lesions the
dihedral group (flips and 90-degree rotations) is exactly that: a lesion has no
canonical orientation, so all eight symmetries are equally valid views of the same
lesion. That makes ``d4`` the natural maximal policy here, and it is unusually
effective — the eight views are genuinely decorrelated rather than near-duplicates.

Averaging happens in **logit space** by default. Averaging probabilities pulls the
mean toward 0.5 and flattens the very tails that per-class thresholding depends on;
averaging logits then applying one sigmoid preserves confident predictions and
composes cleanly with the threshold rescaling in ``validation/thresholds.py``.
"""

from __future__ import annotations

import torch

# Each policy is a list of (horizontal_flip, vertical_flip, n_rot90) view specs.
TTA_POLICIES: dict[str, list[tuple[bool, bool, int]]] = {
    # No augmentation: a single forward pass.
    "none": [(False, False, 0)],
    # Horizontal flip only: the cheap 2x.
    "hflip": [(False, False, 0), (True, False, 0)],
    # Four flips: usually captures most of the available gain at 4x cost.
    "flips": [(False, False, 0), (True, False, 0), (False, True, 0), (True, True, 0)],
    # Full dihedral group of order 8: the maximal label-preserving set.
    "d4": [
        (False, False, 0),
        (True, False, 0),
        (False, True, 0),
        (True, True, 0),
        (False, False, 1),
        (True, False, 1),
        (False, True, 1),
        (True, True, 1),
    ],
}


def apply_view(images: torch.Tensor, hflip: bool, vflip: bool, rot90: int) -> torch.Tensor:
    """Apply one dihedral view to a ``(B, C, H, W)`` batch."""
    out = images
    if hflip:
        out = torch.flip(out, dims=[-1])
    if vflip:
        out = torch.flip(out, dims=[-2])
    if rot90:
        out = torch.rot90(out, k=rot90, dims=[-2, -1])
    return out


@torch.no_grad()
def tta_predict(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    policy: str = "d4",
    average: str = "logit",
    amp_dtype: torch.dtype = torch.bfloat16,
    amp_enabled: bool = True,
) -> torch.Tensor:
    """Predict one batch under a TTA policy. Returns ``(B, num_classes)`` probabilities.

    Both images of a lesion receive the **same** view. They are separate
    photographs, so this is not strictly required, but keeping the pair consistent
    means the ensemble averages over 8 coherent views instead of 64 mismatched
    combinations — cheaper and empirically no worse.

    ``average`` is ``"logit"`` (default, recommended) or ``"prob"``.
    """
    if policy not in TTA_POLICIES:
        raise ValueError(f"Unknown TTA policy {policy!r}; expected one of {sorted(TTA_POLICIES)}")
    if average not in ("logit", "prob"):
        raise ValueError("average must be 'logit' or 'prob'")

    model.eval()
    accumulated: torch.Tensor | None = None
    views = TTA_POLICIES[policy]

    for hflip, vflip, rot90 in views:
        view_batch = dict(batch)
        for key in ("derm", "clin"):
            if key in batch:
                view_batch[key] = apply_view(batch[key], hflip, vflip, rot90).contiguous(
                    memory_format=torch.channels_last
                )

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits = model.forward_batch(view_batch).float()

        contribution = logits if average == "logit" else torch.sigmoid(logits)
        accumulated = contribution if accumulated is None else accumulated + contribution

    mean = accumulated / len(views)
    return torch.sigmoid(mean) if average == "logit" else mean
