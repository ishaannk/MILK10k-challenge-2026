"""Albumentations pipelines for clinical and dermoscopic lesion images.

Augmentation choices are driven by what is actually invariant for skin lesions:

* **Full dihedral symmetry.** A lesion has no canonical orientation, so
  horizontal/vertical flips and 90-degree rotations are all label-preserving and
  free capacity. This is the single highest-value augmentation for dermoscopy.
* **Colour jitter, used with restraint.** Colour genuinely carries diagnostic
  signal (pigment network, erythema, vascular blush), so we perturb it enough to
  cover camera/illumination variation without erasing the signal.
* **Scale/translate crops.** Lesion framing varies a lot between operators,
  especially in clinical close-ups, so ``RandomResizedCrop`` is applied more
  aggressively than one would use on, say, ImageNet-style object photos.
* **Coarse dropout** as a light occlusion prior, standing in for hair, rulers,
  ink markings and gel bubbles.

Clinical and dermoscopic images get *separate* pipelines: clinical photos vary far
more in framing and lighting, so they receive stronger geometric/photometric
jitter than dermoscopy, which is captured through a contact instrument.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from ..constants import IMAGENET_MEAN, IMAGENET_STD
from ..utils.config import Config


def _normalize_block(mean=IMAGENET_MEAN, std=IMAGENET_STD) -> list:
    """Final stage shared by every pipeline: scale, normalise, to CHW tensor."""
    return [A.Normalize(mean=mean, std=std, max_pixel_value=255.0), ToTensorV2()]


def build_train_transform(
    image_size: int,
    modality: str = "dermoscopic",
    strength: float = 1.0,
    scale: tuple[float, float] = (0.7, 1.0),
    ratio: tuple[float, float] = (0.8, 1.25),
    use_coarse_dropout: bool = True,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> A.Compose:
    """Training-time augmentation pipeline.

    Parameters
    ----------
    image_size:
        Output side length (square).
    modality:
        ``"clinical"`` or ``"dermoscopic"``; clinical gets stronger jitter.
    strength:
        Global multiplier on augmentation probabilities/magnitudes. ``0`` gives a
        near-deterministic resize, which is useful for debugging and for the
        final fine-tuning epochs.
    scale, ratio:
        ``RandomResizedCrop`` area and aspect-ratio ranges.
    """
    s = float(np.clip(strength, 0.0, 2.0))
    is_clinical = modality == "clinical"

    # Clinical photographs vary more in framing and white balance.
    colour_p = 0.7 * s if is_clinical else 0.5 * s
    brightness = (0.25 if is_clinical else 0.15) * s
    hue = (0.05 if is_clinical else 0.03) * s
    affine_p = 0.7 * s if is_clinical else 0.5 * s

    return A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size), scale=scale, ratio=ratio, p=1.0),
            # Dihedral group: lesions have no preferred orientation.
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(1 - 0.1 * s, 1 + 0.1 * s),
                translate_percent=(-0.06 * s, 0.06 * s),
                rotate=(-30 * s, 30 * s),
                shear=(-8 * s, 8 * s),
                border_mode=0,
                p=affine_p,
            ),
            A.OneOf(
                [
                    A.ColorJitter(brightness=brightness, contrast=brightness, saturation=brightness, hue=hue),
                    A.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=brightness),
                    A.HueSaturationValue(
                        hue_shift_limit=int(12 * s), sat_shift_limit=int(25 * s), val_shift_limit=int(20 * s)
                    ),
                ],
                p=colour_p,
            ),
            # Mild blur/noise: stands in for focus and sensor variation.
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.GaussNoise(std_range=(0.02, 0.1)), A.Sharpen()], p=0.2 * s),
            # Occlusion prior: hair, ruler marks, ink, gel bubbles.
            *(
                [
                    A.CoarseDropout(
                        num_holes_range=(1, 4),
                        hole_height_range=(0.05, 0.2),
                        hole_width_range=(0.05, 0.2),
                        p=0.25 * s,
                    )
                ]
                if use_coarse_dropout
                else []
            ),
            *_normalize_block(mean, std),
        ]
    )


def build_eval_transform(
    image_size: int,
    center_crop_pct: float = 1.0,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> A.Compose:
    """Deterministic validation/test pipeline.

    ``center_crop_pct < 1.0`` resizes to ``image_size / center_crop_pct`` then
    centre-crops, mirroring the train-time crop distribution's mean framing. At
    ``1.0`` (the default) it is a plain resize, which matches these already
    lesion-centred images well.
    """
    if center_crop_pct >= 1.0:
        stages = [A.Resize(height=image_size, width=image_size)]
    else:
        resized = int(round(image_size / center_crop_pct))
        stages = [
            A.Resize(height=resized, width=resized),
            A.CenterCrop(height=image_size, width=image_size),
        ]
    return A.Compose([*stages, *_normalize_block(mean, std)])


def build_transforms_from_config(cfg: Config, train: bool) -> dict[str, A.Compose]:
    """Build ``{"clinical": ..., "dermoscopic": ...}`` pipelines from a config.

    Returning both modalities unconditionally keeps the dataset code simple: it
    just indexes the dict by whichever modality it is about to load.
    """
    image_size = int(cfg.data.image_size)
    aug = cfg.get("augment", Config())
    mean = tuple(cfg.data.get("mean", IMAGENET_MEAN))
    std = tuple(cfg.data.get("std", IMAGENET_STD))

    if not train:
        transform = build_eval_transform(
            image_size, center_crop_pct=float(cfg.data.get("center_crop_pct", 1.0)), mean=mean, std=std
        )
        return {"clinical": transform, "dermoscopic": transform}

    return {
        modality: build_train_transform(
            image_size,
            modality=modality,
            strength=float(aug.get("strength", 1.0)),
            scale=tuple(aug.get("scale", (0.7, 1.0))),
            ratio=tuple(aug.get("ratio", (0.8, 1.25))),
            use_coarse_dropout=bool(aug.get("coarse_dropout", True)),
            mean=mean,
            std=std,
        )
        for modality in ("clinical", "dermoscopic")
    }
