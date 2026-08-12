"""Image encoder construction on top of ``timm``.

Every encoder is created with ``num_classes=0``, so ``forward`` returns a pooled
feature vector rather than logits. That is what lets the same encoder be dropped
into a single-image head, a two-image fusion module, or a metadata-fused model
without any changes.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from ..utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class ImageEncoder(nn.Module):
    """A pooled-feature image encoder wrapping a timm backbone.

    Parameters
    ----------
    name:
        timm model name, e.g. ``convnext_tiny.fb_in22k_ft_in1k``.
    pretrained:
        Load ImageNet weights. Transfer learning is essential here -- 5,240
        training lesions is far too few to learn general visual features from
        scratch, and the rare classes have double-digit sample counts.
    drop_rate:
        Classifier-level dropout inside the backbone (applies to the pooled
        features we consume).
    drop_path_rate:
        Stochastic depth. A modest value (0.1-0.3) is one of the more reliable
        regularisers for ConvNeXt/ViT on small datasets.
    in_chans:
        Input channels; 3 for RGB.
    """

    def __init__(
        self,
        name: str = "convnext_tiny.fb_in22k_ft_in1k",
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        in_chans: int = 3,
    ):
        super().__init__()
        self.name = name
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,          # -> pooled features, no classifier
            global_pool="avg",
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            in_chans=in_chans,
        )
        self.num_features: int = int(self.backbone.num_features)
        LOGGER.info(
            "Encoder %s | pretrained=%s | features=%d | drop_path=%.2f",
            name,
            pretrained,
            self.num_features,
            drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> ``(B, num_features)``."""
        return self.backbone(x)

    # -- transfer-learning utilities ---------------------------------------
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Trade compute for memory; lets us push image size or batch size up."""
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable)

    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_encoder(cfg, in_chans: int = 3) -> ImageEncoder:
    """Create an :class:`ImageEncoder` from the ``model`` section of a config."""
    return ImageEncoder(
        name=cfg.get("backbone", "convnext_tiny.fb_in22k_ft_in1k"),
        pretrained=bool(cfg.get("pretrained", True)),
        drop_rate=float(cfg.get("drop_rate", 0.0)),
        drop_path_rate=float(cfg.get("drop_path_rate", 0.1)),
        in_chans=in_chans,
    )
