"""The MILK10k classifier, covering all three stages of the progression.

Rather than three unrelated classes, the three stages are three configurations of
one model. That keeps the training loop, checkpoint format and inference path
identical across stages, so a stage-to-stage comparison isolates the architecture
change instead of confounding it with pipeline differences.

===========  =====================  ==============  =============  ===================
stage        config                 clinical img    metadata       fusion
===========  =====================  ==============  =============  ===================
1 baseline   ``stage1_derm.yaml``   no              no             none (single tower)
2 dual       ``stage2_dual.yaml``   yes             no             concat/gated/attn
3 + meta     ``stage3_meta.yaml``   yes             yes            concat/gated/attn
===========  =====================  ==============  =============  ===================
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import NUM_CLASSES
from ..utils.config import Config
from ..utils.logging_utils import get_logger
from .backbones import ImageEncoder, build_encoder
from .heads import ClassifierHead, MetadataMLP, build_fusion

LOGGER = get_logger(__name__)


class LesionNet(nn.Module):
    """Configurable single- or dual-encoder lesion classifier with metadata fusion.

    Parameters
    ----------
    backbone:
        timm backbone name for the dermoscopy tower.
    clinical_backbone:
        Backbone for the clinical tower. Defaults to ``backbone``. Ignored when
        ``share_encoder`` is true.
    use_clinical:
        Add the second (clinical) image tower.
    use_metadata:
        Add the metadata MLP as an extra fusion input.
    share_encoder:
        Use one set of encoder weights for both modalities. Halves parameters and
        acts as a regulariser, but the two modalities have genuinely different
        low-level statistics (contact dermoscopy vs. handheld photo), so separate
        towers usually win when data allows. Exposed as a config switch to test.
    fusion:
        ``concat`` | ``gated`` | ``attention``.
    outputs:
        Raw logits of shape ``(B, 11)``. Loss is ``BCEWithLogitsLoss``, so no
        activation is applied here; ``predict_proba`` applies the sigmoid.
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny.fb_in22k_ft_in1k",
        clinical_backbone: str | None = None,
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        use_clinical: bool = False,
        use_metadata: bool = False,
        meta_dim: int = 0,
        meta_hidden: tuple[int, ...] = (128, 128),
        meta_embed_dim: int = 128,
        meta_dropout: float = 0.3,
        fusion: str = "concat",
        fusion_dim: int = 512,
        head_hidden: int = 0,
        head_dropout: float = 0.3,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        share_encoder: bool = False,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_clinical = use_clinical
        self.use_metadata = use_metadata
        self.share_encoder = share_encoder
        self.num_classes = num_classes

        # --- dermoscopy tower (always present; it is the stronger modality)
        self.derm_encoder = ImageEncoder(
            backbone, pretrained=pretrained, drop_rate=drop_rate, drop_path_rate=drop_path_rate
        )
        feature_dims = [self.derm_encoder.num_features]

        # --- clinical tower
        self.clin_encoder: ImageEncoder | None = None
        if use_clinical:
            if share_encoder:
                self.clin_encoder = self.derm_encoder  # tied weights
            else:
                self.clin_encoder = ImageEncoder(
                    clinical_backbone or backbone,
                    pretrained=pretrained,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                )
            feature_dims.append(self.clin_encoder.num_features)

        # --- metadata tower
        self.meta_encoder: MetadataMLP | None = None
        if use_metadata:
            if meta_dim <= 0:
                raise ValueError("use_metadata=True requires meta_dim > 0")
            self.meta_encoder = MetadataMLP(
                in_dim=meta_dim,
                hidden_dims=tuple(meta_hidden),
                out_dim=meta_embed_dim,
                dropout=meta_dropout,
            )
            feature_dims.append(self.meta_encoder.out_dim)

        # --- fusion: skipped entirely when there is only one input stream, so the
        # Stage 1 baseline stays a clean encoder -> head model with no extra layers.
        if len(feature_dims) > 1:
            self.fusion: nn.Module | None = build_fusion(fusion, feature_dims, fusion_dim, dropout=head_dropout)
            head_in = fusion_dim
        else:
            self.fusion = None
            head_in = feature_dims[0]

        self.head = ClassifierHead(head_in, num_classes, hidden_dim=head_hidden, dropout=head_dropout)

        if grad_checkpointing:
            self.derm_encoder.set_grad_checkpointing(True)
            if self.clin_encoder is not None and not share_encoder:
                self.clin_encoder.set_grad_checkpointing(True)

        LOGGER.info(
            "LesionNet | clinical=%s metadata=%s fusion=%s shared=%s params=%.1fM",
            use_clinical,
            use_metadata,
            fusion if self.fusion is not None else "none",
            share_encoder,
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        derm: torch.Tensor,
        clin: torch.Tensor | None = None,
        meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = [self.derm_encoder(derm)]

        if self.use_clinical:
            if clin is None:
                raise ValueError("Model was configured with use_clinical=True but no clinical image was given")
            features.append(self.clin_encoder(clin))

        if self.use_metadata:
            if meta is None:
                raise ValueError("Model was configured with use_metadata=True but no metadata was given")
            features.append(self.meta_encoder(meta))

        fused = features[0] if self.fusion is None else self.fusion(features)
        return self.head(fused)

    def forward_batch(self, batch: dict) -> torch.Tensor:
        """Convenience wrapper: pull the tensors this model needs out of a batch."""
        return self.forward(
            derm=batch["derm"],
            clin=batch.get("clin") if self.use_clinical else None,
            meta=batch.get("meta") if self.use_metadata else None,
        )

    @torch.no_grad()
    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """Per-class independent probabilities via sigmoid (matches BCE training)."""
        return torch.sigmoid(self.forward(*args, **kwargs))

    # -- parameter groups ---------------------------------------------------
    def param_groups(self, backbone_lr: float, head_lr: float, weight_decay: float = 0.05) -> list[dict]:
        """Split parameters into backbone / head groups with separate LRs.

        Two things are happening here:

        1. **Discriminative learning rates.** Pretrained encoders want a small LR
           so transferred features are refined rather than destroyed; the randomly
           initialised fusion and head want a much larger one. ``head_lr`` is
           typically 10x ``backbone_lr``.
        2. **No weight decay on norms and biases.** Decaying BatchNorm/LayerNorm
           scales and biases measurably hurts; it is standard practice to exclude
           them, and it matters more on small datasets.
        """
        encoder_params = {"decay": [], "no_decay": []}
        head_params = {"decay": [], "no_decay": []}

        encoder_prefixes = ("derm_encoder", "clin_encoder")
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            bucket = encoder_params if name.startswith(encoder_prefixes) else head_params
            # 1-D params are biases and norm weights.
            key = "no_decay" if param.ndim <= 1 or name.endswith(".bias") else "decay"
            bucket[key].append(param)

        groups = [
            {"params": encoder_params["decay"], "lr": backbone_lr, "weight_decay": weight_decay, "name": "encoder"},
            {"params": encoder_params["no_decay"], "lr": backbone_lr, "weight_decay": 0.0, "name": "encoder_no_decay"},
            {"params": head_params["decay"], "lr": head_lr, "weight_decay": weight_decay, "name": "head"},
            {"params": head_params["no_decay"], "lr": head_lr, "weight_decay": 0.0, "name": "head_no_decay"},
        ]
        return [g for g in groups if g["params"]]

    def freeze_encoders(self) -> None:
        """Freeze image towers -- used for an optional warmup epoch."""
        self.derm_encoder.freeze()
        if self.clin_encoder is not None:
            self.clin_encoder.freeze()

    def unfreeze_encoders(self) -> None:
        self.derm_encoder.unfreeze()
        if self.clin_encoder is not None:
            self.clin_encoder.unfreeze()


def build_model(cfg: Config, meta_dim: int = 0) -> LesionNet:
    """Instantiate :class:`LesionNet` from the ``model`` section of a config."""
    model_cfg = cfg.model
    return LesionNet(
        backbone=model_cfg.get("backbone", "convnext_tiny.fb_in22k_ft_in1k"),
        clinical_backbone=model_cfg.get("clinical_backbone"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        num_classes=int(cfg.get("num_classes", NUM_CLASSES)),
        use_clinical=bool(model_cfg.get("use_clinical", False)),
        use_metadata=bool(model_cfg.get("use_metadata", False)),
        meta_dim=meta_dim,
        meta_hidden=tuple(model_cfg.get("meta_hidden", (128, 128))),
        meta_embed_dim=int(model_cfg.get("meta_embed_dim", 128)),
        meta_dropout=float(model_cfg.get("meta_dropout", 0.3)),
        fusion=model_cfg.get("fusion", "concat"),
        fusion_dim=int(model_cfg.get("fusion_dim", 512)),
        head_hidden=int(model_cfg.get("head_hidden", 0)),
        head_dropout=float(model_cfg.get("head_dropout", 0.3)),
        drop_rate=float(model_cfg.get("drop_rate", 0.0)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.1)),
        share_encoder=bool(model_cfg.get("share_encoder", False)),
        grad_checkpointing=bool(model_cfg.get("grad_checkpointing", False)),
    )
