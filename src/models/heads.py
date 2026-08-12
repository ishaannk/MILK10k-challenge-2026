"""Metadata encoder, multimodal fusion blocks and the classification head."""

from __future__ import annotations

import torch
import torch.nn as nn


class MetadataMLP(nn.Module):
    """Encode the tabular feature vector into a dense embedding.

    Kept deliberately small (two hidden layers, heavy dropout). The metadata is
    ~36 low-cardinality features against 5,240 training lesions, so a large MLP
    memorises rather than generalises. ``BatchNorm`` before the activation keeps
    the mixed one-hot/standardised inputs on a comparable scale.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        out_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers += [
                nn.Linear(prev, hidden),
                nn.BatchNorm1d(hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = hidden
        layers += [nn.Linear(prev, out_dim), nn.BatchNorm1d(out_dim), nn.SiLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConcatFusion(nn.Module):
    """Concatenate modality features and project them down.

    The straightforward baseline, and a strong one: it preserves every input
    dimension and lets the projection learn any linear mixing it wants.
    """

    def __init__(self, dims: list[int], out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(sum(dims)),
            nn.Dropout(dropout),
            nn.Linear(sum(dims), out_dim),
            nn.SiLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(features, dim=1))


class GatedFusion(nn.Module):
    """Learn a per-dimension gate that mixes modalities adaptively.

    Motivation: which modality is informative is lesion-dependent. Dermoscopy
    dominates for pigmented lesions, while clinical close-ups carry more signal
    for large, ulcerated or inflammatory presentations. A gate lets the network
    make that call per example instead of committing to a fixed weighting.

    All inputs are first projected to a common width so the gate can act
    element-wise, then combined as a convex mixture whose weights sum to one
    across modalities.
    """

    def __init__(self, dims: list[int], out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim)) for d in dims]
        )
        self.gate = nn.Sequential(
            nn.Linear(sum(dims), out_dim * len(dims)),
        )
        self.n_modalities = len(dims)
        self.out_dim = out_dim
        self.post = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.SiLU(inplace=True))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        projected = torch.stack([proj(f) for proj, f in zip(self.projections, features)], dim=1)
        # (B, n_modalities, out_dim) softmax weights over modalities
        weights = self.gate(torch.cat(features, dim=1))
        weights = weights.view(-1, self.n_modalities, self.out_dim).softmax(dim=1)
        return self.post((projected * weights).sum(dim=1))


class AttentionFusion(nn.Module):
    """Treat each modality as a token and fuse with self-attention.

    Scales naturally to more than two modalities (clinical + dermoscopy +
    metadata) and lets modalities condition on one another rather than only being
    summed. A learned ``[CLS]`` token reads out the fused representation.
    """

    def __init__(self, dims: list[int], out_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim)) for d in dims]
        )
        # Learned modality embeddings so attention can tell the tokens apart.
        self.modality_embed = nn.Parameter(torch.zeros(1, len(dims), out_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.norm = nn.LayerNorm(out_dim)
        self.attn = nn.MultiheadAttention(out_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack([proj(f) for proj, f in zip(self.projections, features)], dim=1)
        tokens = tokens + self.modality_embed
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        normed = self.norm(sequence)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        sequence = sequence + attended
        sequence = sequence + self.ffn(sequence)
        return sequence[:, 0]  # the [CLS] readout


def build_fusion(kind: str, dims: list[int], out_dim: int, dropout: float = 0.2) -> nn.Module:
    """Fusion-module factory. ``kind`` is one of concat/gated/attention."""
    kind = kind.lower()
    if kind == "concat":
        return ConcatFusion(dims, out_dim, dropout)
    if kind == "gated":
        return GatedFusion(dims, out_dim, dropout)
    if kind == "attention":
        return AttentionFusion(dims, out_dim, dropout=dropout)
    raise ValueError(f"Unknown fusion kind: {kind!r} (expected concat/gated/attention)")


class ClassifierHead(nn.Module):
    """Final MLP producing the 11 raw logits.

    ``hidden_dim=0`` gives a plain linear probe on the fused features, which is
    the right default when the fusion module already does the mixing.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 0, dropout: float = 0.3):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Dropout(dropout),
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
