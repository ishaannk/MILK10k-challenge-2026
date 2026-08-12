"""Tabular metadata engineering for MILK10k.

The raw metadata CSV has **one row per image** (two per lesion: clinical and
dermoscopic). The model predicts **one label per lesion**, so this module pivots
image-level rows into a single lesion-level feature vector.

Feature groups produced
-----------------------
=========================  ====  =========================================================
group                      dims  notes
=========================  ====  =========================================================
age                        2     scaled age + explicit missing indicator
sex                        3     one-hot: male / female / unknown
anatomical site            8     one-hot over a fixed vocabulary
skin tone class            7     one-hot over levels 0-5 + unknown
MONET concepts             14    7 concepts x 2 modalities, standardised
image manipulation         2     "altered" flag per modality
=========================  ====  =========================================================

Two design decisions worth flagging:

1. **Fixed vocabularies** (see ``constants.py``) rather than vocabularies learned
   from whichever split happens to be loaded. This guarantees the feature layout
   is identical at train and inference time even if a fold contains no ``genital``
   lesions.
2. **Continuous features are standardised using training statistics only**, which
   are persisted alongside the model. ``MetadataProcessor.fit`` must therefore be
   called on the training split, never on train+test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import (
    IMAGE_TYPE_CLINICAL,
    IMAGE_TYPE_DERMOSCOPIC,
    MONET_COLUMNS,
    SEX_VOCAB,
    SITE_VOCAB,
    SKIN_TONE_VOCAB,
)
from ..utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Age is recorded in 5-year intervals spanning roughly 0-90. Centre/scale are
# fixed constants rather than fitted values so the mapping is easy to reason about.
AGE_CENTER = 50.0
AGE_SCALE = 25.0


class MetadataProcessor:
    """Turn lesion-level metadata rows into a dense float32 feature matrix.

    Usage::

        proc = MetadataProcessor().fit(train_df)
        x_train = proc.transform(train_df)   # (n_lesions, proc.dim)
        x_valid = proc.transform(valid_df)   # same columns, train statistics
        proc.save("checkpoints/meta_processor.json")
    """

    def __init__(self, use_monet: bool = True, use_manipulation: bool = True):
        self.use_monet = use_monet
        self.use_manipulation = use_manipulation
        self.monet_mean: np.ndarray | None = None
        self.monet_std: np.ndarray | None = None
        self.feature_names: list[str] = []
        self._fitted = False

    # -- introspection ------------------------------------------------------
    @property
    def dim(self) -> int:
        """Number of output features. Available before ``fit`` via a dry run."""
        if not self.feature_names:
            self.feature_names = self._build_feature_names()
        return len(self.feature_names)

    def _monet_columns(self) -> list[str]:
        """MONET column names, suffixed per modality after the lesion pivot."""
        return [f"{c}_{m}" for m in ("clin", "derm") for c in MONET_COLUMNS]

    def _build_feature_names(self) -> list[str]:
        names = ["age_scaled", "age_missing"]
        names += [f"sex_{v}" for v in SEX_VOCAB]
        names += [f"site_{v}" for v in SITE_VOCAB]
        names += [f"tone_{v}" for v in SKIN_TONE_VOCAB]
        if self.use_monet:
            names += self._monet_columns()
        if self.use_manipulation:
            names += ["altered_clin", "altered_derm"]
        return names

    # -- fit / transform ----------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "MetadataProcessor":
        """Learn standardisation statistics from a *training* lesion table."""
        self.feature_names = self._build_feature_names()
        if self.use_monet:
            cols = self._monet_columns()
            values = df.reindex(columns=cols).to_numpy(dtype=np.float32, na_value=np.nan)
            self.monet_mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
            # Guard against a constant column producing a divide-by-zero.
            self.monet_std = np.where(std < 1e-6, 1.0, std)
        self._fitted = True
        LOGGER.info("MetadataProcessor fitted: %d features", self.dim)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Encode a lesion table into ``(n_lesions, dim)`` float32 features."""
        if not self._fitted:
            raise RuntimeError("Call fit() (or load()) before transform()")

        blocks: list[np.ndarray] = []

        # --- age: scaled value with NaNs mapped to the centre, plus a flag so
        # the network can learn that "missing" is itself informative.
        age = df.reindex(columns=["age_approx"]).to_numpy(dtype=np.float32, na_value=np.nan).ravel()
        age_missing = np.isnan(age).astype(np.float32)
        age_filled = np.where(np.isnan(age), AGE_CENTER, age)
        blocks.append(((age_filled - AGE_CENTER) / AGE_SCALE).reshape(-1, 1))
        blocks.append(age_missing.reshape(-1, 1))

        # --- categoricals over fixed vocabularies
        blocks.append(_one_hot(df.get("sex"), SEX_VOCAB, n=len(df)))
        blocks.append(_one_hot(df.get("site"), SITE_VOCAB, n=len(df)))
        tone = df.get("skin_tone_class")
        # Skin tone arrives as a float (3.0); normalise to the string keys "0".."5".
        tone_str = None if tone is None else tone.map(_tone_to_str)
        blocks.append(_one_hot(tone_str, SKIN_TONE_VOCAB, n=len(df)))

        # --- MONET concept scores, standardised with training statistics
        if self.use_monet:
            cols = self._monet_columns()
            values = df.reindex(columns=cols).to_numpy(dtype=np.float32, na_value=np.nan)
            values = np.where(np.isnan(values), self.monet_mean, values)
            blocks.append((values - self.monet_mean) / self.monet_std)

        # --- acquisition flag: some images were post-processed ("altered")
        if self.use_manipulation:
            for suffix in ("clin", "derm"):
                col = df.get(f"image_manipulation_{suffix}")
                flag = (
                    np.zeros(len(df), dtype=np.float32)
                    if col is None
                    else (col.astype("string").fillna("") == "altered").to_numpy(dtype=np.float32)
                )
                blocks.append(flag.reshape(-1, 1))

        features = np.concatenate(blocks, axis=1).astype(np.float32)
        if features.shape[1] != self.dim:  # pragma: no cover - internal invariant
            raise RuntimeError(f"Feature width {features.shape[1]} != expected {self.dim}")
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "use_monet": self.use_monet,
            "use_manipulation": self.use_manipulation,
            "monet_mean": None if self.monet_mean is None else self.monet_mean.tolist(),
            "monet_std": None if self.monet_std is None else self.monet_std.tolist(),
            "feature_names": self.feature_names,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MetadataProcessor":
        with open(path) as fh:
            payload = json.load(fh)
        proc = cls(use_monet=payload["use_monet"], use_manipulation=payload["use_manipulation"])
        proc.monet_mean = None if payload["monet_mean"] is None else np.array(payload["monet_mean"], dtype=np.float32)
        proc.monet_std = None if payload["monet_std"] is None else np.array(payload["monet_std"], dtype=np.float32)
        proc.feature_names = payload["feature_names"]
        proc._fitted = True
        return proc


def _tone_to_str(value) -> str:
    """Map a skin-tone cell to its vocabulary key, tolerating floats and NaN."""
    if pd.isna(value):
        return "unknown"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "unknown"


def _one_hot(series: pd.Series | None, vocab: list[str], n: int) -> np.ndarray:
    """One-hot encode ``series`` over ``vocab``; anything unseen -> ``unknown``."""
    out = np.zeros((n, len(vocab)), dtype=np.float32)
    index = {v: i for i, v in enumerate(vocab)}
    unknown = index.get("unknown", 0)
    if series is None:
        out[:, unknown] = 1.0
        return out
    for row, value in enumerate(series.to_numpy()):
        key = "unknown" if pd.isna(value) else str(value)
        out[row, index.get(key, unknown)] = 1.0
    return out


def build_lesion_table(metadata_csv: str | Path, ground_truth_csv: str | Path | None = None) -> pd.DataFrame:
    """Pivot the image-level metadata CSV into one row per lesion.

    Returns a dataframe indexed by position with, per lesion: the shared
    demographic columns, the per-modality image ids (``isic_id_clin`` /
    ``isic_id_derm``), per-modality MONET scores and manipulation flags, and — if
    ``ground_truth_csv`` is given — the 11 one-hot label columns plus a
    convenience integer ``label`` column.

    Lesions missing either modality are dropped with a warning; the released data
    has both for every lesion, but a silent half-pair would corrupt training.
    """
    meta = pd.read_csv(metadata_csv)

    modality_map = {IMAGE_TYPE_CLINICAL: "clin", IMAGE_TYPE_DERMOSCOPIC: "derm"}
    meta["modality"] = meta["image_type"].map(modality_map)
    unknown_types = meta.loc[meta["modality"].isna(), "image_type"].unique()
    if len(unknown_types):
        raise ValueError(f"Unrecognised image_type values: {list(unknown_types)}")

    # Columns that describe the lesion (identical across its two images) versus
    # columns that describe a single image (need a per-modality suffix).
    lesion_level = ["age_approx", "sex", "skin_tone_class", "site"]
    image_level = ["isic_id", "image_manipulation", *MONET_COLUMNS]

    base = meta.groupby("lesion_id", as_index=False)[lesion_level].first()

    wide = base
    for modality in ("clin", "derm"):
        subset = meta.loc[meta["modality"] == modality, ["lesion_id", *image_level]]
        subset = subset.drop_duplicates("lesion_id", keep="first")
        subset = subset.rename(columns={c: f"{c}_{modality}" for c in image_level})
        wide = wide.merge(subset, on="lesion_id", how="left")

    incomplete = wide["isic_id_clin"].isna() | wide["isic_id_derm"].isna()
    if incomplete.any():
        LOGGER.warning("Dropping %d lesion(s) without both modalities", int(incomplete.sum()))
        wide = wide.loc[~incomplete].reset_index(drop=True)

    if ground_truth_csv is not None:
        from ..constants import CLASSES

        truth = pd.read_csv(ground_truth_csv)
        wide = wide.merge(truth, on="lesion_id", how="inner")
        # Single-label data stored as one-hot: argmax gives a stratification key.
        wide["label"] = wide[CLASSES].to_numpy().argmax(axis=1)

    return wide.sort_values("lesion_id").reset_index(drop=True)
