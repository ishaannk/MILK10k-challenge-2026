"""Inference, model ensembling and submission-CSV generation.

The submission contract (fixed by the organisers):

* one row per lesion, no duplicates, every test lesion present;
* columns exactly ``lesion,AKIEC,BCC,BEN_OTH,BKL,DF,INF,MAL_OTH,MEL,NV,SCCKA,VASC``
  — note the id column is ``lesion``, *not* ``lesion_id``;
* float probabilities in ``[0, 1]``, binarised at ``>= 0.5`` for scoring.

:func:`write_submission` validates all of that before writing, because a
format-rejected submission costs a leaderboard slot for nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..constants import CLASSES, NUM_CLASSES, SUBMISSION_THRESHOLD
from ..datasets.metadata import MetadataProcessor
from ..models.lesion_net import build_model
from ..utils.checkpoint import load_checkpoint
from ..utils.config import Config
from ..utils.logging_utils import get_logger
from .tta import tta_predict

LOGGER = get_logger(__name__)


class Predictor:
    """Wrap one trained checkpoint for inference with optional TTA.

    Loads the EMA weights when the checkpoint has them (they are usually the
    better model, and they are what validation was scored on), and carries the
    tuned thresholds that were stored alongside them.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        cfg: Config | None = None,
        device: str = "cuda",
        use_ema: bool = True,
        meta_dim: int | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        # A checkpoint embeds the config it was trained with, so inference cannot
        # silently disagree with training about architecture or image size.
        self.cfg = cfg if cfg is not None else Config(ckpt["config"])
        self.thresholds = None if ckpt.get("thresholds") is None else np.asarray(ckpt["thresholds"], dtype=np.float64)

        if meta_dim is None:
            meta_dim = self._infer_meta_dim(ckpt)

        self.model = build_model(self.cfg, meta_dim=meta_dim)

        state = ckpt["ema"]["module"] if (use_ema and ckpt.get("ema")) else ckpt["model"]
        which = "EMA" if (use_ema and ckpt.get("ema")) else "raw"
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        if bool(self.cfg.train.get("channels_last", True)):
            self.model = self.model.to(memory_format=torch.channels_last)

        amp_cfg = str(self.cfg.train.get("amp", "bf16")).lower()
        self.amp_enabled = amp_cfg in ("fp16", "bf16", "true", "1")
        self.amp_dtype = torch.bfloat16 if amp_cfg == "bf16" else torch.float16

        LOGGER.info(
            "Predictor loaded %s (%s weights, epoch %s, val %s=%.4f)",
            self.checkpoint_path,
            which,
            ckpt.get("epoch"),
            self.cfg.train.get("monitor", "macro_f1"),
            float(ckpt.get("metrics", {}).get(self.cfg.train.get("monitor", "macro_f1"), float("nan"))),
        )

    @staticmethod
    def _infer_meta_dim(ckpt: dict) -> int:
        """Recover the metadata input width from the saved weights.

        Avoids having to re-derive the feature layout at inference time: the first
        MLP layer's shape already records it.
        """
        for key, tensor in ckpt["model"].items():
            if key.startswith("meta_encoder") and key.endswith("weight") and tensor.ndim == 2:
                return int(tensor.shape[1])
        return 0

    @torch.no_grad()
    def predict(self, loader: DataLoader, tta: str = "d4", average: str = "logit") -> tuple[np.ndarray, list[str]]:
        """Predict over a loader. Returns ``(probs, lesion_ids)``."""
        all_probs: list[np.ndarray] = []
        all_ids: list[str] = []

        for raw_batch in loader:
            batch = {
                key: (value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value)
                for key, value in raw_batch.items()
            }
            for key in ("derm", "clin"):
                if key in batch:
                    batch[key] = batch[key].contiguous(memory_format=torch.channels_last)

            probs = tta_predict(
                self.model,
                batch,
                policy=tta,
                average=average,
                amp_dtype=self.amp_dtype,
                amp_enabled=self.amp_enabled,
            )
            all_probs.append(probs.cpu().numpy())
            all_ids.extend(raw_batch["lesion_id"])

        return np.concatenate(all_probs, axis=0), all_ids


def ensemble_probabilities(
    predictions: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
    method: str = "mean",
) -> np.ndarray:
    """Combine several models' probability matrices.

    ``method``:

    ``mean``
        Weighted arithmetic mean. Robust default.
    ``gmean``
        Weighted geometric mean. Sharper than the arithmetic mean (a single model's
        near-zero drags the ensemble down), which often helps precision on the
        common classes.
    ``rank``
        Average of per-class rank-normalised scores. Immune to differences in
        calibration between models, which makes it superficially attractive for
        heterogeneous backbones. **Use with care on this challenge.** Ranks are
        uniform by construction, so applying a threshold fitted on OOF ranks forces
        each class's predicted-positive *rate* on the test set to equal its OOF rate
        — i.e. it silently imposes the training prior. The MILK10k test set is
        measurably prior-shifted (enriched in SCCKA/AKIEC, depleted in BCC), and
        rank blending predicted SCCKA for 45 lesions where ~102 are expected and BCC
        for 246 where ~146 are expected: total deviation 247 from estimated test
        counts, against 59 for ``mean``. Prefer ``mean`` unless the test prior is
        known to match training.
    """
    stacked = np.stack([np.asarray(p, dtype=np.float64) for p in predictions], axis=0)
    if stacked.ndim != 3:
        raise ValueError("Each prediction must be a 2-D (n_samples, n_classes) array")

    n_models = stacked.shape[0]
    w = np.ones(n_models) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != n_models:
        raise ValueError(f"Got {len(w)} weights for {n_models} models")
    w = w / w.sum()

    if method == "mean":
        return np.tensordot(w, stacked, axes=(0, 0))
    if method == "gmean":
        logs = np.log(np.clip(stacked, 1e-7, 1.0))
        return np.exp(np.tensordot(w, logs, axes=(0, 0)))
    if method == "rank":
        from scipy.stats import rankdata

        ranked = np.stack(
            [rankdata(model, axis=0) / model.shape[0] for model in stacked],
            axis=0,
        )
        return np.tensordot(w, ranked, axes=(0, 0))
    raise ValueError(f"Unknown ensemble method {method!r} (mean/gmean/rank)")


def build_submission_frame(lesion_ids: Sequence[str], probs: np.ndarray) -> pd.DataFrame:
    """Assemble the submission dataframe in the exact required column order."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[1] != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} columns of probabilities, got {probs.shape[1]}")
    if len(lesion_ids) != len(probs):
        raise ValueError(f"{len(lesion_ids)} lesion ids vs {len(probs)} prediction rows")

    frame = pd.DataFrame(probs, columns=CLASSES)
    frame.insert(0, "lesion", list(lesion_ids))
    return frame[["lesion", *CLASSES]]


def validate_submission(frame: pd.DataFrame, expected_ids: Sequence[str] | None = None) -> list[str]:
    """Check a submission frame against the challenge contract.

    Returns a list of human-readable problems; empty means the file is valid.
    Errors are collected rather than raised one at a time so a single run surfaces
    every issue.
    """
    problems: list[str] = []

    if list(frame.columns) != ["lesion", *CLASSES]:
        problems.append(f"Column order must be {['lesion', *CLASSES]}, got {list(frame.columns)}")

    if frame["lesion"].duplicated().any():
        duplicates = frame.loc[frame["lesion"].duplicated(), "lesion"].unique()[:5]
        problems.append(f"Duplicate lesion ids, e.g. {list(duplicates)}")

    values = frame[CLASSES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        problems.append("Non-finite probabilities present (NaN or inf)")
    if values.min() < 0.0 or values.max() > 1.0:
        problems.append(f"Probabilities out of [0, 1]: min={values.min():.4f} max={values.max():.4f}")

    if expected_ids is not None:
        missing = set(map(str, expected_ids)) - set(frame["lesion"].astype(str))
        extra = set(frame["lesion"].astype(str)) - set(map(str, expected_ids))
        if missing:
            problems.append(f"{len(missing)} expected lesion(s) missing, e.g. {sorted(missing)[:5]}")
        if extra:
            problems.append(f"{len(extra)} unexpected lesion(s) present, e.g. {sorted(extra)[:5]}")

    # Not a hard error, but a lesion with no class above 0.5 contributes to no
    # class's true positives and is almost always a wasted row.
    empty_rows = int((values < SUBMISSION_THRESHOLD).all(axis=1).sum())
    if empty_rows:
        problems.append(
            f"NOTE: {empty_rows}/{len(frame)} lesion(s) have no class >= {SUBMISSION_THRESHOLD} "
            "(they cannot earn credit for any class)"
        )

    return problems


def write_submission(
    path: str | Path,
    lesion_ids: Sequence[str],
    probs: np.ndarray,
    expected_ids: Sequence[str] | None = None,
    float_format: str = "%.6f",
) -> pd.DataFrame:
    """Validate then write the submission CSV. Returns the frame that was written."""
    frame = build_submission_frame(lesion_ids, probs)

    problems = validate_submission(frame, expected_ids)
    hard_errors = [p for p in problems if not p.startswith("NOTE:")]
    for problem in problems:
        (LOGGER.error if not problem.startswith("NOTE:") else LOGGER.warning)("Submission check: %s", problem)
    if hard_errors:
        raise ValueError(f"Submission failed validation: {hard_errors}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format=float_format)

    positives = (frame[CLASSES].to_numpy() >= SUBMISSION_THRESHOLD).sum(axis=0)
    LOGGER.info("Wrote %s | %d lesions", path, len(frame))
    LOGGER.info(
        "Positives per class at >=0.5: %s",
        json.dumps(dict(zip(CLASSES, positives.tolist()))),
    )
    return frame


def load_meta_processor(path: str | Path | None) -> MetadataProcessor | None:
    """Load a persisted :class:`MetadataProcessor`, or ``None`` if absent."""
    if path is None or not Path(path).exists():
        return None
    return MetadataProcessor.load(path)
