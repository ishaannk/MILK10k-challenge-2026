#!/usr/bin/env python
"""Build lesion-level tables, cross-validation folds and a dataset report.

Run this once before training::

    python scripts/prepare_data.py --config configs/base.yaml

Outputs (under ``data/processed/``):

``train_lesions.csv`` / ``test_lesions.csv``
    One row per lesion: demographics, the two image ids, per-modality MONET
    scores, and (train only) the 11 one-hot label columns.
``folds.csv``
    ``lesion_id, label, fold`` -- a stratified k-fold assignment.
``dataset_report.json``
    Class counts, per-fold class counts, missing-value tallies and image checks.

Two things this script deliberately verifies rather than assumes:

* **every referenced image file exists** (``--check-images``), because a missing
  JPEG surfaces as a crash 20 minutes into epoch 1 otherwise;
* **every lesion has both modalities**, since a half-pair would silently corrupt
  the dual-encoder stages.

On stratification: folds are stratified on the argmax class so that each fold sees
a proportional share of the rare classes. With MAL_OTH at 9 lesions, 5 folds means
1-2 per fold; that is unavoidable, but stratifying at least guarantees no fold gets
zero. Splitting is at the lesion level and each lesion's two images always travel
together, so there is no cross-split image leakage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path side effect)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.constants import CLASSES
from src.datasets.metadata import build_lesion_table
from src.utils.config import load_config
from src.utils.logging_utils import get_logger, setup_logging

LOGGER = get_logger("prepare_data")


def make_folds(frame: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Assign a stratified fold index to every lesion."""
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(frame), -1, dtype=int)
    for fold, (_, valid_idx) in enumerate(splitter.split(frame, frame["label"])):
        folds[valid_idx] = fold
    if (folds < 0).any():  # pragma: no cover - invariant
        raise RuntimeError("Some lesions were not assigned a fold")
    return pd.DataFrame({"lesion_id": frame["lesion_id"], "label": frame["label"], "fold": folds})


def check_images(frame: pd.DataFrame, image_root: Path) -> list[str]:
    """Return the paths of any referenced image that is missing from disk."""
    missing: list[str] = []
    for lesion_id, clin_id, derm_id in zip(
        frame["lesion_id"], frame["isic_id_clin"], frame["isic_id_derm"]
    ):
        for isic_id in (clin_id, derm_id):
            path = image_root / str(lesion_id) / f"{isic_id}.jpg"
            if not path.exists():
                missing.append(str(path))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", help="key.path=value overrides")
    parser.add_argument("--check-images", action="store_true", help="verify every referenced JPEG exists")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging(Path(cfg.paths.log_dir) / "prepare_data.log")

    processed = Path(cfg.paths.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    # --- train -------------------------------------------------------------
    LOGGER.info("Building lesion-level training table...")
    train = build_lesion_table(cfg.data.train_metadata, cfg.data.train_ground_truth)
    LOGGER.info("Train: %d lesions", len(train))

    label_counts = train[CLASSES].sum().astype(int).to_dict()
    LOGGER.info("Class counts: %s", json.dumps(label_counts))
    # Flagged explicitly because it drives the loss/sampler/threshold strategy.
    imbalance = max(label_counts.values()) / max(1, min(label_counts.values()))
    LOGGER.info("Imbalance ratio (max/min): %.0f:1", imbalance)

    # --- test --------------------------------------------------------------
    LOGGER.info("Building lesion-level test table...")
    test = build_lesion_table(cfg.data.test_metadata, None)
    LOGGER.info("Test: %d lesions", len(test))

    # --- folds -------------------------------------------------------------
    folds = make_folds(train, int(cfg.data.n_folds), int(cfg.seed))
    fold_table = (
        folds.merge(train[["lesion_id", *CLASSES]], on="lesion_id")
        .groupby("fold")[CLASSES]
        .sum()
        .astype(int)
    )
    LOGGER.info("Per-fold class counts:\n%s", fold_table.to_string())
    empty = [(int(f), c) for f in fold_table.index for c in CLASSES if fold_table.loc[f, c] == 0]
    if empty:
        LOGGER.warning(
            "%d (fold, class) cells have zero validation lesions -- their per-class F1 and "
            "tuned threshold are undefined for that fold: %s",
            len(empty),
            empty,
        )

    # --- integrity checks --------------------------------------------------
    report: dict = {
        "n_train_lesions": int(len(train)),
        "n_test_lesions": int(len(test)),
        "class_counts": label_counts,
        "imbalance_ratio": float(imbalance),
        "per_fold_class_counts": fold_table.to_dict(orient="index"),
        "n_folds": int(cfg.data.n_folds),
        "seed": int(cfg.seed),
        "train_missing_values": {k: int(v) for k, v in train.isna().sum().items() if v > 0},
        "test_missing_values": {k: int(v) for k, v in test.isna().sum().items() if v > 0},
        "train_site_distribution": Counter(train["site"].fillna("unknown")),
        "empty_fold_class_cells": empty,
    }

    if args.check_images:
        for name, frame, root in (
            ("train", train, Path(cfg.data.train_image_root)),
            ("test", test, Path(cfg.data.test_image_root)),
        ):
            LOGGER.info("Checking %s images under %s ...", name, root)
            missing = check_images(frame, root)
            report[f"{name}_missing_images"] = missing[:50]
            report[f"{name}_n_missing_images"] = len(missing)
            if missing:
                LOGGER.error("%s: %d missing image(s), e.g. %s", name, len(missing), missing[:3])
            else:
                LOGGER.info("%s: all %d images present", name, 2 * len(frame))

    # --- write -------------------------------------------------------------
    train.to_csv(processed / "train_lesions.csv", index=False)
    test.to_csv(processed / "test_lesions.csv", index=False)
    folds.to_csv(processed / "folds.csv", index=False)
    with open(processed / "dataset_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    LOGGER.info("Wrote train_lesions.csv, test_lesions.csv, folds.csv, dataset_report.json to %s", processed)


if __name__ == "__main__":
    main()
