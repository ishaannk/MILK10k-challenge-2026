#!/usr/bin/env python
"""Evaluate a checkpoint (or a run's pooled OOF predictions) in detail.

Re-score a checkpoint on its own validation fold, with TTA::

    python scripts/evaluate.py --checkpoint checkpoints/stage1_derm/fold0/best.pt --fold 0 --tta d4

Analyse a completed cross-validation run from its saved OOF predictions (no GPU
work, instant)::

    python scripts/evaluate.py --run-dir checkpoints/stage3_meta

Produces, in the output directory:

* ``eval_metrics.json`` - the full metric bundle including per-class blocks
* ``confusion_matrix.png`` - row-normalised, argmax-based
* ``per_class.csv`` - one row per class, for pasting into notes
* a logged strategy comparison, so the submission choice is evidence-based
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from src.constants import CLASSES
from src.datasets.metadata import MetadataProcessor
from src.datasets.milk10k import MILK10kDataset, build_dataloader
from src.datasets.transforms import build_transforms_from_config
from src.inference.predictor import Predictor
from src.utils.logging_utils import get_logger, setup_logging
from src.validation.metrics import compute_metrics, format_metrics_table, plot_confusion_matrix
from src.validation.thresholds import evaluate_threshold_strategies, optimize_thresholds

LOGGER = get_logger("evaluate")


def predictions_from_checkpoint(
    checkpoint: Path, fold: int, tta: str, use_ema: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Re-run a checkpoint over its validation fold. Returns ``(probs, targets)``."""
    predictor = Predictor(checkpoint, device="cuda", use_ema=use_ema)
    cfg = predictor.cfg

    train_frame = pd.read_csv(Path(cfg.paths.processed_dir) / "train_lesions.csv")
    folds_frame = pd.read_csv(cfg.data.folds_csv)
    valid_ids = set(folds_frame.loc[folds_frame["fold"] == fold, "lesion_id"])
    valid_frame = train_frame.loc[train_frame["lesion_id"].isin(valid_ids)].reset_index(drop=True)
    LOGGER.info("Validation fold %d: %d lesions", fold, len(valid_frame))

    meta_features = None
    if bool(cfg.model.get("use_metadata", False)):
        processor_path = checkpoint.parent / "meta_processor.json"
        if not processor_path.exists():
            raise FileNotFoundError(f"Model uses metadata but {processor_path} is missing")
        meta_features = MetadataProcessor.load(processor_path).transform(valid_frame)

    dataset = MILK10kDataset(
        valid_frame,
        cfg.data.train_image_root,
        transforms=build_transforms_from_config(cfg, train=False),
        mode=cfg.data.get("mode", "dermoscopic"),
        meta_features=meta_features,
        has_targets=True,
    )
    loader = build_dataloader(dataset, cfg, train=False)
    probs, _ = predictor.predict(loader, tta=tta)
    return probs, dataset.targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="checkpoint to re-score on its validation fold")
    parser.add_argument("--fold", type=int, default=0, help="which fold the checkpoint validated on")
    parser.add_argument("--run-dir", default=None, help="analyse pooled OOF predictions saved by train.py")
    parser.add_argument("--tta", default="d4")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--out-dir", default=None, help="defaults next to the checkpoint / run dir")
    args = parser.parse_args()

    setup_logging("logs/evaluate.log")

    if args.run_dir:
        run_dir = Path(args.run_dir)
        probs = np.load(run_dir / "oof_probs.npy")
        targets = np.load(run_dir / "oof_targets.npy")
        out_dir = Path(args.out_dir or run_dir)
        LOGGER.info("Loaded pooled OOF predictions from %s (%d lesions)", run_dir, len(probs))
    elif args.checkpoint:
        checkpoint = Path(args.checkpoint)
        probs, targets = predictions_from_checkpoint(checkpoint, args.fold, args.tta, not args.no_ema)
        out_dir = Path(args.out_dir or checkpoint.parent)
    else:
        parser.error("Provide --checkpoint or --run-dir")

    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = optimize_thresholds(targets, probs, verbose=True)
    metrics = compute_metrics(targets, probs, thresholds=thresholds)
    strategies = evaluate_threshold_strategies(targets, probs, thresholds)

    LOGGER.info("\n--- At the official 0.5 threshold ---\n%s", format_metrics_table(metrics, tuned=False))
    LOGGER.info("\n--- At tuned per-class thresholds ---\n%s", format_metrics_table(metrics, tuned=True))
    LOGGER.info(
        "Headline: macro_f1@0.5=%.4f  macro_f1_tuned=%.4f  macro_f1_argmax=%.4f  macro_auc=%.4f  bal_acc=%.4f",
        metrics["macro_f1"],
        metrics["macro_f1_tuned"],
        metrics["macro_f1_argmax"],
        metrics["macro_auc"],
        metrics["balanced_accuracy"],
    )
    LOGGER.info("Submission strategy comparison:")
    for name, value in sorted(strategies.items(), key=lambda kv: -kv[1]):
        LOGGER.info("  %-18s %.4f", name, value)

    # --- artefacts
    figure = plot_confusion_matrix(np.array(metrics["confusion_matrix"]), CLASSES, title="Confusion matrix (argmax)")
    figure.savefig(out_dir / "confusion_matrix.png", dpi=140, bbox_inches="tight")

    per_class = pd.DataFrame(metrics["per_class_tuned"]).T.reset_index(names="class")
    per_class.to_csv(out_dir / "per_class.csv", index=False)

    payload = {k: v for k, v in metrics.items() if k not in ("per_class", "per_class_tuned")}
    payload["per_class"] = metrics["per_class"]
    payload["per_class_tuned"] = metrics["per_class_tuned"]
    payload["strategies"] = strategies
    with open(out_dir / "eval_metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    np.save(out_dir / "eval_thresholds.npy", thresholds)
    LOGGER.info("Wrote eval_metrics.json, per_class.csv, confusion_matrix.png to %s", out_dir)


if __name__ == "__main__":
    main()
