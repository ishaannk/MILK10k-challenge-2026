#!/usr/bin/env python
"""Run inference on the test set and write a competition submission CSV.

Single model, single fold::

    python scripts/predict.py --checkpoint checkpoints/stage1_derm/fold0/best.pt \
        --out outputs/submission_stage1.csv

Ensemble every fold of a run, using that run's pooled out-of-fold thresholds and
recommended strategy::

    python scripts/predict.py --run-dir checkpoints/stage3_meta \
        --out outputs/submission_stage3_cv.csv

Ensemble across different runs (e.g. two backbones)::

    python scripts/predict.py \
        --checkpoint checkpoints/stage3_meta/fold0/best.pt \
        --checkpoint checkpoints/stage2_dual/fold0/best.pt \
        --thresholds checkpoints/stage3_meta/oof_thresholds.npy \
        --out outputs/submission_blend.csv

Threshold provenance
--------------------
Thresholds are taken, in order of preference, from ``--thresholds``, then the
run's pooled ``oof_thresholds.npy``, then the checkpoint's own stored thresholds.
Pooled OOF thresholds are the ones to prefer: they are fitted on ~5x more
positives per rare class than any single fold's, which is exactly where threshold
overfitting bites.
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
from src.inference.predictor import Predictor, ensemble_probabilities, write_submission
from src.utils.config import Config, load_config
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed import seed_everything
from src.validation.thresholds import apply_strategy

LOGGER = get_logger("predict")


def discover_checkpoints(run_dir: Path, which: str = "best.pt") -> list[Path]:
    """Find each fold's checkpoint inside a run directory, in fold order."""
    found = sorted(run_dir.glob(f"fold*/{which}"), key=lambda p: int(p.parent.name.removeprefix("fold")))
    if not found:
        raise FileNotFoundError(f"No {which} checkpoints under {run_dir}")
    return found


def build_test_loader(cfg: Config, processor: MetadataProcessor | None) -> tuple[MILK10kDataset, "DataLoader"]:  # noqa: F821
    """Build the test dataset/loader using the *training* config's preprocessing."""
    test_frame = pd.read_csv(Path(cfg.paths.processed_dir) / "test_lesions.csv")

    meta_features = None
    if bool(cfg.model.get("use_metadata", False)):
        if processor is None:
            raise ValueError(
                "This model consumes metadata but no meta_processor.json was found. "
                "It must be the one fitted during training -- refitting on test data "
                "would use different standardisation statistics."
            )
        meta_features = processor.transform(test_frame)

    dataset = MILK10kDataset(
        test_frame,
        cfg.data.test_image_root,
        transforms=build_transforms_from_config(cfg, train=False),
        mode=cfg.data.get("mode", "dermoscopic"),
        meta_features=meta_features,
        has_targets=False,
    )
    return dataset, build_dataloader(dataset, cfg, train=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", default=[], help="checkpoint path (repeatable)")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        dest="run_dirs",
        help="ensemble every fold*/best.pt under this run directory (repeatable, to blend runs)",
    )
    parser.add_argument("--out", required=True, help="output submission CSV path")
    parser.add_argument("--tta", default=None, help="none | hflip | flips | d4 (default: from config)")
    parser.add_argument("--strategy", default=None, help="raw | argmax | tuned | tuned_or_argmax")
    parser.add_argument("--thresholds", default=None, help="path to a saved thresholds .npy")
    parser.add_argument("--ensemble-method", default=None, help="mean | gmean | rank")
    parser.add_argument("--weights", default=None, help="comma-separated ensemble weights")
    parser.add_argument("--no-ema", action="store_true", help="use raw weights instead of EMA")
    parser.add_argument("--save-probs", default=None, help="also save raw ensembled probabilities as .npy")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    setup_logging("logs/predict.log")

    # ---------------------------------------------------- resolve checkpoints
    checkpoints = [Path(c) for c in args.checkpoint]
    # Track which run each checkpoint came from. Blend weights produced by
    # scripts/blend.py are *per run* (one OOF matrix per run), but a run expands to
    # one checkpoint per fold -- so a 2-run blend becomes 10 checkpoints here. The
    # group sizes let run-level weights be spread across their folds below; without
    # this, --weights would simply fail a length check.
    checkpoint_groups: list[int] = [0] * len(checkpoints)  # bare --checkpoint = its own group
    for i, c in enumerate(checkpoints):
        checkpoint_groups[i] = i
    next_group = len(checkpoints)

    run_dirs = [Path(r) for r in args.run_dirs]
    for run_dir in run_dirs:
        found = discover_checkpoints(run_dir)
        checkpoints += found
        checkpoint_groups += [next_group] * len(found)
        next_group += 1
    if not checkpoints:
        parser.error("Provide at least one --checkpoint or a --run-dir")
    LOGGER.info("Ensembling %d checkpoint(s): %s", len(checkpoints), [str(c) for c in checkpoints])

    # ------------------------------------------------------ resolve thresholds
    thresholds: np.ndarray | None = None
    threshold_source = "none"
    strategy = args.strategy
    if args.thresholds:
        thresholds = np.load(args.thresholds)
        threshold_source = args.thresholds
    else:
        # Thresholds calibrate the *ensemble*, so when several runs are blended the
        # only correct source is a blend-level file from scripts/blend.py. Silently
        # borrowing one run's thresholds would mis-calibrate the mix.
        candidates = [d for d in run_dirs if (d / "oof_thresholds.npy").exists()]
        if len(candidates) > 1:
            LOGGER.warning(
                "%d run dirs have their own oof_thresholds.npy. Per-run thresholds do not "
                "calibrate a blend -- run scripts/blend.py and pass --thresholds explicitly. "
                "Proceeding without tuned thresholds.",
                len(candidates),
            )
        elif len(candidates) == 1:
            run_dir = candidates[0]
            thresholds = np.load(run_dir / "oof_thresholds.npy")
            threshold_source = str(run_dir / "oof_thresholds.npy")
            # The run also recorded which strategy won on pooled OOF data; honour it
            # unless the user explicitly asked for something else.
            summary_path = run_dir / "oof_summary.json"
            if strategy is None and summary_path.exists():
                strategy = json.loads(summary_path.read_text()).get("recommended_strategy")
                LOGGER.info("Using strategy recommended by pooled OOF: %s", strategy)

    # ------------------------------------------------------------- predict
    all_probs: list[np.ndarray] = []
    reference_ids: list[str] | None = None
    cfg: Config | None = None

    for path in checkpoints:
        predictor = Predictor(path, device="cuda", use_ema=not args.no_ema)
        cfg = predictor.cfg
        for override in args.overrides:
            key, raw = override.split("=", 1)
            import yaml

            cfg.set_path(key, yaml.safe_load(raw))

        seed_everything(int(cfg.get("seed", 42)), deterministic=True)

        # The metadata processor lives next to the checkpoint it was fitted with.
        processor = None
        processor_path = path.parent / "meta_processor.json"
        if processor_path.exists():
            processor = MetadataProcessor.load(processor_path)

        _, loader = build_test_loader(cfg, processor)

        tta = args.tta or str(cfg.get("inference", {}).get("tta", "d4"))
        probs, ids = predictor.predict(
            loader, tta=tta, average=str(cfg.get("inference", {}).get("tta_average", "logit"))
        )
        LOGGER.info("  %s -> %s probs (tta=%s)", path.name, probs.shape, tta)

        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            # Guard against silently averaging misaligned rows.
            raise RuntimeError(f"Lesion id order differs between checkpoints ({path})")

        all_probs.append(probs)
        if thresholds is None and predictor.thresholds is not None:
            thresholds = predictor.thresholds
            threshold_source = f"{path} (checkpoint)"

    # ------------------------------------------------------------- ensemble
    method = args.ensemble_method or str(cfg.get("inference", {}).get("ensemble_method", "mean"))

    weights = [float(w) for w in args.weights.split(",")] if args.weights else None
    if weights is not None and len(weights) != len(all_probs):
        # Interpret them as run-level weights and divide each across that run's folds,
        # so the run keeps its intended total influence on the ensemble.
        n_groups = len(set(checkpoint_groups))
        if len(weights) != n_groups:
            parser.error(
                f"--weights has {len(weights)} entries, which matches neither the "
                f"{len(all_probs)} checkpoints nor the {n_groups} runs"
            )
        group_sizes = {g: checkpoint_groups.count(g) for g in checkpoint_groups}
        group_order = sorted(set(checkpoint_groups), key=checkpoint_groups.index)
        weight_of = {g: weights[i] for i, g in enumerate(group_order)}
        weights = [weight_of[g] / group_sizes[g] for g in checkpoint_groups]
        LOGGER.info(
            "Expanded %d run-level weight(s) across %d checkpoints: %s",
            n_groups,
            len(weights),
            [round(w, 4) for w in weights],
        )
    probs = ensemble_probabilities(all_probs, weights=weights, method=method) if len(all_probs) > 1 else all_probs[0]
    if len(all_probs) > 1:
        LOGGER.info("Ensembled %d models with method=%s weights=%s", len(all_probs), method, weights or "uniform")

    # ------------------------------------------------------------- strategy
    strategy = strategy or str(cfg.get("inference", {}).get("strategy", "tuned_or_argmax"))
    if strategy != "raw" and thresholds is None:
        LOGGER.warning("No thresholds available; falling back to strategy='raw'")
        strategy = "raw"
    LOGGER.info("Strategy=%s | thresholds from %s", strategy, threshold_source)
    if thresholds is not None:
        LOGGER.info("Thresholds: %s", json.dumps(dict(zip(CLASSES, np.round(thresholds, 4).tolist()))))

    final = apply_strategy(probs, strategy=strategy, thresholds=thresholds)

    # ---------------------------------------------------------------- write
    expected = pd.read_csv(Path(cfg.paths.processed_dir) / "test_lesions.csv")["lesion_id"].astype(str).tolist()
    write_submission(args.out, reference_ids, final, expected_ids=expected)

    if args.save_probs:
        # Raw (pre-strategy) probabilities, for blending runs later without re-running inference.
        Path(args.save_probs).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_probs, probs)
        with open(Path(args.save_probs).with_suffix(".ids.json"), "w") as fh:
            json.dump(reference_ids, fh)
        LOGGER.info("Saved raw probabilities to %s", args.save_probs)


if __name__ == "__main__":
    main()
