#!/usr/bin/env python
"""Export an inference-only model bundle for the Docker container.

Training checkpoints are 425-886 MB each because they carry optimiser state,
scheduler state, AMP scaler state and RNG state so a run can be resumed. None of
that is needed to predict. This strips each checkpoint to its EMA weights in
float16 and writes a single self-describing bundle:

    weights_bundle/
      manifest.json                 architecture + blend recipe + thresholds
      stage1_derm_cv_fold0.pt       fp16 EMA state dict
      ...

Size: ~840 MB of training checkpoints per run pair becomes ~56 MB per
ConvNeXt-Tiny fold and ~112 MB per dual-encoder fold.

Usage::

    python scripts/export_weights.py \
        --run checkpoints/stage1_derm_cv --run checkpoints/stage2_dual_cv \
        --blend outputs/blend2 --out weights_bundle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.constants import CLASSES
from src.utils.logging_utils import get_logger, setup_logging

LOGGER = get_logger("export_weights")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, dest="runs")
    parser.add_argument("--blend", default=None, help="blend dir with thresholds/weights (from blend.py)")
    parser.add_argument("--out", default="weights_bundle")
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()

    setup_logging("logs/export_weights.log")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"classes": CLASSES, "runs": [], "blend": None}
    total_bytes = 0

    for run_dir in [Path(r) for r in args.runs]:
        checkpoints = sorted(
            run_dir.glob("fold*/best.pt"), key=lambda p: int(p.parent.name.removeprefix("fold"))
        )
        if not checkpoints:
            raise FileNotFoundError(f"No fold*/best.pt under {run_dir}")

        run_entry: dict = {"name": run_dir.name, "folds": [], "model": None, "image_size": None, "mode": None}

        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            # EMA weights are what validation was scored on, so they are what ships.
            state = ckpt["ema"]["module"] if ckpt.get("ema") else ckpt["model"]
            state = {
                k: (v.half() if (args.fp16 and v.is_floating_point()) else v)
                for k, v in state.items()
            }

            fold = ckpt_path.parent.name
            out_path = out_dir / f"{run_dir.name}_{fold}.pt"
            torch.save(state, out_path)
            size = out_path.stat().st_size
            total_bytes += size
            run_entry["folds"].append(out_path.name)
            LOGGER.info("%-40s -> %s (%.0f MB)", str(ckpt_path), out_path.name, size / 1048576)

            if run_entry["model"] is None:
                model_cfg = cfg["model"]
                run_entry["model"] = {
                    "backbone": model_cfg.get("backbone"),
                    "clinical_backbone": model_cfg.get("clinical_backbone"),
                    "use_clinical": bool(model_cfg.get("use_clinical", False)),
                    "use_metadata": bool(model_cfg.get("use_metadata", False)),
                    "fusion": model_cfg.get("fusion", "concat"),
                    "fusion_dim": int(model_cfg.get("fusion_dim", 512)),
                    "head_hidden": int(model_cfg.get("head_hidden", 0)),
                    "head_dropout": float(model_cfg.get("head_dropout", 0.3)),
                    "share_encoder": bool(model_cfg.get("share_encoder", False)),
                    # Dropout/stochastic-depth are inert at eval time but the
                    # constructor still needs them to build identical modules.
                    "drop_rate": 0.0,
                    "drop_path_rate": 0.0,
                    "meta_hidden": list(model_cfg.get("meta_hidden", (128, 128))),
                    "meta_embed_dim": int(model_cfg.get("meta_embed_dim", 128)),
                }
                run_entry["image_size"] = int(cfg["data"]["image_size"])
                run_entry["mode"] = cfg["data"].get("mode", "dermoscopic")
                run_entry["tta"] = cfg.get("inference", {}).get("tta", "d4")

        manifest["runs"].append(run_entry)

    # --- blend recipe: how to combine the runs and where to cut
    if args.blend:
        blend_dir = Path(args.blend)
        summary = json.loads((blend_dir / "blend_summary.json").read_text())
        per_class_path = blend_dir / "per_class_weights.npy"
        manifest["blend"] = {
            "method": summary["method"],
            "weights": summary["weights"],
            "per_class_weights": (
                np.load(per_class_path).tolist() if per_class_path.exists() else None
            ),
            "thresholds": summary["thresholds"],
            "strategy": summary["recommended_strategy"],
            "oof_macro_f1": summary.get("chosen_score"),
        }
        LOGGER.info(
            "Blend: method=%s strategy=%s per_class=%s",
            summary["method"],
            summary["recommended_strategy"],
            per_class_path.exists(),
        )
    else:
        # Single run: use its own pooled-OOF thresholds.
        run_dir = Path(args.runs[0])
        thr_path = run_dir / "oof_thresholds.npy"
        if thr_path.exists():
            oof = json.loads((run_dir / "oof_summary.json").read_text())
            manifest["blend"] = {
                "method": "mean",
                "weights": None,
                "per_class_weights": None,
                "thresholds": np.load(thr_path).tolist(),
                "strategy": oof.get("recommended_strategy", "tuned"),
                "oof_macro_f1": oof.get("strategies", {}).get("tuned"),
            }

    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    LOGGER.info(
        "Bundle written to %s | %d run(s), %d weight file(s), %.0f MB total",
        out_dir,
        len(manifest["runs"]),
        sum(len(r["folds"]) for r in manifest["runs"]),
        total_bytes / 1048576,
    )


if __name__ == "__main__":
    main()
