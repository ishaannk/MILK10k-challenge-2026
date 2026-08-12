#!/usr/bin/env python
"""Aggregate every completed run into one comparison table.

    python scripts/report.py                        # markdown to stdout
    python scripts/report.py --out RESULTS.md       # and write a file

Scans ``checkpoints/*/summary.json`` and reports, per run, the cross-validated
metrics plus the pooled-OOF submission-strategy comparison. Sorted by the metric
that actually decides the leaderboard: pooled-OOF tuned macro-F1.

Two columns deserve attention when reading the output:

``f1@0.5`` vs ``f1_tuned``
    The gap is the score available from calibration alone. If it is large, threshold
    tuning is doing the heavy lifting and the model is under-calibrated rather than
    under-trained.
``AUC``
    The threshold-free signal. Compare *this* between architectures — ``f1@0.5`` can
    move on calibration noise, so a run can look better while ranking worse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.constants import CLASSES

METRICS = [
    ("cv_macro_f1_mean", "f1@0.5"),
    ("cv_macro_f1_tuned_mean", "f1_tuned"),
    ("cv_macro_f1_argmax_mean", "f1_argmax"),
    ("cv_macro_auc_mean", "AUC"),
    ("cv_balanced_accuracy_mean", "bal_acc"),
]


def collect(checkpoint_dir: Path) -> list[dict]:
    """Load one record per run directory that has a summary.json."""
    runs: list[dict] = []
    for summary_path in sorted(checkpoint_dir.glob("*/summary.json")):
        # Skip symlinked duplicates of a directory we have already read.
        if summary_path.parent.is_symlink():
            continue
        summary = json.loads(summary_path.read_text())
        record: dict = {
            "name": summary_path.parent.name,
            "folds": summary.get("folds", []),
            "strategies": summary.get("oof_strategies", {}),
            "recommended": summary.get("oof_recommended_strategy", "-"),
        }
        for key, label in METRICS:
            record[label] = summary.get(key)
            record[f"{label}_std"] = summary.get(key.replace("_mean", "_std"))

        oof_path = summary_path.parent / "oof_summary.json"
        if oof_path.exists():
            oof = json.loads(oof_path.read_text())
            record["oof_n"] = oof.get("n_lesions")
            record["oof_tuned"] = oof.get("strategies", {}).get("tuned")
            record["per_class_tuned"] = oof.get("per_class_tuned", {})
            record["thresholds"] = oof.get("thresholds")
        # A single-fold run's score is optimistically biased twice over: it is one
        # small sample, and its thresholds were tuned on the very lesions being
        # scored. Only a full-CV run's pooled OOF is comparable to test performance,
        # so single-fold runs are ranked below every CV run regardless of the number
        # they report -- otherwise the table recommends the wrong submission file.
        record["is_cv"] = len(record["folds"]) > 1
        runs.append(record)
    return sorted(
        runs,
        key=lambda r: (r["is_cv"], r.get("oof_tuned") or r.get("f1_tuned") or -1),
        reverse=True,
    )


def fmt(value, width: int = 8, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return f"{'-':>{width}}"
    return f"{value:>{width}.{places}f}"


def render(runs: list[dict]) -> str:
    lines: list[str] = ["# MILK10k experiment results", ""]

    if not runs:
        return "\n".join(lines + ["_No completed runs found under `checkpoints/`._"])

    # --- headline comparison
    lines += [
        "## Run comparison",
        "",
        "Cross-validated runs first, then sorted by pooled out-of-fold tuned macro-F1.",
        "",
        "**Do not compare a single-fold `OOF tuned` against a CV one.** A single-fold score is",
        "biased upward twice: small sample, and thresholds tuned on the same lesions being",
        "scored. Only full-CV rows (marked ✓) estimate test performance, so **submit from a ✓",
        "run** even when a single-fold row shows a higher number.",
        "",
        "| run | CV | folds | f1@0.5 | f1_tuned | f1_argmax | AUC | bal_acc | OOF n | OOF tuned | strategy |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        folds = ",".join(str(f) for f in r["folds"]) or "-"
        cells = [
            f"`{r['name']}`",
            "✓" if r["is_cv"] else "—",
            folds,
            fmt(r["f1@0.5"], 0),
            fmt(r["f1_tuned"], 0),
            fmt(r["f1_argmax"], 0),
            fmt(r["AUC"], 0),
            fmt(r["bal_acc"], 0),
            str(r.get("oof_n", "-")),
            fmt(r.get("oof_tuned"), 0),
            r["recommended"],
        ]
        lines.append("| " + " | ".join(cells) + " |")

    # --- strategy comparison for the leader
    best = runs[0]
    if best.get("strategies"):
        lines += [
            "",
            f"## Submission strategies — `{best['name']}` (pooled OOF)",
            "",
            "| strategy | macro-F1 |",
            "|---|---|",
        ]
        for name, value in sorted(best["strategies"].items(), key=lambda kv: -kv[1]):
            marker = "  **<- recommended**" if name == best["recommended"] else ""
            lines.append(f"| `{name}` | {value:.4f}{marker} |")

    # --- per-class detail for the leader: where the macro average is actually won
    if best.get("per_class_tuned"):
        lines += [
            "",
            f"## Per-class detail — `{best['name']}` at tuned thresholds",
            "",
            "| class | support | predicted | threshold | precision | recall | F1 | AUC |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name in CLASSES:
            m = best["per_class_tuned"].get(name)
            if not m:
                continue
            lines.append(
                f"| {name} | {m['support']:.0f} | {m['n_predicted']:.0f} | {m['threshold']:.3f} | "
                f"{m['precision']:.3f} | {m['recall']:.3f} | **{m['f1']:.3f}** | {m['auc']:.3f} |"
            )
        zero = [n for n in CLASSES if (best["per_class_tuned"].get(n, {}).get("f1", 1) == 0)]
        if zero:
            lines += [
                "",
                f"Classes scoring F1 = 0: **{', '.join(zero)}**. Each costs a full 1/11 = 0.091 "
                "of macro-F1, so these are where any remaining headroom lives.",
            ]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--out", default=None, help="also write the markdown to this path")
    args = parser.parse_args()

    report = render(collect(Path(args.checkpoint_dir)))
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"[written to {args.out}]")


if __name__ == "__main__":
    main()
