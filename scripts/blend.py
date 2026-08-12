#!/usr/bin/env python
"""Find optimal ensemble weights on out-of-fold predictions, then apply them.

Given several completed runs, each with pooled OOF predictions, this searches for
the blend weights that maximise pooled-OOF tuned macro-F1, and writes the winning
weights plus their re-tuned thresholds::

    python scripts/blend.py \
        --run checkpoints/stage2_dual_cv \
        --run checkpoints/stage3_meta_cv \
        --out outputs/blend

Then generate the submission with those weights::

    python scripts/predict.py \
        --run-dir checkpoints/stage2_dual_cv --run-dir checkpoints/stage3_meta_cv \
        --weights "$(cat outputs/blend/weights.csv)" \
        --thresholds outputs/blend/thresholds.npy \
        --out outputs/submission_blend.csv

Why weights are fitted on OOF and not on validation-of-one-fold: blend weights are
themselves parameters, and with five rare classes there is very little signal to fit
them on. Pooled OOF is the largest honest sample available. Even so, the search is
deliberately coarse and the report includes the equal-weight baseline — if tuning
buys less than a noticeable margin over uniform, **take uniform**, because a weight
vector fitted to a few dozen rare-class positives is exactly the kind of thing that
looks good out-of-fold and evaporates on the leaderboard.

Rows are aligned by ``lesion_id``, never by position: two runs can pool their folds
in a different order, and averaging misaligned rows would silently destroy the
ensemble while still producing a plausible-looking score.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.constants import CLASSES
from src.inference.predictor import ensemble_probabilities
from src.utils.logging_utils import get_logger, setup_logging
from src.validation.metrics import compute_metrics, format_metrics_table
from src.validation.thresholds import (
    evaluate_threshold_strategies,
    optimize_threshold_for_class,
    optimize_thresholds,
)

LOGGER = get_logger("blend")


def load_oof(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a run's pooled OOF probabilities, targets and lesion ids."""
    probs = np.load(run_dir / "oof_probs.npy")
    targets = np.load(run_dir / "oof_targets.npy")

    summary_path = run_dir / "oof_summary.json"
    ids: list[str] = []
    if summary_path.exists():
        ids = json.loads(summary_path.read_text()).get("lesion_ids", []) or []
    if len(ids) != len(probs):
        LOGGER.warning(
            "%s has no usable lesion_ids (%d ids for %d rows); falling back to positional "
            "alignment, which is only safe if every run used identical folds",
            run_dir,
            len(ids),
            len(probs),
        )
        ids = [str(i) for i in range(len(probs))]
    return probs, targets, ids


def align_runs(
    loaded: list[tuple[np.ndarray, np.ndarray, list[str]]]
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Reindex every run onto the lesions common to all of them.

    Returns ``(aligned_probs, aligned_targets, lesion_ids)``. **The row order is
    sorted lesion_id, which is NOT the fold-concatenation order of the per-run
    ``oof_probs.npy``/``oof_targets.npy`` files.** Anything that consumes the blended
    OOF later must use the targets and ids returned here — pairing the blended
    probabilities with a raw ``oof_targets.npy`` silently scrambles the rows and
    produces a plausible-looking but meaningless score (measured: 0.13 instead of
    0.54). That is why the ids are returned and persisted alongside the blend.
    """
    id_sets = [set(ids) for _, _, ids in loaded]
    common = sorted(set.intersection(*id_sets))
    if not common:
        raise ValueError("The runs share no lesion ids; cannot blend them")

    for i, (_, _, ids) in enumerate(loaded):
        dropped = len(ids) - len(common)
        if dropped:
            LOGGER.warning("Run %d: dropping %d lesion(s) not present in every run", i, dropped)

    aligned_probs: list[np.ndarray] = []
    reference_targets: np.ndarray | None = None
    for probs, targets, ids in loaded:
        index = {lesion: row for row, lesion in enumerate(ids)}
        rows = [index[lesion] for lesion in common]
        aligned_probs.append(probs[rows])
        aligned_targets = targets[rows]
        if reference_targets is None:
            reference_targets = aligned_targets
        elif not np.array_equal(reference_targets, aligned_targets):
            raise ValueError("Aligned targets disagree between runs -- check the fold definitions")

    LOGGER.info("Aligned %d run(s) on %d common lesions", len(loaded), len(common))
    return aligned_probs, reference_targets, common


def score_blend(
    probs_list: list[np.ndarray],
    targets: np.ndarray,
    weights: np.ndarray,
    method: str,
    max_predict_multiple: float,
) -> tuple[float, np.ndarray]:
    """Tuned macro-F1 of one weighted blend, plus the thresholds it used."""
    blended = ensemble_probabilities(probs_list, weights=weights, method=method)
    thresholds = optimize_thresholds(
        targets, blended, max_predict_multiple=max_predict_multiple, verbose=False
    )
    from src.validation.metrics import macro_f1

    return macro_f1(targets, blended, thresholds), thresholds


def fit_per_class_weights(
    probs_list: list[np.ndarray],
    targets: np.ndarray,
    grid: list[np.ndarray],
    method: str,
    max_predict_multiple: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a separate blend weight vector for **each class**.

    Returns ``(weights (n_models, n_classes), thresholds (n_classes,))``.

    Motivation, measured rather than assumed: on fold 0 the dual-encoder model beats
    the dermoscopy-only model on 8 of 11 classes, yet loses so badly on INF (-0.112)
    and VASC (-0.074) that its overall macro-F1 ends up *worse*. A single global
    weight has to compromise across that split; per-class weights let INF and VASC
    take from the dermoscopy model while the other nine take from the dual model.

    This is legitimate and cheap because macro-F1 is **separable**: ``F1_c`` depends
    only on class ``c``'s own blended scores and threshold, so optimising each class
    independently is globally optimal — exactly the argument that makes the
    threshold search in ``validation/thresholds.py`` exact.

    The catch is variance, not correctness: 11 weight vectors are 11x the fitted
    parameters, and the rare classes have 44-50 positives. Never enable this without
    :func:`validate_per_class_honestly` clearing it on held-out data.
    """
    n_models, n_classes = len(probs_list), targets.shape[1]
    weights = np.zeros((n_models, n_classes), dtype=np.float64)
    thresholds = np.full(n_classes, 0.5, dtype=np.float64)

    for c in range(n_classes):
        # Blend just this class's column under each candidate weight vector.
        columns = [p[:, [c]] for p in probs_list]
        best = (-1.0, grid[0], 0.5)
        for w in grid:
            blended_col = ensemble_probabilities(columns, weights=w, method=method)
            thr, f1 = optimize_threshold_for_class(
                targets[:, c],
                blended_col[:, 0],
                max_predict_multiple=max_predict_multiple,
            )
            if f1 > best[0]:
                best = (f1, w, thr)
        weights[:, c] = best[1]
        thresholds[c] = best[2]

    return weights, thresholds


def apply_per_class_weights(
    probs_list: list[np.ndarray], weights: np.ndarray, method: str
) -> np.ndarray:
    """Blend with a ``(n_models, n_classes)`` weight matrix, column by column."""
    n_classes = weights.shape[1]
    out = np.zeros_like(np.asarray(probs_list[0], dtype=np.float64))
    for c in range(n_classes):
        columns = [p[:, [c]] for p in probs_list]
        out[:, [c]] = ensemble_probabilities(columns, weights=weights[:, c], method=method)
    return out


def validate_per_class_honestly(
    probs_list: list[np.ndarray],
    targets: np.ndarray,
    grid: list[np.ndarray],
    method: str,
    max_predict_multiple: float,
    n_splits: int = 6,
    seed: int = 0,
) -> dict:
    """Does per-class weighting still win when fitted and scored on different data?

    Fits both global and per-class weights (and their thresholds) on one half of the
    OOF set and scores both on the other half, over several random splits. Fitting
    and evaluating on the same rows would always favour per-class weighting simply
    because it has more parameters — this is the only way to tell whether the gain
    is real.
    """
    from src.validation.metrics import macro_f1

    rng = np.random.default_rng(seed)
    n = len(targets)
    global_scores, per_class_scores, single_scores = [], [], []

    for _ in range(n_splits):
        order = rng.permutation(n)
        fit_idx, eval_idx = order[: n // 2], order[n // 2 :]
        fit_probs = [p[fit_idx] for p in probs_list]
        eval_probs = [p[eval_idx] for p in probs_list]

        # --- arm 0: the best SINGLE model, chosen and calibrated on the fit half.
        # Without this reference, "blend beats uniform" can look like progress while
        # the blend is actually worse than just shipping one model.
        single_best = (-1.0, 0, None)
        for m in range(len(fit_probs)):
            s, t_ = score_blend([fit_probs[m]], targets[fit_idx], np.array([1.0]), method, max_predict_multiple)
            if s > single_best[0]:
                single_best = (s, m, t_)
        _, best_m, single_thr = single_best
        single_scores.append(macro_f1(targets[eval_idx], eval_probs[best_m], single_thr))

        # --- global weights, fitted on the fit half
        best = (-1.0, grid[0], None)
        for w in grid:
            score, thr = score_blend(fit_probs, targets[fit_idx], w, method, max_predict_multiple)
            if score > best[0]:
                best = (score, w, thr)
        _, global_w, global_thr = best
        global_scores.append(
            macro_f1(targets[eval_idx], ensemble_probabilities(eval_probs, weights=global_w, method=method), global_thr)
        )

        # --- per-class weights, fitted on the same fit half
        pc_w, pc_thr = fit_per_class_weights(
            fit_probs, targets[fit_idx], grid, method, max_predict_multiple
        )
        per_class_scores.append(
            macro_f1(targets[eval_idx], apply_per_class_weights(eval_probs, pc_w, method), pc_thr)
        )

    g, p, s = np.array(global_scores), np.array(per_class_scores), np.array(single_scores)
    return {
        "n_splits": int(len(g)),
        "single_best_mean": float(s.mean()),
        "global_mean": float(g.mean()),
        "per_class_mean": float(p.mean()),
        "mean_gain": float((p - g).mean()),
        "win_rate": float((p > g).mean()),
        # Does blending beat simply shipping the best single model?
        "global_gain_over_single": float((g - s).mean()),
        "global_win_rate_over_single": float((g > s).mean()),
    }


def weight_grid(n_models: int, step: float = 0.1) -> list[np.ndarray]:
    """All simplex weight vectors on a coarse grid (coarse on purpose, see module docs)."""
    levels = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    out: list[np.ndarray] = []
    for combo in product(levels, repeat=n_models):
        total = sum(combo)
        if total <= 0:
            continue
        candidate = np.array(combo, dtype=np.float64) / total
        if not any(np.allclose(candidate, existing) for existing in out):
            out.append(candidate)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, dest="runs", help="run directory (repeatable)")
    parser.add_argument("--out", default="outputs/blend", help="output directory")
    parser.add_argument("--method", default="mean", help="mean | gmean | rank")
    parser.add_argument("--step", type=float, default=0.1, help="weight grid resolution")
    parser.add_argument("--max-predict-multiple", type=float, default=4.0)
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.005,
        help="keep tuned weights only if they beat uniform by at least this much",
    )
    parser.add_argument(
        "--per-class",
        choices=["auto", "on", "off"],
        default="auto",
        help="per-class blend weights: 'auto' uses them only if they win held-out validation",
    )
    parser.add_argument(
        "--test-probs",
        action="append",
        default=[],
        help="test-set probability .npy per run, in --run order; enables --submission",
    )
    parser.add_argument("--submission", default=None, help="write a submission CSV using the chosen blend")
    args = parser.parse_args()

    setup_logging("logs/blend.log")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(r) for r in args.runs]
    loaded = [load_oof(d) for d in run_dirs]
    probs_list, targets, aligned_ids = align_runs(loaded)

    # --- individual members, for context
    LOGGER.info("Individual run scores (pooled OOF, tuned):")
    for run_dir, probs in zip(run_dirs, probs_list):
        score, _ = score_blend([probs], targets, np.array([1.0]), args.method, args.max_predict_multiple)
        LOGGER.info("  %-45s %.4f", run_dir.name, score)

    # --- uniform baseline
    n_models = len(probs_list)
    uniform = np.ones(n_models) / n_models
    uniform_score, uniform_thresholds = score_blend(
        probs_list, targets, uniform, args.method, args.max_predict_multiple
    )
    LOGGER.info("Uniform blend: %.4f", uniform_score)

    # --- weight search
    best_weights, best_score, best_thresholds = uniform, uniform_score, uniform_thresholds
    if n_models > 1:
        grid = weight_grid(n_models, args.step)
        LOGGER.info("Searching %d weight combination(s)...", len(grid))
        for weights in grid:
            score, thresholds = score_blend(probs_list, targets, weights, args.method, args.max_predict_multiple)
            if score > best_score:
                best_weights, best_score, best_thresholds = weights, score, thresholds
        LOGGER.info("Best searched blend: %.4f with weights %s", best_score, np.round(best_weights, 3).tolist())

    # --- decide: only accept tuned weights if the gain is worth the overfitting risk
    gain = best_score - uniform_score
    if gain < args.min_gain:
        LOGGER.info(
            "Tuned weights beat uniform by only %.4f (< --min-gain %.4f); keeping UNIFORM weights, "
            "which are far less likely to be fitted to rare-class noise",
            gain,
            args.min_gain,
        )
        chosen_weights, chosen_score, chosen_thresholds = uniform, uniform_score, uniform_thresholds
    else:
        LOGGER.info("Accepting tuned weights (+%.4f over uniform)", gain)
        chosen_weights, chosen_score, chosen_thresholds = best_weights, best_score, best_thresholds

    # --- per-class weights, gated on held-out validation
    per_class_weights: np.ndarray | None = None
    per_class_report: dict | None = None
    if n_models > 1 and args.per_class != "off":
        grid = weight_grid(n_models, args.step)
        per_class_report = validate_per_class_honestly(
            probs_list, targets, grid, args.method, args.max_predict_multiple
        )
        LOGGER.info(
            "Held-out validation (fitted and scored on DIFFERENT halves, %d splits):",
            per_class_report["n_splits"],
        )
        LOGGER.info("  best single model  %.4f", per_class_report["single_best_mean"])
        LOGGER.info(
            "  global-weight blend %.4f  (%+.4f vs single, wins %.0f%%)",
            per_class_report["global_mean"],
            per_class_report["global_gain_over_single"],
            100 * per_class_report["global_win_rate_over_single"],
        )
        LOGGER.info(
            "  per-class blend     %.4f  (%+.4f vs global, wins %.0f%%)",
            per_class_report["per_class_mean"],
            per_class_report["mean_gain"],
            100 * per_class_report["win_rate"],
        )
        # If blending does not even beat the best single model out-of-sample, say so
        # loudly -- the pooled-OOF number above will still look like an improvement
        # because its weights were fitted on those same rows.
        if per_class_report["global_gain_over_single"] <= 0:
            LOGGER.warning(
                "  !! blending does NOT beat the best single model on held-out data "
                "(%+.4f). The pooled-OOF blend score is optimistic -- prefer the single "
                "model unless adding a genuinely different architecture/data source.",
                per_class_report["global_gain_over_single"],
            )
        accept = args.per_class == "on" or (
            per_class_report["mean_gain"] > args.min_gain and per_class_report["win_rate"] >= 0.6
        )
        if accept:
            LOGGER.info("Accepting PER-CLASS weights")
            per_class_weights, chosen_thresholds = fit_per_class_weights(
                probs_list, targets, grid, args.method, args.max_predict_multiple
            )
            for c, name in enumerate(CLASSES):
                LOGGER.info("  %-9s weights=%s", name, np.round(per_class_weights[:, c], 2).tolist())
        else:
            LOGGER.info(
                "Rejecting per-class weights: they do not generalise to held-out data "
                "(11x the fitted parameters against 44-50 positives for the rare classes)"
            )

    # --- report
    blended = (
        apply_per_class_weights(probs_list, per_class_weights, args.method)
        if per_class_weights is not None
        else ensemble_probabilities(probs_list, weights=chosen_weights, method=args.method)
    )
    if per_class_weights is not None:
        from src.validation.metrics import macro_f1

        chosen_score = macro_f1(targets, blended, chosen_thresholds)
        LOGGER.info("Per-class blend pooled-OOF tuned macro-F1: %.4f", chosen_score)
    metrics = compute_metrics(targets, blended, thresholds=chosen_thresholds)
    strategies = evaluate_threshold_strategies(targets, blended, chosen_thresholds)

    LOGGER.info("Chosen blend per-class metrics:\n%s", format_metrics_table(metrics, tuned=True))
    LOGGER.info("Strategy comparison for the blend:")
    for name, value in sorted(strategies.items(), key=lambda kv: -kv[1]):
        LOGGER.info("  %-18s %.4f", name, value)
    recommended = max(strategies, key=strategies.get)

    # --- artefacts
    np.save(out_dir / "thresholds.npy", chosen_thresholds)
    np.save(out_dir / "oof_blended_probs.npy", blended)
    # Persist the targets and ids in the SAME row order as the blended probabilities,
    # so downstream analysis cannot accidentally pair them with a differently-ordered
    # per-run oof_targets.npy (see align_runs docstring).
    np.save(out_dir / "oof_blended_targets.npy", targets)
    (out_dir / "oof_blended_lesion_ids.json").write_text(json.dumps(list(aligned_ids)))
    # A bare comma-separated line, so it can be dropped straight into --weights.
    (out_dir / "weights.csv").write_text(",".join(f"{w:.6f}" for w in chosen_weights))
    with open(out_dir / "blend_summary.json", "w") as fh:
        json.dump(
            {
                "runs": [str(d) for d in run_dirs],
                "method": args.method,
                "weights": chosen_weights.tolist(),
                "uniform_score": uniform_score,
                "best_searched_score": best_score,
                "chosen_score": chosen_score,
                "gain_over_uniform": gain,
                "used_uniform": bool(gain < args.min_gain),
                "thresholds": chosen_thresholds.tolist(),
                "class_names": CLASSES,
                "strategies": strategies,
                "recommended_strategy": recommended,
                "per_class_weights": None if per_class_weights is None else per_class_weights.tolist(),
                "per_class_validation": per_class_report,
            },
            fh,
            indent=2,
            default=float,
        )
    if per_class_weights is not None:
        np.save(out_dir / "per_class_weights.npy", per_class_weights)
    LOGGER.info("Wrote weights.csv, thresholds.npy, blend_summary.json to %s", out_dir)
    LOGGER.info("Blend OOF tuned macro-F1: %.4f (recommended strategy: %s)", chosen_score, recommended)

    # --- optional: write the submission here.
    # Doing it in this script (rather than handing weights to predict.py) is the only
    # way to ship *per-class* weights, and it reuses the already-computed test
    # probabilities so no GPU inference is repeated.
    if args.submission:
        if len(args.test_probs) != len(run_dirs):
            parser.error(
                f"--submission needs one --test-probs per --run "
                f"({len(args.test_probs)} given for {len(run_dirs)} runs)"
            )
        from src.inference.predictor import write_submission
        from src.validation.thresholds import apply_strategy

        test_list, test_ids = [], None
        for path in args.test_probs:
            test_list.append(np.load(path))
            ids_path = Path(path).with_suffix(".ids.json")
            ids = json.loads(ids_path.read_text()) if ids_path.exists() else None
            if test_ids is None:
                test_ids = ids
            elif ids is not None and ids != test_ids:
                raise RuntimeError(f"Test lesion id order differs in {path}")
        if test_ids is None:
            raise RuntimeError("No .ids.json alongside the --test-probs files; cannot label rows safely")

        test_blend = (
            apply_per_class_weights(test_list, per_class_weights, args.method)
            if per_class_weights is not None
            else ensemble_probabilities(test_list, weights=chosen_weights, method=args.method)
        )
        final = apply_strategy(test_blend, strategy=recommended, thresholds=chosen_thresholds)
        write_submission(args.submission, test_ids, final)
        LOGGER.info("Wrote submission %s (strategy=%s)", args.submission, recommended)


if __name__ == "__main__":
    main()
