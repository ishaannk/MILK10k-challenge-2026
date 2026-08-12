"""Per-class threshold optimisation, and how to ship it under a fixed 0.5 cut.

Why this module matters more than it looks
------------------------------------------
The leaderboard binarises at exactly ``0.5`` and then macro-averages F1 over 11
classes. The training distribution is extremely skewed::

    BCC 2522 | NV 746 | BKL 544 | SCCKA 473 | MEL 450 | AKIEC 303
    DF 52 | INF 50 | VASC 47 | BEN_OTH 44 | MAL_OTH 9

Under macro-F1 every class is worth 1/11 of the score. The five rarest classes are
3.9% of the lesions but **45% of the metric**. A well-calibrated model almost never
emits ``p > 0.5`` for a class with 50 training examples, so those five classes score
F1 = 0 and cap the achievable macro-F1 near 0.55 no matter how good the encoder is.

Lowering a rare class's effective threshold trades a little precision for a lot of
recall, and for a rare class that trade is strongly F1-positive. Concretely: a class
with 4 true positives that currently predicts nothing scores 0. Predicting 8 lesions
and catching 3 of them scores ``2*3 / (2*3 + 5 + 1) = 0.50``. Meanwhile BCC giving up
a few points of precision costs a fraction of its own F1. That asymmetry is the whole
game.

Two facts make this tractable:

1. **The optimisation is separable.** ``macro_f1 = mean_c F1_c``, and ``F1_c`` depends
   only on class ``c``'s own threshold. So an independent 1-D search per class is
   *globally* optimal — no coordinate ascent or joint search is needed.
2. **Any threshold can be folded back into the probabilities.**
   :func:`rescale_probabilities` applies a monotone, piecewise-linear map that sends
   ``t_c -> 0.5`` while preserving within-class ranking. So we can honour the fixed
   0.5 submission rule and still submit tuned decisions, with ROC-AUC untouched.

The remaining risk is overfitting the thresholds themselves: with ~10 validation
positives for a rare class, the argmax threshold is noisy. Mitigations used here:
tune on **pooled out-of-fold predictions** rather than a single fold, and select the
**centre of the widest near-optimal plateau** instead of the raw argmax.
"""

from __future__ import annotations

import numpy as np

from ..constants import CLASSES, NUM_CLASSES, SUBMISSION_THRESHOLD
from ..utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Thresholds are clamped away from the open ends so the rescaling map stays finite.
MIN_THRESHOLD = 1e-4
MAX_THRESHOLD = 1.0 - 1e-4


def f1_curve_for_class(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All achievable F1 values for one class, and the thresholds that achieve them.

    Sort scores descending; predicting the top ``k`` as positive gives
    ``TP_k = cumsum(y)[k]`` and, because ``FP + FN = (k - TP) + (P - TP)``,

        ``F1_k = 2 * TP_k / (k + P)``

    which is an O(N log N) sweep over every distinct decision boundary rather than a
    grid search. Returns ``(thresholds, f1_values)`` for ``k = 1..N``.
    """
    order = np.argsort(-y_score, kind="mergesort")
    scores_sorted = y_score[order]
    truth_sorted = y_true[order]

    positives = float(truth_sorted.sum())
    if positives == 0:
        return np.array([]), np.array([])

    k = np.arange(1, len(y_score) + 1, dtype=np.float64)
    tp = np.cumsum(truth_sorted, dtype=np.float64)
    f1 = 2.0 * tp / (k + positives)

    # A threshold that admits exactly the top k items: midway between score k-1
    # and score k (and just below the last score for k = N).
    next_scores = np.append(scores_sorted[1:], scores_sorted[-1] - 1e-6)
    thresholds = (scores_sorted + next_scores) / 2.0
    return thresholds, f1


def optimize_threshold_for_class(
    y_true: np.ndarray,
    y_score: np.ndarray,
    plateau_tolerance: float = 0.01,
    default: float = SUBMISSION_THRESHOLD,
    min_threshold: float = 0.02,
    max_threshold: float = 0.95,
    max_predict_multiple: float | None = 4.0,
) -> tuple[float, float]:
    """Best threshold for one class, chosen robustly. Returns ``(threshold, f1)``.

    Instead of the raw argmax, this takes every threshold whose F1 is within
    ``plateau_tolerance`` (relative) of the best, finds the widest contiguous run of
    those, and returns its midpoint. On rare classes the F1 curve is a step function
    with wide flat tops, and the argmax often sits at the very edge of a step — one
    validation lesion moving would push it off. The plateau centre is the same F1 on
    validation and materially more stable on unseen data.

    ``min_threshold`` / ``max_threshold`` bound the result. The lower bound stops a
    degenerate "predict this class for everything" solution, which can genuinely
    maximise F1 for an ultra-rare class on validation but is a bad bet on test.

    ``max_predict_multiple`` is the stronger guard, and it matters most for the
    classes that matter most. It caps the number of predicted positives at that
    multiple of the class's own support. Without it, a class with 2 validation
    positives can "win" by predicting 111 of 1,048 lesions: that scores F1 = 0.018
    on validation (better than 0) but is pure noise-fitting and transfers as
    ~0.005 on test, while also polluting every other class's precision. Capping at
    4x support keeps the recall gain that threshold tuning is *for* while bounding
    the false positives it can buy. Set to ``None`` to disable.
    """
    if y_true.sum() == 0:
        # No positives to fit: keep the default and let the prior-based fallback
        # in `optimize_thresholds` decide whether to move it.
        return float(default), 0.0

    thresholds, f1 = f1_curve_for_class(y_true, y_score)
    if len(thresholds) == 0:
        return float(default), 0.0

    bounds_ok = (thresholds >= min_threshold) & (thresholds <= max_threshold)

    # Position i in the curve corresponds to predicting the top (i+1) scores
    # positive, so the cap is a simple prefix constraint.
    count_ok = np.ones_like(bounds_ok)
    if max_predict_multiple is not None:
        max_positives = max(1, int(np.ceil(max_predict_multiple * y_true.sum())))
        count_ok = np.arange(1, len(thresholds) + 1) <= max_positives

    valid = bounds_ok & count_ok
    if not valid.any():
        # The two constraints can be jointly unsatisfiable: if the top-k scores all
        # sit above `max_threshold`, no threshold both respects the count cap and
        # falls inside the bounds. The count cap is the constraint that actually
        # protects against overfitting, so relax the cosmetic bounds first and keep
        # it -- relaxing the cap instead would silently restore the unbounded
        # behaviour this guard exists to prevent.
        valid = count_ok if count_ok.any() else bounds_ok
    if not valid.any():
        valid = np.ones_like(thresholds, dtype=bool)

    best_f1 = float(f1[valid].max())
    if best_f1 <= 0:
        # The model cannot rank this class at all: no admissible threshold catches a
        # single true positive. Returning `default` (0.5) would be actively harmful --
        # with uninformative scores roughly half the lesions sit above 0.5, so the
        # class would flood the submission with false positives and drag down every
        # other class's precision for zero gain. Fall back to the tightest
        # cap-respecting threshold instead: F1 is 0 either way, but the collateral
        # damage is bounded.
        if max_predict_multiple is not None and count_ok.any():
            return float(np.clip(thresholds[count_ok.nonzero()[0][-1]], MIN_THRESHOLD, MAX_THRESHOLD)), 0.0
        return float(default), 0.0

    near_optimal = valid & (f1 >= best_f1 * (1.0 - plateau_tolerance))

    # Widest contiguous run of near-optimal thresholds -> take its midpoint.
    best_run, current_start = (0, 0, 0), None
    for i, flag in enumerate(near_optimal):
        if flag and current_start is None:
            current_start = i
        elif not flag and current_start is not None:
            if i - current_start > best_run[0]:
                best_run = (i - current_start, current_start, i - 1)
            current_start = None
    if current_start is not None and len(near_optimal) - current_start > best_run[0]:
        best_run = (len(near_optimal) - current_start, current_start, len(near_optimal) - 1)

    _, start, end = best_run
    threshold = float((thresholds[start] + thresholds[end]) / 2.0)
    return float(np.clip(threshold, MIN_THRESHOLD, MAX_THRESHOLD)), best_f1


def optimize_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    plateau_tolerance: float = 0.01,
    min_threshold: float = 0.02,
    max_threshold: float = 0.95,
    max_predict_multiple: float | None = 4.0,
    prior_fallback: np.ndarray | None = None,
    class_names: list[str] | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Optimise all 11 thresholds independently (globally optimal, see module docs).

    Parameters
    ----------
    y_true, y_prob:
        ``(N, 11)`` one-hot truth and predicted probabilities. Prefer pooled
        out-of-fold predictions: more positives per rare class means less
        threshold overfitting.
    max_predict_multiple:
        Cap on predicted positives as a multiple of each class's support; see
        :func:`optimize_threshold_for_class`. The main defence against threshold
        overfitting on the ultra-rare classes.
    prior_fallback:
        Optional expected per-class positive *rate*. For a class with no validation
        positives at all, there is nothing to fit, so the threshold is instead set
        to the quantile that predicts roughly the prior number of positives. Without
        this such a class is guaranteed F1 = 0.

    Returns
    -------
    np.ndarray
        ``(11,)`` thresholds.
    """
    class_names = class_names or CLASSES
    num_classes = y_prob.shape[1]
    thresholds = np.full(num_classes, SUBMISSION_THRESHOLD, dtype=np.float64)
    achieved = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        if y_true[:, c].sum() == 0 and prior_fallback is not None:
            # Predict about as many positives as the prior implies.
            rate = float(np.clip(prior_fallback[c], 1.0 / len(y_prob), 1.0))
            thresholds[c] = float(np.clip(np.quantile(y_prob[:, c], 1.0 - rate), min_threshold, max_threshold))
            continue
        thresholds[c], achieved[c] = optimize_threshold_for_class(
            y_true[:, c],
            y_prob[:, c],
            plateau_tolerance=plateau_tolerance,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            max_predict_multiple=max_predict_multiple,
        )

    if verbose:
        LOGGER.info("Tuned thresholds:")
        for c, name in enumerate(class_names):
            LOGGER.info(
                "  %-8s thr=%.4f  f1=%.4f  (support=%d, predicted=%d)",
                name,
                thresholds[c],
                achieved[c],
                int(y_true[:, c].sum()),
                int((y_prob[:, c] >= thresholds[c]).sum()),
            )
    return thresholds


def rescale_probabilities(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Re-map probabilities so that a fixed 0.5 cut reproduces the tuned decisions.

    For each class with tuned threshold ``t``, apply the monotone piecewise-linear map

        ``p <= t  ->  0.5 * p / t``
        ``p >  t  ->  0.5 + 0.5 * (p - t) / (1 - t)``

    which sends ``t -> 0.5``, ``0 -> 0`` and ``1 -> 1``. Two consequences:

    * ``p' >= 0.5`` exactly when ``p >= t``, so the submission's mandated 0.5
      threshold now implements the tuned decision boundary.
    * The map is strictly increasing, so within-class ranking — and therefore
      ROC-AUC and average precision — is unchanged.

    This is how threshold tuning is *shipped* rather than merely measured.
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)
    thresholds = np.clip(np.asarray(thresholds, dtype=np.float64), MIN_THRESHOLD, MAX_THRESHOLD)
    thresholds = np.broadcast_to(thresholds, (y_prob.shape[1],))[None, :]

    below = 0.5 * y_prob / thresholds
    above = 0.5 + 0.5 * (y_prob - thresholds) / (1.0 - thresholds)
    return np.clip(np.where(y_prob <= thresholds, below, above), 0.0, 1.0)


def nested_threshold_estimate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    holdout_size: int = 479,
    n_repeats: int = 200,
    seed: int = 0,
    **tuning_kwargs,
) -> dict[str, float]:
    """Unbiased estimate of the score tuned thresholds will actually achieve.

    Why this exists
    ---------------
    The headline ``macro_f1_tuned`` everywhere else in this codebase fits thresholds
    on a set and then scores them *on that same set*. That is optimistically biased,
    and on MILK10k the bias is not small. Measured on the Stage 1 5-fold pooled OOF:

        fit & score on the same 5,240 lesions   0.5635   <- the number normally quoted
        fit on 4,761, score on a disjoint 479   0.5327 +/- 0.0400   <- honest

    The 0.031 gap decomposes into about -0.018 from scoring only 479 lesions (rare
    classes land 1-5 positives, so their F1 is violently noisy) and about -0.013 from
    the thresholds themselves being fitted rather than known.

    This function reproduces the real setup: fit thresholds on everything except a
    random ``holdout_size`` slice, score on that slice, repeat. ``holdout_size``
    defaults to 479, the size of the MILK10k test set, so the spread it reports is
    also a direct estimate of leaderboard noise — which is large enough that gaps
    under roughly 0.05 between adjacent leaderboard ranks are not meaningful.

    One caveat in the optimistic direction: OOF predictions come from a single model
    per lesion, whereas the submission averages all folds with TTA, so the real score
    should land somewhat above this estimate.

    Returns mean/std and the 5th-95th percentile band.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    from .metrics import macro_f1

    rng = np.random.default_rng(seed)
    n = len(y_prob)
    holdout_size = int(min(holdout_size, max(1, n // 2)))

    scores = []
    for _ in range(n_repeats):
        order = rng.permutation(n)
        holdout, fit = order[:holdout_size], order[holdout_size:]
        thresholds = optimize_thresholds(y_true[fit], y_prob[fit], verbose=False, **tuning_kwargs)
        scores.append(macro_f1(y_true[holdout], y_prob[holdout], thresholds))

    scores = np.asarray(scores)
    return {
        "honest_macro_f1_mean": float(scores.mean()),
        "honest_macro_f1_std": float(scores.std()),
        "honest_macro_f1_p5": float(np.percentile(scores, 5)),
        "honest_macro_f1_p95": float(np.percentile(scores, 95)),
        "holdout_size": float(holdout_size),
        "n_repeats": float(n_repeats),
    }


def evaluate_threshold_strategies(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Compare the submission strategies we could actually choose between.

    Reported strategies:

    ``default_0.5``
        Submit raw probabilities. The honest baseline.
    ``tuned``
        Submit rescaled probabilities from per-class tuned thresholds.
    ``argmax``
        Submit a hard one-hot at the argmax class. Guarantees exactly one positive
        per lesion; strong for common classes, but scores 0 on any class that never
        wins the argmax.
    ``tuned_or_argmax``
        Union of the two: tuned decisions, plus the argmax class forced positive so
        every lesion gets at least one prediction. Usually the best of the three
        because it keeps argmax's precision on common classes and adds tuned recall
        on rare ones.

    Whichever wins here on out-of-fold data is the one to ship; ``predict.py``
    exposes the same choice via ``--strategy``.
    """
    from .metrics import macro_f1

    out = {"default_0.5": macro_f1(y_true, y_prob, SUBMISSION_THRESHOLD)}

    argmax_onehot = np.zeros_like(y_prob)
    argmax_onehot[np.arange(len(y_prob)), y_prob.argmax(axis=1)] = 1.0
    out["argmax"] = macro_f1(y_true, argmax_onehot, 0.5)

    if thresholds is not None:
        rescaled = rescale_probabilities(y_prob, thresholds)
        out["tuned"] = macro_f1(y_true, rescaled, SUBMISSION_THRESHOLD)
        # Force the argmax class above 0.5 so no lesion is left with no prediction.
        combined = rescaled.copy()
        rows = np.arange(len(combined))
        argmax_idx = y_prob.argmax(axis=1)
        combined[rows, argmax_idx] = np.maximum(combined[rows, argmax_idx], 0.5)
        out["tuned_or_argmax"] = macro_f1(y_true, combined, SUBMISSION_THRESHOLD)

    return out


def apply_strategy(
    y_prob: np.ndarray,
    strategy: str = "tuned_or_argmax",
    thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """Produce the probability matrix to submit, under the chosen strategy.

    Mirrors :func:`evaluate_threshold_strategies` so the strategy validated
    out-of-fold is exactly the one applied at inference.
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if strategy == "raw":
        return y_prob

    if strategy == "argmax":
        out = np.zeros_like(y_prob)
        out[np.arange(len(y_prob)), y_prob.argmax(axis=1)] = 1.0
        return out

    if thresholds is None:
        raise ValueError(f"strategy={strategy!r} requires tuned thresholds")

    rescaled = rescale_probabilities(y_prob, thresholds)
    if strategy == "tuned":
        return rescaled

    if strategy == "tuned_or_argmax":
        rows = np.arange(len(y_prob))
        argmax_idx = y_prob.argmax(axis=1)
        rescaled[rows, argmax_idx] = np.maximum(rescaled[rows, argmax_idx], 0.5)
        return rescaled

    raise ValueError(f"Unknown strategy {strategy!r} (raw/argmax/tuned/tuned_or_argmax)")
