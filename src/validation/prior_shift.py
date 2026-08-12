"""Transductive prior-shift analysis for the test set.

VERDICT: MEASURED, REJECTED. DO NOT ENABLE FOR SUBMISSIONS.
----------------------------------------------------------
The prior shift documented below is **real**. The correction for it is **not
worth applying** — it was validated against known synthetic shifts using
:func:`validate_on_simulated_shift` and lost in every scenario tested, including
the no-shift control:

======================================  =========  ==========  =======
scenario                                 baseline   corrected     gain
======================================  =========  ==========  =======
count-matching, realistic shift            0.5567      0.5045   -0.052
count-matching, aggressive shift           0.5589      0.5091   -0.050
count-matching, no shift (control)         0.5499      0.5130   -0.037
importance-weighted F1, realistic          0.5567      0.5448   -0.012
importance-weighted F1, no shift            0.5502      0.5447   -0.006
======================================  =========  ==========  =======

Two independent mechanisms were tried and both lost:

1. **Count matching** (:func:`count_matched_thresholds`) replaces F1-tuned
   thresholds with rank-based ones that hit a target positive count. It loses even
   with no shift at all, which shows the mechanism itself is the problem: the
   F1-optimal operating point is *not* the count-matching one, and giving up
   F1-optimality costs more than fixing the prior gains.
2. **Importance-weighted F1 tuning** — reweight OOF samples to the estimated test
   prior and re-tune. This preserves F1-optimality and is the theoretically correct
   fix, yet still loses slightly. The reason is variance: threshold estimates on a
   few hundred samples are already noisy, and reweighting adds more variance than
   the bias it removes. Critically, the five rare classes that carry 45% of
   macro-F1 have an estimated ratio of ~1.0, so there was little bias to remove
   there in the first place, while the reweighting perturbs the well-populated
   classes whose thresholds were already good.

**Conclusion: ship plain tuned thresholds.** The reliable way to improve those
thresholds is more data per estimate — pooled 5-fold OOF rather than one fold —
not a prior correction. This module is retained because the shift is a real
property of the data worth knowing about, and because
:func:`validate_on_simulated_shift` is the reusable harness that produced the
verdict. Nothing here is wired into ``predict.py``.

The shift itself (for the record)
---------------------------------
Per-class thresholds are tuned on out-of-fold predictions, so they are implicitly
tuned for the **training** class prior. If the test set was drawn with a different
class mix, those thresholds are in principle mis-set: too strict for classes that
became more common, too loose for classes that became rarer.

The shift is measurable. Comparing the model's mean predicted probability per class
on OOF against the test set implies:

===========  =====  ==========================================
class        ratio  effect
===========  =====  ==========================================
SCCKA        ~2.2   far more common in test than in training
AKIEC        ~1.6   more common
BKL          ~1.3   more common
BCC          ~0.6   much less common (48% of train, ~28% of test)
NV           ~0.75  less common
rare five    ~1.0   essentially unchanged
===========  =====  ==========================================

AKIEC, SCCKA and BCC are 3/11 of the metric, which is what made this worth
investigating. It turned out not to be worth acting on — see the verdict above.

What is and is not legitimate here
---------------------------------
This is a **transductive** correction: it uses the unlabelled test *inputs* (the
model's own predictions on them) and no test labels. That is standard practice and
is not leakage — the test images are given to us. What it must never do is peek at
test labels, and it does not.

The estimate is noisy, so the correction was built to be conservative — which
turned out not to be enough to make it profitable:

* ``estimate_prior_ratio`` shrinks every ratio toward 1.0 by a configurable factor,
  so a wild estimate cannot swing an operating point far.
* :func:`count_matched_thresholds` targets a predicted-positive count derived from
  the *OOF-validated* ratio of optimal-predictions-to-support, so it inherits the
  per-class conservatism that threshold tuning already discovered.
* :func:`validate_on_simulated_shift` exists so the correction can be checked
  against a known synthetic shift **before** trusting it on the real test set. It
  is what produced the negative verdict above. Any future variant of this idea must
  clear the same gate — including the no-shift control — before being shipped.
"""

from __future__ import annotations

import numpy as np

from ..constants import CLASSES
from ..utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def estimate_prior_ratio(
    oof_probs: np.ndarray,
    oof_targets: np.ndarray,
    test_probs: np.ndarray,
    shrinkage: float = 0.5,
    clip: tuple[float, float] = (0.4, 2.5),
) -> np.ndarray:
    """Estimate per-class prior ratio ``pi_test / pi_oof`` from predictions alone.

    Uses the fact that a reasonably calibrated model's **mean predicted probability**
    for a class tracks that class's prevalence. The ratio of mean probabilities on
    test versus OOF therefore estimates the prevalence ratio, without needing the
    model to be perfectly calibrated in absolute terms — only consistently biased
    across the two sets, which it is, since it is the same model.

    Parameters
    ----------
    shrinkage:
        Pull each ratio toward 1.0 by this fraction. ``0`` trusts the raw estimate,
        ``1`` disables the correction entirely. The default of 0.5 takes half the
        indicated correction, which is the right posture for an estimate this noisy:
        half of a correct adjustment recovers most of the gain, while half of a
        wrong one costs little.
    clip:
        Hard bounds applied after shrinkage.

    Returns
    -------
    np.ndarray
        ``(n_classes,)`` shrunk ratios.
    """
    oof_mean = oof_probs.mean(axis=0)
    test_mean = test_probs.mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(oof_mean > 1e-6, test_mean / oof_mean, 1.0)
    raw = np.nan_to_num(raw, nan=1.0, posinf=1.0, neginf=1.0)

    shrunk = 1.0 + (1.0 - float(shrinkage)) * (raw - 1.0)
    return np.clip(shrunk, clip[0], clip[1])


def estimate_test_counts(
    oof_targets: np.ndarray,
    ratio: np.ndarray,
    n_test: int,
    renormalise: bool = True,
) -> np.ndarray:
    """Expected number of test lesions per class, given the estimated ratios.

    ``renormalise`` rescales the counts so they sum to ``n_test``. The labels are
    single-label one-hot, so exactly one class is true per lesion and the counts
    must sum to the number of lesions — enforcing that removes a chunk of the
    estimate's error for free.
    """
    oof_prior = oof_targets.mean(axis=0)
    counts = oof_prior * np.asarray(ratio, dtype=np.float64) * n_test
    if renormalise and counts.sum() > 0:
        counts = counts * (n_test / counts.sum())
    return counts


def oof_prediction_ratios(
    oof_targets: np.ndarray,
    oof_probs: np.ndarray,
    thresholds: np.ndarray,
    default: float = 0.9,
    clip: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    """How many positives the tuned thresholds predicted, relative to true support.

    Threshold tuning discovers, per class, whether F1 is maximised by over- or
    under-predicting. On MILK10k it settles near 0.8-1.05 for the well-populated
    classes (slight under-prediction) and above 1 for the very rare ones (buying
    recall at the cost of precision). Carrying those learned ratios over to the test
    set preserves that per-class posture instead of naively targeting exact counts.
    """
    support = oof_targets.sum(axis=0)
    predicted = (oof_probs >= np.asarray(thresholds)[None, :]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(support > 0, predicted / np.maximum(support, 1e-9), default)
    ratios = np.nan_to_num(ratios, nan=default, posinf=default, neginf=default)
    return np.clip(ratios, clip[0], clip[1])


def count_matched_thresholds(
    test_probs: np.ndarray,
    target_counts: np.ndarray,
    min_threshold: float = 1e-4,
    max_threshold: float = 1.0 - 1e-4,
) -> np.ndarray:
    """Per-class thresholds that make each class predict its target number of positives.

    For class ``c`` with target count ``k``, the threshold is placed just below the
    ``k``-th highest test score, so exactly ``k`` lesions clear it. This operates on
    *ranks*, which makes it immune to the model's absolute calibration — only the
    ordering matters, and ordering is what the model is actually good at (per-class
    AUC 0.93-0.99 here versus badly-scaled probabilities).
    """
    test_probs = np.asarray(test_probs, dtype=np.float64)
    n_test, n_classes = test_probs.shape
    thresholds = np.full(n_classes, 0.5, dtype=np.float64)

    for c in range(n_classes):
        k = int(round(float(target_counts[c])))
        k = max(0, min(n_test, k))
        if k == 0:
            thresholds[c] = max_threshold
            continue
        descending = np.sort(test_probs[:, c])[::-1]
        kth = descending[k - 1]
        next_value = descending[k] if k < n_test else kth - 1e-6
        thresholds[c] = float(np.clip((kth + next_value) / 2.0, min_threshold, max_threshold))
    return thresholds


def prior_corrected_thresholds(
    oof_probs: np.ndarray,
    oof_targets: np.ndarray,
    test_probs: np.ndarray,
    tuned_thresholds: np.ndarray,
    shrinkage: float = 0.5,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """End-to-end prior-shift correction. Returns ``(test_thresholds, diagnostics)``.

    Pipeline: estimate the prior ratio, convert it to expected test counts, scale
    those by the OOF-validated over/under-prediction ratios, then place thresholds
    by rank so each class predicts that many positives.
    """
    ratio = estimate_prior_ratio(oof_probs, oof_targets, test_probs, shrinkage=shrinkage)
    counts = estimate_test_counts(oof_targets, ratio, n_test=len(test_probs))
    pred_ratio = oof_prediction_ratios(oof_targets, oof_probs, tuned_thresholds)
    target_counts = counts * pred_ratio
    thresholds = count_matched_thresholds(test_probs, target_counts)

    diagnostics = {
        "prior_ratio": ratio.tolist(),
        "estimated_test_counts": counts.tolist(),
        "oof_prediction_ratios": pred_ratio.tolist(),
        "target_predicted_counts": target_counts.tolist(),
        "thresholds": thresholds.tolist(),
        "shrinkage": float(shrinkage),
    }

    if verbose:
        LOGGER.info("Prior-shift correction (shrinkage=%.2f):", shrinkage)
        LOGGER.info(
            "  %-9s %6s %9s %8s %8s %8s", "class", "ratio", "est_count", "pred_r", "target", "thr"
        )
        for c, name in enumerate(CLASSES):
            LOGGER.info(
                "  %-9s %6.2f %9.1f %8.2f %8.1f %8.3f",
                name,
                ratio[c],
                counts[c],
                pred_ratio[c],
                target_counts[c],
                thresholds[c],
            )
    return thresholds, diagnostics


# ---------------------------------------------------------------------------
# Validation: does the correction actually help under a known shift?
# ---------------------------------------------------------------------------
def validate_on_simulated_shift(
    oof_probs: np.ndarray,
    oof_targets: np.ndarray,
    shift: np.ndarray,
    n_repeats: int = 20,
    fraction: float = 0.5,
    shrinkage: float = 0.5,
    seed: int = 0,
) -> dict:
    """Check the correction against a synthetic prior shift with known labels.

    Splits the OOF set into a "reference" half and a "shifted" half, resamples the
    shifted half to the requested class mix, and compares macro-F1 from thresholds
    tuned on the reference half **with** and **without** the correction.

    This is the gate for using the correction in a real submission: if it does not
    beat uncorrected thresholds on a shift of a size we actually believe, it should
    not be trusted on the real test set either.

    Returns a dict with mean/std macro-F1 for both arms and the mean gain.
    """
    from .thresholds import optimize_thresholds
    from .metrics import macro_f1

    rng = np.random.default_rng(seed)
    labels = oof_targets.argmax(axis=1)
    n = len(oof_probs)
    shift = np.asarray(shift, dtype=np.float64)

    baseline_scores, corrected_scores = [], []

    for repeat in range(n_repeats):
        # --- split into reference / shifted pools
        order = rng.permutation(n)
        n_ref = int(n * fraction)
        ref_idx, pool_idx = order[:n_ref], order[n_ref:]

        # --- tune thresholds on the reference half (stands in for pooled OOF)
        thresholds = optimize_thresholds(oof_targets[ref_idx], oof_probs[ref_idx], verbose=False)

        # --- build a shifted sample from the remaining pool
        pool_labels = labels[pool_idx]
        selected: list[int] = []
        for c in range(oof_targets.shape[1]):
            members = pool_idx[pool_labels == c]
            if len(members) == 0:
                continue
            take = int(round(len(members) * shift[c]))
            take = max(0, min(take, len(members)) if shift[c] <= 1 else take)
            replace = take > len(members)
            selected.extend(rng.choice(members, size=take, replace=replace).tolist())
        if len(selected) < 50:
            continue
        selected_idx = np.array(selected)
        shifted_probs = oof_probs[selected_idx]
        shifted_targets = oof_targets[selected_idx]

        # --- arm 1: reference thresholds applied as-is
        baseline_scores.append(macro_f1(shifted_targets, shifted_probs, thresholds))

        # --- arm 2: thresholds corrected using only the shifted *inputs*
        corrected, _ = prior_corrected_thresholds(
            oof_probs[ref_idx],
            oof_targets[ref_idx],
            shifted_probs,
            thresholds,
            shrinkage=shrinkage,
            verbose=False,
        )
        corrected_scores.append(macro_f1(shifted_targets, shifted_probs, corrected))

    baseline = np.array(baseline_scores)
    corrected = np.array(corrected_scores)
    result = {
        "n_repeats": int(len(baseline)),
        "baseline_mean": float(baseline.mean()) if len(baseline) else float("nan"),
        "baseline_std": float(baseline.std()) if len(baseline) else float("nan"),
        "corrected_mean": float(corrected.mean()) if len(corrected) else float("nan"),
        "corrected_std": float(corrected.std()) if len(corrected) else float("nan"),
        "mean_gain": float((corrected - baseline).mean()) if len(baseline) else float("nan"),
        "win_rate": float((corrected > baseline).mean()) if len(baseline) else float("nan"),
        "shrinkage": float(shrinkage),
    }
    return result
