"""Evaluation metrics for MILK10k.

The official metric is **macro-F1 over the 11 diagnostic categories**, with
probabilities binarised at ``>= 0.5``. Everything else here exists to explain
*why* that number moves:

* ``macro_f1`` at the official 0.5 cut - the number the leaderboard reports.
* ``macro_f1_tuned`` - macro-F1 after per-class threshold optimisation. The gap
  between the two is the score currently being left on the table by calibration
  alone, and it is usually large on this dataset.
* ``macro_auc`` - threshold-free ranking quality. Use this to judge whether the
  *model* improved, since macro-F1 at a fixed cut can swing on calibration alone.
* Per-class precision/recall/F1/AUC/support - with five classes under 55 lesions,
  aggregate numbers hide almost everything that matters.
* Confusion matrix over ``argmax`` predictions - for reading which pairs the model
  actually confuses (AKIEC/SCCKA and BKL/MEL being the usual suspects).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

from ..constants import CLASSES, MALIGNANT_CLASSES, NUM_CLASSES, SUBMISSION_THRESHOLD
from ..utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 for one binary column. Returns 0.0 when a class is never predicted.

    Note the convention: if a class has no true positives *and* none predicted,
    F1 is defined here as 0.0 rather than 1.0. That is the conservative reading
    and matches how the challenge's macro average behaves for absent classes.
    """
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def macro_f1(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | float = SUBMISSION_THRESHOLD) -> float:
    """Macro-F1 across all classes at the given threshold(s)."""
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (y_prob.shape[1],))
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int8)
    return float(np.mean([binary_f1(y_true[:, c], y_pred[:, c]) for c in range(y_prob.shape[1])]))


def safe_roc_auc(y_true_col: np.ndarray, y_score_col: np.ndarray) -> float:
    """Per-class ROC-AUC, returning NaN when the column is single-class.

    A fold with zero MAL_OTH lesions has an undefined AUC; propagating NaN and
    using ``nanmean`` downstream is honest, whereas substituting 0.5 would quietly
    drag the macro average toward chance.
    """
    if len(np.unique(y_true_col)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true_col, y_score_col))
    except ValueError:  # pragma: no cover
        return float("nan")


def per_class_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | float = SUBMISSION_THRESHOLD,
    class_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Precision / recall / F1 / AUC / AP / support / n_predicted, per class."""
    class_names = class_names or CLASSES
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (y_prob.shape[1],))
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int8)

    out: dict[str, dict[str, float]] = {}
    for c, name in enumerate(class_names):
        truth, pred = y_true[:, c], y_pred[:, c]
        tp = float(np.sum((truth == 1) & (pred == 1)))
        fp = float(np.sum((truth == 0) & (pred == 1)))
        fn = float(np.sum((truth == 1) & (pred == 0)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": binary_f1(truth, pred),
            "auc": safe_roc_auc(truth, y_prob[:, c]),
            "ap": float(average_precision_score(truth, y_prob[:, c])) if truth.sum() > 0 else float("nan"),
            "support": float(truth.sum()),
            "n_predicted": float(pred.sum()),
            "threshold": float(thresholds[c]),
        }
    return out


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Full metric bundle for a set of predictions.

    Parameters
    ----------
    y_true:
        ``(N, 11)`` one-hot ground truth.
    y_prob:
        ``(N, 11)`` predicted probabilities in [0, 1].
    thresholds:
        Optional per-class thresholds. When given, ``macro_f1_tuned`` and the
        per-class block at those thresholds are added alongside the official
        0.5-threshold numbers.

    Returns
    -------
    dict
        Scalar metrics at the top level (so it can be fed straight to
        TensorBoard), with nested ``per_class`` / ``confusion_matrix`` entries.
    """
    class_names = class_names or CLASSES
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    metrics: dict[str, Any] = {}

    # --- the official metric
    metrics["macro_f1"] = macro_f1(y_true, y_prob, SUBMISSION_THRESHOLD)

    # --- threshold-free model quality
    aucs = np.array([safe_roc_auc(y_true[:, c], y_prob[:, c]) for c in range(y_prob.shape[1])])
    metrics["macro_auc"] = float(np.nanmean(aucs))
    metrics["n_classes_with_auc"] = float(np.sum(~np.isnan(aucs)))

    aps = np.array(
        [
            float(average_precision_score(y_true[:, c], y_prob[:, c])) if y_true[:, c].sum() > 0 else np.nan
            for c in range(y_prob.shape[1])
        ]
    )
    metrics["macro_ap"] = float(np.nanmean(aps))

    # --- single-label view: the ground truth is one-hot, so argmax accuracy and
    # balanced accuracy are meaningful and easy to reason about.
    true_idx = y_true.argmax(axis=1)
    pred_idx = y_prob.argmax(axis=1)
    metrics["accuracy"] = float(np.mean(true_idx == pred_idx))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(true_idx, pred_idx))
    # Macro-F1 if we submitted a hard one-hot argmax instead of probabilities --
    # a useful reference point, and occasionally the better submission.
    onehot_argmax = np.zeros_like(y_prob)
    onehot_argmax[np.arange(len(pred_idx)), pred_idx] = 1.0
    metrics["macro_f1_argmax"] = macro_f1(y_true, onehot_argmax, 0.5)

    # --- clinically meaningful secondary read: malignant vs. benign
    malignant_idx = [CLASSES.index(c) for c in MALIGNANT_CLASSES]
    mal_true = y_true[:, malignant_idx].max(axis=1)
    mal_score = y_prob[:, malignant_idx].sum(axis=1).clip(0, 1)
    metrics["malignant_auc"] = safe_roc_auc(mal_true, mal_score)

    metrics["per_class"] = per_class_metrics(y_true, y_prob, SUBMISSION_THRESHOLD, class_names)
    metrics["confusion_matrix"] = confusion_matrix(
        true_idx, pred_idx, labels=list(range(len(class_names)))
    ).tolist()

    if thresholds is not None:
        thresholds = np.asarray(thresholds, dtype=np.float64)
        metrics["macro_f1_tuned"] = macro_f1(y_true, y_prob, thresholds)
        metrics["per_class_tuned"] = per_class_metrics(y_true, y_prob, thresholds, class_names)
        metrics["thresholds"] = thresholds.tolist()

    return metrics


def flatten_metrics(metrics: dict[str, Any], include_per_class: bool = True) -> dict[str, float]:
    """Flatten the bundle into scalars suitable for TensorBoard / W&B."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            flat[key] = float(value)
        elif include_per_class and key in ("per_class", "per_class_tuned") and isinstance(value, dict):
            suffix = "_tuned" if key.endswith("tuned") else ""
            for class_name, class_metrics in value.items():
                for metric_name, metric_value in class_metrics.items():
                    if metric_name in ("f1", "recall", "precision", "auc"):
                        flat[f"class{suffix}/{class_name}/{metric_name}"] = float(metric_value)
    return flat


def format_metrics_table(metrics: dict[str, Any], tuned: bool = False) -> str:
    """Render the per-class block as a fixed-width table for the log file."""
    key = "per_class_tuned" if tuned and "per_class_tuned" in metrics else "per_class"
    per_class = metrics.get(key, {})
    if not per_class:
        return "(no per-class metrics)"

    header = f"{'class':<9} {'sup':>5} {'npred':>6} {'thr':>5} {'prec':>6} {'rec':>6} {'f1':>6} {'auc':>6}"
    lines = [header, "-" * len(header)]
    for name, m in per_class.items():
        lines.append(
            f"{name:<9} {m['support']:>5.0f} {m['n_predicted']:>6.0f} {m['threshold']:>5.2f} "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['auc']:>6.3f}"
        )
    f1_key = "macro_f1_tuned" if tuned and "macro_f1_tuned" in metrics else "macro_f1"
    lines.append("-" * len(header))
    lines.append(
        f"{'MACRO':<9} {'':>5} {'':>6} {'':>5} {'':>6} {'':>6} {metrics[f1_key]:>6.3f} {metrics['macro_auc']:>6.3f}"
    )
    return "\n".join(lines)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str] | None = None,
    normalize: bool = True,
    title: str = "Confusion matrix",
):
    """Render a confusion matrix as a matplotlib figure (row-normalised by default).

    Returns the figure so callers can log it to TensorBoard/W&B or save it. Uses
    the non-interactive Agg backend, which is what you want on a headless box.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_names = class_names or CLASSES
    matrix = np.asarray(matrix, dtype=np.float64)
    display = matrix.copy()
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums > 0)

    figure, axis = plt.subplots(figsize=(8.5, 7.5))
    image = axis.imshow(display, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    figure.colorbar(image, ax=axis, fraction=0.046)

    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)

    # Annotate cells; white text on dark cells for readability.
    threshold = display.max() / 2 if display.max() > 0 else 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{display[i, j]:.2f}" if normalize else f"{int(matrix[i, j])}"
            if normalize and display[i, j] == 0:
                text = "."
            axis.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if display[i, j] > threshold else "black",
            )
    figure.tight_layout()
    return figure
