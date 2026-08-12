#!/usr/bin/env python
"""Sanity tests for the invariants that would silently corrupt a submission.

Run with::

    /opt/milkenv/bin/python -m pytest tests/ -q
    /opt/milkenv/bin/python tests/test_pipeline.py     # no pytest needed

These are deliberately focused on the properties where a bug is *silent* — where
training would complete, metrics would look plausible, and the submission would be
quietly wrong. Ordinary shape errors surface on their own; label misalignment and
threshold-rescaling errors do not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import CLASSES, NUM_CLASSES, SUBMISSION_THRESHOLD
from src.datasets.metadata import MetadataProcessor, build_lesion_table
from src.datasets.transforms import build_eval_transform, build_train_transform
from src.inference.predictor import build_submission_frame, validate_submission
from src.inference.tta import TTA_POLICIES, apply_view
from src.utils.config import Config, deep_merge, load_config
from src.validation.metrics import binary_f1, compute_metrics, macro_f1
from src.validation.thresholds import (
    apply_strategy,
    f1_curve_for_class,
    optimize_thresholds,
    rescale_probabilities,
)

RNG = np.random.default_rng(0)
PROCESSED = Path("data/processed")


# ---------------------------------------------------------------------------
# Threshold rescaling: the property the whole submission strategy rests on
# ---------------------------------------------------------------------------
def test_rescale_moves_decision_boundary_to_half():
    """``p' >= 0.5`` must hold exactly when ``p >= t``, for every class."""
    probs = RNG.random((500, NUM_CLASSES))
    thresholds = RNG.uniform(0.05, 0.9, NUM_CLASSES)
    rescaled = rescale_probabilities(probs, thresholds)

    expected = probs >= thresholds[None, :]
    actual = rescaled >= SUBMISSION_THRESHOLD
    assert np.array_equal(expected, actual), "rescaling changed which lesions are positive"


def test_rescale_is_strictly_monotone_so_auc_is_preserved():
    """Ranking within each class must survive rescaling, or AUC would move."""
    probs = RNG.random((300, NUM_CLASSES))
    targets = np.zeros_like(probs)
    targets[np.arange(300), RNG.integers(0, NUM_CLASSES, 300)] = 1.0
    thresholds = RNG.uniform(0.05, 0.9, NUM_CLASSES)

    rescaled = rescale_probabilities(probs, thresholds)
    for c in range(NUM_CLASSES):
        order_before = np.argsort(probs[:, c], kind="stable")
        order_after = np.argsort(rescaled[:, c], kind="stable")
        assert np.array_equal(order_before, order_after), f"class {c} ranking changed"

    before = compute_metrics(targets, probs)["macro_auc"]
    after = compute_metrics(targets, rescaled)["macro_auc"]
    assert abs(before - after) < 1e-9, f"AUC moved: {before} -> {after}"


def test_rescale_keeps_probabilities_in_range():
    probs = np.concatenate([np.zeros((1, NUM_CLASSES)), np.ones((1, NUM_CLASSES)), RNG.random((50, NUM_CLASSES))])
    rescaled = rescale_probabilities(probs, RNG.uniform(0.01, 0.99, NUM_CLASSES))
    assert rescaled.min() >= 0.0 and rescaled.max() <= 1.0
    assert np.isfinite(rescaled).all()


def test_tuned_f1_at_half_equals_raw_f1_at_thresholds():
    """Tuning then rescaling must not change the score it promised."""
    probs = RNG.random((400, NUM_CLASSES))
    targets = np.zeros_like(probs)
    targets[np.arange(400), RNG.integers(0, NUM_CLASSES, 400)] = 1.0

    thresholds = optimize_thresholds(targets, probs, verbose=False)
    direct = macro_f1(targets, probs, thresholds)
    shipped = macro_f1(targets, rescale_probabilities(probs, thresholds), SUBMISSION_THRESHOLD)
    assert abs(direct - shipped) < 1e-12, f"{direct} != {shipped}"


# ---------------------------------------------------------------------------
# Threshold search correctness
# ---------------------------------------------------------------------------
def test_f1_curve_matches_brute_force():
    """The O(N log N) sweep must agree with a naive per-threshold computation."""
    scores = RNG.random(200)
    truth = (RNG.random(200) < 0.15).astype(np.int8)

    thresholds, f1_fast = f1_curve_for_class(truth, scores)
    for i in RNG.choice(len(thresholds), size=25, replace=False):
        brute = binary_f1(truth, (scores >= thresholds[i]).astype(np.int8))
        assert abs(brute - f1_fast[i]) < 1e-12, f"index {i}: {brute} vs {f1_fast[i]}"


def test_tuned_thresholds_beat_the_default_on_imbalanced_data():
    """The whole point: tuning must not be worse than submitting raw probabilities.

    Simulates the real situation — a well-ranked but poorly-calibrated model whose
    rare-class probabilities never reach 0.5.
    """
    n = 1200
    labels = RNG.choice(NUM_CLASSES, size=n, p=[0.06, 0.48, 0.01, 0.10, 0.01, 0.01, 0.002, 0.086, 0.14, 0.09, 0.012])
    targets = np.zeros((n, NUM_CLASSES))
    targets[np.arange(n), labels] = 1.0

    # Informative but squashed scores: the rare classes rank well yet sit under 0.5.
    probs = RNG.random((n, NUM_CLASSES)) * 0.35
    probs[np.arange(n), labels] += RNG.uniform(0.15, 0.45, n)
    probs = np.clip(probs, 0, 1)

    thresholds = optimize_thresholds(targets, probs, verbose=False)
    baseline = macro_f1(targets, probs, SUBMISSION_THRESHOLD)
    tuned = macro_f1(targets, probs, thresholds)
    assert tuned >= baseline, f"tuning made things worse: {tuned} < {baseline}"


def test_max_predict_multiple_bounds_rare_class_false_positives():
    """The overfit guard must actually cap predicted positives."""
    n, rare = 1000, 3
    targets = np.zeros((n, NUM_CLASSES))
    targets[np.arange(n), RNG.integers(0, NUM_CLASSES, n)] = 0.0
    targets[:, 0] = 0.0
    targets[:rare, 0] = 1.0                       # class 0 has exactly 3 positives
    targets[rare:, 1] = 1.0                       # everything else is class 1
    probs = RNG.random((n, NUM_CLASSES))          # pure noise -> no real signal

    thresholds = optimize_thresholds(targets, probs, max_predict_multiple=4.0, verbose=False)
    n_predicted = int((probs[:, 0] >= thresholds[0]).sum())
    assert n_predicted <= 4 * rare, f"predicted {n_predicted} positives for {rare} true (cap 4x)"


def test_strategies_produce_valid_probability_matrices():
    probs = RNG.random((100, NUM_CLASSES))
    thresholds = RNG.uniform(0.1, 0.9, NUM_CLASSES)
    for strategy in ("raw", "argmax", "tuned", "tuned_or_argmax"):
        out = apply_strategy(probs, strategy=strategy, thresholds=thresholds)
        assert out.shape == probs.shape
        assert out.min() >= 0.0 and out.max() <= 1.0, strategy
        assert np.isfinite(out).all(), strategy

    # tuned_or_argmax must leave every lesion with at least one positive.
    combined = apply_strategy(probs, "tuned_or_argmax", thresholds)
    assert ((combined >= SUBMISSION_THRESHOLD).sum(axis=1) >= 1).all()

    # argmax must leave exactly one.
    hard = apply_strategy(probs, "argmax")
    assert ((hard >= SUBMISSION_THRESHOLD).sum(axis=1) == 1).all()


# ---------------------------------------------------------------------------
# Submission contract
# ---------------------------------------------------------------------------
def test_submission_columns_match_the_official_order():
    ids = [f"IL_{i:07d}" for i in range(10)]
    frame = build_submission_frame(ids, RNG.random((10, NUM_CLASSES)))
    assert list(frame.columns) == ["lesion", *CLASSES]
    assert not validate_submission(frame, expected_ids=ids) or all(
        p.startswith("NOTE:") for p in validate_submission(frame, expected_ids=ids)
    )


def test_submission_validation_catches_real_problems():
    ids = [f"IL_{i:07d}" for i in range(5)]
    good = build_submission_frame(ids, RNG.random((5, NUM_CLASSES)))

    # Out-of-range probabilities
    bad = good.copy()
    bad.loc[0, "MEL"] = 1.5
    assert any("out of [0, 1]" in p for p in validate_submission(bad))

    # Duplicate lesion ids
    bad = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    assert any("Duplicate" in p for p in validate_submission(bad))

    # A missing lesion relative to the expected list
    assert any("missing" in p for p in validate_submission(good.iloc[:3], expected_ids=ids))

    # NaN
    bad = good.copy()
    bad.loc[1, "NV"] = np.nan
    assert any("Non-finite" in p for p in validate_submission(bad))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_perfect_and_inverted_predictions_score_as_expected():
    n = 200
    labels = RNG.integers(0, NUM_CLASSES, n)
    targets = np.zeros((n, NUM_CLASSES))
    targets[np.arange(n), labels] = 1.0

    assert abs(macro_f1(targets, targets, 0.5) - 1.0) < 1e-12
    metrics = compute_metrics(targets, targets)
    assert abs(metrics["accuracy"] - 1.0) < 1e-12
    assert abs(metrics["macro_auc"] - 1.0) < 1e-12

    # Predicting nothing gives macro-F1 of exactly 0 under our convention.
    assert macro_f1(targets, np.zeros_like(targets), 0.5) == 0.0


def test_absent_class_yields_nan_auc_not_a_silent_half():
    """A class with no positives must yield NaN, not be quietly scored as chance.

    Only AKIEC and BCC are present here, so exactly those two have a defined AUC
    (a column that is all-ones is just as undefined as one that is all-zeros).
    """
    n = 100
    targets = np.zeros((n, NUM_CLASSES))
    targets[: n // 2, 0] = 1.0                # AKIEC
    targets[n // 2 :, 1] = 1.0                # BCC
    probs = RNG.random((n, NUM_CLASSES))
    metrics = compute_metrics(targets, probs)

    assert metrics["n_classes_with_auc"] == 2.0, metrics["n_classes_with_auc"]
    assert np.isnan(metrics["per_class"]["MEL"]["auc"]), "absent class should have NaN AUC"
    assert not np.isnan(metrics["per_class"]["BCC"]["auc"])
    # nanmean must ignore the NaNs rather than propagate them.
    assert np.isfinite(metrics["macro_auc"])


# ---------------------------------------------------------------------------
# Config system
# ---------------------------------------------------------------------------
def test_deep_merge_overrides_leaves_and_preserves_siblings():
    base = Config({"a": {"b": 1, "c": 2}, "d": 3})
    merged = deep_merge(base, {"a": {"b": 99}})
    assert merged.a.b == 99 and merged.a.c == 2 and merged.d == 3
    assert base.a.b == 1, "deep_merge mutated its input"


def test_config_inheritance_and_cli_overrides():
    cfg = load_config("configs/stage3_meta.yaml", ["train.epochs=7", "model.fusion=concat"])
    assert cfg.train.epochs == 7
    assert cfg.model.fusion == "concat"
    assert cfg.model.use_clinical is True and cfg.model.use_metadata is True
    # Inherited from base.yaml, untouched by the child config
    assert cfg.optim.name == "adamw"
    assert cfg.num_classes == NUM_CLASSES


# ---------------------------------------------------------------------------
# Metadata / dataset wiring
# ---------------------------------------------------------------------------
def test_metadata_processor_roundtrips_and_is_deterministic(tmp_path=Path("/tmp")):
    frame = pd.read_csv(PROCESSED / "train_lesions.csv").head(400)
    processor = MetadataProcessor().fit(frame)
    first = processor.transform(frame)

    assert first.shape == (len(frame), processor.dim)
    assert np.isfinite(first).all(), "metadata features contain NaN/inf"
    assert np.array_equal(first, processor.transform(frame)), "transform is not deterministic"

    path = tmp_path / "meta_processor_test.json"
    processor.save(path)
    reloaded = MetadataProcessor.load(path)
    assert reloaded.dim == processor.dim
    assert np.allclose(reloaded.transform(frame), first), "save/load changed the encoding"
    path.unlink(missing_ok=True)


def test_test_metadata_encodes_with_train_statistics_and_same_width():
    """Test features must have identical width to train features, or the model breaks."""
    train = pd.read_csv(PROCESSED / "train_lesions.csv")
    test = pd.read_csv(PROCESSED / "test_lesions.csv")
    processor = MetadataProcessor().fit(train)
    assert processor.transform(test).shape[1] == processor.transform(train).shape[1]


def test_lesion_table_pairs_exactly_one_image_per_modality():
    """Every lesion must carry two distinct image ids — a half-pair is silent corruption."""
    frame = build_lesion_table("data/raw/MILK10k_Training_Metadata.csv", "data/raw/MILK10k_Training_GroundTruth.csv")
    assert len(frame) == 5240, f"expected 5240 lesions, got {len(frame)}"
    assert frame["lesion_id"].is_unique
    assert frame["isic_id_clin"].notna().all() and frame["isic_id_derm"].notna().all()
    assert (frame["isic_id_clin"] != frame["isic_id_derm"]).all(), "clinical and dermoscopic ids collide"
    # Labels are one-hot, exactly one class per lesion.
    assert np.array_equal(frame[CLASSES].to_numpy().sum(axis=1), np.ones(len(frame)))


def test_folds_partition_the_training_set_without_overlap():
    folds = pd.read_csv("data/processed/folds.csv")
    train = pd.read_csv(PROCESSED / "train_lesions.csv")
    assert len(folds) == len(train)
    assert folds["lesion_id"].is_unique
    assert set(folds["lesion_id"]) == set(train["lesion_id"])
    # Stratification should give every fold at least one lesion of every class.
    counts = folds.groupby("fold")["label"].value_counts().unstack(fill_value=0)
    assert (counts > 0).all().all(), f"a fold is missing a class entirely:\n{counts}"


# ---------------------------------------------------------------------------
# Transforms and TTA
# ---------------------------------------------------------------------------
def test_transforms_produce_correctly_shaped_normalised_tensors():
    image = (RNG.random((450, 600, 3)) * 255).astype(np.uint8)
    for modality in ("clinical", "dermoscopic"):
        out = build_train_transform(224, modality=modality)(image=image)["image"]
        assert tuple(out.shape) == (3, 224, 224), out.shape
        assert out.dtype.is_floating_point
    out = build_eval_transform(224)(image=image)["image"]
    assert tuple(out.shape) == (3, 224, 224)


def test_eval_transform_is_deterministic():
    image = (RNG.random((450, 600, 3)) * 255).astype(np.uint8)
    transform = build_eval_transform(224)
    a = transform(image=image)["image"]
    b = transform(image=image)["image"]
    assert np.array_equal(a.numpy(), b.numpy()), "eval transform is not deterministic"


def test_tta_views_are_distinct_and_invertible():
    import torch

    batch = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    seen = set()
    for hflip, vflip, rot in TTA_POLICIES["d4"]:
        view = apply_view(batch, hflip, vflip, rot)
        assert view.shape == batch.shape
        key = view.numpy().tobytes()
        assert key not in seen, f"duplicate TTA view: {(hflip, vflip, rot)}"
        seen.add(key)
    assert len(seen) == 8, "D4 policy should give 8 distinct views"


# ---------------------------------------------------------------------------
# Runner (so the file works without pytest installed)
# ---------------------------------------------------------------------------
def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("\nFailures:")
        for name, exc in failures:
            print(f"  {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
