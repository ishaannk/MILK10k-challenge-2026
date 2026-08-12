# MILK10k 2026 — Multimodal Skin Lesion Classification

11-way diagnosis of skin lesions from **paired clinical + dermoscopic images** for the
[ISIC MILK10k benchmark](https://challenge.isic-archive.com/landing/milk10k/).
5,240 training lesions → 479 blind test lesions, scored by **macro-F1 at a fixed 0.5 threshold**.

**Result: 0.5444 macro-F1** (unbiased nested estimate) from a 5-fold × 2-architecture
ensemble with per-class threshold calibration. No external data.

---

## Why this task is won on calibration, not accuracy

The class distribution spans 280:1:

```
BCC 2522 │ NV 746 │ BKL 544 │ SCCKA 473 │ MEL 450 │ AKIEC 303
DF 52 │ INF 50 │ VASC 47 │ BEN_OTH 44 │ MAL_OTH 9
```

Macro-F1 weights all 11 classes equally, so **the five rarest classes are 3.9% of the
data but 45% of the score**. A calibrated model rarely emits `p > 0.5` for a class with
50 examples, so those classes score F1 = 0 and cap macro-F1 near 0.55 regardless of
encoder quality.

Fixing that is worth **+0.050 macro-F1** — more than any backbone change measured here:

| submission strategy | macro-F1 |
| --- | --- |
| raw probabilities @ 0.5 | 0.5135 |
| hard argmax one-hot | 0.5348 |
| **per-class tuned thresholds** | **0.5635** |

Macro-AUC is 0.880, so the model already *ranks* well — the gap was pure calibration.

**The mechanism.** Since `macro_f1 = mean_c F1_c` and `F1_c` depends only on class `c`'s
threshold, the problem is **separable**: an independent 1-D search per class is globally
optimal. `f1_curve_for_class` sweeps every distinct boundary in one `O(N log N)` pass via
`F1_k = 2·TP_k / (k + P)`. Tuned thresholds are then folded *back into the probabilities*
with a monotone map so the mandated 0.5 cut reproduces them:

```
p ≤ t  →  0.5 · p/t
p > t  →  0.5 + 0.5 · (p − t)/(1 − t)
```

This sends `t → 0.5` while preserving within-class ranking, so ROC-AUC is unchanged.

---

## Results

Pooled out-of-fold over all 5,240 lesions, 384 px, D4 TTA:

| model | pooled OOF | honest estimate |
| --- | --- | --- |
| Stage 1 — dermoscopy only, ConvNeXt-Tiny | 0.5635 | 0.5325 |
| Stage 2 — dual encoder, gated fusion | 0.5611 | — |
| Stage 3 — dual + metadata MLP *(single fold)* | 0.5562 | — |
| **Stage 1 + Stage 2 ensemble** | **0.5801** | **0.5444** |

Per-class F1 at tuned thresholds (Stage 1, pooled OOF):

| BCC | VASC | NV | SCCKA | DF | MEL | BKL | AKIEC | INF | BEN_OTH | MAL_OTH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.899 | 0.813 | 0.771 | 0.701 | 0.600 | 0.625 | 0.561 | 0.554 | 0.306 | 0.268 | 0.100 |

**MAL_OTH has 9 training lesions and OOF AUC 0.492 — chance.** It is effectively
unlearnable and caps macro-F1 at 0.909.

---

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # install torch for your GPU arch first

bash scripts/reproduce.sh              # data → train → TTA → ensemble → submission.csv
```

~2.7 h on one RTX PRO 4000 (24 GB). `FOLDS=0 bash scripts/reproduce.sh` for a fast
single-fold smoke run. The individual steps:

```bash
# Data → data/raw/, then lesion tables + stratified folds
bash scripts/download_data.sh
python scripts/prepare_data.py --config configs/base.yaml --check-images

# Train (each stage is one config of the same model)
python scripts/train.py --config configs/stage1_derm.yaml --folds all --name stage1_derm_cv
python scripts/train.py --config configs/stage2_dual.yaml --folds all --name stage2_dual_cv

# Test-time inference with D4 TTA + OOF-tuned thresholds
python scripts/predict.py --run-dir checkpoints/stage1_derm_cv \
    --out outputs/submission_stage1.csv --save-probs outputs/probs_stage1.npy
python scripts/predict.py --run-dir checkpoints/stage2_dual_cv \
    --out outputs/submission_stage2.csv --save-probs outputs/probs_stage2.npy

# Ensemble → final submission
python scripts/blend.py \
    --run checkpoints/stage1_derm_cv --run checkpoints/stage2_dual_cv \
    --test-probs outputs/probs_stage1.npy --test-probs outputs/probs_stage2.npy \
    --submission outputs/submission_final.csv --method mean --out outputs/blend

# Reports and invariant tests
python scripts/report.py --out RESULTS.md
python tests/test_pipeline.py          # 21/21
```

Single fold for fast iteration: `--fold 0` instead of `--folds all`.
Override anything without editing YAML: `--set train.epochs=40 data.image_size=448`.

---

## Architecture

Three progressive stages are **one configurable `LesionNet`**, so a stage comparison
isolates the architecture change rather than pipeline differences.

| stage | clinical image | metadata | fusion |
| --- | --- | --- | --- |
| 1 | — | — | single tower |
| 2 | ✓ | — | concat / gated / attention |
| 3 | ✓ | ✓ | attention over 3 streams |

**Recipe.** ConvNeXt-Tiny (IN-22k→1k) · `BCEWithLogitsLoss` with damped `pos_weight`
`((N−n_c)/n_c)^0.5` clipped at 20 · AdamW with discriminative LRs (backbone 1e-4,
head 1e-3) · per-iteration cosine + warmup · label smoothing 0.02 · stochastic depth 0.2
· mixup 0.4 disabled for the final 3 epochs so probabilities calibrate · EMA 0.999 ·
bf16 · class-balanced sampling at `power=0.5` · D4 TTA averaged in logit space.

```
configs/     YAML with `defaults:` inheritance — every knob documented inline
src/
  datasets/  lesion-level pivot, per-modality augmentation, paired-image Dataset
  models/    timm encoders, metadata MLP, concat/gated/attention fusion
  training/  losses, optim, EMA, trainer (AMP, accumulation, early stopping)
  validation/ metrics, threshold optimisation, prior-shift analysis
  inference/ D4 TTA, ensembling, submission writer
scripts/     reproduce · download_data · prepare_data · train · evaluate · predict · blend · report
docker/      ISIC submission container
tests/       21 invariant tests
```

Every checkpoint embeds its config **and** its tuned thresholds, so inference can never
silently disagree with training. Folds split at lesion level with both images travelling
together — no cross-split leakage.

---

## Docker submission

```bash
python scripts/export_weights.py \
    --run checkpoints/stage1_derm_cv --run checkpoints/stage2_dual_cv \
    --blend outputs/blend --out weights_bundle          # 13 GB → 822 MB

docker build -f docker/Dockerfile -t <user>/milk10k:latest .
docker run --rm -v /path/to/MILK10k_Test_Input:/images <user>/milk10k:latest > submission.csv
docker push <user>/milk10k:latest
```

Reads every JPEG under `/images`, writes CSV to stdout. Handles both the nested
`<lesion_id>/<isic_id>.jpg` layout and a flat directory; for pairs it averages both
tower orderings so mis-identifying which image is dermoscopic cannot change the result.

---

## What we measured, including what failed

Four plausible ideas were built, validated against held-out data, and **three were
rejected**. Kept here so they are not retried blind.

**1. Rank-averaged ensembling — rejected.** Calibration-free across architectures, so it
looked ideal. But ranks are uniform by construction, so an OOF-fitted rank threshold
forces each class's predicted-positive *rate* onto the training prior. The test set is
prior-shifted (est. SCCKA ×2.2, AKIEC ×1.6, BCC ×0.58), and rank blending predicted
SCCKA for 45 lesions where ~102 are expected, BCC for 246 where ~146 are expected —
deviation 247 from estimated test counts versus **59 for `mean`**. `mean` also scored
higher honestly (0.5444 vs 0.5374). Caught only by checking predicted counts against an
independent prior estimate; the OOF score cannot see it, because on OOF the prior matches
by construction.

**2. Prior-shift correction — rejected.** Two mechanisms, both validated against *known
synthetic shifts including a no-shift control*, both lost: count-matched thresholds
(−0.052) and importance-weighted F1 tuning (−0.012). Threshold-estimation variance
exceeds the bias removed, and the rare classes that dominate macro-F1 have ratio ≈ 1.0
anyway. `src/validation/prior_shift.py`.

**3. Per-class blend weights — rejected at small scale, accepted at large.** The dual
model beats dermoscopy-only on 8 of 11 classes but loses on INF (−0.112) and VASC
(−0.074), so per-class weights are theoretically well-motivated (macro-F1 is separable).
On 1,048-lesion single folds, held-out validation rejected them (−0.005, 33% win rate);
on 5,240 pooled OOF lesions it accepted them (+0.011, 100%). Sample size, not cleverness.

**4. Our own reported scores were biased — corrected.** Every `macro_f1_tuned` fits
thresholds on a set and scores them *on that same set*:

| estimator | macro-F1 |
| --- | --- |
| fit & score on the same 5,240 lesions | 0.5635 |
| **fit on 4,761, score on a disjoint 479** | **0.5327 ± 0.040** |

The 0.031 gap is ≈ −0.018 from scoring only 479 lesions (rare classes get 1–5 positives)
and ≈ −0.013 from thresholds being estimated. `nested_threshold_estimate()` now reports
the unbiased number for every run. The **±0.040** spread is also an estimate of
leaderboard noise: gaps under ~0.05 between adjacent ranks are not meaningful.

**Where the remaining headroom is.** BEN_OTH (44), INF (50) and MAL_OTH (9) are 3/11 of
the metric and score 0.268 / 0.306 / 0.100. No public dataset contains these MILK10k-specific
categories, so external data (ISIC 2019 maps cleanly onto the other 8 classes, ×5.8 BKL,
×6.4 VASC, ×5.6 DF) lifts everything *except* the three that hurt most. Next steps, in
expected order of payoff: external pretraining, a transformer tower for ensemble
diversity, then token-level cross-attention between the paired views instead of
pooled-feature late fusion.

---

## License

Code MIT. MILK10k data is CC-BY-NC 4.0 and not redistributed here.
