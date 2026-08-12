# ISIC "Create New Approach" form — text to paste

## Name

```
Threshold-Calibrated Dual-Encoder Ensemble (ConvNeXt, 5-fold)
```

## Uses external data

**Select "No".**

Only the data provided for MILK10k 2025 Lesion Diagnosis was used. ISIC 2019 was
evaluated as a possible pretraining source but **no external-data model contributed to
this submission**.

## Description

```
We treat MILK10k as 11 independent binary decisions per lesion and optimise the
operating point of each one, because macro-F1 weights all classes equally while the
data spans a 280:1 imbalance: the five rarest categories are 3.9% of the lesions but
45% of the metric.

The model is a ConvNeXt-Tiny encoder (ImageNet-22k to 1k) applied to the dermoscopic
view, trained with BCEWithLogitsLoss under damped inverse-frequency positive weighting
(exponent 0.5, clipped at 20) and class-balanced sampling (exponent 0.5). Undamped
weighting assigns the 9-lesion MAL_OTH class a weight near 580 and destabilises
training. Optimisation is AdamW with discriminative learning rates (backbone 1e-4,
randomly initialised head 1e-3), a per-iteration cosine schedule with warmup, label
smoothing 0.02, stochastic depth 0.2, mixup 0.4 disabled for the final three epochs so
probabilities calibrate, and EMA weight averaging. Inputs are 384x384 with per-modality
augmentation exploiting the full dihedral symmetry of lesion images. A second model adds
a parallel encoder over the paired clinical close-up, fused by a learned per-dimension
gate so the network can decide per lesion which modality to trust.

Both models are trained with 5-fold lesion-level cross-validation and ensembled with
8-view dihedral test-time augmentation, averaged in logit space to preserve confident
tails.

The decisive component is calibration. Because macro-F1 decomposes as a mean of
per-class F1 scores and each depends only on its own threshold, the optimisation is
separable and an independent one-dimensional search per class is globally optimal; we
sweep every distinct decision boundary in O(N log N) using F1_k = 2*TP_k/(k+P) on pooled
out-of-fold predictions over all 5,240 training lesions. Thresholds are selected at the
centre of the widest near-optimal plateau rather than the argmax, and predicted positives
are capped at four times class support to prevent an ultra-rare class from fitting noise.
Because the challenge binarises at a fixed 0.5, each tuned threshold t is folded back
into the probabilities by a monotone piecewise-linear map (p<=t maps to 0.5p/t, p>t maps
to 0.5+0.5(p-t)/(1-t)), which sends t to 0.5 while preserving within-class ranking, so
ROC-AUC is unchanged and the fixed cut implements the tuned decision. This is worth
+0.050 macro-F1 over submitting raw probabilities, more than any architectural change we
measured.

Ensemble weights are fitted on pooled out-of-fold predictions and accepted only if they
beat uniform weighting on held-out data. We deliberately use probability averaging rather
than rank averaging: rank averaging is calibration-free but forces each class's predicted
positive rate onto the training prior, and the test distribution appears prior-shifted
relative to training.

Reported performance uses a nested estimate (thresholds fitted on 4,761 lesions and
scored on a disjoint 479, matching the test-set size) rather than fitting and scoring on
the same data, giving 0.5444 macro-F1 with a standard deviation of 0.045. Pooled
out-of-fold macro-F1 with in-sample thresholds is 0.5801, and macro-AUC is 0.880. MAL_OTH
(9 training lesions, out-of-fold AUC 0.492) is not learnable from the provided data and
bounds attainable macro-F1 at 0.909.
```

## Manuscript

Optional; not submitted.

## Docker container

Optional. Tag after pushing to Docker Hub:

```
<dockerhub-user>/milk10k:latest
```

---

### Numbers referenced above

| quantity | value |
| --- | --- |
| honest (nested) macro-F1 | 0.5444 ± 0.045 |
| pooled OOF macro-F1, in-sample thresholds | 0.5801 |
| macro-AUC | 0.880 |
| gain from threshold calibration | +0.050 |
| submitted file | `outputs/submission_blend_mean.csv` |
