# MILK10k experiment results

## Run comparison

Cross-validated runs first, then sorted by pooled out-of-fold tuned macro-F1.

**Do not compare a single-fold `OOF tuned` against a CV one.** A single-fold score is
biased upward twice: small sample, and thresholds tuned on the same lesions being
scored. Only full-CV rows (marked ✓) estimate test performance, so **submit from a ✓
run** even when a single-fold row shows a higher number.

| run | CV | folds | f1@0.5 | f1_tuned | f1_argmax | AUC | bal_acc | OOF n | OOF tuned | strategy |
|---|---|---|---|---|---|---|---|---|---|---|
| `stage1_derm_cv` | ✓ | 0,1,2,3,4 | 0.5128 | 0.5832 | 0.5314 | 0.8798 | 0.5630 | 5240 | 0.5635 | tuned |
| `stage2_dual_cv` | ✓ | 0,1,2,3,4 | 0.5417 | 0.5882 | 0.5384 | 0.8619 | 0.5412 | 5240 | 0.5611 | tuned |
| `stage1_derm_convnext_tiny` | — | 0 | 0.4953 | 0.5994 | 0.5226 | 0.9150 | 0.5801 | 1048 | 0.5994 | tuned |
| `stage2_dual_convnext_tiny` | — | 0 | 0.5232 | 0.5943 | 0.5529 | 0.8814 | 0.5660 | 1048 | 0.5943 | tuned |
| `stage3_dual_meta_convnext_tiny` | — | 0 | 0.4689 | 0.5562 | 0.4615 | 0.8944 | 0.5334 | 1048 | 0.5562 | tuned |

## Submission strategies — `stage1_derm_cv` (pooled OOF)

| strategy | macro-F1 |
|---|---|
| `tuned` | 0.5635  **<- recommended** |
| `tuned_or_argmax` | 0.5431 |
| `argmax` | 0.5348 |
| `default_0.5` | 0.5135 |

## Per-class detail — `stage1_derm_cv` at tuned thresholds

| class | support | predicted | threshold | precision | recall | F1 | AUC |
|---|---|---|---|---|---|---|---|
| AKIEC | 303 | 260 | 0.768 | 0.600 | 0.515 | **0.554** | 0.927 |
| BCC | 2522 | 2642 | 0.387 | 0.879 | 0.920 | **0.899** | 0.959 |
| BEN_OTH | 44 | 38 | 0.464 | 0.289 | 0.250 | **0.268** | 0.727 |
| BKL | 544 | 551 | 0.473 | 0.557 | 0.564 | **0.561** | 0.863 |
| DF | 52 | 48 | 0.823 | 0.625 | 0.577 | **0.600** | 0.943 |
| INF | 50 | 35 | 0.849 | 0.371 | 0.260 | **0.306** | 0.914 |
| MAL_OTH | 9 | 11 | 0.461 | 0.091 | 0.111 | **0.100** | 0.492 |
| MEL | 450 | 443 | 0.799 | 0.630 | 0.620 | **0.625** | 0.939 |
| NV | 746 | 859 | 0.525 | 0.721 | 0.830 | **0.771** | 0.962 |
| SCCKA | 473 | 448 | 0.801 | 0.721 | 0.683 | **0.701** | 0.950 |
| VASC | 47 | 44 | 0.947 | 0.841 | 0.787 | **0.813** | 0.982 |
