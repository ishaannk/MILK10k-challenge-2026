#!/usr/bin/env bash
# =============================================================================
# Reproduce the submitted result end to end.
#
#   bash scripts/reproduce.sh                 # full 5-fold, both models
#   FOLDS=0 bash scripts/reproduce.sh         # single fold, for a fast smoke run
#
# Environment:
#   PY=python                                 interpreter to use
#   CUDA_VISIBLE_DEVICES=0                    GPU to train on
#   IMAGE_SIZE=384  EPOCHS=30  WORKERS=16     overrides
#
# Runtime on one RTX PRO 4000 (24 GB): ~50 min for Stage 1, ~110 min for Stage 2.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

FOLDS=${FOLDS:-all}
IMAGE_SIZE=${IMAGE_SIZE:-384}
EPOCHS=${EPOCHS:-30}
WORKERS=${WORKERS:-16}

mkdir -p logs outputs checkpoints
log() { echo "[$(date '+%H:%M:%S')] $*"; }

COMMON=(
  "data.image_size=${IMAGE_SIZE}"
  "train.epochs=${EPOCHS}"
  "loader.num_workers=${WORKERS}"
)

# --- 1. data ----------------------------------------------------------------
log "data"
bash scripts/download_data.sh
$PY scripts/prepare_data.py --config configs/base.yaml --check-images

# --- 2. train ---------------------------------------------------------------
# Stage 1 (dermoscopy only) and Stage 2 (dual encoder) are the two ensemble
# members. Stage 2 uses a smaller batch because it runs two encoders per lesion.
log "stage 1: dermoscopy-only ConvNeXt-Tiny, ${FOLDS} fold(s)"
$PY scripts/train.py --config configs/stage1_derm.yaml --folds "$FOLDS" \
  --name stage1_derm_cv --set "${COMMON[@]}" "train.batch_size=24"

log "stage 2: dual encoder, gated fusion, ${FOLDS} fold(s)"
$PY scripts/train.py --config configs/stage2_dual.yaml --folds "$FOLDS" \
  --name stage2_dual_cv --set "${COMMON[@]}" "train.batch_size=16"

# --- 3. inference -----------------------------------------------------------
# Each run's pooled-OOF thresholds are stored with it, so predict.py applies the
# same calibration that was validated.
for run in stage1_derm_cv stage2_dual_cv; do
  log "inference: ${run}"
  $PY scripts/predict.py --run-dir "checkpoints/${run}" \
    --out "outputs/submission_${run}.csv" \
    --save-probs "outputs/probs_${run}.npy"
done

# --- 4. ensemble ------------------------------------------------------------
# `mean`, not `rank`: rank averaging forces each class's predicted-positive rate
# onto the training prior, and the test set is prior-shifted (README, item 1).
log "ensemble -> submission.csv"
$PY scripts/blend.py \
  --run checkpoints/stage1_derm_cv --run checkpoints/stage2_dual_cv \
  --test-probs outputs/probs_stage1_derm_cv.npy \
  --test-probs outputs/probs_stage2_dual_cv.npy \
  --submission submission.csv \
  --method mean --out outputs/blend

# --- 5. report --------------------------------------------------------------
$PY scripts/report.py --out RESULTS.md
$PY tests/test_pipeline.py

log "done -> submission.csv"
