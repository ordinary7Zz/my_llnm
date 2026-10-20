#!/usr/bin/env bash
# B9 LNM on Lymph_Node_Metastasis_fake: 5-fold CV (C1) + full C1 train + C2 external eval.
# Usage:
#   bash new_code/run_b9_lnm_fake.sh
#   bash new_code/run_b9_lnm_fake.sh --smoke
#   CUDA_VISIBLE_DEVICES=1 RUN_TAG=my_fake bash new_code/run_b9_lnm_fake.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/LNM_ThyroidXAgent_ZJH_LymphUs_fake}"
IMG="${IMG:-/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/images}"
RUN_TAG="${RUN_TAG:-b9_fake}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EPOCHS="${EPOCHS:-50}"
N_SPLITS="${N_SPLITS:-5}"

SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      echo "Env: DATA IMG RUN_TAG DEVICE BATCH_SIZE LEARNING_RATE EPOCHS N_SPLITS CUDA_VISIBLE_DEVICES"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try --smoke or --help)" >&2
      exit 1
      ;;
  esac
done

if [[ "$SMOKE" == 1 ]]; then
  N_SPLITS=2
  EPOCHS=1
  RUN_TAG="${RUN_TAG}_smoke"
  echo "[smoke] n_splits=$N_SPLITS epochs=$EPOCHS run_tag=$RUN_TAG"
fi

OUT_CV="./new_code/runs/${RUN_TAG}_cv"
OUT_TRAIN="./new_code/runs/${RUN_TAG}_full_train"

echo "=== B9 fake | DATA=$DATA ==="
echo "CV output:    $OUT_CV"
echo "Train output: $OUT_TRAIN"
echo "device=$DEVICE batch_size=$BATCH_SIZE lr=$LEARNING_RATE epochs=$EPOCHS n_splits=$N_SPLITS"

echo ""
echo ">>> Step 1/3: patient-level ${N_SPLITS}-fold CV (lymphus C1, n=180)"
python new_code/run_b9_patient_cv.py \
  --train_manifest "$DATA/manifests/lymphus_center1_train.json" \
  --image_roots_json "$DATA/image_roots.json" \
  --output_dir "$OUT_CV" \
  --n_splits "$N_SPLITS" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --device "$DEVICE"

echo ""
echo ">>> Step 2/3: train on C1 (full manifest + image_roots, same patient_id rules as CV)"
python new_code/train.py \
  --train_json "$DATA/manifests/lymphus_center1_train.json" \
  --test_json "$DATA/manifests/lymphus_center2_test.json" \
  --image_roots_json "$DATA/image_roots.json" \
  --output_dir "$OUT_TRAIN" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --device "$DEVICE"

echo ""
echo ">>> Step 3/3: external eval on C2"
python new_code/evaluate.py \
  --test_json "$DATA/manifests/lymphus_center2_test.json" \
  --image_root "$IMG" \
  --image_roots_json "$DATA/image_roots.json" \
  --weights "$OUT_TRAIN/best_model.pth" \
  --norm_stats "$OUT_TRAIN/train_norm_stats.pkl" \
  --output "$OUT_TRAIN/c2_eval.json" \
  --device "$DEVICE"

echo ""
echo "Done."
echo "  CV:        $OUT_CV/cv_summary.json"
echo "  C2 eval:   $OUT_TRAIN/c2_eval.json"
