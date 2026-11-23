#!/usr/bin/env bash
# Rebuild SignGen HDF5 data using the preprocessing described in the
# SignGen supplemental: center-crop each video frame, then resize to 128x128.
#
# This script intentionally uses separate cache roots so pose/depth/motion are
# regenerated from the center-cropped RGB frames instead of reusing old resize-only
# caches.
set -euo pipefail

REPO_DIR="${REPO_DIR:-.}"
SIGNGEN_PYTHON="${SIGNGEN_PYTHON:-python3}"
SIGNGEN_DATA_ROOT="${SIGNGEN_DATA_ROOT:-data/raw_rwth}"
SIGNGEN_OUT_ROOT="${SIGNGEN_OUT_ROOT:-data/rwth}"
SIGNGEN_SPLITS="${SIGNGEN_SPLITS:-train dev test}"
SIGNGEN_NUM_WORKERS="${SIGNGEN_NUM_WORKERS:-16}"
SIGNGEN_TEXT_BATCH_SIZE="${SIGNGEN_TEXT_BATCH_SIZE:-128}"
SIGNGEN_TORCH_NUM_THREADS="${SIGNGEN_TORCH_NUM_THREADS:-32}"
SIGNGEN_OPENCV_NUM_THREADS="${SIGNGEN_OPENCV_NUM_THREADS:-0}"
SIGNGEN_H5_COMPRESSION="${SIGNGEN_H5_COMPRESSION:-none}"
SIGNGEN_VIDS_PER_SHARD="${SIGNGEN_VIDS_PER_SHARD:-100000}"
SIGNGEN_EAGER_INIT_FEATURES="${SIGNGEN_EAGER_INIT_FEATURES:-1}"
SIGNGEN_DWPOSE_INFER_SIZE="${SIGNGEN_DWPOSE_INFER_SIZE:-384}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export SIGNGEN_POSE_CACHE_ROOT="${SIGNGEN_POSE_CACHE_ROOT:-$REPO_DIR/cache_pose_center_crop_128}"
export SIGNGEN_POSE_KEYPOINT_CACHE_ROOT="${SIGNGEN_POSE_KEYPOINT_CACHE_ROOT:-$REPO_DIR/cache_pose_keypoints_center_crop_384}"
export SIGNGEN_MOTION_CACHE_ROOT="${SIGNGEN_MOTION_CACHE_ROOT:-$REPO_DIR/cache_motion_center_crop_128}"
export SIGNGEN_DEPTH_CACHE_ROOT="${SIGNGEN_DEPTH_CACHE_ROOT:-$REPO_DIR/cache_depth_center_crop_128}"
export SIGNGEN_TORCH_NUM_THREADS
export SIGNGEN_OPENCV_NUM_THREADS
export SIGNGEN_DWPOSE_INFER_SIZE
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF

cd "$REPO_DIR"

EAGER_INIT_ARGS=()
if [[ "$SIGNGEN_EAGER_INIT_FEATURES" == "1" || "$SIGNGEN_EAGER_INIT_FEATURES" == "true" || "$SIGNGEN_EAGER_INIT_FEATURES" == "yes" ]]; then
  EAGER_INIT_ARGS+=(--eager_init_features)
fi

echo "================================================================"
echo " Rebuild SignGen HDF5 with paper preprocessing"
echo " Raw root:   $SIGNGEN_DATA_ROOT"
echo " Output:     $SIGNGEN_OUT_ROOT"
echo " Splits:     $SIGNGEN_SPLITS"
echo " Preprocess: center-crop -> resize 128x128"
echo " CUDA:       $CUDA_VISIBLE_DEVICES"
echo " Workers:    $SIGNGEN_NUM_WORKERS"
echo " Text batch: $SIGNGEN_TEXT_BATCH_SIZE"
echo " Eager init: $SIGNGEN_EAGER_INIT_FEATURES"
echo " DWPose in.: $SIGNGEN_DWPOSE_INFER_SIZE -> 128"
echo " Cache pose: $SIGNGEN_POSE_CACHE_ROOT"
echo " Cache kpts: $SIGNGEN_POSE_KEYPOINT_CACHE_ROOT"
echo " Cache mot.: $SIGNGEN_MOTION_CACHE_ROOT"
echo " Cache dep.: $SIGNGEN_DEPTH_CACHE_ROOT"
echo "================================================================"

for split in $SIGNGEN_SPLITS; do
  echo ""
  echo "==================== split=$split ===================="
  "$SIGNGEN_PYTHON" sign_language_convert.py \
    --sl_dir "${SIGNGEN_DATA_ROOT}/videos/${split}" \
    --split "$split" \
    --out_dir "${SIGNGEN_OUT_ROOT}/${split}" \
    --image_size 128 \
    --num_workers "$SIGNGEN_NUM_WORKERS" \
    --text_batch_size "$SIGNGEN_TEXT_BATCH_SIZE" \
    --h5_compression "$SIGNGEN_H5_COMPRESSION" \
    --force_h5 True \
    --center_crop_before_resize True \
    --vids_per_shard "$SIGNGEN_VIDS_PER_SHARD" \
    --text_path "${SIGNGEN_DATA_ROOT}/manual/PHOENIX-2014-T.${split}.corpus.csv" \
    "${EAGER_INIT_ARGS[@]}"
done

echo ""
echo "================================================================"
echo " Done. New data root: $SIGNGEN_OUT_ROOT"
echo " Validate with:"
echo "   $SIGNGEN_PYTHON check_center_crop_hdf5.py --raw-root $SIGNGEN_DATA_ROOT --hdf5-root $SIGNGEN_OUT_ROOT"
echo "================================================================"
