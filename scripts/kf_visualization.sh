#!/usr/bin/env bash
set -e

# ==============================================
# 1) Run with OpenCV SIFT detector + BF matcher:
# ==============================================
python3 slam/core/keyframe.py \
  --dataset tum-rgbd \
  --base_dir ./Dataset \
  --detector akaze \
  --matcher bf \
  --fps 10 \
  --ransac_thresh 1.0

# # ==============================================
# # 2) (Alternative) Run with ALIKED + LightGlue:
# # ==============================================
# python3 slam/core/keyframe.py \
#   --dataset kitti \
#   --base_dir ./Dataset \
#   --use_lightglue \
#   --fps 10 \
#   --ransac_thresh 1.0
