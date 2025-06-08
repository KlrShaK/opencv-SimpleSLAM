#!/usr/bin/env bash
set -e

# ==============================================
# 1) Run with OpenCV SIFT detector + BF matcher:
# ==============================================
# python3 slam/core/visualize_tracking.py \
#   --dataset malaga \
#   --base_dir ./Dataset \
#   --detector orb \
#   --matcher bf \
#   --fps 10 \
#   --ransac_thresh 1.0

# # ==============================================
# # 2) (Alternative) Run with ALIKED + LightGlue:
# # ==============================================
python3 slam/core/visualize_tracking.py \
  --dataset kitti \
  --base_dir ./Dataset \
  --use_lightglue \
  --fps 10 \
  --ransac_thresh 1.0
