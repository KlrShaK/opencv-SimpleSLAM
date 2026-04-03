# OpenCV-SimpleSLAM
### Implementing a SLAM framework from scratch

https://github.com/user-attachments/assets/f489a554-299e-41ad-a4b4-436e32d8cbd5

## Preface
This project was was done for **OpenCV** under the guidance of *Gary Bradski* and *Reza Amayeh* supported by **Google Summer of Code(GSOC) 2025**. This project presents a methodological approach to designing, implementing, and documenting a **Simultaneous Localization and Mapping(SLAM) framework from scratch in Python**. The primary objective is to build a fully functional, OpenCV-based SLAM system with clear documentation to facilitate reproducibility, future extensions, and adoption by the broader computer vision community. Traditional SLAM systems that rely on hand-crafted features (e.g., ORB, SIFT) often exhibit reduced robustness under challenging conditions such as viewpoint shifts, illumination changes, or motion blur. To address these limitations, we integrate modern learned features—ALIKED keypoints and descriptors *(Zhao et al., 2022)* combined with the LightGlue matcher *(Lindenberger et al., 2023)*—into a streamlined SLAM pipeline. This integration improves tracking stability and relocalization performance, especially in environments with significant photometric and geometric variations.

In addition, the system **extends feature extraction and matching using [LightGlue](https://github.com/cvg/LightGlue)** *(Lindenberger et al., 2023)* and leverages **[PyCeres](https://github.com/cvg/pyceres)** for non-linear optimization, enabling efficient **bundle adjustment and graph optimization** within the pipeline.

Python feature-based SLAM / visual odometry experiments built around OpenCV, ALIKED + LightGlue, PyCeres, and Open3D.

The current primary runtime path is [`slam/monocular/main_revamped.py`](slam/monocular/main_revamped.py). It implements an incremental monocular pipeline with delayed two-view bootstrap, frame-to-map tracking, keyframe insertion, triangulation, local bundle adjustment, and live visualization. Older scripts are still present in the repository as legacy or experimental variants.

## Current Status

This repository has evolved beyond the original standalone SfM prototypes. The code that best reflects the current project state is:

- `slam/monocular/main_revamped.py`: main monocular SLAM entrypoint
- `slam/core/*.py`: reusable tracking, mapping, bootstrap, BA, and visualization modules
- `tests/`: focused unit tests for geometry and helper utilities

The project is still experimental and educational in nature. It is useful for understanding and iterating on a Python SLAM pipeline, but it is not a production-ready SLAM system.

## What The Current Monocular Pipeline Does

`slam/monocular/main_revamped.py` performs the following steps:

1. Load a dataset, camera intrinsics, and ground truth if available.
2. Extract features using either:
   - OpenCV detectors and matchers (`ORB`, `SIFT`, `AKAZE`, `BF`, `FLANN`), or
   - `ALIKED + LightGlue`
3. Delay initialization until a good two-view pair is found.
4. Bootstrap the initial map by competing homography vs. fundamental-matrix models.
5. Track each new frame against the map using:
   - constant-velocity pose prediction
   - reprojection of active landmarks
   - small-window 2D-3D matching
   - `solvePnPRansac`
6. Fall back to frame-to-frame 2D-2D tracking if PnP fails.
7. Decide whether the current frame should become a new keyframe.
8. Triangulate new landmarks between the latest keyframes.
9. Run local bundle adjustment when enough new landmarks were added.
10. Visualize:
    - track-debug reprojection overlays
    - current frame plus recent keyframes
    - frame-to-frame feature matches
    - 2D trajectory `(x-z)` against ground truth when GT is available
    - 3D map view with Open3D

At the end of a run, the script always saves a trajectory image named `trajectory_<dataset>.png`.

## Repository Structure

```text
.
├── slam/
│   ├── core/
│   │   ├── ba_utils.py              # local/global bundle-adjustment helpers
│   │   ├── dataloader.py            # sequence, calibration, and GT loaders
│   │   ├── features_utils.py        # feature extraction and matching
│   │   ├── keyframe_utils.py        # keyframe policy + thumbnail helpers
│   │   ├── landmark_utils.py        # Map / MapPoint data structures
│   │   ├── pnp_utils.py             # frame-to-map tracking and PnP helpers
│   │   ├── pose_utils.py            # SE(3) pose conversions/utilities
│   │   ├── trajectory_utils.py      # GT alignment helpers
│   │   ├── triangulation_utils.py   # keyframe-to-keyframe triangulation
│   │   ├── two_view_bootstrap.py    # monocular initialization
│   │   ├── visualization_utils.py   # 2D/3D visualization utilities
│   │   └── visualize_ba.py          # BA visualization helpers
│   ├── monocular/
│   │   ├── main_revamped.py         # current monocular entrypoint
│   │   ├── main.py                  # older monocular pipeline
│   │   ├── main4.py                 # alternate legacy monocular path
│   │   └── Notes.txt
│   └── stereo/
│       └── ROUGHstereo_tracker.py   # early stereo experiment
├── config/
│   ├── calibrate_camera/            # scripts and example files for custom calibration
│   ├── monocular.yaml
│   └── stereo.yaml
├── docs/
│   └── system_design.md
├── scripts/
│   └── run_tracker_visualization.sh # example launch commands
├── tests/                           # utility and geometry-focused tests
├── tools/                           # misc tooling scripts
├── refrences/                       # older reference / prototype code
├── requirements.txt
└── setup.py
```

## Dependencies

Recommended:

- Python `3.10+`
- `opencv-python`
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `torch`
- `lightglue`
- `lz4`
- `open3d`
- `pyceres`
- `pycolmap`

Important notes about the current codebase:

- `slam/core/features_utils.py` imports `LightGlue` and `ALIKED` at module import time, so `LightGlue` must currently be installed even if you intend to run with OpenCV features only.
- `slam/core/ba_utils.py` imports `pyceres` and `pycolmap` at module import time, so those packages are also required for `main_revamped.py`.
- `Open3D` is optional in practice if you always use `--no_viz3d` or `--headless`; the visualizer degrades gracefully when `open3d` is unavailable.

## Installation

Create an environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If `lightglue==0.0` does not resolve cleanly in your environment, install it from upstream and then continue:

```bash
pip install "lightglue @ git+https://github.com/cvg/LightGlue.git"
```

## Expected Dataset Layout

The current loader in [`slam/core/dataloader.py`](slam/core/dataloader.py) expects specific folder names and, for some datasets, specific hardcoded sequences.

### KITTI

```text
<base_dir>/kitti/
├── 05/
│   └── image_0/
│       └── *.png
└── poses/
    └── 05.txt
```

Notes:

- The current monocular loader is hardcoded to KITTI sequence `05`.

### Malaga

```text
<base_dir>/malaga/
├── malaga-urban-dataset-extract-07_rectified_800x600_Images/
│   ├── *_left.jpg
│   └── *_right.jpg
└── malaga-urban-dataset-extract-07_all-sensors_GPS.txt
```

Notes:

- The monocular path uses the left images.
- The current loader is hardcoded to Malaga extract `07`.

### TUM RGB-D

```text
<base_dir>/tum-rgbd/
└── rgbd_dataset_freiburg3_long_office_household/
    ├── rgb/
    │   └── *.png
    ├── rgb.txt
    └── groundtruth.txt
```

Notes:

- The current loader is hardcoded to `rgbd_dataset_freiburg3_long_office_household`.
- The monocular pipeline only uses the RGB frames.

### Custom

```text
<base_dir>/custom/
├── custom_compress.mp4
└── calibration.pkl
```

Notes:

- `calibration.pkl` is loaded by [`slam/core/dataloader.py`](slam/core/dataloader.py) and is expected to contain the camera matrix as the first item in the pickled tuple.
- The scripts in [`config/calibrate_camera/`](config/calibrate_camera/) are intended to help generate the calibration files for custom data.

## Running The Current Monocular Pipeline

The recommended way to run the current code is from the repository root:

### 1. KITTI with ALIKED + LightGlue

```bash
python3 -m slam.monocular.main_revamped \
  --dataset kitti \
  --base_dir ./Dataset \
  --use_lightglue
```

### 2. KITTI headless

This disables all GUI windows and saves the trajectory plot at the end.

```bash
python3 -m slam.monocular.main_revamped \
  --dataset kitti \
  --base_dir ./Dataset \
  --use_lightglue \
  --headless \
  --no_viz3d
```

### 3. TUM RGB-D with OpenCV features

```bash
python3 -m slam.monocular.main_revamped \
  --dataset tum-rgbd \
  --base_dir ./Dataset \
  --detector akaze \
  --matcher bf \
  --no_viz3d
```

### 4. Custom video input

```bash
python3 -m slam.monocular.main_revamped \
  --dataset custom \
  --base_dir ./Dataset \
  --use_lightglue \
  --no_viz3d
```

The shell script [`scripts/run_tracker_visualization.sh`](scripts/run_tracker_visualization.sh) also contains example launch commands.

## Useful CLI Arguments

The main entrypoint is [`slam/monocular/main_revamped.py`](slam/monocular/main_revamped.py). Key options:

| Argument | Meaning |
| --- | --- |
| `--dataset {kitti,malaga,tum-rgbd,custom}` | Select dataset loader |
| `--base_dir PATH` | Root folder that contains the dataset subdirectory |
| `--use_lightglue` | Use ALIKED + LightGlue instead of OpenCV detectors/matchers |
| `--detector {orb,sift,akaze}` | OpenCV detector when not using LightGlue |
| `--matcher {bf,flann}` | OpenCV matcher when not using LightGlue |
| `--max_features INT` | Max features / keypoints for OpenCV detectors and ALIKED |
| `--min_conf FLOAT` | Minimum LightGlue confidence |
| `--ransac_thresh FLOAT` | Shared RANSAC reprojection threshold |
| `--pnp_min_inliers INT` | Minimum inliers required to accept PnP |
| `--proj_radius FLOAT` | Small-window matching radius for landmark reprojection |
| `--kf_min_inliers INT` | Keyframe insertion threshold on matched inliers |
| `--kf_min_ratio FLOAT` | Minimum inlier ratio to previous keyframe |
| `--kf_max_disp FLOAT` | Keyframe insertion threshold based on image displacement |
| `--kf_min_rot_deg FLOAT` | Keyframe insertion threshold based on rotation |
| `--local_ba_window INT` | Local BA window size |
| `--local_ba_min_new_points INT` | Only run local BA when enough new landmarks were triangulated |
| `--local_ba_max_points INT` | Cap landmarks used by local BA |
| `--local_ba_max_iters INT` | Max Ceres iterations for local BA |
| `--no_viz3d` | Disable the Open3D 3D map window |
| `--headless` | Disable all interactive visualization and only save final trajectory |
| `--gba_every INT` | Global BA milestone hook; the full GBA call is currently scaffolded in code but not actively executed in the main loop |

## Visualization And Outputs

When not running headless, the current pipeline may open:

- `Track debug`: current-frame reprojection overlay
- `Strip: img2 + last 3 KFs`: current frame plus recent keyframe thumbnails
- `img2 + prev→cur matches`: frame-to-frame matched features
- `Trajectory 2D (x-z)`: estimated trajectory over ground truth when GT exists
- `SLAM Map`: Open3D view of landmarks and camera path

At the end of a run:

- the trajectory figure is saved as `trajectory_<dataset>.png`
- in non-headless mode, the final trajectory figure is also shown interactively

## Tests

The repository includes focused utility and geometry tests under [`tests/`](tests/). A representative example:

```bash
pytest -q tests/test_pnp_utils.py
```

The tests are mainly unit-level checks for geometry helpers and matching / pose utilities, not a full end-to-end dataset regression suite.

## Legacy And Experimental Files

Not every file in the repository is part of the active path:

- `slam/monocular/main.py` and `slam/monocular/main4.py` are older monocular variants.
- `slam/stereo/ROUGHstereo_tracker.py` is an early stereo experiment.
- `refrences/` contains reference and prototype scripts.
- Some files in `docs/` and `tools/` are placeholders or work-in-progress.

If you are extending the current code, start from:

- [`slam/monocular/main_revamped.py`](slam/monocular/main_revamped.py)
- [`slam/core/`](slam/core/)

## Performance & Future Directions

- **Python Prototyping**: Great for algorithmic experimentation but can be slower for large-scale or real-time tasks.
- **Production-Grade**: Offload heavy steps (bundle adjustment, large-scale optimization) to C++.
- **Loop Closure & Full SLAM**: These scripts focus on **Visual Odometry**. Future expansions may include place recognition, pose graph optimization, etc.

