"""Tests for multi_view_utils.py: multi_view_triangulation() and
MultiViewTriangulator using synthetic data.

Run with:
    pytest test_multi_view_utils.py
"""

import importlib.util
import pathlib
import sys
import types
import numpy as np

# --------------------------------------------------------------------------- #
#  Minimal stub of the SLAM Map & Point interfaces required by the utils
# --------------------------------------------------------------------------- #
class _DummyPoint:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, dtype=float)
        self._obs = []

    def add_observation(self, kf_idx: int, kp_idx: int):
        self._obs.append((kf_idx, kp_idx))


class DummyMap:
    """Drop‑in replacement for slam.core.landmark_utils.Map for testing."""

    def __init__(self):
        self.points = {}
        self._next_id = 0

    # Identity alignment – no global bundle‑adjustment in the unit test
    def align_points_to_map(self, pts: np.ndarray, *, radius: float = 0.0):
        return pts

    def add_points(self, pts: np.ndarray, rgb: np.ndarray):
        ids = []
        for p in pts:
            pid = self._next_id
            self.points[pid] = _DummyPoint(p)
            ids.append(pid)
            self._next_id += 1
        return ids


# --------------------------------------------------------------------------- #
#  Monkey‑patch the missing package hierarchy expected by multi_view_utils
# --------------------------------------------------------------------------- #
# The production code lives in slam/core/multi_view_utils.py and does a
# `from .landmark_utils import Map` relative import.  We create the expected
# modules on‑the‑fly and insert our DummyMap.
_landmark_mod = types.ModuleType("slam.core.landmark_utils")
_landmark_mod.Map = DummyMap

# Build the `slam` -> `slam.core` package chain
sys.modules.setdefault("slam", types.ModuleType("slam"))
sys.modules.setdefault("slam.core", types.ModuleType("slam.core"))
sys.modules["slam.core.landmark_utils"] = _landmark_mod
sys.modules["landmark_utils"] = _landmark_mod   # fall‑back for absolute import

# --------------------------------------------------------------------------- #
#  Dynamically load the module under test
# --------------------------------------------------------------------------- #
# _mvu_path = pathlib.Path(__file__).with_name("multi_view_utils.py")
# spec = importlib.util.spec_from_file_location("slam.core.multi_view_utils", _mvu_path)
# mvu = importlib.util.module_from_spec(spec)
# sys.modules[spec.name] = mvu
# spec.loader.exec_module(mvu)
import slam.core.multi_view_utils as mvu


# Convenience aliases
multi_view_triangulation = mvu.multi_view_triangulation
MultiViewTriangulator = mvu.MultiViewTriangulator

# --------------------------------------------------------------------------- #
#  Synthetic scene generation helpers
# --------------------------------------------------------------------------- #

def _make_camera_pose(tx: float) -> np.ndarray:
    """Camera looking down +Z, translated along +X by *tx* (c→w)."""
    T = np.eye(4)
    T[0, 3] = tx
    return T


def _generate_scene(
    n_views: int = 5,
    n_pts: int = 20,
    noise_px: float = 0.3,
    seed: int = 0,
):
    """Return K, list[poses_w_c], 3‑D points, and noisy 2‑D projections."""
    rng = np.random.default_rng(seed)
    # ---------- intrinsics (640×480 pin‑hole) ----------
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0,   0.0,   1.0]])

    # ---------- camera trajectory ----------
    poses = [_make_camera_pose(tx) for tx in np.linspace(0.0, 1.0, n_views)]

    # ---------- random 3‑D points in front of the cameras ----------
    pts_w = np.stack([
        rng.uniform(-1.0, 1.0, n_pts),          # x
        rng.uniform(-1.0, 1.0, n_pts),          # y
        rng.uniform(4.0, 6.0, n_pts),           # z – all in front
    ], axis=1)

    # ---------- project + add pixel noise ----------
    pts2d_all = []
    for T_w_c in poses:
        P = np.linalg.inv(T_w_c)                # w→c
        uv = []
        for X_w in pts_w:
            pc = P @ np.append(X_w, 1.0)
            uv_pt = (K @ pc[:3])[:2] / pc[2]
            uv_pt += rng.normal(0.0, noise_px, 2)
            uv.append(uv_pt)
        pts2d_all.append(np.asarray(uv, dtype=np.float32))

    return K, poses, pts_w, pts2d_all


# --------------------------------------------------------------------------- #
#  1. Direct multi‑view triangulation
# --------------------------------------------------------------------------- #

def test_multi_view_triangulation_accuracy():
    K, poses, pts_w, pts2d = _generate_scene()

    errs = []
    for j in range(pts_w.shape[0]):
        uv_track = [view[j] for view in pts2d]          # one point across all views
        X_hat = multi_view_triangulation(
            K, poses, np.float32(uv_track),
            min_depth=0.1, max_depth=10.0, max_rep_err=2.0,
        )
        assert X_hat is not None, "Triangulation failed unexpectedly"
        errs.append(np.linalg.norm(X_hat - pts_w[j]))

    mean_err = float(np.mean(errs))
    assert mean_err < 5e-2, f"Mean 3‑D error too high: {mean_err:.4f} m"


# --------------------------------------------------------------------------- #
#  2. End‑to‑end test of MultiViewTriangulator
# --------------------------------------------------------------------------- #
class _KeyPointStub:
    """Mimics cv2.KeyPoint – only `.pt` is accessed by the triangulator."""
    def __init__(self, x: float, y: float):
        self.pt = (float(x), float(y))


def test_multiview_triangulator_pipeline():
    K, poses, pts_w, pts2d = _generate_scene()

    tri = MultiViewTriangulator(
        K,
        min_views=3,
        merge_radius=0.1,
        max_rep_err=2.0,
        min_depth=0.1,
        max_depth=10.0,
    )
    world_map = DummyMap()

    # Feed key‑frames one by one
    for frame_idx, (pose, uv_view) in enumerate(zip(poses, pts2d)):
        kps, track_map = [], {}
        for pid, (u, v) in enumerate(uv_view):
            kps.append(_KeyPointStub(u, v))
            track_map[pid] = pid                # 1‑to‑1 track id
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        tri.add_keyframe(frame_idx, pose, kps, track_map, dummy_img)

    new_ids = tri.triangulate_ready_tracks(world_map)
    assert len(new_ids) == pts_w.shape[0], "Not all points triangulated"

    errs = [
        np.linalg.norm(world_map.points[pid].xyz - pts_w[pid])
        for pid in new_ids
    ]
    mean_err = float(np.mean(errs))
    assert mean_err < 5e-2, f"Mean 3‑D error too high: {mean_err:.4f} m"
