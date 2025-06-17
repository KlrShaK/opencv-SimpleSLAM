from __future__ import annotations

"""
landmark_utils.py
~~~~~~~~~~~~~~~~~
Classes and helper functions for managing 3‑D landmarks and camera poses
in an incremental VO / SLAM pipeline.

* MapPoint ─ encapsulates a single 3‑D landmark.
* Map      ─ container for all landmarks + camera trajectory.
* triangulate_points ─ convenience wrapper around OpenCV triangulation.

The module is intentionally lightweight and free of external dependencies
beyond NumPy + OpenCV; this makes it easy to unit‑test without a heavy
visualisation stack.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
#  MapPoint
# --------------------------------------------------------------------------- #
@dataclass
class MapPoint:
    """A single triangulated 3‑D landmark."""

    id: int
    position: np.ndarray  # shape (3,)
    observations: List[Tuple[int, int]] = field(default_factory=list)  # (frame_idx, kp_idx)

    def add_observation(self, frame_idx: int, kp_idx: int) -> None:
        """Register that *kp_idx* in *frame_idx* observes this landmark."""
        self.observations.append((frame_idx, kp_idx))


# --------------------------------------------------------------------------- #
#  Map container
# --------------------------------------------------------------------------- #
class Map:
    """A minimalistic map: 3‑D points + camera trajectory."""

    def __init__(self) -> None:
        self.points: Dict[int, MapPoint] = {}
        self.poses: List[np.ndarray] = []  # List of 4×4 camera‑to‑world matrices
        self._next_pid: int = 0

    # ---------------- Camera trajectory ---------------- #
    def add_pose(self, pose_w_c: np.ndarray) -> None:
        """Append a 4×4 *pose_w_c* (camera‑to‑world) to the trajectory."""
        assert pose_w_c.shape == (4, 4), "Pose must be 4×4 homogeneous matrix"
        self.poses.append(pose_w_c.copy())

    # ---------------- Landmarks ------------------------ #
    def add_points(self, pts3d: np.ndarray) -> List[int]:
        """Add a set of 3‑D points and return the list of newly assigned ids."""
        if pts3d.ndim != 2 or pts3d.shape[1] != 3:
            raise ValueError("pts3d must be (N,3)")
        new_ids: List[int] = []
        for p in pts3d:
            pid = self._next_pid
            self.points[pid] = MapPoint(pid, p.astype(np.float64))
            new_ids.append(pid)
            self._next_pid += 1
        return new_ids

    # ---------------- Convenience accessors ------------ #
    def get_point_array(self) -> np.ndarray:
        """Return all landmark positions as an (N,3) array (N may be 0)."""
        if not self.points:
            return np.empty((0, 3))
        return np.stack([mp.position for mp in self.points.values()], axis=0)

    def __len__(self) -> int:
        return len(self.points)


# --------------------------------------------------------------------------- #
#  Geometry helpers (stay here to avoid cyclic imports)
# --------------------------------------------------------------------------- #
def triangulate_points(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Triangulate corresponding *pts1* ↔ *pts2* given (R, t).

    Parameters
    ----------
    K
        3×3 camera intrinsic matrix.
    R, t
        Rotation + translation from *view‑1* to *view‑2*.
    pts1, pts2
        Nx2 arrays of pixel coordinates (dtype float32/float64).
    Returns
    -------
    pts3d
        Nx3 array in *view‑1* camera coordinates (not yet in world frame).
    """
    if pts1.shape != pts2.shape:
        raise ValueError("pts1 and pts2 must be the same shape")
    if pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError("pts1/pts2 must be (N,2)")

    proj1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    proj2 = K @ np.hstack((R, t.reshape(3, 1)))

    pts4d_h = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
    pts3d = (pts4d_h[:3] / pts4d_h[3]).T  # → (N,3)
    return pts3d
