# slam/core/multi_view_utils.py
"""Utilities for deferred (multi-view) triangulation in the SLAM pipeline.

This module is written for a **camera-from-world** pose convention (Tcw):
    X_c = R_cw * X_w + t_cw
Projection uses: P = K [R_cw | t_cw]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import logging
import cv2
import numpy as np

from .landmark_utils import Map

logger = logging.getLogger("multi_view")

# --------------------------------------------------------------------------- #
#  Robust linear triangulation across ≥ 2 views (inputs/outputs in world)
# --------------------------------------------------------------------------- #
# NOTE: For numeric robustness, you could Hartley-normalize the image points
# before DLT and denormalize after. Kept as TODO to keep this simple & fast.
def multi_view_triangulation(
    K: np.ndarray,
    poses_cw: List[np.ndarray],               # M × 4×4  (camera-from-world)
    pts2d: np.ndarray,                        # M × 2    (pixels)
    *,
    min_depth: float,
    max_depth: float,
    max_rep_err: float,
    eps: float = 1e-6
) -> Optional[np.ndarray]:
    """Triangulate one world point from ≥2 views.

    Parameters
    ----------
    K : (3,3)
        Camera intrinsics.
    poses_cw : List[(4,4)]
        Poses **camera-from-world** for each observing view.
    pts2d : (M,2)
        Pixel coordinates aligned with poses_cw (same order).
    min_depth, max_depth : float
        Acceptable mean depth in each observing camera (cheirality+range).
    max_rep_err : float
        Max *mean* reprojection error (pixels).

    Returns
    -------
    X_w : (3,) or None
        Triangulated world point, or None if validation fails.
    """
    M = len(poses_cw)
    assert M == len(pts2d) and M >= 2, "Need ≥ 2 consistent views"
    if M < 2:
        logger.debug("Triangulation skipped: only %d view(s).", M)
        return None

    # Build linear system A X_h = 0 (DLT), using P = K [R|t] with **Tcw** directly.
    A_rows = []
    for Tcw, (u, v) in zip(poses_cw, pts2d):
        P = K @ Tcw[:3, :4]           # world → pixels via camera projection
        A_rows.append(u * P[2] - P[0])
        A_rows.append(v * P[2] - P[1])
    A = np.stack(A_rows)

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < eps:
        logger.debug("[Triang] Degenerate homogeneous solution (w≈0).")
        return None
    X_w = X_h[:3] / X_h[3]

    # Validation: cheirality, depth range, reprojection error
    reproj_errs, depths = [], []
    for Tcw, (u, v) in zip(poses_cw, pts2d):
        pc = (Tcw @ np.append(X_w, 1.0))[:3]  # camera frame z = depth
        if pc[2] <= 0:
            logger.debug("[Triang] Cheirality failed (z<=0). z=%.4f", float(pc[2]))
            return None
        depths.append(pc[2])

        uv_hat_h = K @ pc
        uv_hat = uv_hat_h[:2] / (uv_hat_h[2] + 1e-12)
        reproj_errs.append(float(np.linalg.norm(uv_hat - (u, v))))

    mean_depth = float(np.mean(depths))
    mean_rep   = float(np.mean(reproj_errs))

    if not (min_depth <= mean_depth <= max_depth):
        logger.debug("[Triang] Depth check failed: mean=%.3f not in [%.3f, %.3f]",
                     mean_depth, min_depth, max_depth)
        return None
    if mean_rep > max_rep_err:
        logger.debug("[Triang] Reproj error failed: mean=%.3f > %.3f",
                     mean_rep, max_rep_err)
        return None

    logger.debug("[Triang] OK: views=%d  meanDepth=%.3f  meanRep=%.3f px",
                 M, mean_depth, mean_rep)
    return X_w


# --------------------------------------------------------------------------- #
#  Track manager – accumulates 2-D key-frame observations
# --------------------------------------------------------------------------- #
@dataclass
class _Obs:
    kf_idx: int
    kp_idx: int
    uv: Tuple[float, float]
    descriptor: np.ndarray

def update_and_prune_tracks(matches, prev_map, tracks,
                            kp_curr, frame_idx, next_track_id,
                            prune_age=30):
    """
    Continuation of simple 2-D point tracks across frames (for KF gating).
    """
    curr_map = {}

    for m in matches:
        q, t = m.queryIdx, m.trainIdx
        x, y = map(int, kp_curr[t].pt)
        tid   = prev_map.get(q, next_track_id)
        if tid == next_track_id:
            tracks[tid] = []
            next_track_id += 1
        curr_map[t] = tid
        tracks[tid].append((frame_idx, x, y))

    # prune dead tracks
    for tid, pts in list(tracks.items()):
        if frame_idx - pts[-1][0] > prune_age:
            del tracks[tid]
    return curr_map, tracks, next_track_id


class MultiViewTriangulator:
    """
    Accumulate feature tracks (key-frames only) and triangulate once a track
    appears in ≥ `min_views` distinct key-frames.

    Conventions:
      - Stored poses are **Tcw** (camera-from-world).
      - Triangulated points are returned/inserted in **world** coordinates.
    """

    def __init__(self,
                 K: np.ndarray,
                 *,
                 min_views:    int,
                 merge_radius: float,
                 max_rep_err:  float,
                 min_depth:    float,
                 max_depth:    float):
        # All thresholds come from the caller – no magic numbers inside.
        self.K            = K
        self.min_views    = max(2, min_views)
        self.merge_radius = merge_radius
        self.max_rep_err  = max_rep_err
        self.min_depth    = min_depth
        self.max_depth    = max_depth

        self._track_obs: Dict[int, List[_Obs]] = {}
        self._kf_poses:  Dict[int, np.ndarray] = {}   # frame_idx → Tcw
        self._kf_imgs:   Dict[int, np.ndarray] = {}   # BGR uint8
        self._triangulated: set[int]           = set()

    # ------------------------------------------------------------------ #
    def add_keyframe(self,
                     frame_idx: int,
                     Tcw_pose: np.ndarray,
                     kps: List,                       # List[cv2.KeyPoint]
                     track_map: Dict[int, int],
                     img_bgr: np.ndarray,
                     descriptors: Optional[List[np.ndarray]]) -> None:
        """Register observations (and keep the full-res image for colour sampling)."""
        self._kf_poses[frame_idx] = Tcw_pose.copy()
        self._kf_imgs[frame_idx]  = img_bgr
        default_desc = np.zeros(32, dtype=np.uint8)

        num_add = 0
        for kp_idx, tid in track_map.items():
            u, v = kps[kp_idx].pt
            desc_raw = descriptors[kp_idx] if descriptors is not None else default_desc
            # tolerate torch tensors (desc.cpu().numpy()) without importing torch
            try:
                desc = np.array(desc_raw).copy()
            except Exception:
                try:
                    desc = np.array(desc_raw.cpu()).copy()  # type: ignore[attr-defined]
                except Exception:
                    desc = default_desc.copy()
            self._track_obs.setdefault(tid, []).append(_Obs(frame_idx, kp_idx, (u, v), desc))
            num_add += 1

        logger.info("[MVTri] KF add: frame=%d  Tcw set  obs+=%d  tracks=%d",
                    frame_idx, num_add, len(self._track_obs))

    # ------------------------------------------------------------------ #
    def triangulate_ready_tracks(self, world_map: Map) -> List[int]:
        """Triangulate mature tracks, insert them into the map, and return new ids."""
        new_ids: List[int] = []
        ready = 0

        for tid, obs in list(self._track_obs.items()):
            if tid in self._triangulated or len(obs) < self.min_views:
                continue
            ready += 1

            obs_sorted = sorted(obs, key=lambda o: o.kf_idx)
            Tcw_poses, pts2d = [], []
            for o in obs_sorted:
                pose = self._kf_poses.get(o.kf_idx)
                if pose is None:
                    logger.debug("[MVTri] Missing pose for KF frame %d", o.kf_idx)
                    break
                Tcw_poses.append(pose)
                pts2d.append(o.uv)
            else:
                X_w = multi_view_triangulation(
                    self.K, Tcw_poses, np.float32(pts2d),
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                    max_rep_err=self.max_rep_err,
                )
                if X_w is None:
                    continue

                # --------- colour sampling (pick first obs with an image) -------
                rgb = (1.0, 1.0, 1.0)  # default white
                for o in obs_sorted:
                    img = self._kf_imgs.get(o.kf_idx)
                    if img is None:
                        continue
                    h, w, _ = img.shape
                    x, y = int(round(o.uv[0])), int(round(o.uv[1]))
                    if 0 <= x < w and 0 <= y < h:
                        b, g, r = img[y, x]
                        rgb = (r / 255.0, g / 255.0, b / 255.0)
                        break

                # --------------- map insertion (+ optional merging) -------------
                # Choose most recent KF index if not tracked separately by map
                kf_idx = world_map.keyframe_indices[-1] if world_map.keyframe_indices else (len(world_map.poses) - 1)
                pid = world_map.add_points(X_w[None, :], np.float32([[*rgb]]), keyframe_idx=kf_idx)[0]
                for o in obs_sorted:
                    world_map.points[pid].add_observation(o.kf_idx, o.kp_idx, o.descriptor)

                new_ids.append(pid)
                self._triangulated.add(tid)
                self._track_obs.pop(tid, None)  # free memory

        if new_ids:
            logger.info("[MVTri] Added %d new point(s) to the map (ready=%d, tracks=%d)",
                        len(new_ids), ready, len(self._track_obs))
        else:
            logger.debug("[MVTri] No new points (ready=%d, tracks=%d)", ready, len(self._track_obs))

        return new_ids


# --------------------------------------------------------------------------- #
#  Two-view triangulation helpers (fallbacks)
# --------------------------------------------------------------------------- #
def triangulate_points(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Two-view triangulation using relative motion (R, t) from view-1 → view-2.

    This is **compatible** with the camera-from-world convention if you choose the
    world frame = camera-1. In that case:
        Tcw1 = [I|0],  Tcw2 = [R|t]
    and the resulting points are in **world** coordinates (numerically equal to cam-1).

    Parameters
    ----------
    K : (3,3) intrinsics
    R, t : relative rotation/translation (cam1→cam2)
    pts1, pts2 : (N,2) pixel coordinates

    Returns
    -------
    pts3d_w : (N,3) world coordinates (with world ≡ cam-1 at bootstrap)
    """
    if pts1.shape != pts2.shape:
        raise ValueError("pts1 and pts2 must be the same shape")
    if pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError("pts1/pts2 must be (N,2)")

    # Build projection matrices using **Tcw** (world=cam1)
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))   # Tcw1 = [I|0]
    P2 = K @ np.hstack((R, t.reshape(3, 1)))           # Tcw2 = [R|t]

    pts4d_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d_w = (pts4d_h[:3] / (pts4d_h[3] + 1e-12)).T
    logger.debug("[Triang-2v RT] Triangulated %d points.", pts3d_w.shape[0])
    return pts3d_w
