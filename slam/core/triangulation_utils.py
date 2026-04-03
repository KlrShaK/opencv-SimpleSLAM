# slam/core/triangulation_utils.py
"""
2-View Triangulation (compatible with Map API)
- Matches prev_kf <-> cur_kf
- cv2.triangulatePoints with P = K @ Tcw[:3,:]
- Gates: parallax (optional), cheirality/depth, reprojection
- Inserts points via Map.add_points(...) + MapPoint.add_observation(...)
- Rich debug logging & reason counters
"""
import numpy as np
import cv2
import logging
from collections import Counter

from slam.core.features_utils import (
    feature_matcher,
    filter_matches_ransac
)
from slam.core.two_view_bootstrap import pts_from_matches

log_tri = logging.getLogger("triangulation")

# ---------- Helpers ----------

def _RCt_from_Tcw(Tcw: np.ndarray):
    """Decompose 4x4 Tcw into (R, C, t), where x_c = R (X - C) = R X + t."""
    R = Tcw[:3, :3]
    t = Tcw[:3, 3]
    C = -R.T @ t
    return R, C, t

def _P_from_K_Tcw(K: np.ndarray, Tcw: np.ndarray) -> np.ndarray:
    """3x4 projection: P = K @ Tcw[:3,:] (world -> camera homogeneous)."""
    return K @ Tcw[:3, :]

def _project_px(K, R, t, X):
    """Project world point X with camera (R,t). Returns 2D pixel or None if behind camera."""
    Xc = R @ X + t
    if Xc[2] <= 1e-6:
        return None
    u = K @ (Xc / Xc[2])
    return u[:2]

def _angle_parallax_deg(K, uv1, uv2):
    """Angle between unit rays (for parallax gating)."""
    x1 = np.linalg.inv(K) @ np.array([uv1[0], uv1[1], 1.0], dtype=float)
    x2 = np.linalg.inv(K) @ np.array([uv2[0], uv2[1], 1.0], dtype=float)
    v1 = x1 / (np.linalg.norm(x1) + 1e-12)
    v2 = x2 / (np.linalg.norm(x2) + 1e-12)
    c = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _angle_parallax_deg_batch(K_inv: np.ndarray, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """Vectorised parallax angles for Nx2 pixel coordinate arrays."""
    rays1 = (K_inv @ np.c_[uv1, np.ones(len(uv1), dtype=np.float64)].T).T
    rays2 = (K_inv @ np.c_[uv2, np.ones(len(uv2), dtype=np.float64)].T).T
    rays1 /= np.linalg.norm(rays1, axis=1, keepdims=True) + 1e-12
    rays2 /= np.linalg.norm(rays2, axis=1, keepdims=True) + 1e-12
    cosang = np.clip(np.sum(rays1 * rays2, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _map_add_point(world_map, Xw, desc, obs_list):
    """
    Insert one 3-D point into Map and add its observations.

    Uses Map.add_points(...) (your Map stores points in a dict, not a list),
    then calls MapPoint.add_observation(...) for each view.
    """
    # 1) create the point (returns [pid])
    ids = world_map.add_points(np.asarray([Xw], dtype=np.float64))
    if not ids:
        raise RuntimeError("Map.add_points returned no ids.")
    pid = ids[0]

    # 2) add observations
    try:
        for ob in obs_list:
            kf_idx = int(ob["kf_idx"])
            kp_idx = int(ob["kp_idx"])
            if desc is None and "desc" in ob and ob["desc"] is not None:
                d = ob["desc"]
            else:
                d = desc
            world_map.points[pid].add_observation(kf_idx, kp_idx, d if d is not None else np.zeros((1,), np.uint8))
    except Exception as e:
        # If anything goes wrong, remove the half-created landmark to keep map clean
        world_map.points.pop(pid, None)
        raise e

    return pid


# ---------- Public: triangulate between two keyframes ----------

def triangulate_between_kfs_2view(
    args, K, world_map, prev_kf, cur_kf, matcher, log,
    use_parallax_gate: bool = True, parallax_min_deg: float = 2.0,
    reproj_px_max: float | None = None,
    debug_max_examples: int = 10
):
    """
    Triangulate new points from matches between two keyframes.

    Args:
      args: CLI args (uses min_depth, max_depth, ransac_thresh)
      K: (3,3) intrinsics
      world_map: Map (points: dict[int, MapPoint], add_points(...))
      prev_kf, cur_kf: Keyframe objects with .idx, .kps, .desc, .pose (Tcw)
      matcher: feature matcher from your pipeline
      log: logger (we also use 'triangulation' logger for extra debug)
    """
    # 0) Match + geometric filtering
    raw = feature_matcher(args, prev_kf.kps, cur_kf.kps, prev_kf.desc, cur_kf.desc, matcher)
    matches = filter_matches_ransac(prev_kf.kps, cur_kf.kps, raw, args.ransac_thresh)

    log.info("[TRI] KF %d→%d  raw=%d  after_RANSAC=%d (th=%.2f px)",
             prev_kf.idx, cur_kf.idx, len(raw), len(matches), float(args.ransac_thresh))
    log_tri.debug("prev_kf: idx=%d, kps=%d | cur_kf: idx=%d, kps=%d",
                  prev_kf.idx, len(prev_kf.kps), cur_kf.idx, len(cur_kf.kps))

    if len(matches) == 0:
        log.info("[TRI] No matches to triangulate for KFs %d→%d.", prev_kf.idx, cur_kf.idx)
        return []

    # 1) Build point arrays (Nx2) in pixel coords
    pts1, pts2 = pts_from_matches(prev_kf.kps, cur_kf.kps, matches)
    K_inv = np.linalg.inv(K)

    # 2) Projection matrices
    P1 = _P_from_K_Tcw(K, prev_kf.pose)
    P2 = _P_from_K_Tcw(K, cur_kf.pose)

    # 3) Triangulate (homogeneous 4xN -> Nx3)
    X4 = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))
    w = X4[3, :]
    valid_w = np.isfinite(w) & (np.abs(w) > 1e-12)
    if valid_w.sum() == 0:
        log.warning("[TRI] cv2.triangulatePoints produced no finite depths (w).")
        return []

    X = (X4[:3, valid_w] / w[valid_w]).T   # Nx3 valid
    idx_valid = np.flatnonzero(valid_w)

    # 4) Decompose poses for gating
    R1, C1, t1 = _RCt_from_Tcw(prev_kf.pose)
    R2, C2, t2 = _RCt_from_Tcw(cur_kf.pose)

    # Choose reprojection threshold
    if reproj_px_max is None:
        reproj_px_max = float(args.ransac_thresh)

    # Debug counters
    reasons = Counter()
    kept_ids = []
    examples_shown = 0

    # Precompute optional parallax stats for a quick sense of geometry
    all_parallaxes = None
    if use_parallax_gate:
        all_parallaxes = _angle_parallax_deg_batch(K_inv, pts1.astype(np.float64), pts2.astype(np.float64))
        sample_parallaxes = all_parallaxes[:200]
        if sample_parallaxes.size:
            log.info("[TRI] Parallax(sample of %d): med=%.2f°, p25=%.2f°, p75=%.2f°",
                     len(sample_parallaxes),
                     float(np.median(sample_parallaxes)),
                     float(np.percentile(sample_parallaxes, 25)),
                     float(np.percentile(sample_parallaxes, 75)))

    # 5) Iterate matches and corresponding 3D points
    min_d = float(getattr(args, "min_depth", 0.0))
    max_d = float(getattr(args, "max_depth", 1e6))
    pts1_valid = pts1[idx_valid]
    pts2_valid = pts2[idx_valid]

    Xc1 = (R1 @ X.T + t1.reshape(3, 1)).T
    Xc2 = (R2 @ X.T + t2.reshape(3, 1)).T
    z1_all = Xc1[:, 2]
    z2_all = Xc2[:, 2]
    front1 = z1_all > 1e-6
    front2 = z2_all > 1e-6

    e1_all = np.full(len(X), np.inf, dtype=np.float64)
    e2_all = np.full(len(X), np.inf, dtype=np.float64)
    if np.any(front1):
        u1h = (K @ (Xc1[front1] / z1_all[front1, None]).T).T[:, :2]
        e1_all[front1] = np.linalg.norm(u1h - pts1_valid[front1], axis=1)
    if np.any(front2):
        u2h = (K @ (Xc2[front2] / z2_all[front2, None]).T).T[:, :2]
        e2_all[front2] = np.linalg.norm(u2h - pts2_valid[front2], axis=1)

    for out_idx, m_idx in enumerate(idx_valid):
        Xw = X[out_idx]
        m = matches[m_idx]
        i1, i2 = m.queryIdx, m.trainIdx
        uv1 = pts1_valid[out_idx]
        uv2 = pts2_valid[out_idx]

        # (a) Optional parallax gate
        if use_parallax_gate:
            par = float(all_parallaxes[m_idx])
            if par < parallax_min_deg:
                reasons["low_parallax"] += 1
                if examples_shown < debug_max_examples:
                    log_tri.debug("reject: low_parallax (%.2f°)  match=(%d,%d)", par, i1, i2)
                continue

        # (b) Cheirality + depth window
        z1 = float(z1_all[out_idx])
        z2 = float(z2_all[out_idx])
        if not (min_d <= z1 <= max_d and min_d <= z2 <= max_d):
            reasons["bad_depth"] += 1
            if examples_shown < debug_max_examples:
                log_tri.debug("reject: bad_depth z1=%.3f z2=%.3f in [%.3f, %.3f]", z1, z2, min_d, max_d)
            continue

        # (c) Reprojection gate
        if (not front1[out_idx]) or (not front2[out_idx]):
            reasons["behind_cam"] += 1
            if examples_shown < debug_max_examples:
                log_tri.debug("reject: behind_cam  z1=%.3f z2=%.3f", z1, z2)
            continue

        e1 = float(e1_all[out_idx])
        e2 = float(e2_all[out_idx])
        if max(e1, e2) > reproj_px_max:
            reasons["high_reproj"] += 1
            if examples_shown < debug_max_examples:
                log_tri.debug("reject: high_reproj e1=%.2f e2=%.2f > %.2f", e1, e2, reproj_px_max)
            continue

        # (d) Insert MapPoint (descriptor: take from current KF)
        desc_cur = cur_kf.desc[i2] if cur_kf.desc is not None else None
        obs_list = [
            {"kf_idx": prev_kf.idx, "kp_idx": i1, "uv": uv1, "desc": prev_kf.desc[i1] if prev_kf.desc is not None else None},
            {"kf_idx": cur_kf.idx,  "kp_idx": i2, "uv": uv2, "desc": desc_cur},
        ]
        pid = _map_add_point(world_map, Xw, desc_cur, obs_list)
        kept_ids.append(pid)
        reasons["kept"] += 1

        if examples_shown < debug_max_examples:
            log_tri.debug("kept pid=%d  par=%.2f°  z1=%.2f z2=%.2f  e1=%.2f e2=%.2f", pid,
                          float(all_parallaxes[m_idx]) if use_parallax_gate else -1.0, z1, z2, e1, e2)
        examples_shown += 1

    # 6) Summary
    log.info("[TRI] KF %d<->%d: kept=%d of %d valid w  | reasons: %s  (reproj<=%.1fpx, depth∈[%.2f,%.2f])",
             prev_kf.idx, cur_kf.idx, len(kept_ids), int(valid_w.sum()),
             dict(reasons), reproj_px_max, min_d, max_d)

    return kept_ids
