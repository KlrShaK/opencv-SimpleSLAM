# keyframe.py
from dataclasses import dataclass
import cv2
import numpy as np
import lz4.frame

# --------------------------------------------------------------------------- #
#  Dataclass
# --------------------------------------------------------------------------- #
@dataclass
class Keyframe:
    idx:   int                    # global frame index
    path:  str                    # "" for in-memory frames
    kps:   list[cv2.KeyPoint]
    desc:  np.ndarray
    thumb: bytes                  # lz4-compressed JPEG


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def make_thumb(bgr, hw=(640, 360)):
    th = cv2.resize(bgr, hw)
    ok, enc = cv2.imencode('.jpg', th,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return lz4.frame.compress(enc.tobytes()) if ok else b''


def is_new_keyframe(frame_idx,
                    matches_to_kf,
                    n_kf_features,
                    kp_curr,
                    kp_kf,
                    kf_max_disp=30.0,
                    kf_min_inliers=125,
                    kf_cooldown=5,
                    last_kf_idx=-999):
    """
    Decide whether current frame should be promoted to a Keyframe.
    """
    if frame_idx - last_kf_idx < kf_cooldown:
        return False
    if not matches_to_kf or not n_kf_features:
        return True
    if len(matches_to_kf) < kf_min_inliers:
        return True

    disp = [np.hypot(kp_curr[m.trainIdx].pt[0] - kp_kf[m.queryIdx].pt[0],
                     kp_curr[m.trainIdx].pt[1] - kp_kf[m.queryIdx].pt[1])
            for m in matches_to_kf]
    return np.mean(disp) > kf_max_disp


# --------------------------------------------------------------------------- #
#  Track maintenance
# --------------------------------------------------------------------------- #
def update_and_prune_tracks(matches, prev_map, tracks,
                            kp_curr, frame_idx, next_track_id,
                            prune_age=30):
    """
    Continuation of simple 2-D point tracks across frames.
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
