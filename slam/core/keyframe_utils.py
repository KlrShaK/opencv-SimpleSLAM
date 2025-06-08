from dataclasses import dataclass
import lz4.frame
import cv2
import numpy as np


@dataclass
class Keyframe:
    idx: int                   # global frame index
    path: str                  # on-disk image file  OR "" if custom video
    kps:  list[cv2.KeyPoint]   # keypoints (needed for geometric checks)
    desc: np.ndarray           # descriptors (uint8/float32)
    thumb: bytes               # lz4-compressed thumbnail for UI

def is_new_keyframe(frame_idx: int,
                    matches_to_kf: list[cv2.DMatch],
                    n_kf_features: int,
                    kp_curr: list[cv2.KeyPoint],
                    kp_kf: list[cv2.KeyPoint],
                    kf_max_disp: float = 30.0,
                    kf_min_inliers: int = 125,
                    kf_cooldown: int = 5,
                    last_kf_idx: int = -999) -> bool:
    """
    Decide whether the *current* frame should become a new key-frame.

    Parameters
    ----------
    frame_idx      : index of the current frame in the sequence
    matches_to_kf  : list of inlier cv2.DMatch between LAST KF and current frame
    n_kf_features  : total number of keypoints that existed in the last KF
    kp_curr        : keypoints detected in the current frame
    kp_kf          : keypoints of the last key-frame
    kf_max_disp    : pixel-space parallax threshold (avg. L2) to trigger a new KF
    kf_min_inliers   : minimum *surviving-match* ratio; below → new KF
    kf_cooldown    : minimum #frames to wait after last KF before creating another
    last_kf_idx    : frame index of the most recent key-frame

    Returns
    -------
    bool           : True ⇒ promote current frame to key-frame
    """

    # 0) Cool-down guard                                                 #
    if frame_idx - last_kf_idx < kf_cooldown:
        return False            # too soon to spawn another KF

    # 1) Overlap / track-survival test                                   #
    # If the key-frame had zero features (should never happen) → force new KF
    if not matches_to_kf or not n_kf_features:       # no matches or features at all ⇒ we must create a new KF
        return True
    
    if len(matches_to_kf) < kf_min_inliers:
        return True            # too few matches survived, so we must create a new KF
    
    # OVERLAP RATIO TEST (not used anymore, see below) # This Logic is not working well, so we use a different metric 
    # overlap_ratio = len(matches_to_kf) / float(n_kf_features)
    # print(f"[KF] Overlap ratio: {overlap_ratio:.2f} (matches={len(matches_to_kf)})")
    # if overlap_ratio < kf_min_ratio:
    #     return True             

    # 2) Parallax test (average pixel displacement)                      #TODO replace with a more robust metric
    # Compute mean Euclidean displacement between matched pairs
    disp = [
        np.hypot(
            kp_curr[m.trainIdx].pt[0] - kp_kf[m.queryIdx].pt[0],
            kp_curr[m.trainIdx].pt[1] - kp_kf[m.queryIdx].pt[1]
        )
        for m in matches_to_kf
    ]

    if np.mean(disp) > kf_max_disp:
        return True
    return False

def make_thumb(bgr, hw=(854,480)):
    """Return lz4-compressed JPEG thumbnail (bytes)."""
    th = cv2.resize(bgr, hw)
    ok, enc = cv2.imencode('.jpg', th, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return lz4.frame.compress(enc.tobytes()) if ok else b''
