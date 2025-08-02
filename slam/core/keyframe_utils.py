# keyframe.py
from dataclasses import dataclass
import cv2
import numpy as np
import lz4.frame
from typing import List, Tuple
from slam.core.features_utils import feature_matcher, filter_matches_ransac

# --------------------------------------------------------------------------- #
#  Dataclass
# --------------------------------------------------------------------------- #
@dataclass
class Keyframe:
    idx:   int                    # global frame index
    frame_idx: int                # actual frame number (0-based), where this KF was created
    path:  str                    # "" for in-memory frames
    kps:   list[cv2.KeyPoint]
    desc:  np.ndarray
    pose: np.ndarray              # 4×4 camera-to-world pose
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
                    kp_curr,
                    kp_kf,
                    kf_max_disp=30.0,
                    kf_min_inliers=125,
                    kf_cooldown=5,
                    last_kf_frame_no=-999):
    """
    Decide whether current frame should be promoted to a Keyframe.
    """
    if frame_idx - last_kf_frame_no > kf_cooldown: 
        return True  # enough time has passed since last KF
    if not matches_to_kf:
        return True
    if len(matches_to_kf) < kf_min_inliers:
        return True
    # TODO as a ratio of frame rather than pixels
    disp = [np.hypot(kp_curr[m.trainIdx].pt[0] - kp_kf[m.queryIdx].pt[0],
                     kp_curr[m.trainIdx].pt[1] - kp_kf[m.queryIdx].pt[1])
            for m in matches_to_kf]
    return np.mean(disp) > kf_max_disp

def select_keyframe(
    args,
    seq: List[str],
    frame_idx: int,
    img2, kp2, des2,
    pose2,
    matcher,
    kfs: List[Keyframe],
    last_kf_frame_no: int
) -> Tuple[List[Keyframe], int]:
    """
    Decide whether to add a new Keyframe at this iteration.

    Parameters
    ----------
    args
        CLI namespace (provides use_lightglue, ransac_thresh, kf_* params).
    seq
        Original sequence list (so we can grab file paths if needed).
    frame_idx
        zero-based index of the *first* of the pair.  Keyframes use i+1 as frame number.
    img2
        BGR image for frame i+1 (for thumbnail if we promote).
    kp2, des2
        KPs/descriptors of frame i+1.
    pose2
        Current camera-to-world pose estimate for frame i+1 (4×4).  May be None.
    matcher
        Either the OpenCV BF/FLANN matcher or the LightGlue matcher.
    kfs
        Current list of Keyframe objects.
    last_kf_frame_no
        Frame number (1-based) of the last keyframe added; or -inf if none.

    Returns
    -------
    kfs
        Possibly-extended list of Keyframe objects.
    last_kf_frame_no
        Updated last keyframe frame number (still the same if we didn’t add one).
    """
    frame_no = frame_idx + 1
    prev_kf = kfs[-1]
    # 1) match descriptors from the old KF to the new frame
    raw_matches = feature_matcher(
        args, prev_kf.kps, kp2, prev_kf.desc, des2, matcher
    )
    # 2) drop outliers
    matches = filter_matches_ransac(
        prev_kf.kps, kp2, raw_matches, args.ransac_thresh
    )

    # 3) promotion test
    if is_new_keyframe(frame_no, matches, kp2, prev_kf.kps, args.kf_max_disp, args.kf_min_inliers,
                       args.kf_cooldown, last_kf_frame_no):
        thumb = make_thumb(img2, tuple(args.kf_thumb_hw))
        path  = seq[frame_idx + 1] if isinstance(seq[frame_idx + 1], str) else ""
        seq_id = len(kfs)          # 0-based sequential KF ID; use +1 for 1-based
        kfs.append(Keyframe(seq_id, frame_no, path, kp2, des2, pose2, thumb))
        last_kf_frame_no = frame_no

    return kfs, last_kf_frame_no
