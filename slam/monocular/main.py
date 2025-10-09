# main.py
#  MAIN: Frame-by-frame tracking using Essential/Fundamental only (NO PnP)
"""
Entry-point: high-level processing loop
--------------------------------------
$ python main.py --dataset kitti --base_dir ../Dataset

Pipeline:
  1) Feature detection + matching (OpenCV or LightGlue)
  2) Essential-matrix estimation + pose recovery (every frame)
     - Fall back to Fundamental → Essential if needed
     - Scale handling: inherit previous baseline length
  3) Pose integration (camera trajectory in world frame)
  4) Keyframe selection + Two-view triangulation
  5) Optional local BA, 2-D/3-D visualization

Notes:
- Entire PnP / 2D–3D reprojection path is removed.
- We keep your two-view bootstrap + KF/Map machinery intact.
"""

import argparse
from copy import deepcopy
import cv2
import lz4.frame
import numpy as np
from tqdm import tqdm
from typing import List
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s:%(funcName)s: %(message)s")
log = logging.getLogger("main")
log.setLevel(logging.DEBUG)

from slam.core.pose_utils import _pose_inverse, _pose_rt_to_homogenous

from slam.core.dataloader import (
    load_sequence,
    load_frame_pair,
    load_calibration,
    load_groundtruth
)

from slam.core.features_utils import (
    init_feature_pipeline,
    feature_extractor,
    feature_matcher,
    filter_matches_ransac
)

from slam.core.keyframe_utils import (
    Keyframe,
    select_keyframe,
    make_thumb
)

from slam.core.visualization_utils import draw_tracks, Visualizer3D, Trajectory2D, VizUI
from slam.core.trajectory_utils import compute_gt_alignment, apply_alignment
from slam.core.landmark_utils import Map

from slam.core.two_view_bootstrap import (
    InitParams,
    pts_from_matches,
    evaluate_two_view_bootstrap_with_masks,
    bootstrap_two_view_map
)

# --- Removed all PnP / 2D-3D imports ---
# from slam.core.pnp_utils import predict_pose_const_vel, reproject_and_match_2d3d, solve_pnp_ransac, draw_reprojection_debug

from slam.core.triangulation_utils import triangulate_between_kfs_2view
from slam.core.ba_utils import pose_only_ba, local_bundle_adjustment, global_bundle_adjustment


class BootstrapState:
    def __init__(self):
        self.has_ref = False
        self.kps_ref = None
        self.des_ref = None
        self.img_ref = None
        self.frame_id_ref = -1
    def seed(self, kps, des, img, frame_id):
        self.has_ref = True
        self.kps_ref, self.des_ref = kps, des
        self.img_ref, self.frame_id_ref = img, frame_id
        log.info(f"[Init] Seeded reference @frame={frame_id} (kps={len(kps)})")
    def clear(self):
        log.info("[Init] Clearing reference (bootstrap succeeded).")
        self.__init__()

def _refresh_ref_needed(matches, min_matches=80, max_age=30, cur_id=0, ref_id=0):
    too_few = len(matches) < min_matches
    too_old = (cur_id - ref_id) > max_age
    if too_few: log.info(f"[Init] Refresh ref: few matches ({len(matches)}<{min_matches})")
    if too_old: log.info(f"[Init] Refresh ref: age={cur_id-ref_id}>{max_age}")
    return too_few or too_old

def _safe_norm(v):
    n = np.linalg.norm(v)
    return n if n > 1e-12 else 1e-12

# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Feature tracking with key-frames (E/F-only, no PnP)")
    p.add_argument('--dataset',
                   choices=['kitti', 'malaga', 'tum-rgbd', 'custom'],
                   required=True)
    p.add_argument('--base_dir', default='../Dataset')

    # feature/detector settings
    p.add_argument('--detector', choices=['orb', 'sift', 'akaze'],
                   default='orb')
    p.add_argument('--matcher', choices=['bf', 'flann'], default='bf')
    p.add_argument('--use_lightglue', action='store_true')
    p.add_argument('--min_conf', type=float, default=0.7,
                   help='Minimum LightGlue confidence for a match')

    # runtime
    p.add_argument('--fps', type=float, default=10)

    # RANSAC
    p.add_argument('--ransac_thresh', type=float, default=2.0)

    # key-frame params (unchanged)
    p.add_argument('--kf_max_disp', type=float, default=45)
    p.add_argument('--kf_min_inliers', type=float, default=150)
    p.add_argument('--kf_min_ratio', type=float, default=0.35,
                   help='Min inlier ratio (to prev KF kps) before promoting KF')
    p.add_argument('--kf_min_rot_deg', type=float, default=8.0,
                   help='Min rotation (deg) wrt prev KF to trigger KF')
    p.add_argument('--kf_cooldown', type=int, default=1)
    p.add_argument('--kf_thumb_hw', type=int, nargs=2,
                   default=[640, 360])

    # 3-D visualisation toggle
    p.add_argument("--no_viz3d", action="store_true", help="Disable 3-D visualization window")

    # triangulation depth filtering
    p.add_argument("--min_depth", type=float, default=0.40)
    p.add_argument("--max_depth", type=float, default=100.0)
    p.add_argument('--mvt_rep_err', type=float, default=2.0,
                   help='Max mean reprojection error (px) for multi-view triangulation')

    # Bundle Adjustment
    p.add_argument('--local_ba_window', type=int, default=10, help='Window size (number of keyframes) for local BA')

    # Global BA
    p.add_argument('--gba_every', type=int, default=100, help='Run global BA every N frames')
    p.add_argument('--gba_max_points', type=int, default=None, help='Cap points in GBA (None = all)')
    p.add_argument('--gba_max_iters', type=int, default=30, help='Ceres iterations for GBA')
    p.add_argument('--gba_fix_first', type=int, default=1, help='1=fix first KF to anchor gauge, 0=free')

    return p

import cv2
import numpy as np
from typing import Tuple, Union

def shift_image(
    img: np.ndarray,
    dx: int = 10,            # +right, -left
    dy: int = 0,             # +down,  -up
    border: str = "constant",
    value: Union[int, Tuple[int, int, int]] = 0
) -> np.ndarray:
    """
    Shift an image by (dx, dy) pixels.

    Args:
        img: np.ndarray (H,W) or (H,W,C), any dtype.
        dx: pixels to shift right (negative shifts left).
        dy: pixels to shift down (negative shifts up).
        border: how to fill exposed areas: "constant", "edge", "reflect", "wrap".
        value: fill value if border=="constant". Use (B,G,R) tuple for color images.

    Returns:
        Shifted image with same shape as input.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy.ndarray")

    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])

    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "edge":     cv2.BORDER_REPLICATE,
        "reflect":  cv2.BORDER_REFLECT_101,
        "wrap":     cv2.BORDER_WRAP,
    }
    bm = border_modes.get(border, cv2.BORDER_CONSTANT)

    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=bm,
        borderValue=value
    )

# --------------------------------------------------------------------------- #
#  Main processing loop
# --------------------------------------------------------------------------- #
def main():
    args = _build_parser().parse_args()

    # --- Data loading ---
    seq = load_sequence(args)
    calib       = load_calibration(args)        # dict with K_l, P_l, ...
    groundtruth = load_groundtruth(args)        # None or Nx3x4 array
    K = calib["K_l"]  # intrinsic matrix for left camera
    P = calib["P_l"]  # projection matrix for left camera

    # ------ build 4×4 GT poses + alignment matrix (once) ----------------
    gt_T = None
    R_align = t_align = None
    if groundtruth is not None:
        gt_T = np.pad(groundtruth, ((0, 0), (0, 1), (0, 0)), constant_values=0.0)
        gt_T[:, 3, 3] = 1.0
        R_align, t_align = compute_gt_alignment(gt_T)

    # --- feature pipeline (OpenCV / LightGlue) ---
    detector, matcher = init_feature_pipeline(args)

    bs = BootstrapState()

    # --- tracking state ---
    initialised = False

    # --- World Map Initialization ---
    world_map = Map()
    Tcw_cur_pose = np.eye(4)  # camera-from-world (identity at t=0)

    # --- Visualization  ---
    viz3d = None if args.no_viz3d else Visualizer3D(color_axis="y")
    traj2d = Trajectory2D(gt_T_list=gt_T if groundtruth is not None else None)
    ui = VizUI()  # p: pause/resume, n: step, q/Esc: quit

    # --- Keyframe Initialization ---
    kfs: list[Keyframe] = []
    last_kf_frame_no = -999

    new_ids:  list[int] = []  # for viz
    total = len(seq) - 1

    # --------------- RECIPE: Frame-to-Frame tracking (copy/paste) -----------------
    # Every iteration (img1->img2):
    #   kp1,des1 = feature_extractor(img1); kp2,des2 = feature_extractor(img2)
    #   matches  = feature_matcher(kp1,kp2,des1,des2)  # + filter_matches_ransac
    #   pts1,pts2 = np.float32([...]), np.float32([...])
    #   E, mE = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, ransac_px)
    #   if E invalid → try F: F,mF=cv2.findFundamentalMat(...);  E = K.T @ F @ K
    #   _, R, t, mPose = cv2.recoverPose(E, pts1, pts2, K, mask=mE_or_mF)
    #   # scale handling: set |t| to previous baseline length
    #   t = (t/||t||) * last_baseline_or_1
    #   T_rel = [R|t];  Tcw_cur = T_rel @ Tcw_prev
    #   world_map.add_pose(Tcw_cur, is_keyframe=False); traj2d.push(...)
    # ------------------------------------------------------------------------------

    for i in tqdm(range(total), desc='Tracking'):

        # --- load image pair ---
        img1, img2 = load_frame_pair(args, seq, i)
        # img2 = img1.copy() # shift img2 by few pixels to simulate motion
        # img2 = shift_image(img2, dx=10, dy=0, border="edge")  # simulate rightward motion
        # print(type(img2))
        # --- feature extraction / matching ---
        kp1, des1 = feature_extractor(args, img1, detector)
        kp2, des2 = feature_extractor(args, img2, detector)

        # --------------------------------------------------------------------------- #
        # ------------------------ Bootstrap (delayed two-view) ----------------------#
        # --------------------------------------------------------------------------- #
        if not initialised:
            if not bs.has_ref:
                bs.seed(kp1, des1, img1, frame_id=i)   # reference = frame i (img1)
                continue

            matches_bs = feature_matcher(args, bs.kps_ref, kp2, bs.des_ref, des2, matcher)
            matches_bs = filter_matches_ransac(bs.kps_ref, kp2, matches_bs, args.ransac_thresh)
            log.info(f"[Init] Matches ref→cur: raw={len(matches_bs)}  ransac_th={args.ransac_thresh:.2f}px")

            if _refresh_ref_needed(matches_bs, min_matches=80, max_age=30, cur_id=i+1, ref_id=bs.frame_id_ref):
                bs.seed(kp2, des2, img2, frame_id=i+1)
                continue

            pts_ref, pts_cur = pts_from_matches(bs.kps_ref, kp2, matches_bs)
            init_params = InitParams(
                ransac_px=float(args.ransac_thresh),
                min_posdepth=0.90,
                min_parallax_deg=1.5,
                score_ratio_H=0.45
            )

            decision = evaluate_two_view_bootstrap_with_masks(K, pts_ref, pts_cur, init_params)
            if decision is None:
                log.info("[Init] Pair rejected → waiting for a better one.")
                continue

            log.info(f"[Init] Accepted pair via {'H' if decision.pose.model.name=='HOMOGRAPHY' else 'F/E'}: "
                     f"posdepth={decision.pose.posdepth:.3f}, parallax={decision.pose.parallax_deg:.2f}°")

            ok, T0_cw, T1_cw = bootstrap_two_view_map(
                K,
                bs.kps_ref, bs.des_ref,
                kp2, des2,
                matches_bs,
                args,
                world_map,
                params=init_params,
                decision=decision
            )
            if not ok:
                log.warning("[Init] bootstrap_two_view_map() failed — keep searching.")
                continue

            world_map.add_pose(T0_cw, is_keyframe=True)   # KF0
            world_map.add_pose(T1_cw, is_keyframe=True)   # KF1
            traj2d.push(bs.frame_id_ref, T0_cw)
            traj2d.push(i + 1, T1_cw)

            try:
                kf0_idx = 0 if len(kfs) == 0 else len(kfs)
                kf1_idx = kf0_idx + 1
                ref_fidx = bs.frame_id_ref
                cur_fidx = i + 1
                ref_path = seq[ref_fidx] if isinstance(seq[ref_fidx], str) else ""
                cur_path = seq[cur_fidx] if isinstance(seq[cur_fidx], str) else ""
                thumb0 = make_thumb(bs.img_ref, tuple(args.kf_thumb_hw)) if 'make_thumb' in globals() else None
                thumb1 = make_thumb(img2,       tuple(args.kf_thumb_hw)) if 'make_thumb' in globals() else None
                kf0 = Keyframe(idx=kf0_idx, frame_idx=ref_fidx, path=ref_path, kps=bs.kps_ref, desc=bs.des_ref, pose=T0_cw, thumb=thumb0)
                kf1 = Keyframe(idx=kf1_idx, frame_idx=cur_fidx, path=cur_path, kps=kp2, desc=des2, pose=T1_cw, thumb=thumb1)
                kfs.extend([kf0, kf1])
                last_kf_frame_no = cur_fidx
                raw01 = feature_matcher(args, kfs[0].kps, kfs[1].kps, kfs[0].desc, kfs[1].desc, matcher)  # seed first pair links
            except Exception as e:
                log.exception("[Init] Failed to create initial keyframes: %s", e)

            print("-----BOOTSTRAPPED SUCCESSFULLY-----")
            if viz3d:
                viz3d.update(world_map, new_ids=world_map.point_ids())

            Tcw_cur_pose = T1_cw.copy()
            bs.clear()
            initialised = True
            continue

        # ------------------------------------------------------------------- #
        # --------------------- Frame-to-Frame Tracking (E/F) --------------- #
        # ------------------------------------------------------------------- #
        # 1) Match prev frame (img1) → current frame (img2)
        tracking_matches = feature_matcher(args, kp1, kp2, des1, des2, matcher)
        tracking_matches = filter_matches_ransac(kp1, kp2, tracking_matches, args.ransac_thresh)

        if len(tracking_matches) < 8:
            log.warning(f"[Track] Too few matches for E/F: {len(tracking_matches)}")
            # Keep previous pose (dead-reckon); still push to traj for continuity
            world_map.add_pose(Tcw_cur_pose, is_keyframe=False)
            traj2d.push(i + 1, Tcw_cur_pose)
        else:
            pts0 = np.float32([kp1[m.queryIdx].pt for m in tracking_matches])
            pts1 = np.float32([kp2[m.trainIdx].pt for m in tracking_matches])

            # Prefer Essential (calibrated); fall back to Fundamental
            E, inlE = cv2.findEssentialMat(pts0, pts1, K, cv2.RANSAC, 0.999, args.ransac_thresh)
            mask_pose = None
            if E is not None and inlE is not None and int(inlE.sum()) >= 5:
                _, R, t, pose_mask = cv2.recoverPose(E, pts0, pts1, K, mask=inlE)
                mask_pose = pose_mask
            else:
                # Fundamental fallback
                F, inlF = cv2.findFundamentalMat(pts0, pts1, cv2.USAC_MAGSAC, 0.999, args.ransac_thresh)
                if F is None or inlF is None or int(inlF.sum()) < 7:
                    log.warning("[Track] E and F failed — keeping previous pose.")
                    world_map.add_pose(Tcw_cur_pose, is_keyframe=False)
                    traj2d.push(i + 1, Tcw_cur_pose)
                    # proceed to KF logic / viz as usual
                else:
                    # Upgrade F → E using calibration
                    E = K.T @ F @ K
                    _, R, t, pose_mask = cv2.recoverPose(E, pts0, pts1, K, mask=inlF)
                    mask_pose = pose_mask

            if E is not None and mask_pose is not None:
                # --- scale handling: inherit last baseline length ---
                if len(world_map.poses) >= 2:
                    Tcw_prevprev = world_map.poses[-2]
                    Tcw_prev     = world_map.poses[-1]
                    T_prev_rel   = Tcw_prev @ np.linalg.inv(Tcw_prevprev)   # T_cw(k-1) * inv(T_cw(k-2))
                    last_baseline = _safe_norm(T_prev_rel[:3, 3])
                else:
                    last_baseline = 1.0

                t = t.reshape(3, 1)
                t = (t / _safe_norm(t)) * last_baseline

                T_rel = _pose_rt_to_homogenous(R, t)      # T_c2c1
                Tcw_cur_pose = T_rel @ world_map.poses[-1]  # T_cw2 = T_c2c1 @ T_cw1

                world_map.add_pose(Tcw_cur_pose, is_keyframe=False)
                traj2d.push(i + 1, Tcw_cur_pose)

                log.debug(f"[Track] E/F inliers={int(mask_pose.sum())}")

                # (Optional) quick visual of inlier tracks on img2
                try:
                    vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2.copy()
                    inl_idx = np.where(mask_pose.ravel() > 0)[0]
                    for j in inl_idx:
                        x0, y0 = map(int, pts0[j])
                        x1, y1 = map(int, pts1[j])
                        cv2.circle(vis, (x1, y1), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                        cv2.line(vis, (x0, y0), (x1, y1), (0, 255, 0), 1, lineType=cv2.LINE_AA)
                    cv2.imshow("E/F inlier tracks", vis)
                except Exception:
                    pass

            # --- Pose-only BA on most recent KF (still useful; not PnP) ---
            try:
                if len(kfs) > 0:
                    pose_only_ba(world_map, K, kfs, kf_idx=(len(kfs) - 1), max_iters=8, huber_thr=2.0)
            except Exception as e:
                log.warning(f"[BA] Pose-only BA failed: {e}")

        # ------------------------------------------------------------------- #
        # ---------------------     Keyframe Selection  --------------------- #
        # ------------------------------------------------------------------- #
        prev_len = len(kfs)
        kfs, last_kf_frame_no = select_keyframe(
            args, seq, i, img2, kp2, des2, Tcw_cur_pose, matcher, kfs, last_kf_frame_no
        )
        is_kf = (len(kfs) > prev_len)
        if is_kf:
            log.debug(f"[KF] Added KF #{kfs[-1].idx} @frame={kfs[-1].frame_idx} (total KFs={len(kfs)})")

        # ------------------------------------------------------------------- #
        # --------------------- Map Growth (Triangulation) ------------------ #
        # ------------------------------------------------------------------- #
        if is_kf and len(kfs) >= 2:
            prev_kf = kfs[-2]
            curr_kf = kfs[-1]
            new_ids = triangulate_between_kfs_2view(
                args, K, world_map, prev_kf, curr_kf, matcher, log,
                use_parallax_gate=True, parallax_min_deg=2.0,
                reproj_px_max=float(args.ransac_thresh)
            )
            if new_ids:
                log.info("[Map] Triangulated %d new points after KF %d.", len(new_ids), curr_kf.idx)
                try:
                    local_bundle_adjustment(
                        world_map, K, kfs,
                        center_kf_idx=curr_kf.idx,
                        window_size=int(getattr(args, "local_ba_window", 6)),
                        max_points=10000,
                        max_iters=15
                    )
                except Exception as e:
                    log.warning(f"[BA] Local BA failed: {e}")

                if viz3d:
                    viz3d.update(world_map, new_ids=new_ids)

        # ------------------------------------------------------------------- #
        # --------------------- Visualization ----------------------- #
        # ------------------------------------------------------------------- #
        if viz3d:
            viz3d.update(world_map, new_ids=new_ids)

        # ---- helpers: LZ4->JPEG->BGR and safe conversions ------------------
        import lz4.frame as lz4f

        def _ensure_u8(img):
            if img is None:
                return None
            if img.dtype == np.uint8:
                return img
            if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.01:
                img = img * 255.0
            return np.clip(img, 0, 255).astype(np.uint8)

        def _decode_jpeg(buf_like):
            arr = np.frombuffer(buf_like, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        def _im_from_any(im):
            """Return np.uint8 BGR image from: np.ndarray | (LZ4-)bytes | PIL."""
            if isinstance(im, (bytes, bytearray, memoryview)):
                try:
                    dec = lz4f.decompress(im)
                    img = _decode_jpeg(dec)
                except Exception:
                    img = _decode_jpeg(im)
                if img is None:
                    return None
            elif isinstance(im, np.ndarray):
                img = im
            else:
                try:
                    from PIL import Image
                    if isinstance(im, Image.Image):
                        img = np.array(im)
                    else:
                        return None
                except Exception:
                    return None

            img = _ensure_u8(img)
            if img is None:
                return None
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.ndim == 3:
                if img.shape[2] == 3:
                    return img
                if img.shape[2] == 4:
                    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return None

        def _thumb(im, size_wh):
            bgr = _im_from_any(im)
            if bgr is None:
                w, h = size_wh
                return np.zeros((h, w, 3), dtype=np.uint8)
            w, h = size_wh
            return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

        # --- Make sure HighGUI is available and windows are created once ---
        if 'HAS_HIGHGUI' not in globals():
            globals()['HAS_HIGHGUI'] = True
            try:
                cv2.namedWindow("Strip: img2 + last 3 KFs", cv2.WINDOW_NORMAL)
                cv2.namedWindow("img2 + prev→cur matches", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Strip: img2 + last 3 KFs", 1200, 380)
                cv2.resizeWindow("img2 + prev→cur matches", 900, 600)
                cv2.moveWindow("Strip: img2 + last 3 KFs", 40, 40)
                cv2.moveWindow("img2 + prev→cur matches", 40, 460)
            except Exception as e:
                log.warning("[Viz] OpenCV HighGUI unavailable, disabling cv2 windows: %s", e)
                globals()['HAS_HIGHGUI'] = False

        # --- 1) Horizontal strip: [ current img2 | last 3 keyframes ] ---
        # --- 2) img2 overlaid with ONLY prev→cur matched features       ---
        if globals().get('HAS_HIGHGUI', False):
            try:
                # Thumb size config (args.kf_thumb_hw expected as [w, h])
                W_thumb, H_thumb = map(int, args.kf_thumb_hw)

                # Current frame (bytes-safe)
                cur_tile = _thumb(img2, (W_thumb, H_thumb))
                cv2.putText(cur_tile, "cur frame", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Last 3 KFs — their .thumb is LZ4-compressed JPEG bytes
                recent_kfs = kfs[-1:-4:-1] if len(kfs) > 0 else []
                tiles = [cur_tile]
                for kf in recent_kfs:
                    if getattr(kf, "thumb", None) is not None and len(kf.thumb) > 0:
                        th = _thumb(kf.thumb, (W_thumb, H_thumb))
                    elif getattr(kf, "path", ""):
                        raw = cv2.imread(kf.path, cv2.IMREAD_UNCHANGED)
                        th = _thumb(raw, (W_thumb, H_thumb))
                    else:
                        th = np.zeros((H_thumb, W_thumb, 3), dtype=np.uint8)

                    label = f"KF{getattr(kf, 'idx', '?')} (f{getattr(kf, 'frame_idx', '?')})"
                    cv2.putText(th, label, (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    tiles.append(th)

                while len(tiles) < 4:
                    tiles.append(np.zeros((H_thumb, W_thumb, 3), dtype=np.uint8))

                strip = cv2.hconcat([_im_from_any(t) for t in tiles])
                cv2.imshow("Strip: img2 + last 3 KFs", strip)

                # --- img2 with ONLY features matched to previous frame (img1) ---
                feat_vis = _im_from_any(img2)
                if feat_vis is None:
                    feat_vis = np.zeros((H_thumb, W_thumb, 3), dtype=np.uint8)

                match_count = 0
                if (kp1 is not None and kp2 is not None and
                    des1 is not None and des2 is not None and
                    len(kp1) > 0 and len(kp2) > 0):

                    # Compute prev→cur matches and filter geometrically
                    viz_matches = feature_matcher(args, kp1, kp2, des1, des2, matcher)
                    viz_matches = filter_matches_ransac(kp1, kp2, viz_matches, args.ransac_thresh)

                    # Draw only the matched keypoints in img2
                    for m in viz_matches:
                        x, y = map(int, kp2[m.trainIdx].pt)
                        cv2.circle(feat_vis, (x, y), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)

                    match_count = len(viz_matches)

                cv2.putText(feat_vis, f"prev→cur matches: {match_count}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("img2 + prev→cur matches", feat_vis)

                # Make the windows actually refresh
                cv2.waitKey(1)

            except Exception as e:
                log.exception("[Viz] HighGUI failed; disabling further cv2 windows.")
                globals()['HAS_HIGHGUI'] = False

        # --- draw & UI control (end of iteration) ---
        traj2d.draw(paused=ui.paused)
        ui.poll(1)
        if ui.should_quit():
            break
        if ui.paused:
            _ = ui.wait_if_paused()

    if viz3d:
        viz3d.close()

if __name__ == '__main__':
    main()
