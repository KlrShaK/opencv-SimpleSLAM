# main.py
"""
WITH MULTI-VIEW TRIANGULATION - New PnP

Entry-point: high-level processing loop
--------------------------------------
$ python main.py --dataset kitti --base_dir ../Dataset


The core loop now performs:
  1) Feature detection + matching (OpenCV or LightGlue)
  2) Essential‑matrix estimation + pose recovery
  3) Landmarks triangulation (with Z‑filtering)
  4) Pose integration (camera trajectory in world frame)
  5) Optional 3‑D visualisation via Open3D 

The script shares most command‑line arguments with the previous version
but adds `--no_viz3d` to disable the 3‑D window.

"""
import argparse
from copy import deepcopy
import cv2
import lz4.frame
import numpy as np
from tqdm import tqdm
from typing import List
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s:%(funcName)s: %(message)s") # INFO for clean summary, DEBUG for deep dive
# logging.getLogger("two_view_bootstrap").setLevel(logging.DEBUG)
# logging.getLogger("pnp").setLevel(logging.DEBUG)
# logging.getLogger("multi_view").setLevel(logging.DEBUG)
# logging.getLogger("triangulation").setLevel(logging.DEBUG)
# logging.getLogger("ba").setLevel(logging.DEBUG)
log = logging.getLogger("main")
log.setLevel(logging.INFO)


from slam.core.pose_utils import _pose_inverse, _pose_rt_to_homogenous

from slam.core.dataloader import (
                            load_sequence, 
                            load_frame_pair, 
                            load_calibration, 
                            load_groundtruth)

from slam.core.features_utils import (
                                init_feature_pipeline, 
                                feature_extractor, 
                                feature_matcher, 
                                filter_matches_ransac)

from slam.core.keyframe_utils import (
    Keyframe, 
    select_keyframe, 
    make_thumb)

from slam.core.visualization_utils import draw_tracks, Visualizer3D, Trajectory2D, VizUI
from slam.core.trajectory_utils import compute_gt_alignment, apply_alignment
from slam.core.landmark_utils import Map

from slam.core.two_view_bootstrap import (
    InitParams,
    pts_from_matches,                        # used to build point arrays
    evaluate_two_view_bootstrap_with_masks,  # NEW (returns inlier mask)
    bootstrap_two_view_map                   # NEW (builds KF0+KF1+points)
)

from slam.core.pnp_utils import (
    predict_pose_const_vel,
    reproject_and_match_2d3d,
    solve_pnp_ransac,
    draw_reprojection_debug
)

from slam.core.triangulation_utils import (
    triangulate_between_kfs_2view
)

from slam.core.ba_utils import local_bundle_adjustment, global_bundle_adjustment



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


def _load_frame(seq, frame_idx: int, dataset: str):
    if dataset == "custom":
        return seq[frame_idx]
    img = cv2.imread(seq[frame_idx], cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to load frame {frame_idx}: {seq[frame_idx]}")
    return img


def _match_with_ransac(args, kp0, kp1, des0, des1, matcher):
    if (
        kp0 is None or kp1 is None or des0 is None or des1 is None
        or len(kp0) == 0 or len(kp1) == 0
        or len(des0) == 0 or len(des1) == 0
    ):
        return [], []
    raw = feature_matcher(args, kp0, kp1, des0, des1, matcher)
    return raw, filter_matches_ransac(kp0, kp1, raw, args.ransac_thresh)


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
            dec = lz4.frame.decompress(im)
            img = _decode_jpeg(dec)
            if img is None:
                return None
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
        c = img.shape[2]
        if c == 3:
            return img
        if c == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return None


def _thumb(im, size_wh):
    bgr = _im_from_any(im)
    if bgr is None:
        w, h = size_wh
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.resize(bgr, size_wh, interpolation=cv2.INTER_AREA)

# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Feature tracking with key-frames")
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
    p.add_argument('--max_features', type=int, default=4000,
                   help='Max features/keypoints for OpenCV detectors and ALIKED')
    # runtime
    p.add_argument('--fps', type=float, default=10)

    # RANSAC
    p.add_argument('--ransac_thresh', type=float, default=2.5)

    # key-frame params
    p.add_argument('--kf_max_disp', type=float, default=45)
    p.add_argument('--kf_min_inliers', type=float, default=150)
    p.add_argument('--kf_min_ratio', type=float, default=0.35,
               help='Min inlier ratio (to prev KF kps) before promoting KF')
    p.add_argument('--kf_min_rot_deg', type=float, default=8.0,
               help='Min rotation (deg) wrt prev KF to trigger KF')
    p.add_argument('--kf_cooldown', type=int, default=5)
    p.add_argument('--kf_thumb_hw', type=int, nargs=2,
                   default=[640, 360])
    
    # 3‑D visualisation toggle
    p.add_argument("--no_viz3d", action="store_true", help="Disable 3‑D visualization window")
    p.add_argument("--headless", action="store_true",
                   help="Disable ALL visualization during the run; save trajectory graph at the end")

    # triangulation depth filtering
    p.add_argument("--min_depth", type=float, default=0.40) # 0.40
    p.add_argument("--max_depth", type=float, default=100.0) # float('inf')
    p.add_argument('--mvt_rep_err', type=float, default=2.0,
               help='Max mean reprojection error (px) for multi-view triangulation')

    #  PnP / map-maintenance
    p.add_argument('--pnp_min_inliers', type=int, default=30)
    p.add_argument('--proj_radius',     type=float, default=10.0) # 12
    p.add_argument('--merge_radius',    type=float, default=0.10)

    # Bundle Adjustment
    p.add_argument('--local_ba_window', type=int, default=10, help='Window size (number of keyframes) for local BA')
    p.add_argument('--local_ba_min_new_points', type=int, default=60,
                   help='Only run local BA when at least this many new landmarks were triangulated')
    p.add_argument('--local_ba_max_points', type=int, default=5000,
                   help='Cap landmarks included in local BA for runtime')
    p.add_argument('--local_ba_max_iters', type=int, default=12,
                   help='Maximum Ceres iterations for local BA')

    # Global BA
    p.add_argument('--gba_every', type=int, default=100, help='Run global BA every N frames')
    p.add_argument('--gba_max_points', type=int, default=None, help='Cap points in GBA (None = all)')
    p.add_argument('--gba_max_iters', type=int, default=30, help='Ceres iterations for GBA')
    p.add_argument('--gba_fix_first', type=int, default=1, help='1=fix first KF to anchor gauge, 0=free')

    return p


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

    # --- Undistort images if distortion coefficients are available ---
    D = calib.get("D_l", None)
    undist_maps = None

    # ------ build 4×4 GT poses + alignment matrix (once) ----------------
    gt_T = None
    if groundtruth is not None:
        gt_T = np.pad(groundtruth, ((0, 0), (0, 1), (0, 0)), constant_values=0.0)
        gt_T[:, 3, 3] = 1.0                             # homogeneous 1s
        compute_gt_alignment(gt_T)

    # --- feature pipeline (OpenCV / LightGlue) ---
    detector, matcher = init_feature_pipeline(args)
    bs = BootstrapState()

    initialised = False
    tracking_lost = False

    # --- World Map Initialization ---
    world_map = Map()
    Tcw_cur_pose = np.eye(4)  # camera-from-world (identity at t=0)

    # --- Visualization  ---
    headless = getattr(args, "headless", False)
    viz3d = None if (args.no_viz3d or headless) else Visualizer3D(color_axis="y")
    traj2d = Trajectory2D(gt_T_list=gt_T if groundtruth is not None else None)
    ui = None if headless else VizUI()  # p: pause/resume, n: step, q/Esc: quit

    # --- Keyframe Initialization ---
    kfs: list[Keyframe] = []
    last_kf_frame_no = -999

    # --- Global BA ---
    last_gba_kf_count = -1
    do_gba = False

    total_frames = len(seq)
    img_prev = _load_frame(seq, 0, args.dataset)
    # Build undistortion maps once (if D is non-zero) and apply to first frame
    if D is not None and np.any(D != 0):
        H0, W0 = img_prev.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (W0, H0), alpha=0, newImgSize=(W0, H0))
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (W0, H0), cv2.CV_32FC1)
        K = new_K
        undist_maps = (mapx, mapy)
        img_prev = cv2.remap(img_prev, mapx, mapy, cv2.INTER_LINEAR)
        log.info("[Init] Undistortion enabled — updated K to new optimal camera matrix.")
    
    kp_prev, des_prev = feature_extractor(args, img_prev, detector)

    for frame_idx in tqdm(range(1, total_frames), desc='Tracking'):
        img_cur = _load_frame(seq, frame_idx, args.dataset)
        if undist_maps is not None:
            img_cur = cv2.remap(img_cur, undist_maps[0], undist_maps[1], cv2.INTER_LINEAR)
        kp_cur, des_cur = feature_extractor(args, img_cur, detector)
        new_ids: list[int] = []
        # Compute frame-to-frame matches (prev→cur) once and reuse them for fallback/debug views.
        _, prev_cur_matches = _match_with_ransac(
            args, kp_prev, kp_cur, des_prev, des_cur, matcher
        )

        # --------------------------------------------------------------------------- #
        # ------------------------ Bootstrap (delayed two-view) ------------------------------------------------
        # --------------------------------------------------------------------------- #
        if not initialised:
            # 1) Seed a reference frame the first time we get here
            if not bs.has_ref:
                bs.seed(kp_prev, des_prev, img_prev, frame_id=frame_idx - 1)
                img_prev, kp_prev, des_prev = img_cur, kp_cur, des_cur
                continue

            # 2) Match reference ↔ current using your existing pipeline
            raw_matches_bs, matches_bs = _match_with_ransac(
                args, bs.kps_ref, kp_cur, bs.des_ref, des_cur, matcher
            )
            log.info("[Init] Matches ref→cur: raw=%d  ransac=%d  ransac_th=%.2fpx",
                     len(raw_matches_bs), len(matches_bs), float(args.ransac_thresh))

            # Optional: refresh reference if matches are weak or ref is stale
            if _refresh_ref_needed(matches_bs, min_matches=80, max_age=30, cur_id=frame_idx, ref_id=bs.frame_id_ref):
                bs.seed(kp_cur, des_cur, img_cur, frame_id=frame_idx)
                img_prev, kp_prev, des_prev = img_cur, kp_cur, des_cur
                continue

            # 3) Build point arrays and evaluate the pair
            pts_ref, pts_cur = pts_from_matches(bs.kps_ref, kp_cur, matches_bs)
            init_params = InitParams(
                ransac_px=float(args.ransac_thresh),
                min_posdepth=0.90,
                min_parallax_deg=0.5,
                score_ratio_H=0.45
            )
            # Decide with masks (no redundancy later)
            decision = evaluate_two_view_bootstrap_with_masks(K, pts_ref, pts_cur, init_params)
            if decision is None:
                log.info("[Init] Pair rejected → waiting for a better one.")
                img_prev, kp_prev, des_prev = img_cur, kp_cur, des_cur
                continue

            log.info(f"[Init] Accepted pair via {'H' if decision.pose.model.name=='HOMOGRAPHY' else 'F/E'}: "
                    f"posdepth={decision.pose.posdepth:.3f}, parallax={decision.pose.parallax_deg:.2f}°")

            # Build the initial map (KF0 = I, KF1 = [R|t]) once
            ok, T0_cw, T1_cw = bootstrap_two_view_map(
                K,
                bs.kps_ref, bs.des_ref,   # reference frame kps/desc
                kp_cur, des_cur,          # current frame kps/desc
                matches_bs,
                args,
                world_map,
                params=init_params,
                decision=decision         # pass decision to avoid recomputing/gating
            )
            if not ok:
                log.warning("[Init] bootstrap_two_view_map() failed — keep searching.")
                img_prev, kp_prev, des_prev = img_cur, kp_cur, des_cur
                continue
            
            # -------------BOOKKEEPING for Map and Keyframes---------------------
            world_map.add_pose(T0_cw, is_keyframe=True)   # KF0
            world_map.add_pose(T1_cw, is_keyframe=True)   # KF1

            # --- 2D Trajectory visualization bookkeeping ---
            traj2d.push(bs.frame_id_ref, T0_cw)
            traj2d.push(frame_idx, T1_cw)
            
            # --- Create the two initial Keyframes to mirror world_map KFs ---
            try:
                if len(kfs) != 0:
                    log.warning("[Init] Expected empty kfs at bootstrap, found len=%d. "
                                "Proceeding but indices may misalign.", len(kfs))

                kf0_idx = 0 if len(kfs) == 0 else len(kfs)
                kf1_idx = kf0_idx + 1

                # Resolve frame indices and image paths
                ref_fidx = bs.frame_id_ref
                cur_fidx = frame_idx
                ref_path = seq[ref_fidx] if isinstance(seq[ref_fidx], str) else ""
                cur_path = seq[cur_fidx] if isinstance(seq[cur_fidx], str) else ""

                # Build thumbnails (guard if you disabled UI)
                thumb0 = make_thumb(bs.img_ref, tuple(args.kf_thumb_hw)) if 'make_thumb' in globals() else None
                thumb1 = make_thumb(img_cur,    tuple(args.kf_thumb_hw)) if 'make_thumb' in globals() else None

                # Create KF0 = Identity pose
                kf0 = Keyframe(idx=kf0_idx, frame_idx=ref_fidx, path=ref_path,
                               kps=bs.kps_ref, desc=bs.des_ref, pose=T0_cw, thumb=thumb0)

                # Create KF1 = bootstrap pose
                kf1 = Keyframe(idx=kf1_idx, frame_idx=cur_fidx, path=cur_path,
                               kps=kp_cur, desc=des_cur, pose=T1_cw, thumb=thumb1)

                kfs.extend([kf0, kf1])
                last_kf_frame_no = cur_fidx   # the most recently added keyframe
                log.info("[Init] Inserted initial keyframes: KF0(frame=%d, idx=%d) & KF1(frame=%d, idx=%d).",
                        ref_fidx, kf0_idx, cur_fidx, kf1_idx)

            except Exception as e:
                log.exception("[Init] Failed to create initial keyframes: %s", e)

            # SET INITIALISED FLAG
            initialised = True
            print("-----BOOTSTRAPPED SUCCESSFULLY-----")
            
            if viz3d:
                # highlight everything once (all points are new at bootstrap)
                viz3d.update(world_map, new_ids=world_map.point_ids())

            Tcw_cur_pose = T1_cw.copy()   # current camera pose (optional)
            bs.clear()
            img_prev, kp_prev, des_prev = img_cur, kp_cur, des_cur
            continue
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----

        # ------------------------------------------------------------------- #
        # --------------------- Frame-to-Map Tracking ----------------------- #
        # ------------------------------------------------------------------- #
        # 1) Predict the current pose from the last two camera poses.
        if len(world_map.poses) >= 2:
            Tcw_prevprev = world_map.poses[-2]
            Tcw_prev     = world_map.poses[-1]
            Tcw_pred     = predict_pose_const_vel(Tcw_prevprev, Tcw_prev)
        else:
            Tcw_pred     = Tcw_cur_pose

        # 2) Reproject active map points and do small-window matching.
        H, W = img_cur.shape[:2]
        tracking_lost = False
        m23d = reproject_and_match_2d3d(
            world_map, K, Tcw_pred, kp_cur, des_cur, W, H,
            radius_px=float(args.proj_radius),  # CLI param
            max_hamm=64
        )

        log.debug(f"[Track] candidates for PnP: {len(m23d.pts3d)}")

        # 3) PnP+RANSAC
        if len(m23d.pts3d) >= int(args.pnp_min_inliers):
            Tcw_est, inl_mask = solve_pnp_ransac(
                m23d.pts3d, m23d.pts2d, K,
                ransac_px=float(args.ransac_thresh),
                Tcw_init=Tcw_pred,
                iters=300, conf=0.999
            )
            ninl = int(inl_mask.sum()) if inl_mask.size else 0
           # NOTE Could be converted to a function - because it is used again below in the tracking-lost case
            if Tcw_est is not None and ninl >= int(args.pnp_min_inliers):
                Tcw_cur_pose = Tcw_est
                world_map.add_pose(Tcw_cur_pose, is_keyframe=False)

                # --- 2D Trajectory visualization bookkeeping ---
                traj2d.push(frame_idx, Tcw_cur_pose)

                log.debug(f"[Track] PnP inliers={ninl}  (th={args.ransac_thresh:.1f}px)")

                if not headless:
                    try:
                        dbg = draw_reprojection_debug(img_cur, K, Tcw_cur_pose, m23d, inl_mask)
                        cv2.imshow("Track debug", dbg)
                    except Exception:
                        pass

            else:
                log.warning(f"[Track] PnP failed or too few inliers (got {ninl}). Tracking lost?")
                tracking_lost = True
        else:
            log.warning(f"[Track] Too few 2D–3D matches for PnP ({len(m23d.pts3d)} < {args.pnp_min_inliers}).")
            tracking_lost = True

        # (Optional) if tracking_lost: trigger relocalization here in the future.
        if tracking_lost:                      # fallback to 2-D-2-D if PnP failed
            log.info("[Track] PnP failed, fallback to 2D-2D tracking.")
            fallback_inliers = 0
            # 1) Frame-to-frame matching (prev frame -> current).
            tracking_matches = prev_cur_matches

            if len(tracking_matches) >= 5:
                pts0 = np.float32([kp_prev[m.queryIdx].pt for m in tracking_matches])
                pts1 = np.float32([kp_cur[m.trainIdx].pt  for m in tracking_matches])

                E, inlE = cv2.findEssentialMat(pts0, pts1, K, cv2.RANSAC, 0.999, args.ransac_thresh)
                if E is not None and inlE is not None and int(inlE.sum()) >= 5:
                    fallback_inliers, R, t, inlPose = cv2.recoverPose(E, pts0, pts1, K, mask=inlE)
                    fallback_inliers = int(fallback_inliers)
                    log.debug(
                        "[Track] Fallback 2D-2D matches: raw=%d  inlE=%d  inl=%d",
                        len(tracking_matches),
                        int(inlE.sum()) if inlE is not None else 0,
                        fallback_inliers,
                    )
                    if fallback_inliers >= 5:
                        T_rel = _pose_rt_to_homogenous(R, t)   # c₁ → c₂
                        # Scale fallback translation by expected magnitude from
                        # constant velocity — recoverPose returns unit-norm t
                        # which injects huge fake translations in rotation-dominant
                        # motion (e.g. TUM-RGBD indoor scenes).
                        if len(world_map.poses) >= 2:
                            T_rp = world_map.poses[-1] @ _pose_inverse(world_map.poses[-2])
                            expected_t = np.linalg.norm(T_rp[:3, 3])
                            T_rel[:3, 3] = t.ravel() * expected_t
                        else:
                            T_rel[:3, 3] = 0.0
                        # 2) Compose the relative motion with the last world pose.
                        Tcw_cur_pose = T_rel @ world_map.poses[-1]
                        world_map.add_pose(Tcw_cur_pose, is_keyframe=False)
                        traj2d.push(frame_idx, Tcw_cur_pose)
                        log.debug("[Track] Pushed fallback pose, inliers=%d", fallback_inliers)
                        tracking_lost = False

            if tracking_lost:
                log.warning("[Track] Fallback 2D-2D tracking failed.")
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----
        
        
        # ------------------------------------------------------------------- #
        # ---------------------     Keyframe Selection  --------------------- #
        # ------------------------------------------------------------------- #
        # Compare the current frame against the latest KF for keyframe insertion.
        prev_len = len(kfs)
        kfs, last_kf_frame_no = select_keyframe(
            args, seq, frame_idx - 1, img_cur, kp_cur, des_cur, Tcw_cur_pose, matcher, kfs, last_kf_frame_no
        )
        is_kf = (len(kfs) > prev_len)
        if is_kf:
            log.debug(f"[KF] Added KF #{kfs[-1].idx} @frame={kfs[-1].frame_idx} (total KFs={len(kfs)})")
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----

        # ------------------------------------------------------------------- #
        # --------------------- Map Growth (Triangulation) --------------------- #
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

                # Run local BA only when the new keyframe contributed enough landmark growth.
                if len(new_ids) >= int(getattr(args, "local_ba_min_new_points", 0)):
                    try:
                        local_bundle_adjustment(
                            world_map, K, kfs,
                            center_kf_idx=curr_kf.idx,
                            window_size=int(getattr(args, "local_ba_window", 6)),
                            max_points=int(getattr(args, "local_ba_max_points", 5000)),
                            max_iters=int(getattr(args, "local_ba_max_iters", 12))
                        )
                    except Exception as e:
                        log.warning(f"[BA] Local BA failed: {e}")
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----


        # --------------------------------------MISCELLANEOUS INFO ---------------------------------------------------
        # print(f"[traj2d] est={len(traj2d.est_xyz)} gt={len(traj2d.gt_xyz)}")
        # print(f"Frame {frame_idx}/{total_frames - 1}  |  KFs: {len(kfs)}  |  Map points: {len(world_map.points)}")
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----

        # ------------------------------------------------------------------- #
        # --------------------- Global Bundle Adjustment ----------------------- #
        # ------------------------------------------------------------------- #

        gba_every = int(getattr(args, "gba_every", 0))
        if gba_every > 0 and len(kfs) > 0 and len(kfs) % gba_every == 0 and last_gba_kf_count != len(kfs):
            do_gba = True
        
        if do_gba:
            try:
                pass
                # log.info(f"[Global BA] Running global BA on {len(kfs)} KFs and {len(world_map.points)} points.")
                # global_bundle_adjustment(
                #     world_map, K, kfs,
                #     fix_first=bool(int(args.gba_fix_first)),
                #     max_points=(None if int(args.gba_max_points) <= 0 else int(args.gba_max_points)),
                #     max_iters=int(args.gba_max_iters)
                # )
                do_gba = False
                last_gba_kf_count = len(kfs)
            except Exception as e:
                log.warning(f"[BA] Global BA failed: {e}")
            
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----

        
        # ------------------------------------------------------------------- #
        # --------------------- Visualization ----------------------- #
        # ------------------------------------------------------------------- #
        if not headless:
            if viz3d:
                viz3d.update(world_map, new_ids=new_ids)

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
                    cur_tile = _thumb(img_cur, (W_thumb, H_thumb))
                    cv2.putText(cur_tile, "cur frame", (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    # Last 3 KFs — their .thumb is LZ4-compressed JPEG bytes
                    recent_kfs = kfs[-1:-4:-1] if len(kfs) > 0 else []
                    tiles = [cur_tile]
                    for kf in recent_kfs:
                        th = None
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

                    # Pad to 4 tiles total if <3 KFs exist
                    while len(tiles) < 4:
                        tiles.append(np.zeros((H_thumb, W_thumb, 3), dtype=np.uint8))

                    strip = cv2.hconcat([_im_from_any(t) for t in tiles])
                    cv2.imshow("Strip: img2 + last 3 KFs", strip)

                    # --- img2 with ONLY features matched to previous frame (img1) ---
                    feat_vis = _im_from_any(img_cur)
                    if feat_vis is None:
                        feat_vis = np.zeros((H_thumb, W_thumb, 3), dtype=np.uint8)

                    # Reuse the cached frame-to-frame matches computed at the top of the loop.
                    for m in prev_cur_matches:
                        x, y = map(int, kp_cur[m.trainIdx].pt)
                        cv2.circle(feat_vis, (x, y), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)

                    cv2.putText(feat_vis, f"prev→cur matches: {len(prev_cur_matches)}", (10, 24),
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
                # if user pressed 'n', did_step=True -> allow this iteration to exit;
                # next iteration will immediately pause again (nice for stepping).

        img_prev, kp_prev, des_prev = img_cur, kp_cur, des_cur
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----

    if viz3d:
        viz3d.close()

    # --- Final trajectory graph (always drawn; saved to file in headless mode) ---
    import matplotlib.pyplot as plt
    traj2d.draw()
    out_path = f"trajectory_{args.dataset}.png"
    traj2d.fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("[Done] Trajectory saved to %s", out_path)
    if headless:
        plt.close(traj2d.fig)
    else:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
