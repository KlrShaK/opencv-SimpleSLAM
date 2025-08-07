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
import cv2
import lz4.frame
import numpy as np
from tqdm import tqdm
from typing import List, Sequence, Tuple

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

from slam.core.visualization_utils import draw_tracks, Visualizer3D, TrajectoryPlotter
from slam.core.trajectory_utils import compute_gt_alignment, apply_alignment
from slam.core.landmark_utils import Map
from slam.core.triangulation_utils import update_and_prune_tracks, MultiViewTriangulator, triangulate_points
from slam.core.pnp_utils import associate_landmarks, refine_pose_pnp
from slam.core.ba_utils import (
    two_view_ba,
    pose_only_ba,
    local_bundle_adjustment
)

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
    p.add_argument('--matcher', choices=['bf'], default='bf')
    p.add_argument('--use_lightglue', action='store_true')
    p.add_argument('--min_conf', type=float, default=0.7,
                   help='Minimum LightGlue confidence for a match')
    # runtime
    p.add_argument('--fps', type=float, default=10)
    # RANSAC
    p.add_argument('--ransac_thresh', type=float, default=1.0)
    # key-frame params
    p.add_argument('--kf_max_disp', type=float, default=45)
    p.add_argument('--kf_min_inliers', type=float, default=150)
    p.add_argument('--kf_cooldown', type=int, default=5)
    p.add_argument('--kf_thumb_hw', type=int, nargs=2,
                   default=[640, 360])
    
    # 3‑D visualisation toggle
    p.add_argument("--no_viz3d", action="store_true", help="Disable 3‑D visualization window")
    # triangulation depth filtering
    p.add_argument("--min_depth", type=float, default=0.60)
    p.add_argument("--max_depth", type=float, default=50.0)
    p.add_argument('--mvt_rep_err', type=float, default=30.0,
               help='Max mean reprojection error (px) for multi-view triangulation')

    #  PnP / map-maintenance
    p.add_argument('--pnp_min_inliers', type=int, default=30)
    p.add_argument('--proj_radius',     type=float, default=3.0)
    p.add_argument('--merge_radius',    type=float, default=0.10)

    # Bundle Adjustment
    p.add_argument('--local_ba_window', type=int, default=5, help='Window size (number of keyframes) for local BA')

    return p


# --------------------------------------------------------------------------- #
#  Bootstrap initialisation
# --------------------------------------------------------------------------- #
def try_bootstrap(K, kp0, descs0, kp1, descs1, matches, args, world_map):
    """Return (success, T_cam0_w, T_cam1_w) and add initial landmarks."""
    if len(matches) < 50:
        return False, None, None

    # 1. pick a model: Essential (general) *or* Homography (planar)
    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])

    E, inl_E = cv2.findEssentialMat(
        pts0, pts1, K, cv2.RANSAC, 0.999, args.ransac_thresh)

    H, inl_H = cv2.findHomography(
        pts0, pts1, cv2.RANSAC, args.ransac_thresh)

    # score = #inliers (ORB-SLAM uses a more elaborate model-selection score)
    use_E = (inl_E.sum() > inl_H.sum())

    if use_E and E is None:
        return False, None, None
    if not use_E and H is None:
        return False, None, None

    if use_E:
        _, R, t, inl_pose = cv2.recoverPose(E, pts0, pts1, K)
        mask = (inl_E.ravel() & inl_pose.ravel()).astype(bool)
    else:
        print("[BOOTSTRAP] Using Homography for initialisation")
        R, t = cv2.decomposeHomographyMat(H, K)[1:3]  # take the best hypothesis
        mask = inl_H.ravel().astype(bool)

    # 2. triangulate those inliers – exactly once
    p0 = pts0[mask]
    p1 = pts1[mask]
    pts3d = triangulate_points(K, R, t, p0, p1)

    z0 = pts3d[:, 2]
    pts3d_cam1 = (R @ pts3d.T + t.reshape(3, 1)).T
    z1 = pts3d_cam1[:, 2]

    # both cameras see the point in front of them
    ok = ( (z0 > args.min_depth) & (z0 < args.max_depth) &     # in front of cam‑0
    (z1 > args.min_depth) & (z1 < args.max_depth) )       # in front of cam‑1)
    pts3d = pts3d[ok]
    print(f"[BOOTSTRAP] Triangulated {len(pts3d)} points. Status: {ok.sum()} inliers")

    if len(pts3d) < 80:
        print("[BOOTSTRAP] Not enough points to bootstrap the map.")
        return False, None, None

    T1_cw = _pose_rt_to_homogenous(R, t) # camera-from-world
    T1_wc = _pose_inverse(T1_cw)    # world-from-camera
    # 3. fill the map
    world_map.add_pose(T1_wc, is_keyframe=True)  # Keyframe because we only bootstrap on keyframes

    cols = np.full((len(pts3d), 3), 0.7)   # grey – colour is optional here
    ids = world_map.add_points(pts3d, cols, keyframe_idx=0) #TODO 0 or 1

    # -----------------------------------------------
    # add (frame_idx , kp_idx) pairs for each new MP
    # -----------------------------------------------
    inlier_kp_idx = np.where(mask)[0][ok]   # kp indices that survived depth
    for pid, kp_idx in zip(ids, inlier_kp_idx):
        world_map.points[pid].add_observation(0, kp_idx, descs0[kp_idx])   # img0 side
        world_map.points[pid].add_observation(1, kp_idx, descs1[kp_idx])   # img1 side


    print(f"[BOOTSTRAP] Map initialised with {len(ids)} landmarks.")
    return True, T1_wc


# --------------------------------------------------------------------------- #
#  Continuous pose tracking (PnP)
# --------------------------------------------------------------------------- #
def visualize_pnp_reprojection(img_bgr, K, T_wc, pts3d_w, pts2d_px, inlier_mask=None,
                               win_name="PnP debug", thickness=2):
    """
    Draw projected 3‑D landmarks (from world) on top of the current image and connect
    them to the actual detected keypoints.

    img_bgr     : current image (BGR)
    K           : 3x3 intrinsics
    T_wc        : 4x4 pose (cam->world)
    pts3d_w     : (N,3) 3D points in *world* coords
    pts2d_px    : (N,2) measured pixel locations that took part in PnP
    inlier_mask : optional boolean array (len=N). If given, only inliers get lines.
    """
    import cv2
    import numpy as np

    img = img_bgr.copy()
    # world -> cam
    T_cw = np.linalg.inv(T_wc)
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    rvec, _ = cv2.Rodrigues(R_cw)
    tvec = t_cw.reshape(3, 1)

    proj, _ = cv2.projectPoints(pts3d_w.astype(np.float32), rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)

    if inlier_mask is None:
        inlier_mask = np.ones(len(proj), dtype=bool)

    for (u_meas, v_meas), (u_proj, v_proj), ok in zip(pts2d_px, proj, inlier_mask):
        color = (0, 255, 0) if ok else (0, 0, 255)  # green for inlier, red for outlier
        cv2.circle(img, (int(round(u_proj)), int(round(v_proj))), 4, (255, 0, 0), -1)    # projected (blue)
        cv2.circle(img, (int(round(u_meas)), int(round(v_meas))), 4, color, -1)          # measured
        if ok:
            cv2.line(img, (int(round(u_meas)), int(round(v_meas))),
                          (int(round(u_proj)), int(round(v_proj))), color, thickness)

    cv2.imshow(win_name, img)
    cv2.waitKey(1)
    return img

# pnp_utils.py  (NEW)

# ------------ parameters (mimic ORB-SLAM) -------------
GRID_ROWS = 48          # 640×480 → ≈13 px cells
GRID_COLS = 64
MIN_VIEW_COS = 0.5      # landmark normal · viewing dir  (optional)

# ------------------------------------------------------ #
#  Image-space grid holding indices of key-points
# ------------------------------------------------------ #
def _build_grid(kps: Sequence[cv2.KeyPoint],
                img_h: int, img_w: int
) -> Tuple[List[List[List[int]]], float, float]:
    cell_w = img_w / GRID_COLS
    cell_h = img_h / GRID_ROWS
    grid = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    for idx, kp in enumerate(kps):
        c, r = int(kp.pt[0] / cell_w), int(kp.pt[1] / cell_h)
        if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
            grid[r][c].append(idx)
    return grid, cell_w, cell_h

# pnp_utils.py  (REPLACES associate_landmarks)
from scipy.spatial.transform import Rotation as Rot

def associate_landmarks(
    K:          np.ndarray,
    pose_w_c:   np.ndarray,        # camera-to-world 4×4
    pts_w:      np.ndarray,        # N×3  map points
    kps:        Sequence[cv2.KeyPoint],
    img_size:   Tuple[int, int],
    search_rad: float = 8.0,
    normals_w:  np.ndarray | None = None,     # optional (N×3)
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    ORB-SLAM-like projection search with a 2-D grid instead of brute-force.

    1) *Frustum cull* + optional view-angle check.
    2) Project surviving landmarks (vectorised).
    3) Use the feature grid to fetch *candidate* key-points in O(1).
    4) Choose the nearest candidate within `search_rad` px.

    Returns 3-D/2-D correspondences **without duplicates**.
    """
    if pts_w.size == 0 or not kps:
        return np.empty((0, 3)), np.empty((0, 2)), []

    img_h, img_w = img_size
    grid, cell_w, cell_h = _build_grid(kps, img_h, img_w)
    kp_xy  = np.float32([kp.pt for kp in kps])           # K×2

    # -------------------------------------------------- #
    # 1. visibility + view-angle filter  (vectorised)
    # -------------------------------------------------- #
    T_c_w  = np.linalg.inv(pose_w_c)
    R_c_w  = T_c_w[:3, :3]
    t_c_w  = T_c_w[:3, 3]
    pts_c  = (R_c_w @ pts_w.T + t_c_w[:, None]).T        # world → camera
    z = pts_c[:, 2]
    visible = (z > 0.1)                                   # in front of cam
    if normals_w is not None:
        view_cos = np.einsum('ij,ij->i', normals_w, pts_w - pose_w_c[:3, 3])
        visible &= (view_cos / (np.linalg.norm(normals_w, axis=1) *
                                 np.linalg.norm(pts_w - pose_w_c[:3, 3], axis=1)) > MIN_VIEW_COS)

    pts_w = pts_w[visible]
    pts_c = pts_c[visible]
    if pts_w.size == 0:
        return np.empty((0, 3)), np.empty((0, 2)), []

    # -------------------------------------------------- #
    # 2. project (vectorised)
    # -------------------------------------------------- #
    proj = (K @ pts_c.T).T
    proj = (proj[:, :2] / proj[:, 2:]).astype(np.float32)   # N_vis×2

    # restrict to image rectangle with a margin
    inside = (proj[:, 0] >= 0) & (proj[:, 0] < img_w) & \
             (proj[:, 1] >= 0) & (proj[:, 1] < img_h)
    proj   = proj[inside]
    pts_w  = pts_w[inside]

    # -------------------------------------------------- #
    # 3. grid lookup
    # -------------------------------------------------- #
    used_kp: set[int] = set()
    out3d, out2d, out_kidx = [], [], []
    r2 = search_rad ** 2
    for pw, (u, v) in zip(pts_w, proj):
        # which grid cells intersect the search circle?
        min_c = int(max((u - search_rad) / cell_w, 0))
        max_c = int(min((u + search_rad) / cell_w, GRID_COLS - 1))
        min_r = int(max((v - search_rad) / cell_h, 0))
        max_r = int(min((v + search_rad) / cell_h, GRID_ROWS - 1))

        cand: list[int] = []
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                cand.extend(grid[r][c])
        if not cand:
            continue

        cand = np.array(cand, dtype=int)
        # squared distances only on *candidates*
        d2 = np.sum((kp_xy[cand] - (u, v)) ** 2, axis=1)
        idx = cand[np.argmin(d2)]
        if d2.min() < r2 and idx not in used_kp:
            out3d.append(pw)
            out2d.append(kp_xy[idx])
            out_kidx.append(idx)
            used_kp.add(idx)

    if not out3d:
        return np.empty((0, 3)), np.empty((0, 2)), []

    return np.float32(out3d), np.float32(out2d), out_kidx


# --------------------------------------------------------------------------- #
#  Continuous pose tracking (PnP) – FAST version
# --------------------------------------------------------------------------- #
def track_with_pnp(K,
                   kp_prev, kp_cur, desc_prev, desc_cur, matches, img2,
                   frame_no,                  # 1-based index of *current* frame
                   Twc_prev,                  # pose for frame –1  (4×4)
                   world_map, args):
    """
    Estimate the camera-to-world pose T_wc for *frame_no* using Perspective-n-Point
    and already-triangulated landmarks.

    Returns
    -------
    ok           : bool
    Twc_cur      : (4,4) np.ndarray – pose for the current frame
    used_cur_idx : set[int]         – indices of key-points on *kp_cur* that
                                      participated in the inlier PnP solution.
    """
    import cv2
    import numpy as np
    from slam.core.pose_utils import _pose_rt_to_homogenous, _pose_inverse

    # ------------------------------------------------------------------ #
    # 1. Build 3-D ↔ 2-D correspondences via projection-grid search
    # ------------------------------------------------------------------ #
    pts_w_all = world_map.get_point_array()          # (N,3)
    pid_all   = world_map.point_ids()                # list length N
    if pts_w_all.size == 0:
        return False, None, set()

    img_h, img_w = img2.shape[:2]
    GRID_ROWS, GRID_COLS = 48, 64                    # ≈13 px cells @ 640×480
    cell_w, cell_h = img_w / GRID_COLS, img_h / GRID_ROWS

    grid = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    for idx, kp in enumerate(kp_cur):
        c, r = int(kp.pt[0] / cell_w), int(kp.pt[1] / cell_h)
        if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
            grid[r][c].append(idx)

    # ---- project all landmarks with *predicted* pose ------------------ #
    Twc_pred = Twc_prev                                   # constant-motion prior
    T_cw = np.linalg.inv(Twc_pred)
    P = K @ T_cw[:3, :4]                                  # 3×4
    pts_h = np.hstack([pts_w_all, np.ones((len(pts_w_all), 1))]).T
    uvw = P @ pts_h                                       # 3×N (homogeneous)
    proj = (uvw[:2] / uvw[2]).T.astype(np.float32)        # N×2  (u,v)

    obj_pts, img_pts = [], []
    obj_pids, kp_cur_ids = [], []
    used_kp = set()
    r2 = args.proj_radius ** 2

    for i, (u, v) in enumerate(proj):
        z = uvw[2, i]
        if z <= 0:                                        # behind camera
            continue
        if not (0 <= u < img_w and 0 <= v < img_h):       # outside frame
            continue

        # which grid cells intersect the search circle?
        min_c = int(max((u - args.proj_radius) / cell_w, 0))
        max_c = int(min((u + args.proj_radius) / cell_w, GRID_COLS - 1))
        min_r = int(max((v - args.proj_radius) / cell_h, 0))
        max_r = int(min((v + args.proj_radius) / cell_h, GRID_ROWS - 1))

        cand = []
        for rr in range(min_r, max_r + 1):
            for cc in range(min_c, max_c + 1):
                cand.extend(grid[rr][cc])
        if not cand:
            continue

        kp_xy = np.float32([kp_cur[j].pt for j in cand])
        d2 = np.sum((kp_xy - (u, v)) ** 2, axis=1)
        j_local = int(np.argmin(d2))
        if d2[j_local] < r2:
            kp_idx = cand[j_local]
            if kp_idx in used_kp:
                continue
            obj_pts.append(pts_w_all[i])
            img_pts.append(kp_xy[j_local])
            obj_pids.append(pid_all[i])
            kp_cur_ids.append(kp_idx)
            used_kp.add(kp_idx)

    if len(obj_pts) < args.pnp_min_inliers:
        print(f"[PNP]  Not enough 3-D/2-D pairs ({len(obj_pts)}).")
        return False, None, set()

    obj_pts = np.asarray(obj_pts, np.float32)
    img_pts = np.asarray(img_pts, np.float32)

    # ------------------------------------------------------------------ #
    # 2. Robust PnP  (EPnP + RANSAC → iterative refinement)
    # ------------------------------------------------------------------ #
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj_pts, img_pts, K, None,
        iterationsCount=100,
        reprojectionError=args.ransac_thresh,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)

    if not ok or inl is None or len(inl) < args.pnp_min_inliers:
        print(f"[PNP]  RANSAC failed – {0 if inl is None else len(inl)} inliers")
        return False, None, set()

    # optional refinement on inliers
    inl = inl.reshape(-1)
    cv2.solvePnP(obj_pts[inl], img_pts[inl], K, None,
                 rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)

    # camera→world pose
    R, _  = cv2.Rodrigues(rvec)
    T_cw  = _pose_rt_to_homogenous(R, tvec)
    Twc_cur = _pose_inverse(T_cw)

    # ------------------------------------------------------------------ #
    # 3. Map maintenance – add fresh observations
    # ------------------------------------------------------------------ #
    for arr_idx, pid in enumerate(obj_pids):
        if arr_idx in inl:                                 # inlier only
            kp_idx = kp_cur_ids[arr_idx]
            world_map.points[pid].add_observation(frame_no,
                                                   kp_idx,
                                                   desc_cur[kp_idx])

    used_cur_idx = {kp_cur_ids[i] for i in inl}
    print(f"[PNP]  Pose @ frame {frame_no} refined with {len(inl)} inliers")

    # ------------------------------------------------------------------ #
    # 4. Visual debug (unchanged)
    # ------------------------------------------------------------------ #
    visualize_pnp_reprojection(
        img2, K, Twc_cur,
        obj_pts[inl], img_pts[inl],
        inlier_mask=np.ones(len(inl), dtype=bool),
        win_name="PnP reprojection")

    return True, Twc_cur, used_cur_idx


# --------------------------------------------------------------------------- #
#  Main processing loop
# --------------------------------------------------------------------------- #
def main():
    PAUSED = False
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
        gt_T[:, 3, 3] = 1.0                             # homogeneous 1s
        R_align, t_align = compute_gt_alignment(gt_T)

    # --- feature pipeline (OpenCV / LightGlue) ---
    detector, matcher = init_feature_pipeline(args)

    mvt = MultiViewTriangulator(
        K,
        min_views=2,                             # ← “every 3 key-frames”
        merge_radius=args.merge_radius,
        max_rep_err=args.mvt_rep_err,
        min_depth=args.min_depth,
        max_depth=args.max_depth)
    
    # --- tracking state ---
    prev_map, tracks = {}, {}
    next_track_id = 0
    initialised = False
    tracking_lost = False


    world_map = Map()
    Twc_cur_pose = np.eye(4)  # camera‑to‑world (identity at t=0)
    world_map.add_pose(Twc_cur_pose, is_keyframe=True)  # initial pose
    viz3d = None if args.no_viz3d else Visualizer3D(color_axis="y")
    plot2d = TrajectoryPlotter()           

    kfs: list[Keyframe] = []
    last_kf_frame_no = -999

    # TODO FOR BUNDLE ADJUSTMENT
    frame_keypoints: List[List[cv2.KeyPoint]] = []  #CHANGE
    frame_keypoints.append([])   # placeholder to keep indices aligned

    # --- visualisation ---
    achieved_fps = 0.0
    last_time = cv2.getTickCount() / cv2.getTickFrequency()

    # cv2.namedWindow('Feature Tracking', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Feature Tracking', 1200, 600)

    total = len(seq) - 1
    


    new_ids:  list[int] = [] # CHANGE

    for i in tqdm(range(total), desc='Tracking'):
        # --- load image pair ---
        img1, img2 = load_frame_pair(args, seq, i)

        # --- feature extraction / matching ---
        kp1, des1 = feature_extractor(args, img1, detector)
        kp2, des2 = feature_extractor(args, img2, detector)
        matches = feature_matcher(args, kp1, kp2, des1, des2, matcher)
        # --- filter matches with RANSAC ---
        matches = filter_matches_ransac(kp1, kp2, matches, args.ransac_thresh)

        if len(matches) < 12:
            print(f"[WARN] Not enough matches at frame {i}. Skipping.")
            continue
        
        # ---------------- 2D - Feature Tracking --------------------------
        frame_no = i + 1
        curr_map, tracks, next_track_id = update_and_prune_tracks(
                matches, prev_map, tracks, kp2, frame_no, next_track_id)
        prev_map = curr_map                       # for the next iteration

        # ---------------- Key-frame decision -------------------------- # TODO make every frame a keyframe
        
        if i == 0:
            kfs.append(Keyframe(idx=0, frame_idx=1, path=seq[0] if isinstance(seq[0], str) else "",
                        kps=kp1, desc=des1, pose=Twc_cur_pose, thumb=make_thumb(img1, tuple(args.kf_thumb_hw))))
            last_kf_frame_no = kfs[-1].frame_idx
            prev_len = len(kfs)
            is_kf = False
            continue
        else:
            prev_len = len(kfs)
            kfs, last_kf_frame_no = select_keyframe(
                args, seq, i, img2, kp2, des2, Twc_cur_pose, matcher, kfs, last_kf_frame_no)
            is_kf = len(kfs) > prev_len
        
        if is_kf:
            frame_keypoints.append(kp2.copy())
        # print("len(kfs) = ", len(kfs), "last_kf_frame_no = ", last_kf_frame_no)
        # ------------------------------------------------ bootstrap ------------------------------------------------ # TODO FIND A BETTER WAY TO manage index
        if not initialised:
            if len(kfs) < 2:
                continue
            bootstrap_matches = feature_matcher(args, kfs[0].kps, kfs[-1].kps, kfs[0].desc, kfs[-1].desc, matcher)
            bootstrap_matches = filter_matches_ransac(kfs[0].kps, kfs[-1].kps, bootstrap_matches, args.ransac_thresh)
            ok, Twc_temp_pose = try_bootstrap(K, kfs[0].kps, kfs[0].desc, kfs[-1].kps, kfs[-1].desc, bootstrap_matches, args, world_map)
            if ok:
                frame_keypoints[0] = kfs[0].kps.copy()    # BA (img1 is frame-0)
                frame_keypoints[-1] = kfs[-1].kps.copy()        # BA (img2 is frame-1)
                # print("POSES: " ,world_map.poses)
                # two_view_ba(world_map, K, frame_keypoints, max_iters=25) # BA

                initialised = True
                Twc_cur_pose = world_map.poses[-1].copy()               # we are at frame i+1
                continue
            else:
                print("******************BOOTSTRAP FAILED**************")
                continue           # keep trying with next frame

        # ------------------------------------------------ tracking -------------------------------------------------
        Twc_pose_pred = Twc_cur_pose.copy()         
        # ok_pnp, Twc_cur_pose, used_idx = solve_pnp_step(
        #     K, Twc_pose_pred, world_map, kp2, args) # kp2 is the current frame keypoints
        ok_pnp, Twc_cur_pose, used_idx = track_with_pnp(K, kp1, kp2, des1, des2, matches,
                                                        frame_no=i + 1, img2=img2,
                                                        Twc_prev=Twc_cur_pose,        # pose from the *previous* iteration
                                                        world_map=world_map,
                                                        args=args)
        

        if not ok_pnp:                      # fallback to 2-D-2-D if PnP failed
            print(f"[WARN] PnP failed at frame {i}. Using 2D-2D tracking.")
            # raise  Exception(f"[WARN] PnP failed at frame {i}. Using 2D-2D tracking.")
            if not is_kf:
                last_kf = kfs[-1]
            else:
                last_kf = kfs[-2] if len(kfs) > 1 else kfs[0]

            tracking_matches = feature_matcher(args, last_kf.kps, kp2, last_kf.desc, des2, matcher)
            tracking_matches = filter_matches_ransac(last_kf.kps, kp2, tracking_matches, args.ransac_thresh)

            pts0 = np.float32([last_kf.kps[m.queryIdx].pt for m in tracking_matches])
            pts1 = np.float32([kp2[m.trainIdx].pt  for m in tracking_matches])
            E, mask = cv2.findEssentialMat(pts0, pts1, K, cv2.RANSAC,
                                        0.999, args.ransac_thresh)
            if E is None:
                tracking_lost = True
                continue
            _, R, t, mpose = cv2.recoverPose(E, pts0, pts1, K)
            T_rel = _pose_rt_to_homogenous(R, t)   # c₁ → c₂
            Twc_cur_pose = last_kf.pose @ np.linalg.inv(T_rel)   # c₂ → world
            tracking_lost = False

        if is_kf:
            world_map.add_pose(Twc_cur_pose, is_keyframe=is_kf)        # always push *some* pose

        # pose_only_ba(world_map, K, frame_keypoints,    # FOR BA
            #  frame_idx=len(world_map.poses)-1)

        # ------------------------------------------------ map growth ------------------------------------------------
        if  is_kf:
            # 1) hand the new KF to the multi-view triangulator
            kf_pose_idx = len(world_map.poses) - 1       # this is the new pose’s index
            mvt.add_keyframe(
                frame_idx=kf_pose_idx,            # global frame number of this KF
                Twc_pose=Twc_cur_pose,
                kps=kp2,
                track_map=curr_map,
                img_bgr=img2,
                descriptors=des2)               # new key-frame

            # 2) try triangulating all tracks that now have ≥3 distinct KFs
            new_mvt_ids = mvt.triangulate_ready_tracks(world_map)

            # 3) visualisation hook
            new_ids = new_mvt_ids                # keeps 3-D viewer in sync

        # ------------------------------------------------ Local Bundle Adjustment ------------------------------------------------
        # if is_kf and (len(kfs) % args.local_ba_window == 0): # or len(world_map.keyframe_indices) > args.local_ba_window
        #     pose_prev = Twc_cur_pose.copy()
        #     center_kf_idx = kfs[-1].idx
        #     print(f"[BA] Running local BA around key-frame {center_kf_idx} (window size = {args.local_ba_window}) , current = {len(world_map.poses) - 1}")
        #     print(f'len keyframes = {len(kfs)}, len frame_keypoints = {len(frame_keypoints)}, len poses = {len(world_map.poses)}')
        #     # print(f"world_map.poses = {len(world_map.poses)}, \n raw: {world_map.poses} \n keyframe_indices= {len(world_map.keyframe_indices)},\n raw: {world_map.keyframe_indices}")
        #     local_bundle_adjustment(
        #         world_map, K, frame_keypoints,
        #         center_kf_idx=len(world_map.poses) - 1,
        #         window_size=args.local_ba_window)

        # p = Twc_cur_pose[:3, 3]
        # p_gt = gt_T[i + 1, :3, 3]
        # print(f"Cam position z = {p}, GT = {p_gt}  (should decrease on KITTI)")

        # --- 2-D path plot (cheap) ----------------------------------------------
        est_pos = Twc_cur_pose[:3, 3]
        gt_pos  = None
        if gt_T is not None and i + 1 < len(gt_T):
            p_gt = gt_T[i + 1, :3, 3]                     # raw GT
            gt_pos = apply_alignment(p_gt, R_align, t_align)
        plot2d.append(est_pos, gt_pos, mirror_x=True)


        # # --- 2-D track maintenance (for GUI only) ---
        # frame_no = i + 1
        # prev_map, tracks, next_track_id = update_and_prune_tracks(
        #     matches, prev_map, tracks, kp2, frame_no, next_track_id)


        # --- 3-D visualisation ---
        if viz3d is not None:
            viz3d.update(world_map, new_ids)


        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('p') and viz3d is not None:
            viz3d.paused = not viz3d.paused


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()