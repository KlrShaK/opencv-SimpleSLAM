# -*- coding: utf-8 -*-
"""
ba_utils.py
===========

ALWAYS PASS POSE as T_wc (world→camera) (camera-from-world) pose convention to pyceres.

Light-weight Bundle-Adjustment helpers built on **pyceres**.
Implements
  • two_view_ba(...)
  • pose_only_ba(...)
  • local_bundle_adjustment(...)
and keeps the older run_bundle_adjustment (full BA) for
back-compatibility.

A *blue-print* for global_bundle_adjustment is included but not wired.
"""
from __future__ import annotations
import cv2
import numpy as np
import pyceres
from pycolmap import cost_functions, CameraModelId
import math
from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------------- #
#  Small pose ⇄ parameter converters
# --------------------------------------------------------------------- #

def project_to_SO3(M):
    """
    Project a near-rotation 3x3 matrix M onto SO(3) via SVD.
    Returns the closest rotation (Frobenius norm).
    """
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

# TODO: USES camera-to-world pose convention T_wc (camera→world) because of PyCERES, REMEMBER TO CONVERT, use below functions
def invert_homogeneous_scipy(T, validate=True):
    """
    Invert a 4x4 homogeneous rigid transform using SciPy.

    Parameters
    ----------
    T : array-like (4,4)
        Homogeneous transform [[R, t],[0,0,0,1]].
    validate : bool
        If True, re-project rotation onto SO(3) via SciPy (robust to mild drift).

    Returns
    -------
    T_inv : (4,4) ndarray
        Inverse transform.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4,4):
        raise ValueError("T must be 4x4.")
    Rmat = T[:3,:3]
    t    = T[:3, 3]

    if validate:
        # Produces closest rotation in Frobenius norm
        # Re-projects matrix to ensure perfect orthonormality
        Rmat = project_to_SO3(Rmat)

    R_inv = Rmat.T
    t_inv = -R_inv @ t

    T_inv = np.eye(4)
    T_inv[:3,:3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def _pose_to_quat_trans(T, ordering="xyzw"): # pyceres uses maybe (xyzw), optionally you could use "wxyz" 
    """
    Convert a camera-from-world pose matrix T_cw (4x4) into (quat_cw, t_cw)

    Parameters
    ----------
    T : (4,4) array-like
        Homogeneous transform.
    ordering : str
        "wxyz" (default) or "xyzw".
    Returns
    -------
    q : (4,) ndarray
        Quaternion in requested ordering, unit norm (w>=0 if wxyz).
    t : (3,) ndarray
        Translation vector.
    """
    T = np.asarray(T, dtype=float)
    assert T.shape == (4,4)
    Rmat = T[:3,:3]
    t = T[:3,3].copy()
    
    # Re-orthonormalize (optional but good hygiene)
    U, _, Vt = np.linalg.svd(Rmat)
    Rmat = U @ Vt
    if np.linalg.det(Rmat) < 0:     # enforce right-handed
        U[:, -1] *= -1
        Rmat = U @ Vt
    
    rot = R.from_matrix(Rmat)
    # SciPy gives quaternions in (x, y, z, w)
    q_xyzw = rot.as_quat()
    
    # Normalize (usually already unit)
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw)
    
    if ordering.lower() == "xyzw":
        q = q_xyzw
        # optional consistent sign: enforce w>=0
        if q[-1] < 0: q = -q
    else:  # wxyz
        w = q_xyzw[3]
        q = np.array([w, *q_xyzw[:3]])
        if q[0] < 0: q = -q
    return q, t

def _quat_trans_to_pose(q, t, ordering="xyzw"):
    q = np.asarray(q, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    if ordering.lower() == "wxyz":
        w, x, y, z = q
        q_xyzw = np.array([x, y, z, w])
    else:
        x, y, z, w = q
        q_xyzw = q
    # Normalize in case
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw)
    Rmat = R.from_quat(q_xyzw).as_matrix()
    T = np.eye(4)
    T[:3,:3] = Rmat
    T[:3,3] = t
    return T



# def _pose_to_quat_trans(T_cw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Convert a camera-from-world pose matrix T_cw (4x4) into (quat_cw, t_cw)
#     with X_c = R_cw * X_w + t_cw.

#     Returns
#     -------
#     quat_cw : np.ndarray shape (4,) (w,x,y,z)
#     t_cw    : np.ndarray shape (3,)
#     """
#     R_cw = T_cw[:3, :3]
#     t_cw = T_cw[:3, 3]

#     # axis-angle from rotation
#     aa, _ = cv2.Rodrigues(R_cw)
#     theta = np.linalg.norm(aa)
#     if theta < 1e-8:
#         quat = np.array([1.0, 0.0, 0.0, 0.0], np.float64)
#     else:
#         axis = aa.flatten() / theta
#         s = math.sin(theta / 2.0)
#         quat = np.array([math.cos(theta / 2.0), axis[0]*s, axis[1]*s, axis[2]*s], np.float64)
#     return quat, t_cw.copy()

# def _quat_trans_to_pose(quat_cw: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
#     """
#     Inverse of _pose_to_quat_trans: (quat_cw, t_cw) -> 4x4 T_cw
#     """
#     qw, qx, qy, qz = quat_cw
#     R_cw = np.array([
#         [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
#         [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
#         [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
#     ], dtype=np.float64)
#     T = np.eye(4, dtype=np.float64)
#     T[:3, :3] = R_cw
#     T[:3, 3] = t_cw
#     return T


# --------------------------------------------------------------------- #
#  Shared helper to add one reprojection residual
# --------------------------------------------------------------------- #
def _add_reproj_edge(problem, loss_fn,
                     kp_uv: tuple[float, float],
                     quat_param, trans_param,
                     point_param, intr_param):
    """Create a pyceres residual block for one observation."""
    cost = cost_functions.ReprojErrorCost(
        CameraModelId.PINHOLE,
        np.asarray(kp_uv, np.float64)
    )
    problem.add_residual_block(
        cost, loss_fn,
        [quat_param, trans_param, point_param, intr_param]
    )

# --------------------------------------------------------------------- #
#  1) Two-view BA  (bootstrap refinement)
# --------------------------------------------------------------------- #
def two_view_ba(world_map, K, keypoints, max_iters: int = 20):
    """
    Refine the two initial camera poses + all bootstrap landmarks.

    Assumes `world_map` has exactly *two* poses (T_0w, T_1w) and that
    each MapPoint already stores **two** observations (frame-0 and
    frame-1).  Called once right after initialisation.
    """
    assert len(world_map.poses) == 2, "two_view_ba expects exactly 2 poses"

    _core_ba(world_map, K, keypoints,
             opt_kf_idx=[0, 1],
             fix_kf_idx=[],
             max_iters=max_iters,
             info_tag="[2-view BA]")


# --------------------------------------------------------------------- #
#  2) Pose-only BA   (current frame refinement)
# --------------------------------------------------------------------- #
def pose_only_ba(world_map, K, keypoints,
                 frame_idx: int, max_iters: int = 8,
                 huber_thr: float = 2.0):
    """
    Optimise **only one pose** (SE3) while keeping all 3-D points fixed.
    Mimics ORB-SLAM's `Optimizer::PoseOptimization`.
    """
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    intr = np.array([fx, fy, cx, cy], np.float64)

    quat, trans = _pose_to_quat_trans(world_map.poses[frame_idx])

    problem = pyceres.Problem()
    problem.add_parameter_block(quat, 4)
    problem.add_parameter_block(trans, 3)
    problem.set_manifold(quat, pyceres.EigenQuaternionManifold())
    problem.add_parameter_block(intr, 4)
    problem.set_parameter_block_constant(intr)

    loss_fn = pyceres.HuberLoss(huber_thr)

    # for pid, mp in world_map.points.items():
    #     for f_idx, kp_idx in mp.observations:
    #         if f_idx != frame_idx:
    #             continue
    #         u, v = keypoints[f_idx][kp_idx].pt
    #         _add_reproj_edge(problem, loss_fn,
    #                          (u, v), quat, trans, mp.position, intr)
    for mp in world_map.points.values():
        problem.add_parameter_block(mp.position, 3)
        problem.set_parameter_block_constant(mp.position)
        for f_idx, kp_idx in mp.observations:
            if f_idx != frame_idx:
                continue
            u, v = keypoints[f_idx][kp_idx].pt
            _add_reproj_edge(problem, loss_fn,
                             (u, v), quat, trans, mp.position, intr)


    if problem.num_residual_blocks() < 10:
        print(f"POSE-ONLY BA skipped – not enough residuals")
        return  # too few observations

    opts = pyceres.SolverOptions()
    opts.max_num_iterations = max_iters
    summary = pyceres.SolverSummary()
    pyceres.solve(opts, problem, summary)

    world_map.poses[frame_idx][:] = _quat_trans_to_pose(quat, trans)
    # print(f"[Pose-only BA] iters={summary.iterations_used}"
    #       f"  inliers={problem.num_residual_blocks()}")
    print(f"[Pose-only BA] iters={summary.num_successful_steps}"
          f"  inliers={problem.num_residual_blocks()}")


# --------------------------------------------------------------------- #
#  3) Local BA  (sliding window)
# --------------------------------------------------------------------- #
def local_bundle_adjustment(world_map, K, keypoints,
                            center_kf_idx: int,
                            window_size: int = 8,
                            max_points  : int = 3000,
                            max_iters   : int = 15):
    """
    Optimise the *last* `window_size` key-frames around
    `center_kf_idx` plus all landmarks they observe.
    Older poses are kept fixed (gauge).
    """
    first_opt = max(0, center_kf_idx - window_size + 1)
    opt_kf    = list(range(first_opt, center_kf_idx + 1))
    fix_kf    = list(range(0, first_opt))

    _core_ba(world_map, K, keypoints,
             opt_kf_idx=opt_kf,
             fix_kf_idx=fix_kf,
             max_points=max_points,
             max_iters=max_iters,
             info_tag=f"[Local BA (kf {center_kf_idx})]")


# --------------------------------------------------------------------- #
#  4) Global BA  (blue-print only)
# --------------------------------------------------------------------- #
def global_bundle_adjustment_blueprint(world_map, K, keypoints):
    """
    *** NOT WIRED YET ***

    Outline:
      • opt_kf_idx = all key-frames
      • fix_kf_idx = []  (maybe fix the very first to anchor gauge)
      • run _core_ba(...) with a robust kernel
      • run asynchronously (thread) and allow early termination
        if tracking thread needs the map
    """
    raise NotImplementedError


# --------------------------------------------------------------------- #
#  Shared low-level BA engine
# --------------------------------------------------------------------- #
def _core_ba(world_map, K, keypoints,
             *,
             opt_kf_idx: list[int],
             fix_kf_idx: list[int],
             max_points: int | None = None,
             max_iters : int = 20,
             info_tag  : str = ""):
    """
    Generic sparse BA over a **subset** of poses + points.
    """
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    intr = np.array([fx, fy, cx, cy], np.float64)

    problem = pyceres.Problem()
    loss_fn = pyceres.HuberLoss(2.0)
    # TODO DONT add intrinsics if they are fixed
    problem.add_parameter_block(intr, 4)
    problem.set_parameter_block_constant(intr)

    # --- pose blocks ---------------------------------------------------
    #TODO why for looop, optimize relative poses instead of absolute, 
    quat_params, trans_params = {}, {}
    for k in opt_kf_idx:
        quat, tr = _pose_to_quat_trans(world_map.poses[k])
        quat_params[k] = quat
        trans_params[k] = tr
        problem.add_parameter_block(quat, 4)
        problem.set_manifold(quat, pyceres.EigenQuaternionManifold())
        problem.add_parameter_block(tr, 3)

    for k in fix_kf_idx:
        quat, tr = _pose_to_quat_trans(world_map.poses[k])
        quat_params[k] = quat
        trans_params[k] = tr
        problem.add_parameter_block(quat, 4)
        problem.set_manifold(quat, pyceres.EigenQuaternionManifold())
        problem.add_parameter_block(tr, 3)
        problem.set_parameter_block_constant(quat)
        problem.set_parameter_block_constant(tr)

    # --- point blocks --------------------------------------------------
    added_pts = 0
    for mp in world_map.points.values():
        # keep only points seen by at least one optimisable KF
        if not any(f in opt_kf_idx for f, _ in mp.observations):
            continue
        if max_points and added_pts >= max_points:
            continue
        problem.add_parameter_block(mp.position, 3)
        added_pts += 1

        for f_idx, kp_idx in mp.observations:
            if f_idx not in opt_kf_idx and f_idx not in fix_kf_idx:
                continue
            u, v = keypoints[f_idx][kp_idx].pt
            # print("Should be xyz:  ",mp.position)
            _add_reproj_edge(problem, loss_fn,
                             (u, v),
                             quat_params[f_idx],
                             trans_params[f_idx],
                             mp.position,
                             intr)
    print(problem.num_residual_blocks(), "residuals added")
    if problem.num_residual_blocks() < 10:
        print(f"{info_tag} skipped – not enough residuals")
        return

    # --- solve ---------------------------------------------------------
    opts = pyceres.SolverOptions()
    opts.max_num_iterations = max_iters
    opts.minimizer_progress_to_stdout = True
    summary = pyceres.SolverSummary()
    pyceres.solve(opts, problem, summary)

    # --- write poses back ---------------------------------------------
    for k in opt_kf_idx:
        world_map.poses[k][:] = _quat_trans_to_pose(
            quat_params[k], trans_params[k])

    iters = (getattr(summary, "iterations_used", getattr(summary, "num_successful_steps", getattr(summary, "num_iterations", None))))
    print(f"{info_tag}  iters={iters}  "
          f"χ²={summary.final_cost:.2f}  "
          f"res={problem.num_residual_blocks()}")
