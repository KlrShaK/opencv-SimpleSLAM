# visualization_utils.py
from __future__ import annotations
"""
visualization_utils.py
~~~~~~~~~~~~~~~~~~~~~~
Clean, modular utilities for
* drawing 2‑D track overlays with OpenCV, and
* rendering a coloured 3‑D map in an **Open3D** window.

Main entry‑points
-----------------
``draw_tracks(img, tracks, frame_no)``
    Overlay recent feature tracks.

``Visualizer3D``
    Live window that shows
      • the SLAM point cloud, colour‑coded along an axis or PCA‑auto;
      • the camera trajectory (blue line);
      • new landmarks highlighted (default bright‑green).
    Supports WASDQE first‑person navigation when the Open3D build exposes
    key‑callback APIs.
"""

from typing import Dict, List, Tuple, Optional, Literal
import warnings, threading, cv2, numpy as np

import cv2
import numpy as np

try:
    import open3d as o3d  # type: ignore
except Exception as exc:  # pragma: no cover
    o3d = None  # type: ignore
    _OPEN3D_ERR = exc
else:
    _OPEN3D_ERR = None

from slam.core.landmark_utils import Map
import numpy as np
import matplotlib.pyplot as plt


ColourAxis = Literal["x", "y", "z", "auto"]

# --------------------------------------------------------------------------- #
#  3‑D visualiser  (Open3D only)                                             #
# --------------------------------------------------------------------------- #

class Visualizer3D:
    """Open3D window that shows the coloured point-cloud and camera path."""

    def __init__(
        self,
        color_axis: ColourAxis = "z",
        *,
        new_colour: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        window_size: Tuple[int, int] = (1280, 720),
        nav_step: float = 0.25,
    ) -> None:
        self.backend  = "none"
        self._closed  = False
        self._lock    = threading.Lock()
        self.paused   = False

        self.color_axis = color_axis
        self.new_colour = np.asarray(new_colour, dtype=np.float32)
        self.nav_step   = nav_step

        # scalar-to-colour normalisation
        self._v_min: Optional[float] = None
        self._v_max: Optional[float] = None
        self._pc_vec: Optional[np.ndarray] = None  # PCA axis for "auto"

        # ------------------------------------------------------------------ #
        #  Open3D init
        # ------------------------------------------------------------------ #
        if o3d is None:
            warnings.warn(f"[Visualizer3D] Open3D missing → window disabled ({_OPEN3D_ERR})")
            return

        vis_cls = (
            o3d.visualization.VisualizerWithKeyCallback
            if hasattr(o3d.visualization, "VisualizerWithKeyCallback")
            else o3d.visualization.Visualizer
        )
        self.vis = vis_cls()
        self.vis.create_window("SLAM Map", width=window_size[0], height=window_size[1])
        self.backend = "open3d"

        self.pcd   = o3d.geometry.PointCloud()
        self.lines = o3d.geometry.LineSet()
        self._first = True

        if isinstance(self.vis, o3d.visualization.VisualizerWithKeyCallback):
            self._bind_nav_keys()

        print(f"[Visualizer3D] ready | colour_axis={self.color_axis}")

    # ------------------------------------------------------------------ #
    #  WASDQE first-person navigation
    # ------------------------------------------------------------------ #
    def _bind_nav_keys(self):
        vc = self.vis.get_view_control()

        def translate(delta: np.ndarray):
            cam = vc.convert_to_pinhole_camera_parameters()
            T = np.eye(4);  T[:3, 3] = delta * self.nav_step
            cam.extrinsic = T @ cam.extrinsic
            vc.convert_from_pinhole_camera_parameters(cam)
            return False

        key_map = {
            ord("W"): np.array([0, 0, -1]),
            ord("S"): np.array([0, 0,  1]),
            ord("A"): np.array([-1, 0, 0]),
            ord("D"): np.array([ 1, 0, 0]),
            ord("Q"): np.array([0,  1, 0]),
            ord("E"): np.array([0, -1, 0]),
        }
        for k, v in key_map.items():
            self.vis.register_key_callback(k, lambda _v, vec=v: translate(vec))

    # ------------------------------------------------------------------ #
    #  Helpers for colour mapping
    # ------------------------------------------------------------------ #
    def _compute_scalar(self, pts: np.ndarray) -> np.ndarray:
        if self.color_axis in ("x", "y", "z"):
            return pts[:, {"x": 0, "y": 1, "z": 2}[self.color_axis]]
        if self._pc_vec is None:                 # first call → PCA axis
            centred = pts - pts.mean(0)
            _, _, vh = np.linalg.svd(centred, full_matrices=False)
            self._pc_vec = vh[0]
        return pts @ self._pc_vec

    def _normalise(self, scalars: np.ndarray) -> np.ndarray:
        if self._v_min is None:                  # initialise 5th–95th perc.
            self._v_min, self._v_max = np.percentile(scalars, [5, 95])
        else:                                    # expand running min / max
            self._v_min = min(self._v_min, scalars.min())
            self._v_max = max(self._v_max, scalars.max())
        return np.clip((scalars - self._v_min) / (self._v_max - self._v_min + 1e-6), 0, 1)

    def _colormap(self, norm: np.ndarray) -> np.ndarray:
        try:
            import matplotlib.cm as cm
            return cm.get_cmap("turbo")(norm)[:, :3]
        except Exception:
            # fall-back: simple HSV → RGB
            h = (1 - norm) * 240
            c = np.ones_like(h); m = np.zeros_like(h)
            x = c * (1 - np.abs((h / 60) % 2 - 1))
            return np.select(
                [h < 60, h < 120, h < 180, h < 240, h < 300, h >= 300],
                [
                    np.stack([c, x, m], 1), np.stack([x, c, m], 1),
                    np.stack([m, c, x], 1), np.stack([m, x, c], 1),
                    np.stack([x, m, c], 1), np.stack([c, m, x], 1),
                ])

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #
    def update(self, slam_map: Map, new_ids: Optional[List[int]] = None):
        if self.backend != "open3d" or len(slam_map.points) == 0:
            return
        if self.paused:
            with self._lock:
                self.vis.poll_events(); self.vis.update_renderer()
            return

        # -------------------------- build numpy arrays -------------------------
        pts = slam_map.get_point_array()
        col = slam_map.get_color_array() if hasattr(slam_map, "get_color_array") else None
        if col is None or len(col) == 0:            # legacy maps without colour
            scal  = self._compute_scalar(pts)
            col   = self._colormap(self._normalise(scal))
        else:
            col = col.astype(np.float32)

        # keep arrays in sync (pad / trim)
        if len(col) < len(pts):
            diff = len(pts) - len(col)
            col  = np.vstack([col, np.full((diff, 3), 0.8, np.float32)])
        elif len(col) > len(pts):
            col  = col[: len(pts)]

        # highlight newly-added landmarks
        if new_ids:
            id_to_i = {pid: i for i, pid in enumerate(slam_map.point_ids())}
            for nid in new_ids:
                if nid in id_to_i:
                    col[id_to_i[nid]] = self.new_colour

        # ----------------------- Open3D geometry update ------------------------
        with self._lock:
            self.pcd.points  = o3d.utility.Vector3dVector(pts)
            self.pcd.colors  = o3d.utility.Vector3dVector(col)

            self._update_lineset(slam_map)

            if self._first:
                self.vis.add_geometry(self.pcd); self.vis.add_geometry(self.lines)
                self._first = False
            else:
                self.vis.update_geometry(self.pcd); self.vis.update_geometry(self.lines)

            self.vis.poll_events(); self.vis.update_renderer()

    # ------------------------------------------------------------------ #
    #  Blue camera trajectory poly-line
    # ------------------------------------------------------------------ #
    def _update_lineset(self, slam_map: Map):
        if len(slam_map.poses) < 2:
            return
        path = np.asarray([p[:3, 3] for p in slam_map.poses], np.float32)
        self.lines.points = o3d.utility.Vector3dVector(path)
        self.lines.lines  = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(path) - 1)])
        self.lines.colors = o3d.utility.Vector3dVector(np.tile([[0, 0, 1]], (len(path) - 1, 1)))

    # ------------------------------------------------------------------ #
    #  Clean shutdown
    # ------------------------------------------------------------------ #
    def close(self):
        if self.backend == "open3d" and not self._closed:
            self.vis.destroy_window(); self._closed = True
# --------------------------------------------------------------------------- #
#  2‑D overlay helpers
# --------------------------------------------------------------------------- #

def draw_tracks(
    vis: np.ndarray,
    tracks: Dict[int, List[Tuple[int, int, int]]],
    current_frame: int,
    max_age: int = 10,
    sample_rate: int = 5,
    max_tracks: int = 100,
) -> np.ndarray:
    """Draw ageing feature tracks as fading polylines.

    Parameters
    ----------
    vis          : BGR uint8 image (modified *in‑place*)
    tracks       : {track_id: [(frame_idx,x,y), ...]}
    current_frame: index of the frame being drawn
    max_age      : only show segments younger than this (#frames)
    sample_rate  : skip tracks where `track_id % sample_rate != 0` to avoid clutter
    max_tracks   : cap total rendered tracks for speed
    """
    recent = [
        (tid, pts)
        for tid, pts in tracks.items()
        if current_frame - pts[-1][0] <= max_age
    ]
    recent.sort(key=lambda x: x[1][-1][0], reverse=True)

    drawn = 0
    for tid, pts in recent:
        if drawn >= max_tracks:
            break
        if tid % sample_rate:
            continue
        pts = [p for p in pts if current_frame - p[0] <= max_age]
        for j in range(1, len(pts)):
            _, x0, y0 = pts[j - 1]
            _, x1, y1 = pts[j]
            ratio = (current_frame - pts[j - 1][0]) / max_age
            colour = (0, int(255 * (1 - ratio)), int(255 * ratio))
            cv2.line(vis, (x0, y0), (x1, y1), colour, 2)
        drawn += 1
    return vis


# --------------------------------------------------------------------------- #
#  Lightweight 2-D trajectory plotter (Matplotlib)
# --------------------------------------------------------------------------- #

# ---- Trajectory 2D (x–z) + simple pause UI ----
class Trajectory2D:
    """
    Matplotlib-based 2D trajectory plotter (x–z), with GT↔EST Sim(3) alignment
    and a live display of the *scale* (s) used for alignment.

    Public API mirrors the original Trajectory2D:
        - __init__(gt_T_list=None, win="Trajectory 2D (x–z)")
        - push(frame_idx: int, Tcw: np.ndarray)
        - draw(paused: bool=False)
    """
    def __init__(self, gt_T_list=None, win="Trajectory 2D (x–z)"):
        # state
        self.win = win
        self.gt_T = gt_T_list   # list of 4x4 Twc (or None)
        self.est_xyz: list[np.ndarray] = []   # estimated camera centers (world)
        self.gt_xyz:  list[np.ndarray] = []   # paired GT centers
        self.align_ok = False
        self.s = 1.45 # initial guess
        self.R = np.eye(3)
        self.t = np.zeros(3)

        # small OpenCV 'ghost' window so cv2.waitKey in VizUI keeps working
        # (keep it tiny and unobtrusive)
        try:
            cv2.namedWindow("__ghost__", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("__ghost__", 1, 1)
            cv2.imshow("__ghost__", np.zeros((1, 1, 3), np.uint8))
        except Exception:
            pass

        # Matplotlib figure
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        try:
            # not all backends expose this
            self.fig.canvas.manager.set_window_title(win)
        except Exception:
            pass

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("z")
        self.ax.grid(True, which="both", alpha=0.3)
        self.ax.set_aspect("equal", adjustable="box")

        # line objects for fast updates
        (self.line_est,) = self.ax.plot([], [], lw=2, label="estimate")
        (self.line_gt,)  = self.ax.plot([], [], lw=2, label="ground-truth")

        # text overlay for alignment scale + status
        self.info_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            va="top", ha="left", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.8", alpha=0.8)
        )

        self.legend = None

    @staticmethod
    def _cam_center_from_Tcw(Tcw: np.ndarray) -> np.ndarray:
        R, t = Tcw[:3, :3], Tcw[:3, 3]
        return (-R.T @ t).astype(np.float64)

    def _maybe_update_alignment(self, Kpairs: int = 100):
        # Need at least a few pairs
        if len(self.est_xyz) < 6 or len(self.gt_xyz) < 6:
            return
        X = np.asarray(self.gt_xyz[-Kpairs:], np.float64)   # GT
        Y = np.asarray(self.est_xyz[-Kpairs:], np.float64)  # EST
        muX, muY = X.mean(0), Y.mean(0)
        X0, Y0 = X - muX, Y - muY
        cov = (Y0.T @ X0) / X.shape[0]
        U, S, Vt = np.linalg.svd(cov)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R = U @ D @ Vt
        varY = (Y0**2).sum() / X.shape[0]
        s = (S * np.diag(D)).sum() / (varY + 1e-12)
        t = muX - s * (R @ muY)
        self.s, self.R, self.t = float(s), R, t
        self.align_ok = True

    def push(self, frame_idx: int, Tcw: np.ndarray):
        self.est_xyz.append(self._cam_center_from_Tcw(Tcw))
        if self.gt_T is not None and 0 <= frame_idx < len(self.gt_T):
            self.gt_xyz.append(self.gt_T[frame_idx][:3, 3].astype(np.float64))
        # self._maybe_update_alignment(Kpairs=min(100, len(self.est_xyz))) # TODO: TEMPORARY - UNCOMMENT WHEN YOU WANT ALIGNMENT

    def draw(self, paused: bool = False, margin_frac: float = 0.05):
        # nothing to draw yet
        if not self.est_xyz:
            self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
            return
        
        self.align_ok = True # TODO: TEMPORARY - REMOVE WHEN YOU WANT ALIGNMENT
        # Assemble current series
        E = np.asarray(self.est_xyz, np.float64)
        if self.align_ok:
            E = (self.s * (self.R @ E.T)).T + self.t

        have_gt = len(self.gt_xyz) > 0
        if have_gt:
            G = np.asarray(self.gt_xyz, np.float64)
            # axis limits based on both
            all_x = np.concatenate([E[:, 0], G[:, 0]])
            all_z = np.concatenate([E[:, 2], G[:, 2]])
        else:
            all_x = E[:, 0]; all_z = E[:, 2]

        # Pad bounds a little
        minx, maxx = all_x.min(), all_x.max()
        minz, maxz = all_z.min(), all_z.max()
        spanx = max(maxx - minx, 1e-6)
        spanz = max(maxz - minz, 1e-6)
        pad_x = margin_frac * spanx
        pad_z = margin_frac * spanz

        self.ax.set_xlim(minx - pad_x - 50, maxx + pad_x + 50)  # extra horiz. space for legend
        self.ax.set_ylim(minz - pad_z - 30, maxz + pad_z + 30)

        # Update lines: plot x vs z
        self.line_est.set_data(E[:, 0], E[:, 2])
        if have_gt:
            self.line_gt.set_data(G[:, 0], G[:, 2])
            self.line_gt.set_visible(True)
        else:
            self.line_gt.set_visible(False)

        # Keep equal aspect for geometric fidelity
        self.ax.set_aspect("equal", adjustable="box")

        # Info box (scale + status)
        if have_gt and self.align_ok:
            info = f"s = {self.s:.4f}  •  R={self.R.shape[0]}x{self.R.shape[1]}  •  aligned ✓"
        elif have_gt:
            info = "s = 1.0000  •  aligning…"
        else:
            info = "No GT available • showing EST only"
        if paused:
            info += "\nPAUSED  [p: resume | n: step | q/Esc: quit]"
        self.info_text.set_text(info)

        # Legend once
        if self.legend is None:
            handles = [self.line_est] + ([self.line_gt] if have_gt else [])
            labels  = ["estimate"] + (["ground-truth"] if have_gt else [])
            self.legend = self.ax.legend(handles, labels, loc="upper right", framealpha=0.85)

        # Draw
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    # Optional getter if you want to read the current scale from outside
    def current_scale(self) -> float:
        return float(self.s)


# --------------------------------------------------------------------------- #
#  UI for Pausing and stepping through the pipeline
# --------------------------------------------------------------------------- #
class VizUI:
    """Tiny UI state for pausing/stepping the pipeline."""
    def __init__(self, pause_key='p', step_key='n', quit_keys=('q', 27)):
        self.pause_key = self._to_code(pause_key)
        self.step_key  = self._to_code(step_key)
        self.quit_keys = {self._to_code(k) for k in (quit_keys if isinstance(quit_keys, (tuple, list, set)) else (quit_keys,))}
        self.paused = False
        self._request_quit = False
        self._do_step = False

    @staticmethod
    def _to_code(k):
        return k if isinstance(k, int) else ord(k)

    def poll(self, delay_ms=1):
        k = cv2.waitKey(delay_ms) & 0xFF
        if k == 255:  # no key
            return
        if k in self.quit_keys:
            self._request_quit = True
            return
        if k == self.pause_key:
            self.paused = not self.paused
            self._do_step = False
            return
        if k == self.step_key:
            self._do_step = True
            return

    def should_quit(self):
        return self._request_quit

    def wait_if_paused(self):
        """Block while paused, but allow 'n' to step one iteration."""
        if not self.paused:
            return False  # not blocking
        while True:
            k = cv2.waitKey(30) & 0xFF
            if k == self.pause_key:  # resume
                self.paused = False
                return False
            if k in self.quit_keys:
                self._request_quit = True
                return False
            if k == self.step_key:
                # allow one iteration to run, remain paused afterward
                self._do_step = True
                return True  # consume one step

    def consume_step(self):
        """Return True once if a single-step was requested."""
        if self._do_step:
            self._do_step = False
            return True
        return False

