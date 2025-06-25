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
import warnings

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

ColourAxis = Literal["x", "y", "z", "auto"]

# --------------------------------------------------------------------------- #
#  3‑D visualiser  (Open3D only)                                             #
# --------------------------------------------------------------------------- #

class Visualizer3D:
    """Open3D window showing coloured landmarks & trajectory."""

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        color_axis: ColourAxis = "z",
        *,
        new_colour: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        window_size: Tuple[int, int] = (1280, 720),
        nav_step: float = 0.25,
    ) -> None:
        self.backend = "none"
        self._closed = False

        self.color_axis = color_axis
        self.new_colour = np.array(new_colour, float)
        self.nav_step = nav_step

        # running stats for colour normalisation
        self._v_min: Optional[float] = None
        self._v_max: Optional[float] = None
        self._pc_vec: Optional[np.ndarray] = None  # PCA axis for auto mode

        if o3d is None:
            warnings.warn(f"Open3D missing → visualiser disabled ({_OPEN3D_ERR})")
            return

        vis_cls = (
            o3d.visualization.VisualizerWithKeyCallback
            if hasattr(o3d.visualization, "VisualizerWithKeyCallback")
            else o3d.visualization.Visualizer
        )
        self.vis = vis_cls()
        self.vis.create_window("SLAM Map", width=window_size[0], height=window_size[1])
        self.backend = "open3d"

        # geometry holders
        self.pcd = o3d.geometry.PointCloud()
        self.lines = o3d.geometry.LineSet()
        self._first = True

        if isinstance(self.vis, o3d.visualization.VisualizerWithKeyCallback):
            self._bind_nav_keys()

        print(f"[Visualizer3D] ready | colour_axis={self.color_axis}")

    # ------------------------------------------------------------------ #
    #  Navigation keys (WASDQE)
    # ------------------------------------------------------------------ #
    def _bind_nav_keys(self):
        vc = self.vis.get_view_control()

        def translate(delta: np.ndarray):
            cam = vc.convert_to_pinhole_camera_parameters()
            T = np.eye(4)
            T[:3, 3] = delta * self.nav_step
            cam.extrinsic = T @ cam.extrinsic
            vc.convert_from_pinhole_camera_parameters(cam)
            return False

        key_map = {
            ord("W"): np.array([0, 0, -1]),
            ord("S"): np.array([0, 0, 1]),
            ord("A"): np.array([-1, 0, 0]),
            ord("D"): np.array([1, 0, 0]),
            ord("Q"): np.array([0, 1, 0]),
            ord("E"): np.array([0, -1, 0]),
        }
        for k, v in key_map.items():
            self.vis.register_key_callback(k, lambda _v, vec=v: translate(vec))

    # ------------------------------------------------------------------ #
    #  Scalar & colour helpers
    # ------------------------------------------------------------------ #
    def _compute_scalar(self, pts: np.ndarray) -> np.ndarray:
        if self.color_axis in ("x", "y", "z"):
            return pts[:, {"x": 0, "y": 1, "z": 2}[self.color_axis]]
        if self._pc_vec is None:
            centred = pts - pts.mean(axis=0)
            _, _, vh = np.linalg.svd(centred, full_matrices=False)
            self._pc_vec = vh[0]
        return pts @ self._pc_vec

    def _normalise(self, scalars: np.ndarray) -> np.ndarray:
        if self._v_min is None:
            self._v_min, self._v_max = np.percentile(scalars, [5, 95])
        else:
            self._v_min = min(self._v_min, scalars.min())
            self._v_max = max(self._v_max, scalars.max())
        return np.clip((scalars - self._v_min) / (self._v_max - self._v_min + 1e-6), 0, 1)

    def _colormap(self, norm: np.ndarray) -> np.ndarray:
        try:
            import matplotlib.cm as cm
            return cm.get_cmap("turbo")(norm)[:, :3]
        except Exception:
            h = (1 - norm) * 240
            c = np.ones_like(h)
            x = c * (1 - np.abs((h / 60) % 2 - 1))
            m = np.zeros_like(h)
            rgb = np.select(
                [h < 60, h < 120, h < 180, h < 240, h < 300, h >= 300],
                [
                    np.stack([c, x, m], 1),
                    np.stack([x, c, m], 1),
                    np.stack([m, c, x], 1),
                    np.stack([m, x, c], 1),
                    np.stack([x, m, c], 1),
                    np.stack([c, m, x], 1),
                ],
            )
            return rgb

    # ------------------------------------------------------------------ #
    #  Public update
    # ------------------------------------------------------------------ #
    def update(self, slam_map: Map, new_ids: Optional[List[int]] = None):
        if self.backend != "open3d" or len(slam_map.points) == 0:
            return

        pts = np.array([mp.position for mp in slam_map.points.values()])
        if hasattr(next(iter(slam_map.points.values())), "colour"):
            colours = np.array([mp.colour for mp in slam_map.points.values()])
        else:                           # backward-compat
            scalars = self._compute_scalar(pts)
            colours = self._colormap(self._normalise(scalars))

        if new_ids:
            id_to_i = {pid: i for i, pid in enumerate(slam_map.points.keys())}
            for nid in new_ids:
                if nid in id_to_i:
                    colours[id_to_i[nid]] = self.new_colour

        self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd.colors = o3d.utility.Vector3dVector(colours)

        self._update_lineset(slam_map)

        if self._first:
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(self.lines)
            self._first = False
        else:
            self.vis.update_geometry(self.pcd)
            self.vis.update_geometry(self.lines)

        self.vis.poll_events()
        self.vis.update_renderer()

    # ------------------------------------------------------------------ #
    #  Trajectory helper
    # ------------------------------------------------------------------ #
    def _update_lineset(self, slam_map: Map):
        if len(slam_map.poses) < 2:
            return
        path = np.array([p[:3, 3] for p in slam_map.poses])
        self.lines.points = o3d.utility.Vector3dVector(path)
        self.lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(path) - 1)])
        self.lines.colors = o3d.utility.Vector3dVector(np.tile([[0, 0, 1]], (len(path) - 1, 1)))

    # ------------------------------------------------------------------ #
    #  Teardown
    # ------------------------------------------------------------------ #
    def close(self):
        if self.backend == "open3d" and not self._closed:
            self.vis.destroy_window()
            self._closed = True


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
import matplotlib.pyplot as plt

class TrajectoryPlotter:
    """Interactive Matplotlib window showing estimate (blue) and GT (red)."""

    def __init__(self, figsize: Tuple[int, int] = (5, 5)) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True, ls="--", lw=0.5)
        self.line_est, = self.ax.plot([], [], "b-", lw=1.5, label="estimate")
        self.line_gt,  = self.ax.plot([], [], "r-", lw=1.0, label="ground-truth")
        self.ax.legend(loc="upper right")
        self._est_xy: list[tuple[float, float]] = []
        self._gt_xy:  list[tuple[float, float]] = []

    # ------------------------------------------------------------------ #
    def append(self,
               est_pos: np.ndarray,
               gt_pos:  Optional[np.ndarray] = None,
               swap_axes: bool = False) -> None:
        """
        Add one position pair and refresh the plot.

        Parameters
        ----------
        est_pos : (3,)  SLAM position in world coords.
        gt_pos  : (3,) or None  aligned ground-truth, same frame.
        swap_axes : plot x<->z if your dataset convention differs.
        """
        x, z = (est_pos[2], est_pos[0]) if swap_axes else (est_pos[0], est_pos[2])
        self._est_xy.append((x, z))
        ex, ez = zip(*self._est_xy)
        self.line_est.set_data(ex, ez)

        if gt_pos is not None:
            gx, gz = (gt_pos[2], gt_pos[0]) if swap_axes else (gt_pos[0], gt_pos[2])
            self._gt_xy.append((gx, gz))
            gx_s, gz_s = zip(*self._gt_xy)
            self.line_gt.set_data(gx_s, gz_s)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
