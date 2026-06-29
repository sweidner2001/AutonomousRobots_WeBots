"""
mapviz.py
=========
Live visualization of the occupancy grid using matplotlib.

Shows, updating in real time:
  * the occupancy grid   (white = free, black = wall, grey = unknown)
  * the robot            (red dot + heading arrow)
  * the current lidar scan (faint blue points)
  * the planned path     (green line) and frontier target (magenta star)

Runs in-process and is refreshed every few control steps.  It is wrapped in
try/except everywhere so that, if no display is available, the controller keeps
running and simply writes a PNG at the end (see Controller_v1.save_map).
"""

import math

import numpy as np

import Maze5.controllers.Controller_v1.config as C

_OK = True
try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:                      # pragma: no cover
    _OK = False


class MapViz:
    def __init__(self, grid):
        self.grid = grid
        self.ok = _OK
        self.fig = None
        if not self.ok:
            print("[mapviz] matplotlib unavailable -> live view disabled.")
            return
        try:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.ax.set_title("RosBot maze exploration")
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")
            extent = [C.GRID_ORIGIN_X,
                      C.GRID_ORIGIN_X + C.GRID_WIDTH_M,
                      C.GRID_ORIGIN_Y,
                      C.GRID_ORIGIN_Y + C.GRID_HEIGHT_M]
            self._extent = extent
            self.im = self.ax.imshow(
                self._render_rgb(), origin="lower", extent=extent,
                interpolation="nearest")
            self.scan_plot, = self.ax.plot([], [], ".", color="#3a7bd5",
                                           markersize=1.5, alpha=0.5)
            self.path_plot, = self.ax.plot([], [], "-", color="#16a34a", lw=1.8)
            self.target_plot, = self.ax.plot([], [], "*", color="magenta",
                                             markersize=14)
            self.robot_plot, = self.ax.plot([], [], "o", color="red",
                                            markersize=7)
            self.heading_arrow = None
            self.ax.set_aspect("equal")
            self.fig.tight_layout()
        except Exception as e:                 # pragma: no cover
            print("[mapviz] init failed (%s) -> live view disabled." % e)
            self.ok = False

    # ------------------------------------------------------------------ #
    def _render_rgb(self):
        """Build an RGB image: unknown grey, free white, wall black."""
        p = self.grid.prob()
        observed = self.grid.observed
        img = np.full(p.shape + (3,), 0.6, dtype=np.float32)   # unknown grey
        free = observed & (p <= C.P_FREE_THRESH)
        wall = p >= C.P_OCC_THRESH
        img[free] = 1.0          # white
        img[wall] = 0.0          # black
        return img

    # ------------------------------------------------------------------ #
    def update(self, pose, scan_xy=None, world_path=None, target_xy=None):
        if not self.ok:
            return
        try:
            self.im.set_data(self._render_rgb())

            x, y, theta = pose
            self.robot_plot.set_data([x], [y])

            # heading arrow (recreate each frame)
            if self.heading_arrow is not None:
                try:
                    self.heading_arrow.remove()
                except Exception:
                    pass
            self.heading_arrow = self.ax.arrow(
                x, y, 0.3 * math.cos(theta), 0.3 * math.sin(theta),
                head_width=0.12, head_length=0.12, fc="red", ec="red",
                length_includes_head=True)

            if scan_xy is not None and len(scan_xy[0]):
                self.scan_plot.set_data(scan_xy[0], scan_xy[1])
            else:
                self.scan_plot.set_data([], [])

            if world_path:
                xs = [p[0] for p in world_path]
                ys = [p[1] for p in world_path]
                self.path_plot.set_data(xs, ys)
            else:
                self.path_plot.set_data([], [])

            if target_xy is not None:
                self.target_plot.set_data([target_xy[0]], [target_xy[1]])
            else:
                self.target_plot.set_data([], [])

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception as e:                 # pragma: no cover
            print("[mapviz] update failed (%s) -> disabling live view." % e)
            self.ok = False

    # ------------------------------------------------------------------ #
    def save(self, path):
        """Save the current figure (or a fresh render) to PNG."""
        try:
            if self.ok and self.fig is not None:
                self.fig.savefig(path, dpi=130)
                return
            # Headless render fallback.
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(figsize=(7, 7))
            extent = [C.GRID_ORIGIN_X, C.GRID_ORIGIN_X + C.GRID_WIDTH_M,
                      C.GRID_ORIGIN_Y, C.GRID_ORIGIN_Y + C.GRID_HEIGHT_M]
            ax.imshow(self._render_rgb(), origin="lower", extent=extent)
            ax.set_aspect("equal")
            ax.set_title("RosBot maze map")
            fig.savefig(path, dpi=130)
            _plt.close(fig)
        except Exception as e:                 # pragma: no cover
            print("[mapviz] save failed: %s" % e)
