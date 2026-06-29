"""
mapviz.py  --  Live occupancy-grid visualisation using matplotlib.
==================================================================

WHAT DOES THIS FILE DO?
------------------------
During the simulation, MapViz draws a real-time picture of everything
the robot knows so far:

  MAP (background image)
  ------
    White  = known FREE cells    (robot has seen empty space here)
    Black  = occupied cells      (probably a wall)
    Grey   = unknown cells       (lidar has never measured this area)

  ROBOT (red dot + arrow)
  ------
    Shows the estimated robot position and the direction it faces.

  LIDAR SCAN (small blue dots)
  ------
    Shows the raw lidar hit points in the current scan.
    Useful to see what the sensor is actually measuring.

  PLANNED PATH (green line)
  ------
    The A* path the robot intends to follow to the next frontier.

  FRONTIER TARGET (magenta star *)
  ------
    The frontier cluster centroid the robot is heading toward.

IMPLEMENTATION NOTES
---------------------
- We use matplotlib's interactive mode (plt.ion()) so the plot
  updates without blocking the controller.
- If matplotlib is unavailable (e.g. headless server), the controller
  still runs normally and just saves a PNG at the end instead.
- All drawing operations are wrapped in try/except so a display error
  never crashes the robot.
"""

import math

import numpy as np

import Maze4.controllers.Controller_v1.config as C

# Try to import matplotlib.  If the environment has no display (headless
# server, CI runner) the import may succeed but plt.show() would fail.
# We guard against this by wrapping everything in try/except later.
_OK = True
try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:
    _OK = False


class MapViz:
    """Live matplotlib visualisation of the occupancy grid and robot state."""

    def __init__(self, grid):
        """Initialise the figure and all plot elements.

        Args:
            grid (OccupancyGrid): the map being built (shared reference;
                we read from it on every update but never write to it).
        """
        self.grid = grid
        self.ok   = _OK          # False if matplotlib is unavailable
        self.fig  = None

        if not self.ok:
            print("[mapviz] matplotlib unavailable -> live view disabled.")
            return

        try:
            plt.ion()   # turn on interactive mode (non-blocking updates)

            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.ax.set_title("RosBot maze exploration")
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")

            # `extent` tells imshow the world coordinates of the image edges.
            # [left, right, bottom, top] in world metres.
            extent = [
                C.GRID_ORIGIN_X,
                C.GRID_ORIGIN_X + C.GRID_WIDTH_M,
                C.GRID_ORIGIN_Y,
                C.GRID_ORIGIN_Y + C.GRID_HEIGHT_M,
            ]
            self._extent = extent

            # Background image: the occupancy map.
            # We will call im.set_data() every update to refresh it.
            self.im = self.ax.imshow(
                self._render_rgb(),
                origin="lower",           # row 0 = bottom of the image
                extent=extent,
                interpolation="nearest",  # no blurring between cells
            )

            # Blue dots: current lidar scan hits.
            self.scan_plot,   = self.ax.plot([], [], ".",
                                             color="#3a7bd5",
                                             markersize=1.5, alpha=0.5)

            # Green line: planned A* path.
            self.path_plot,   = self.ax.plot([], [], "-",
                                             color="#16a34a", lw=1.8)

            # Magenta star: frontier target centroid.
            self.target_plot, = self.ax.plot([], [], "*",
                                             color="magenta", markersize=14)

            # Red dot: robot position.
            self.robot_plot,  = self.ax.plot([], [], "o",
                                             color="red", markersize=7)

            # Heading arrow: recreated each frame (Arrow objects can't be updated).
            self.heading_arrow = None

            self.ax.set_aspect("equal")  # keep grid cells square
            self.fig.tight_layout()

        except Exception as e:
            print("[mapviz] init failed (%s) -> live view disabled." % e)
            self.ok = False

    # ---------------------------------------------------------------------- #
    def _render_rgb(self):
        """Build an RGB colour image from the current occupancy map.

        Each cell gets one of three colours based on its probability:
          grey  (0.6, 0.6, 0.6) = unknown   (never observed)
          white (1.0, 1.0, 1.0) = free      (observed and probably empty)
          black (0.0, 0.0, 0.0) = occupied  (probably a wall)

        Returns:
            np.ndarray shape (nrows, ncols, 3), dtype float32, values [0, 1].
        """
        p        = self.grid.prob()       # occupancy probability array
        observed = self.grid.observed     # which cells have been measured

        # Start with all cells grey (unknown).
        img = np.full(p.shape + (3,), 0.6, dtype=np.float32)

        # Override free cells with white.
        free = observed & (p <= C.P_FREE_THRESH)
        img[free] = 1.0

        # Override occupied cells with black.
        wall = p >= C.P_OCC_THRESH
        img[wall] = 0.0

        return img

    # ---------------------------------------------------------------------- #
    def update(self, pose, scan_xy=None, world_path=None, target_xy=None):
        """Refresh all plot elements with the latest robot state.

        Called by MazeExplorer every VIZ_EVERY control steps.

        Args:
            pose       : (x, y, theta) from odometry.
            scan_xy    : (xs, ys) arrays of lidar hit world coordinates,
                         or None if no scan is available.
            world_path : list of (x, y) waypoints of the planned path,
                         or None if no path is active.
            target_xy  : (x, y) world position of the frontier target,
                         or None if no target is set.
        """
        if not self.ok:
            return
        try:
            # Refresh the background map image.
            self.im.set_data(self._render_rgb())

            x, y, theta = pose

            # Move the robot dot to the current position.
            self.robot_plot.set_data([x], [y])

            # Redraw the heading arrow.
            # Arrow objects do not support in-place update, so we remove the
            # old one and create a new one each frame.
            if self.heading_arrow is not None:
                try:
                    self.heading_arrow.remove()
                except Exception:
                    pass
            arrow_len = 0.3  # metres
            self.heading_arrow = self.ax.arrow(
                x, y,
                arrow_len * math.cos(theta),
                arrow_len * math.sin(theta),
                head_width=0.12, head_length=0.12,
                fc="red", ec="red",
                length_includes_head=True,
            )

            # Update lidar scan points.
            if scan_xy is not None and len(scan_xy[0]):
                self.scan_plot.set_data(scan_xy[0], scan_xy[1])
            else:
                self.scan_plot.set_data([], [])

            # Update planned path line.
            if world_path:
                xs = [p[0] for p in world_path]
                ys = [p[1] for p in world_path]
                self.path_plot.set_data(xs, ys)
            else:
                self.path_plot.set_data([], [])

            # Update frontier target marker.
            if target_xy is not None:
                self.target_plot.set_data([target_xy[0]], [target_xy[1]])
            else:
                self.target_plot.set_data([], [])

            # Push all changes to the screen.
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)   # tiny pause lets the GUI process events

        except Exception as e:
            print("[mapviz] update failed (%s) -> disabling live view." % e)
            self.ok = False

    # ---------------------------------------------------------------------- #
    def save(self, path):
        """Save the current map figure to a PNG file.

        Falls back to a headless render if the live view is disabled.

        Args:
            path (str): destination file path (e.g. "map_final.png").
        """
        try:
            if self.ok and self.fig is not None:
                # Save the already-open interactive figure.
                self.fig.savefig(path, dpi=130)
                return

            # Headless fallback: create a fresh figure using the Agg backend
            # (no display required), render the map, save, and close.
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

        except Exception as e:
            print("[mapviz] save failed: %s" % e)
