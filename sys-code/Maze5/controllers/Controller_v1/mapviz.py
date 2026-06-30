"""
mapviz.py
=========
Live visualisation of the SLAM state with matplotlib.

It draws, on one figure:
  * the occupancy grid   -- white = free, black = wall, grey = unknown;
  * the current lidar scan (blue dots) -- the walls seen right now;
  * the pose-graph trajectory (green) -- the chain of keyframe poses;
  * loop-closure edges (red)          -- where the robot recognised a place;
  * the robot's pose (red dot + heading line).

Everything here is OPTIONAL eye-candy: if matplotlib cannot open a window
(e.g. a head-less machine) the controller keeps mapping; only the live view
and the saved PNG are skipped.  All drawing is wrapped so a rendering hiccup
can never crash the robot.
"""

import numpy as np


class MapViz:
    """A self-refreshing matplotlib view of the map and pose graph."""

    def __init__(self, grid):
        self.grid = grid
        self.ok = False
        self.plt = None
        self.fig = None

        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()                              # interactive (non-blocking)

            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            extent = [grid.ox, grid.ox + grid.ncols * grid.res,
                      grid.oy, grid.oy + grid.nrows * grid.res]

            # 'gray_r': probability 0 -> white (free), 1 -> black (wall).
            self.im = self.ax.imshow(grid.prob(), cmap="gray_r",
                                     origin="lower", extent=extent,
                                     vmin=0.0, vmax=1.0, interpolation="nearest")
            (self.scan_dots,) = self.ax.plot([], [], ".", color="dodgerblue",
                                             markersize=2, label="lidar")
            (self.traj_line,) = self.ax.plot([], [], "-", color="limegreen",
                                             linewidth=1.0, label="trajectory")
            (self.node_dots,) = self.ax.plot([], [], "o", color="orange",
                                             markersize=3, label="keyframes")
            (self.robot_dot,) = self.ax.plot([], [], "o", color="red",
                                             markersize=7, label="robot")
            (self.heading,) = self.ax.plot([], [], "-", color="red", linewidth=2)

            from matplotlib.collections import LineCollection
            self.loop_lines = LineCollection([], colors="red", linewidths=1.2)
            self.ax.add_collection(self.loop_lines)

            self.ax.set_title("Graph-SLAM map  (RosBot 2)")
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")
            self.ax.set_aspect("equal")
            self.ax.legend(loc="upper right", fontsize=8)
            self.fig.tight_layout()
            self.ok = True
        except Exception as exc:                    # noqa: BLE001
            print("[mapviz] live view disabled (%s: %s)"
                  % (type(exc).__name__, exc))

    # ---------------------------------------------------------------------- #
    def update(self, pose, scan_world, node_xy, loop_pairs):
        """Refresh all artists.

        Args:
            pose       : Pose2D, the live robot pose.
            scan_world : (N, 2) current scan hit points in world coords.
            node_xy    : (M, 2) keyframe-node positions.
            loop_pairs : list of (i, j) index pairs into node_xy (loop edges).
        """
        if not self.ok:
            return
        try:
            self.im.set_data(self.grid.prob())

            if scan_world is not None and len(scan_world):
                self.scan_dots.set_data(scan_world[:, 0], scan_world[:, 1])
            else:
                self.scan_dots.set_data([], [])

            if node_xy is not None and len(node_xy):
                self.traj_line.set_data(node_xy[:, 0], node_xy[:, 1])
                self.node_dots.set_data(node_xy[:, 0], node_xy[:, 1])
                segs = [[node_xy[i], node_xy[j]] for (i, j) in loop_pairs]
                self.loop_lines.set_segments(segs)

            self.robot_dot.set_data([pose.x], [pose.y])
            hx = pose.x + 0.25 * np.cos(pose.theta)
            hy = pose.y + 0.25 * np.sin(pose.theta)
            self.heading.set_data([pose.x, hx], [pose.y, hy])

            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)
        except Exception as exc:                    # noqa: BLE001
            print("[mapviz] update failed, disabling live view (%s)" % exc)
            self.ok = False

    # ---------------------------------------------------------------------- #
    def save(self, path):
        """Save the current figure to ``path`` (PNG)."""
        if self.fig is None:
            return
        try:
            self.im.set_data(self.grid.prob())       # make sure it is current
            self.fig.savefig(path, dpi=150)
            print("[mapviz] saved map image -> %s" % path)
        except Exception as exc:                     # noqa: BLE001
            print("[mapviz] could not save image (%s)" % exc)
