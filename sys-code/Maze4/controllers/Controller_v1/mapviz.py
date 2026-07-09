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

    def __init__(self, grid, vizualization_nav_grid=C.VIZALIZATION_NAV_GRID):
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

        # ---------- second window: 5 internal planner/debug maps ----------
        self._fig2          = None
        self._axes2         = None
        self._drive_ims     = [None] * 5   # AxesImage handles (lazy init)
        self._drive_arrow2  = None          # heading arrow on the nav grid

        self.vizualization_nav_grid = vizualization_nav_grid
        if self.ok and vizualization_nav_grid:
            try:
                subplots = 4
                self._fig2, self._axes2 = plt.subplots(
                    1, subplots, figsize=(15, 5)
                )
                _labels = [
                    # "Blocked cells",
                    # "Reachable area",
                    "Nav grid  (+pose)",
                    "Frontier mask",
                    "Frontier clusters",
                    "A* debug",
                ]
                for ax, lbl in zip(self._axes2, _labels):
                    ax.set_title(lbl, fontsize=9)
                    ax.axis("off")
                self._fig2.suptitle("Internal planner maps", fontsize=11)
                self._fig2.tight_layout()
            except Exception as e:
                print("[mapviz] drive-map fig init failed (%s)" % e)
                self._fig2 = None

    # ---------------------------------------------------------------------- #
    def _render_rgb(self):
        """Build an RGB colour image from the current occupancy map.

        Each cell gets one of six colours based on its state:
          grey        (0.6, 0.6, 0.6) = unknown   (never observed)
          white       (1.0, 1.0, 1.0) = free      (observed and probably empty)
          black       (0.0, 0.0, 0.0) = occupied  (probably a wall)
          bright green(0.1, 0.9, 0.1) = hazard    (camera-detected green floor
                                                    marking -- treated like a wall)
          blue        (0.15,0.35,0.95)= blue tracked object
          yellow      (0.95,0.85,0.10)= yellow tracked object

        Hazard/object cells are drawn LAST so they are always visible even
        on top of a cell that also reads as "free" from the lidar (the
        camera sees things the lidar's flat 2-D scan plane cannot).

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

        # Override camera-detected hazard cells with bright green.
        img[self.grid.hazard_mask()] = (0.1, 0.9, 0.1)

        # Override low, lidar-blind obstacles (camera-only map) with orange,
        # so they are distinguishable from lidar walls (black) while debugging.
        img[self.grid.camera_obstacle_mask()] = (1.0, 0.55, 0.0)

        # Override camera-detected tracked-object cells with their colour.
        # object_mask() is kept clean of wall-fused/speckle false positives
        # by the periodic clean_object_log() call in the main loop (see
        # occupancy_grid.py: OccupancyGrid.clean_object_log()).
        img[self.grid.object_mask("blue")]   = (0.15, 0.35, 0.95)
        img[self.grid.object_mask("yellow")] = (0.95, 0.85, 0.10)

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
    def update_drive_map(self, pose = None, nav=None, fmask=None, fclusters=None, astar_debug=None):
        """Refresh the five-panel debug window with internal planner data.

        Each panel shows one stage of the exploration pipeline:
          1. Blocked cells       -- cells the robot cannot enter (walls + inflation).
          2. Reachable area      -- cells reachable from the robot via flood-fill.
          3. Nav grid + pose     -- 3-value grid (free/unknown/blocked) with the
                                   current robot position and heading overlaid.
          4. Frontier mask       -- cells on the explored/unexplored boundary.
          5. Frontier clusters   -- each cluster drawn in a unique colour;
                                   yellow dot marks the centroid of every cluster.

        Args:
            pose      : (x, y, theta) robot pose in world metres.
            blocked   : bool ndarray -- True = cell is blocked.
            reachable : bool ndarray -- True = cell is reachable.
            nav       : float32 ndarray -- 0.0 / 0.5 / 1.0 navigation grid.
            fmask     : bool ndarray -- True = frontier cell.
            fclusters : list of cluster dicts {'cells', 'centroid', 'size'},
                        or None if no clusters are available yet.
        """
        if self.vizualization_nav_grid is False:
            return
        
        if not self.ok or self._fig2 is None:
            return
        # if blocked is None or reachable is None or nav is None or fmask is None:
        #     return

        try:
            shape = nav.shape   # (nrows, ncols)

            # -------------------------------------------------------------- #
            # Build the 5 RGB images (float32, values in [0, 1]).
            # -------------------------------------------------------------- #

            # Image 1 -- Blocked cells:  white = passable, black = blocked.
            # if blocked is not None:
            #     img1 = np.ones(shape + (3,), dtype=np.float32)
            #     img1[blocked] = 0.0

            # # Image 2 -- Reachable area: dark grey = not reachable,
            # #                            green     = reachable.
            # if reachable is not None:
            #     img2 = np.full(shape + (3,), 0.2, dtype=np.float32)
            #     img2[reachable] = (0.15, 0.80, 0.25)

            # Image 3 -- Nav grid: grey = unknown (0.5),
            #                      white = free (1.0), black = blocked (0.0).
            img3 = np.full(shape + (3,), 0.55, dtype=np.float32)
            img3[nav == 1.0] = 1.0
            img3[nav == 0.0] = 0.0

            # Image 4 -- Frontier mask: black = non-frontier,
            #                           orange = frontier cell.
            img4 = np.zeros(shape + (3,), dtype=np.float32)
            img4[fmask] = (1.0, 0.55, 0.0)

            # Image 5 -- Frontier clusters: each cluster gets a distinct hue;
            #            centroid cell shown as a bright yellow dot.
            img5 = self._render_clusters(shape, fclusters)

            # Image 6 -- A* debug: show blocked/unknown plus latest search
            # state (closed set, open set, final path, start, goal).
            img6 = np.full(shape + (3,), 0.55, dtype=np.float32)
            img6[nav == 1.0] = 1.0
            img6[nav == 0.0] = 0.0

            if astar_debug is not None:
                blocked = astar_debug.get("blocked")
                unknown = astar_debug.get("unknown")
                closed = astar_debug.get("closed")
                opened = astar_debug.get("open")
                path = astar_debug.get("path")
                start = astar_debug.get("start")
                goal = astar_debug.get("goal")

                if blocked is not None and blocked.shape == shape:
                    img6[blocked] = (0.02, 0.02, 0.02)
                if unknown is not None and unknown.shape == shape:
                    img6[unknown] = (0.35, 0.35, 0.35)
                if closed is not None and closed.shape == shape:
                    img6[closed] = (0.25, 0.45, 0.95)
                if opened is not None and opened.shape == shape:
                    img6[opened] = (0.25, 0.85, 0.95)
                if path is not None and path.shape == shape:
                    img6[path] = (0.10, 0.90, 0.25)
                if start is not None:
                    sr, sc = start
                    if 0 <= sr < shape[0] and 0 <= sc < shape[1]:
                        img6[sr, sc] = (1.0, 0.10, 0.10)
                if goal is not None:
                    gr, gc = goal
                    if 0 <= gr < shape[0] and 0 <= gc < shape[1]:
                        img6[gr, gc] = (1.0, 0.10, 1.0)

            # imgs = [img1, img2, img3, img4, img5]
            imgs = [img3, img4, img5, img6]
            # imgs = [img for img in imgs if img is not None]  # drop any Nones

            # -------------------------------------------------------------- #
            # Push images to axes (create imshow on first call, then set_data).
            # -------------------------------------------------------------- #
            for i, (ax, img) in enumerate(zip(self._axes2, imgs)):
                if self._drive_ims[i] is None:
                    self._drive_ims[i] = ax.imshow(
                        img,
                        origin="lower",
                        interpolation="nearest",
                        vmin=0.0, vmax=1.0,
                    )
                else:
                    self._drive_ims[i].set_data(img)

            # -------------------------------------------------------------- #
            # Robot pose overlay on axes[2] (nav grid).
            # Coordinates are in CELL units (col = x-axis, row = y-axis).
            # -------------------------------------------------------------- #
            x, y, theta = pose
            col, row = self.grid.world_to_grid(x, y)

            # Robot dot: reuse a stored Line2D if it exists.
            if not hasattr(self, "_drive_robot_dot") or self._drive_robot_dot is None:
                self._drive_robot_dot, = self._axes2[0].plot(
                    [col], [row], "o", color="red", markersize=2, zorder=5
                )
            else:
                self._drive_robot_dot.set_data([col], [row])

            # Heading arrow in cell units.
            arrow_cells = 3.0
            if self._drive_arrow2 is not None:
                try:
                    self._drive_arrow2.remove()
                except Exception:
                    pass
            self._drive_arrow2 = self._axes2[0].arrow(
                col, row,
                arrow_cells * math.cos(theta),
                arrow_cells * math.sin(theta),
                head_width=1.5, head_length=1.5,
                fc="red", ec="red",
                length_includes_head=True,
                zorder=6,
            )

            # -------------------------------------------------------------- #
            # Flush to screen.
            # -------------------------------------------------------------- #
            self._fig2.canvas.draw_idle()
            self._fig2.canvas.flush_events()

        except Exception as e:
            print("[mapviz] update_drive_map failed (%s)" % e)

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _render_clusters(shape, clusters):
        """Build an RGB image that colour-codes frontier clusters.

        Each cluster is painted with a distinct colour from the tab10 palette.
        The centroid cell of every cluster is overdrawn with bright yellow so
        it is easy to spot.

        Args:
            shape    : (nrows, ncols) of the grid.
            clusters : list of cluster dicts, each containing:
                         'cells'    -- list of (row, col) tuples
                         'centroid' -- (row, col) of the cluster centre
                         'size'     -- number of cells
                       Pass None or an empty list to get a blank image.

        Returns:
            np.ndarray shape (nrows, ncols, 3), dtype float32, values [0, 1].
        """
        # Dark background (unexplored / non-frontier space).
        img = np.full(shape + (3,), 0.12, dtype=np.float32)

        if not clusters:
            return img

        # tab10 gives 10 perceptually distinct colours.
        try:
            import matplotlib.pyplot as _plt
            cmap = _plt.get_cmap("tab10")
        except Exception:
            # Fallback: cycle through a small hard-coded palette.
            _palette = [
                (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17),
                (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
                (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
                (0.09, 0.75, 0.81),
            ]
            cmap = lambda i: _palette[i % len(_palette)]  # noqa: E731

        nrows, ncols = shape
        for idx, cl in enumerate(clusters):
            color = cmap(idx % 10)[:3]   # (R, G, B)
            for r, c in cl["cells"]:
                if 0 <= r < nrows and 0 <= c < ncols:
                    img[r, c] = color
            # Centroid in bright yellow so it stands out.
            cr, cc = cl["centroid"]
            if 0 <= cr < nrows and 0 <= cc < ncols:
                img[cr, cc] = (1.0, 0.95, 0.0)

        return img

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
