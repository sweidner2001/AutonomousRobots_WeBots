"""
occupancy_map.py
================
The digital map: a probabilistic 2-D occupancy grid in LOG-ODDS form.

WHAT IS AN OCCUPANCY GRID?
--------------------------
The world is chopped into a regular array of small square cells.  Each cell
stores how strongly we believe it contains a wall:

    p ~ 1   ->  occupied  (wall)        drawn black
    p ~ 0.5 ->  unknown   (never seen)  drawn grey
    p ~ 0   ->  free      (empty)       drawn white

WHY LOG-ODDS?
-------------
Fusing many noisy measurements with Bayes' rule turns into simple ADDITION if
we store the log-odds  L = log(p / (1 - p))  instead of p directly:

    every "this cell is free"  observation:  L += L_FREE   (L_FREE < 0)
    every "this cell is a wall" observation:  L += L_OCC    (L_OCC  > 0)

No multiplications, no underflow, and recovering p is just p = sigmoid(L).

HOW ONE SCAN IS ADDED (the inverse sensor model)
-------------------------------------------------
For each lidar beam we know the sensor cell and the hit cell.  Using
Bresenham's line algorithm we walk the cells the beam crossed:
    * every cell BEFORE the hit was passed through  -> evidence of FREE space;
    * the cell AT the hit                            -> evidence of a WALL.

THE SLAM CONNECTION
-------------------
``integrate()`` adds the live scan for a responsive view.  After the pose
graph is optimised (a loop closes and every past pose shifts), ``rebuild()``
clears the grid and re-draws every keyframe scan from its CORRECTED pose --
that is the moment the map snaps into global consistency.
"""

import numpy as np

import Maze5.controllers.Controller_v1.config as C


class OccupancyGrid:
    """Log-odds occupancy grid; all public access is in world metres."""

    def __init__(self):
        self.res = C.GRID_RESOLUTION
        self.ox = C.GRID_ORIGIN_X            # world x of column 0
        self.oy = C.GRID_ORIGIN_Y            # world y of row 0
        self.ncols = int(round(C.GRID_WIDTH_M / self.res))
        self.nrows = int(round(C.GRID_HEIGHT_M / self.res))

        # Log-odds (0 = unknown) and an "ever observed" mask.
        self.log = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        self.observed = np.zeros((self.nrows, self.ncols), dtype=bool)

    # ---- coordinate conversion ------------------------------------------ #
    def world_to_grid(self, x, y):
        """World (m) -> (col, row).  May fall outside the array; check bounds."""
        col = int((x - self.ox) / self.res)
        row = int((y - self.oy) / self.res)
        return col, row

    def grid_to_world(self, col, row):
        """(col, row) -> world (m) of the cell CENTRE."""
        return (self.ox + (col + 0.5) * self.res,
                self.oy + (row + 0.5) * self.res)

    def in_bounds(self, col, row):
        return 0 <= col < self.ncols and 0 <= row < self.nrows

    # ---- map building ---------------------------------------------------- #
    def integrate(self, pose, scan):
        """Fuse one scan, taken at ``pose`` (Pose2D), into the grid."""
        c0, r0 = self.world_to_grid(pose.x, pose.y)
        if not self.in_bounds(c0, r0):
            return                       # sensor off the map; nothing to do

        # Place the scan's hit points into the world frame.
        hits = pose.transform_points(scan.points)

        free_c, free_r, occ_c, occ_r = [], [], [], []
        for hx, hy in hits:
            c1, r1 = self.world_to_grid(hx, hy)
            cols, rows = _bresenham(c0, r0, c1, r1)
            # Cells before the hit are free; the last cell is the wall.
            free_c.extend(cols[:-1])
            free_r.extend(rows[:-1])
            occ_c.append(cols[-1])
            occ_r.append(rows[-1])

        self._apply(np.asarray(free_c), np.asarray(free_r), C.L_FREE)
        self._apply(np.asarray(occ_c), np.asarray(occ_r), C.L_OCC)
        np.clip(self.log, -C.L_CLAMP, C.L_CLAMP, out=self.log)

    def rebuild(self, keyframes):
        """Clear the grid and re-integrate every keyframe at its current pose.

        Args:
            keyframes: iterable of (Pose2D, Scan) -- e.g. from
                       ``GraphSlam.keyframe_poses()``.
        """
        self.log.fill(0.0)
        self.observed.fill(False)
        for pose, scan in keyframes:
            self.integrate(pose, scan)

    def _apply(self, cols, rows, delta):
        """Add ``delta`` to the log-odds of the given (in-bounds) cells."""
        if cols.size == 0:
            return
        keep = (cols >= 0) & (cols < self.ncols) & \
               (rows >= 0) & (rows < self.nrows)
        cols, rows = cols[keep], rows[keep]
        # np.add.at accumulates correctly even when a cell appears twice.
        np.add.at(self.log, (rows, cols), delta)
        self.observed[rows, cols] = True

    # ---- views used by visualisation and (later) planning --------------- #
    def prob(self):
        """Occupancy probability per cell, p = sigmoid(log-odds), in [0, 1]."""
        return 1.0 / (1.0 + np.exp(-self.log))

    def occ_mask(self):
        """True where the cell is probably a wall."""
        return self.prob() >= C.P_OCC_THRESH

    def free_mask(self):
        """True where the cell is observed AND probably empty."""
        return (self.prob() <= C.P_FREE_THRESH) & self.observed

    def unknown_mask(self):
        """True where the cell has never been observed."""
        return ~self.observed

    # ---- persistence ----------------------------------------------------- #
    def save(self, npy_path):
        """Save the raw log-odds array as a NumPy ``.npy`` file."""
        np.save(npy_path, self.log)


# ===========================================================================
# Bresenham's line algorithm
# ===========================================================================
def _bresenham(c0, r0, c1, r1):
    """Integer grid cells on the line from (c0, r0) to (c1, r1), inclusive.

    Bresenham's algorithm (1962) steps along the dominant axis using only
    integer additions, choosing at each step whether to also move on the
    other axis.  We use it to find exactly which cells each laser beam
    crossed.

    Returns:
        (cols, rows) -- two lists of equal length, both endpoints included.
    """
    cols, rows = [], []
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc - dr

    c, r = c0, r0
    guard = dc + dr + 2          # hard cap so a bad input can't loop forever
    while True:
        cols.append(c)
        rows.append(r)
        if c == c1 and r == r1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
        guard -= 1
        if guard < 0:
            break
    return cols, rows
