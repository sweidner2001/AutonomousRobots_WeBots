"""
occupancy_grid.py  --  Probabilistic 2-D map using log-odds.
=============================================================

WHAT IS AN OCCUPANCY GRID?
----------------------------
An occupancy grid divides the world into a regular array of square
cells.  Each cell stores the probability that it contains an obstacle
(e.g. a wall).  The robot updates these probabilities every time it
takes a new lidar measurement.

  probability ≈ 1.0  ->  very likely OCCUPIED  (wall is here)
  probability ≈ 0.5  ->  UNKNOWN  (never measured)
  probability ≈ 0.0  ->  very likely FREE       (empty space)

The grid is indexed like a 2-D array:  grid[row][col]
  row  corresponds to the y-axis
  col  corresponds to the x-axis

WHY LOG-ODDS INSTEAD OF PLAIN PROBABILITIES?
----------------------------------------------
We update the map using BAYES' THEOREM:
  P_new = P_old * P_sensor / normalisation_constant

Multiplying probabilities directly can cause numerical underflow
(values get too small for floating point).  Log-odds avoids this:

  log-odds L = log( p / (1-p) )       (ranges from -∞ to +∞)
  probability p = 1 / (1 + e^(-L))

With log-odds, Bayesian updates become simple ADDITIONS:
  L_new = L_old + L_measurement

This is fast and numerically stable.

HOW A LIDAR SCAN IS PROCESSED
-------------------------------
For each laser ray:
  1. Draw a line from the sensor to the hit point (Bresenham's algorithm).
  2. All cells ALONG the line (before the hit) are probably FREE
     -> subtract L_FREE from their log-odds (makes them more free).
  3. The cell AT the hit point is probably OCCUPIED
     -> add L_OCC to its log-odds (makes it more occupied).

After many scans, walls build up high log-odds and empty corridors
build up low log-odds.

THE "OBSERVED" MASK
--------------------
We also keep a boolean array `observed` that records whether each cell
has EVER been touched by a lidar ray.  Frontier detection uses this to
distinguish "free" from "never seen" cells.

COORDINATE CONVENTIONS
------------------------
  world (x, y) in metres  <->  grid (col, row) in integer cell indices
  world_to_grid(x, y) -> (col, row)
  grid_to_world(col, row) -> (x, y)  [returns the cell centre]

The grid origin (col=0, row=0) is at world (GRID_ORIGIN_X, GRID_ORIGIN_Y).
"""

import numpy as np

import Maze4.controllers.Controller_v1.config as C


class OccupancyGrid:
    """2-D log-odds occupancy grid.  All external access via world coordinates."""

    def __init__(self):
        self.res   = C.GRID_RESOLUTION   # metres per cell
        self.ox    = C.GRID_ORIGIN_X     # world x of cell (col=0)
        self.oy    = C.GRID_ORIGIN_Y     # world y of cell (row=0)
        self.ncols = int(round(C.GRID_WIDTH_M  / self.res))  # number of columns
        self.nrows = int(round(C.GRID_HEIGHT_M / self.res))  # number of rows

        # Log-odds array.  0 = unknown (probability 0.5).
        # Shape: (nrows, ncols)  indexed as  [row, col] = [y-index, x-index].
        self.log = np.zeros((self.nrows, self.ncols), dtype=np.float32)

        # Boolean mask: True if this cell has ever been touched by a lidar ray.
        self.observed = np.zeros((self.nrows, self.ncols), dtype=bool)

        # Boolean mask: True where the camera has spotted a green "do not
        # drive here" floor marking.  Unlike the probabilistic wall map,
        # this is STICKY -- once a cell is flagged it is never cleared.
        # A false positive here just makes the robot take a longer path;
        # a false negative would let it drive over a hazard, so we err on
        # the side of never forgetting a detection.
        self.hazard = np.zeros((self.nrows, self.ncols), dtype=bool)

        # Boolean masks: True where the camera has spotted the blue / yellow
        # target object.  Also sticky, for the same reason as `hazard` above
        # -- once we know a cell holds a solid object, we never "un-know" it.
        self.object_masks = {
            "blue":   np.zeros((self.nrows, self.ncols), dtype=bool),
            "yellow": np.zeros((self.nrows, self.ncols), dtype=bool),
        }

    # ---------------------------------------------------------------------- #
    # Coordinate conversion helpers
    # ---------------------------------------------------------------------- #
    def world_to_grid(self, x, y):
        """Convert world position (m) to grid indices (col, row).

        Note: returns (col, row) — x maps to col, y maps to row.
        The result may be outside [0, ncols) or [0, nrows) — use
        in_bounds() to check before indexing the arrays.
        """
        col = int((x - self.ox) / self.res)
        row = int((y - self.oy) / self.res)
        return col, row

    def grid_to_world(self, col, row):
        """Convert grid indices to the world coordinates of the cell CENTRE.

        Returns (x, y) in metres.
        The centre offset (+0.5) converts the corner position to the
        middle of the cell.
        """
        x = self.ox + (col + 0.5) * self.res
        y = self.oy + (row + 0.5) * self.res
        return x, y

    def in_bounds(self, col, row):
        """True if (col, row) is inside the grid array."""
        return 0 <= col < self.ncols and 0 <= row < self.nrows

    # ---------------------------------------------------------------------- #
    # Probability views (derived from the log-odds array)
    # ---------------------------------------------------------------------- #
    def prob(self):
        """Return full (nrows x ncols) array of occupancy probabilities [0, 1].

        p = sigmoid(L) = 1 / (1 + e^(-L))
        """
        return 1.0 - 1.0 / (1.0 + np.exp(self.log))

    def occ_mask(self):
        """Boolean mask: True where p >= P_OCC_THRESH (cell is a wall)."""
        return self.prob() >= C.P_OCC_THRESH

    def free_mask(self):
        """Boolean mask: True where p <= P_FREE_THRESH AND cell has been observed.

        "Free" cells are both probably empty AND have been measured at
        least once.  Unobserved cells are NOT considered free.
        """
        return (self.prob() <= C.P_FREE_THRESH) & self.observed

    def unknown_mask(self):
        """Boolean mask: True where the cell has NEVER been observed.

        Unknown cells are the exploration targets — the frontier sits
        at the boundary between free and unknown cells.
        """
        return ~self.observed

    def hazard_mask(self):
        """Boolean mask: True where a green floor hazard has been detected.

        Returned directly (already boolean, no thresholding needed).
        """
        return self.hazard

    def object_mask(self, color):
        """Boolean mask: True where the given colour ('blue' or 'yellow')
        has been detected."""
        return self.object_masks[color]

    def any_object_mask(self):
        """Boolean mask: True where EITHER tracked colour has been detected.

        Used by the planner to inflate/block both colours' cells in a
        single pass -- it doesn't need to distinguish which colour it is,
        only that the robot cannot drive through it.
        """
        return self.object_masks["blue"] | self.object_masks["yellow"]

    # ---------------------------------------------------------------------- #
    # Camera-based hazard / object marking
    # ---------------------------------------------------------------------- #
    def mark_hazard_world(self, xs, ys):
        """Flag the grid cells at the given world coordinates as hazards.

        Called by MazeExplorer with the (x, y) points that FloorHazardDetector
        found to be green AND on the floor.  Marking is permanent (see the
        `hazard` array's docstring in __init__ for why).

        Args:
            xs, ys : 1-D NumPy arrays of world coordinates (metres).
                     May be empty -- this is a normal "nothing detected" frame.
        """
        self._mark_world(self.hazard, xs, ys)

    def mark_object_world(self, color, xs, ys):
        """Flag the grid cells at the given world coordinates as holding the
        given colour's tracked object.  Same semantics as mark_hazard_world.

        Args:
            color  : "blue" or "yellow".
            xs, ys : 1-D NumPy arrays of world coordinates (metres).
        """
        self._mark_world(self.object_masks[color], xs, ys)

    def _mark_world(self, mask, xs, ys):
        """Shared helper: set `mask` to True at the grid cells under (xs, ys)."""
        if len(xs) == 0:
            return

        cols = ((xs - self.ox) / self.res).astype(np.int32)
        rows = ((ys - self.oy) / self.res).astype(np.int32)

        in_bounds = (
            (cols >= 0) & (cols < self.ncols)
            & (rows >= 0) & (rows < self.nrows)
        )
        mask[rows[in_bounds], cols[in_bounds]] = True

    # ---------------------------------------------------------------------- #
    # Scan integration  (the core mapping function)
    # ---------------------------------------------------------------------- #
    def integrate_scan(self, x, y, theta, ranges, bearings):
        """Fuse one complete lidar scan into the occupancy map.

        This function implements the "inverse sensor model" of a
        range finder: for each laser ray, decide which cells are
        probably FREE and which is probably OCCUPIED.

        Args:
            x, y   : Robot position in the world frame (metres).
            theta  : Robot heading (radians).
            ranges : 1-D NumPy array of measured distances (m).
                     np.inf means the ray hit nothing within range.
            bearings: 1-D NumPy array of per-ray angles in the ROBOT
                     frame (radians), precomputed by robot.py.
        """
        # The lidar is mounted slightly in front of the robot centre.
        # Compute the sensor origin in the world frame.
        sx = x + C.LIDAR_OFFSET_X * np.cos(theta)
        sy = y + C.LIDAR_OFFSET_X * np.sin(theta)
        # Convert sensor origin to grid cell (start cell for Bresenham).
        c0, r0 = self.world_to_grid(sx, sy)

        # Precompute world-frame angles for all rays at once (vectorised).
        world_ang = theta + bearings
        cos_a = np.cos(world_ang)
        sin_a = np.sin(world_ang)

        max_range = C.LIDAR_USE_RANGE  # cap: rays beyond this are "no hit"

        # Process each ray individually.
        for i in range(len(ranges)):
            rng = ranges[i]

            # Determine whether this ray actually hit an obstacle.
            hit = True
            if not np.isfinite(rng):
                # Ray returned inf: the beam hit nothing within sensor range.
                # We have NO information from this ray — skipping it avoids
                # falsely marking cells as free through walls that the lidar
                # beam happened to pass through without reflecting back.
                continue
            elif rng >= max_range:
                # Ray is beyond our usable range: skip (same reasoning).
                continue
            elif rng < C.LIDAR_MIN_RANGE:
                # Too close to be reliable (sensor blind zone); skip entirely.
                continue
            # Clamp to usable range (should never exceed max_range here,
            # but guard against floating-point edge cases).
            rng = min(rng, max_range)

            # Compute the end point of this ray in the world frame.
            ex = sx + rng * cos_a[i]
            ey = sy + rng * sin_a[i]
            c1, r1 = self.world_to_grid(ex, ey)

            # Update all cells along the ray using Bresenham's line algorithm.
            self._ray(c0, r0, c1, r1, hit)

    def _ray(self, c0, r0, c1, r1, hit):
        """Update grid cells along a single laser ray.

        All cells from (c0,r0) to the cell BEFORE the end are marked FREE.
        The end cell (c1,r1) is marked OCCUPIED if `hit` is True,
        or FREE if the beam did not hit anything.

        Args:
            c0, r0 : Start cell (sensor position).
            c1, r1 : End cell (hit or max range point).
            hit    : True if the ray struck an obstacle at (c1, r1).
        """
        cells = _bresenham(c0, r0, c1, r1)
        if not cells:
            return

        # Mark all intermediate cells (between sensor and hit) as FREE.
        for (c, r) in cells[:-1]:
            if self.in_bounds(c, r):
                # Subtract L_FREE (making cell more likely free), then clamp.
                self.log[r, c] = max(self.log[r, c] + C.L_FREE, -C.L_CLAMP)
                self.observed[r, c] = True

        # Update the end cell — always a confirmed hit (inf and max-range rays
        # are skipped in integrate_scan before _ray is called).
        c, r = cells[-1]
        if self.in_bounds(c, r):
            self.log[r, c] = min(self.log[r, c] + C.L_OCC, C.L_CLAMP)
            self.observed[r, c] = True

    # ---------------------------------------------------------------------- #
    def save(self, npy_path):
        """Save the raw log-odds array to a NumPy binary file."""
        np.save(npy_path, self.log)


# ============================================================================
# Bresenham's line algorithm
# ============================================================================
def _bresenham(x0, y0, x1, y1):
    """Return all integer grid cells on the line from (x0,y0) to (x1,y1).

    Bresenham's line algorithm is a classic method (1962) for drawing a
    straight line on a grid using only integer arithmetic.  It finds the
    set of grid cells that best approximates the ideal straight line.

    This is used to trace each laser ray through the grid so we know
    exactly which cells the beam passed through.

    Returns:
        List of (col, row) tuples, including BOTH endpoints.
        Returns an empty list if the inputs are invalid.

    EXAMPLE:
        _bresenham(0, 0, 3, 2) might return:
        [(0,0), (1,0), (1,1), (2,1), (3,2)]
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1   # step direction in x
    sy = 1 if y0 < y1 else -1   # step direction in y
    err = dx - dy                # error term drives which axis steps each iteration

    x, y = x0, y0
    max_steps = dx + dy + 2      # upper bound on path length (guards runaway loops)
    steps = 0

    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break  # reached the destination
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x   += sx
        if e2 < dx:
            err += dx
            y   += sy
        steps += 1
        if steps > max_steps:
            break   # safety guard: exit if we somehow over-run

    return points
