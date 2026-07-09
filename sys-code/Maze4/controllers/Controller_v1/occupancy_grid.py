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

import math

import numpy as np
from scipy import ndimage

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

        # Log-odds maps for the blue / yellow target objects -- ONE per colour.
        # Unlike `hazard` (which is sticky), these behave EXACTLY like the
        # lidar wall map `self.log`: every camera frame adds positive evidence
        # to cells that looked like the colour and negative evidence to cells
        # that were clearly visible but were NOT that colour.  A false
        # detection therefore decays away once the camera looks again and
        # disagrees -- see update_object_observation().  0 = unknown (p=0.5).
        self.object_log = {
            "blue":   np.zeros((self.nrows, self.ncols), dtype=np.float32),
            "yellow": np.zeros((self.nrows, self.ncols), dtype=np.float32),
        }

        # Separate log-odds map for LOW obstacles the RGB-D camera sees but the
        # lidar's fixed-height sweep passes over (see integrate_camera_obstacle
        # for why this must NOT share self.log: the lidar would otherwise erase
        # these cells as "free" on every scan).  0 = unknown (probability 0.5).
        self.camera_obstacle_log = np.zeros((self.nrows, self.ncols), dtype=np.float32)

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

    def object_prob(self, color):
        """Return the (nrows x ncols) occupancy-probability array for `color`.

        p = sigmoid(L) = 1 / (1 + e^(-L)), same relation as prob() uses for
        the lidar wall map.
        """
        return 1.0 - 1.0 / (1.0 + np.exp(self.object_log[color]))

    def camera_obstacle_mask(self):
        """Boolean mask: True where the camera-only obstacle map believes a
        low, lidar-blind obstacle sits (p >= P_OCC_THRESH).

        Kept entirely separate from occ_mask() (the lidar wall map) so the
        lidar cannot erase these cells; the planner ORs the two together when
        building its blocked-cell mask (see planner.get_block_cells_*).
        """
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.camera_obstacle_log))
        return prob >= C.P_OCC_THRESH

    def object_mask(self, color):
        """Boolean mask: True where the given colour ('blue' or 'yellow')
        is currently believed to hold the object (p >= P_OBJ_THRESH).

        This is a LIVE view of the log-odds map, so a cell that was a false
        positive drops back to False on its own once the camera revisits it
        and sees non-matching colour there (see update_object_observation()).
        """
        return self.object_prob(color) >= C.P_OBJ_THRESH

    def object_centroid(self, color):
        """Return the world (x, y) centre of the cells currently believed to
        hold `color`'s object, or None if no cell is above threshold.

        Because this is computed FRESH from object_mask() every call, the
        estimate self-corrects: when a false-positive cell decays back below
        threshold it stops contributing, and the centroid shifts back onto
        the real object -- unlike a running average, which can never forget a
        point it once averaged in.  object_mask() itself is kept clean of
        wall-fused/speckle false positives by the periodic
        clean_object_log() call in the main loop (see that method's
        docstring) -- this method doesn't need to know about that.
        """
        # mask = self.object_mask(color)
        mask = self.reconciled_object_mask(color)
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return None
        # Cell centres, then mean -> world centroid of the believed object.
        x = self.ox + (cols.mean() + 0.5) * self.res
        y = self.oy + (rows.mean() + 0.5) * self.res
        return float(x), float(y)

    def any_object_mask(self):
        """Boolean mask: True where EITHER tracked colour is believed present.

        Used by the planner to inflate/block both colours' cells in a
        single pass -- it doesn't need to distinguish which colour it is,
        only that the robot cannot drive through it.
        """
        return self.object_mask("blue") | self.object_mask("yellow")

    # ---------------------------------------------------------------------- #
    # Periodically cleaning false colour detections out of the object log
    # ---------------------------------------------------------------------- #
    def clean_object_log(self, color):
        """Permanently erase parts of `color`'s log-odds map that don't look
        like a real, standalone object.

        Called periodically from the main loop (every C.OBJECT_CLEAN_EVERY
        control steps, NOT on every read -- see explorer.py) -- unlike a
        live/stateless filter recomputed on every call, this edits
        self.object_log[color] directly, so object_mask(color) itself is
        clean everywhere it's read afterwards and no caller (object_centroid,
        the planner, mapviz, ...) needs to know this cleanup exists.

        THE PROBLEM THIS SOLVES
        --------------------------
        The RGB-D camera and the lidar are two independent sensors.  When
        the robot is close to the tracked object, a small registration/
        parallax error can make a few pixels of a WALL read as the
        object's colour instead of the object itself (see
        colored_objects.py's no-depth fallback for the related camera-side
        fix).  These false positives come in two recognisable shapes:
          - tiny 1-2 pixel speckles from a single noisy frame, or
          - a patch that is really part of the wall, which -- unlike a
            free-standing object -- is fused into the lidar's much larger
            connected network of wall cells.
        A real object is neither: it is a solid handful of camera pixels,
        and physically it sits apart from the wall network.

        WHY WIPING LOG-ODDS COUNTS AS "FOREVER" WITH NO BLACKLIST
        -----------------------------------------------------------------
        Resetting a cell's log-odds to 0 ("unknown") does not, on its own,
        stop some future frame from marking it `color` again -- there is
        no separate sticky per-cell blacklist, on purpose (every other map
        layer in this class self-corrects the same way; hazard_mask() is
        the one deliberate exception, and this isn't meant to become a
        second one). What makes the deletion effectively permanent is that
        this same check re-runs every C.OBJECT_CLEAN_EVERY steps: a
        cluster that is genuinely a wall reading will fail the isolation
        check again next cycle, and the one after that, for as long as the
        underlying wall geometry hasn't changed -- so it is never given
        more than C.OBJECT_CLEAN_EVERY steps to look like the object
        before being wiped again.

        THE TWO CHECKS
        ---------------
          1. Denoise: label object_mask(color) into 8-connected clusters
             and wipe any cluster with C.OBJECT_CLUSTER_MIN_PIXELS pixels
             or fewer -- single-frame camera noise.
          2. Isolation: label occ_mask() (the lidar wall map) into
             8-connected components.  Any surviving cluster that touches a
             wall component larger than C.OBJECT_ISOLATED_BLOB_MAX_CELLS
             cells is fused into the wall network rather than being a
             small, separate object, and gets wiped too.

        Args:
            color : "blue" or "yellow".
        """
        raw = self.object_mask(color)
        if not raw.any():
            return

        structure = np.ones((3, 3), dtype=bool)   # 8-connected

        # Step 1: denoise.
        cluster_labels, n_clusters = ndimage.label(raw, structure=structure)
        if n_clusters <= 1:
            return
        cluster_sizes = ndimage.sum(raw, cluster_labels, index=np.arange(1, n_clusters + 1))
        small     = np.nonzero(cluster_sizes <= C.OBJECT_CLUSTER_MIN_PIXELS)[0] + 1
        survivors = np.nonzero(cluster_sizes >  C.OBJECT_CLUSTER_MIN_PIXELS)[0] + 1
        to_delete = np.isin(cluster_labels, small)

        # Step 2: isolation check, only for clusters that survived step 1.
        if survivors.size:
            blocked = self.occ_mask()
            blocked_labels, n_blocked = ndimage.label(blocked, structure=structure)
            if n_blocked:
                blocked_sizes = ndimage.sum(
                    blocked, blocked_labels, index=np.arange(1, n_blocked + 1))
                for lbl in survivors:
                    cluster_mask = cluster_labels == lbl
                    touched = np.unique(blocked_labels[cluster_mask])
                    touched = touched[touched != 0]
                    if touched.size == 0:
                        continue   # touches no lidar wall -- a standalone object
                    blob_size = blocked_sizes[touched - 1].sum()
                    if blob_size > C.OBJECT_ISOLATED_BLOB_MAX_CELLS:
                        to_delete |= cluster_mask

        if to_delete.any():
            self.object_log[color][to_delete] = 0.0



        # ---------------------------------------------------------------------- #
    # Reconciling the lidar wall map with camera-detected objects
    # ---------------------------------------------------------------------- #
    def reconciled_object_mask(self, color, max_distance_m=None):
        """Build a corrected object mask by reconciling the camera's raw
        colour detections with the lidar's wall map, TRUSTING THE LIDAR
        whenever the two disagree.

        THE PROBLEM THIS SOLVES
        --------------------------
        The RGB-D camera and the lidar are two independent sensors, and the
        camera's colour test frequently only recognises PART of a real
        object -- lighting, viewing angle, or the hue thresholds miss some
        of its surface.  The lidar, meanwhile, sees the object's full
        physical footprint as ordinary occupied cells, indistinguishable
        from a wall.  We want to recover that full footprint for the
        colour that WAS recognised, by pulling in the occ_mask() cells
        that are the same physical object -- without pulling in a nearby
        but physically SEPARATE wall.

        HOW IT WORKS
        -------------
        1. Start from object_mask(color) -- the current camera-only
           detections (a live thresholded view of object_log, never
           modified by this method).
        2. Flood fill outward from there by at most max_distance_m,
           4-CONNECTED steps (up/down/left/right, no diagonals), via
           flood_fill_4connected() -- but each step may only enter a cell
           that is itself part of the raw detection or occ_mask() (walls).
           Growth can therefore only follow an UNBROKEN chain of occupied
           cells: a wall that happens to sit within max_distance_m of the
           seed but has a gap of FREE space in between is never reached,
           no matter how large max_distance_m is.  This is the fix over
           the old plain bounded dilation, which grew by raw distance and
           could jump straight over such a gap.
        3. Intersect the reachable region with occ_mask() to get exactly
           the wall/object-surface cells that are genuinely connected to
           the detection (the raw cells themselves are already colour
           evidence and don't need this filter).
        4. Return the union of the raw detections and those connected
           wall cells.

        WHY A CELL THAT LIDAR LATER PROVES FREE REMOVES ITSELF
        ------------------------------------------------------------
        This method is STATELESS: it recomputes the result FRESH from
        occ_mask() on every call, instead of caching or storing the
        merged result anywhere.  occ_mask() is itself a live view of the
        log-odds grid, and the ordinary Bayesian update in
        integrate_scan() already handles "the lidar changed its mind":
        repeated FREE observations lower a cell's log-odds until it drops
        back below P_OCC_THRESH, at which point occ_mask() stops
        including it -- which also breaks the connected chain leading to
        it, so it drops out of this method's result on the very next call
        too, with no extra bookkeeping.  The camera's own object_log is
        untouched either way.

        Args:
            color          : "blue" or "yellow".
            max_distance_m : how far (world metres), following connected
                              occ_mask() cells, a wall cell may be from a
                              raw detection to count as the same object.
                              Defaults to config.OBJECT_WALL_MATCH_DISTANCE_M.

        Returns:
            np.ndarray bool -- raw camera detections UNION any nearby,
            connected lidar-confirmed wall cells.
        """
        raw = self.object_mask(color)
        if not raw.any():
            return raw

        if max_distance_m is None:
            max_distance_m = C.OBJECT_WALL_MATCH_DISTANCE_M
        radius_cells = max(1, int(round(max_distance_m / self.res)))

        occ          = self.occ_mask()
        reachable    = flood_fill_4connected(raw, raw | occ, radius_cells)
        matched_wall = reachable & occ

        return raw | matched_wall







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

    def fuse_camera_free_space(self, xs, ys):
        """Fill UNKNOWN map holes with free floor the camera directly saw.

        THE PROBLEM THIS SOLVES
        --------------------------
        integrate_scan() only marks cells free along rays that actually HIT
        a wall -- an inf ray (no wall within range) is skipped completely,
        because we cannot tell "open space" from "beam lost".  In open areas
        with no wall behind them, the lidar therefore NEVER confirms the
        floor, and those cells stay UNKNOWN forever: phantom frontiers the
        robot keeps chasing, and holes in the flood-fill reachable set.

        The RGB-D camera does not have this ambiguity: a pixel that passes
        the ground-plane test (see floor_hazard.py) is a direct measurement
        of real, open floor at a known (x, y).  We fuse those points here.

        THE FUSION RULE: ONLY WHERE THE MAP IS STILL UNKNOWN
        -----------------------------------------------------
        A cell the lidar has already observed is left completely alone --
        the lidar is the more reliable geometry sensor, and the camera must
        never overrule it (e.g. soften a wall reading).  Only cells with
        observed == False receive the camera's free evidence:

          1. add L_FREE to the cell's log-odds (once per cell per frame --
             many pixels landing in the same cell still count as ONE
             observation, so a single frame cannot over-commit);
          2. once the accumulated log-odds is confidently free
             (p <= P_FREE_THRESH, i.e. at least two camera confirmations),
             flip observed = True.  From then on the cell counts as known
             free space for the frontier detector and flood fill, and this
             method stops touching it (it is no longer unknown).

        Requiring two confirmations before the flip keeps one noisy frame
        from erasing a genuine frontier.

        Args:
            xs, ys : 1-D NumPy arrays of world coordinates (m) of confirmed
                     free-floor points (may be empty).
        """
        if len(xs) == 0:
            return

        cols = ((np.asarray(xs) - self.ox) / self.res).astype(np.int32)
        rows = ((np.asarray(ys) - self.oy) / self.res).astype(np.int32)

        keep = (
            (cols >= 0) & (cols < self.ncols)
            & (rows >= 0) & (rows < self.nrows)
        )
        rows, cols = rows[keep], cols[keep]
        if rows.size == 0:
            return

        # Only cells the lidar has NEVER observed (the fusion rule above).
        unknown = ~self.observed[rows, cols]
        rows, cols = rows[unknown], cols[unknown]
        if rows.size == 0:
            return

        # De-duplicate: one log-odds step per cell per frame, no matter how
        # many camera pixels landed in it.
        flat = np.unique(rows.astype(np.int64) * self.ncols + cols)
        rows = (flat // self.ncols).astype(np.int32)
        cols = (flat %  self.ncols).astype(np.int32)

        self.log[rows, cols] = np.maximum(
            self.log[rows, cols] + C.L_FREE, -C.L_CLAMP)

        # Flip to "observed" once confidently free.  In log-odds, the
        # p <= P_FREE_THRESH boundary is log(p / (1-p)).
        free_logodds = math.log(C.P_FREE_THRESH / (1.0 - C.P_FREE_THRESH))
        confirmed = self.log[rows, cols] <= free_logodds
        self.observed[rows[confirmed], cols[confirmed]] = True

    def update_object_observation(self, color, hit_xs, hit_ys, free_xs, free_ys):
        """Fold ONE camera frame's colour observation into `color`'s log-odds
        map -- the object-detection twin of integrate_scan()'s inverse sensor
        model.

        Every camera frame produces two sets of world points for each colour:

          * hit points  -- pixels that MATCHED the colour this frame.  Each
            adds L_OBJ_OCC to its cell ("the object is here").
          * free points -- pixels with valid depth that were clearly visible
            but did NOT match the colour.  Each adds L_OBJ_FREE ("something
            is here, and it is not this colour"), which lets a past false
            positive decay back below threshold.

        Cells the camera could not see this frame are simply left untouched,
        so their belief neither grows nor decays -- exactly like a lidar cell
        that no ray passed through.

        Args:
            color                : "blue" or "yellow".
            hit_xs, hit_ys       : world coords of matching pixels (may be empty).
            free_xs, free_ys     : world coords of visible non-matching pixels
                                   (may be empty).
        """
        log = self.object_log[color]
        # Negative evidence first, then positive -- if the SAME cell somehow
        # appears in both (it cannot within one frame, but be safe), the "hit"
        # wins, matching |L_OCC| > |L_FREE| in the lidar model.
        self._add_logodds(log, free_xs, free_ys, C.L_OBJ_FREE, C.L_OBJ_CLAMP)
        self._add_logodds(log, hit_xs,  hit_ys,  C.L_OBJ_OCC,  C.L_OBJ_CLAMP)

    def _add_logodds(self, log, xs, ys, delta, clamp):
        """Shared helper: add `delta` ONCE per cell under (xs, ys), clamped
        to +/- `clamp`.  Out-of-bounds points are dropped.

        WHY DE-DUPLICATE TO ONE UPDATE PER CELL
        ------------------------------------------
        At close range, the camera's sampled pixel grid is FINER than one
        occupancy cell: e.g. at 0.3 m, ~150 sampled pixels can fall inside a
        single 4 cm cell (their physical spacing shrinks with distance).
        Those are NOT 150 independent observations -- they are the SAME
        physical patch seen from the SAME viewpoint in the SAME frame.  A
        naive per-point accumulation (np.add.at) would apply `delta` up to
        150 times in one frame, instantly slamming the cell to the log-odds
        clamp ceiling.  Undoing that later needs ~clamp/|L_FREE| separate
        CORRECTING frames (e.g. ~15 for the object log-odds values) -- which
        in practice rarely happens from the same vantage point, so a wrong
        mark effectively never clears.  Collapsing to one update per cell
        per call fixes this: repeated same-frame sampling of one cell counts
        as exactly ONE observation, matching a sound Bayesian sensor model
        (and exactly the pattern fuse_camera_free_space() already uses).
        """
        if len(xs) == 0:
            return
        cols = ((np.asarray(xs) - self.ox) / self.res).astype(np.int32)
        rows = ((np.asarray(ys) - self.oy) / self.res).astype(np.int32)
        keep = (
            (cols >= 0) & (cols < self.ncols)
            & (rows >= 0) & (rows < self.nrows)
        )
        rows, cols = rows[keep], cols[keep]
        if rows.size == 0:
            return
        flat = np.unique(rows.astype(np.int64) * self.ncols + cols)
        rows = (flat // self.ncols).astype(np.int32)
        cols = (flat %  self.ncols).astype(np.int32)
        log[rows, cols] += delta
        np.clip(log, -clamp, clamp, out=log)

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
            self._ray(c0, r0, c1, r1, hit, observed=self.observed)

    # ---------------------------------------------------------------------- #
    # Scan integration  (RGB-D camera -- catches obstacles the lidar misses)
    # ---------------------------------------------------------------------- #
    def integrate_camera_obstacle(self, x, y, theta, ranges, bearings):
        """Fuse camera-detected low obstacles into the SEPARATE camera-only
        obstacle map (self.camera_obstacle_log) -- NOT the lidar map.

        WHY A SEPARATE MAP INSTEAD OF self.log
        --------------------------------------
        The lidar only sweeps ONE fixed horizontal plane.  A low obstacle
        (a few cm high -- see depth_obstacle.py) sits BELOW that plane, so
        every lidar scan shoots straight over it and marks its cell FREE.
        If we folded the camera's detection into the same self.log array, the
        very next lidar scan would erase it again.  Keeping a dedicated
        camera-only log-odds map means the lidar can never overwrite it; the
        planner then treats camera_obstacle_mask() as impassable alongside
        walls and hazards (see planner.get_block_cells_*).

        It is still a full inverse-sensor-model update (FREE along the ray,
        OCCUPIED at the hit), so a camera false positive is self-corrected
        the next time the camera sees clear depth through that cell -- the
        map only stops updating a cell once it enters the 0.6 m depth dead
        zone, by which point the obstacle is already recorded.

        OCCLUSION PADDING: FILLING IN THE BODY BEHIND THE VISIBLE FACE
        --------------------------------------------------------------
        From any single viewpoint the camera only sees the obstacle's NEAR
        face; the body behind it is occluded and would stay unknown -- and
        A* is allowed to plan through unknown cells, i.e. straight through
        the object.  So each hit is extended CAMERA_OBSTACLE_DEPTH_PAD_M
        further along its ray, marking the occluded cells occupied too.
        The padding is normal log-odds evidence: a later viewpoint whose
        rays pass through those cells to a farther hit erodes it again.

        Args:
            x, y, theta : robot pose, same convention as integrate_scan().
            ranges      : 1-D array of distances (m) from the camera to each
                           detected obstacle point.
            bearings    : 1-D array of matching bearings (rad, robot frame,
                           positive = left, same as robot.bearings).
        """
        # observed=None: this map must not touch the lidar's frontier "observed"
        # bookkeeping -- it lives entirely in camera_obstacle_log.
        self._integrate_camera_rays(
            x, y, theta, ranges, bearings,
            log=self.camera_obstacle_log, observed=None,
            pad_m=C.CAMERA_OBSTACLE_DEPTH_PAD_M,
        )

    def _integrate_camera_rays(self, x, y, theta, ranges, bearings, log, observed,
                               pad_m=0.0):
        """Shared ray-casting core for camera-origin (range, bearing) scans.

        Places the sensor origin at the camera's mount offset (the same
        CAMERA_FORWARD_M / CAMERA_LATERAL_M constants floor_hazard.py and
        colored_objects.py use).  FREE cells along each ray are marked
        per-ray (redundant free evidence is harmless).  OCCUPIED cells (the
        hit itself, plus the occlusion-padding stretch behind it if
        pad_m > 0) are collected across ALL rays and applied ONCE PER CELL
        at the end -- see _add_logodds()'s docstring for why: many candidate
        points from ONE frame can converge on the SAME cell at close range,
        and applying L_OCC once per point (instead of once per cell) can
        slam a cell to the clamp ceiling in a single frame, making it
        effectively impossible to erase later.
        """
        if len(ranges) == 0:
            return

        sx = x + C.CAMERA_FORWARD_M * np.cos(theta) - C.CAMERA_LATERAL_M * np.sin(theta)
        sy = y + C.CAMERA_FORWARD_M * np.sin(theta) + C.CAMERA_LATERAL_M * np.cos(theta)
        c0, r0 = self.world_to_grid(sx, sy)

        world_ang = theta + np.asarray(bearings)
        cos_a = np.cos(world_ang)
        sin_a = np.sin(world_ang)

        occ_rows, occ_cols = [], []
        for i in range(len(ranges)):
            ex = sx + ranges[i] * cos_a[i]
            ey = sy + ranges[i] * sin_a[i]
            c1, r1 = self.world_to_grid(ex, ey)
            self._mark_free_along_ray(c0, r0, c1, r1, log, observed)
            if self.in_bounds(c1, r1):
                occ_rows.append(r1)
                occ_cols.append(c1)

            if pad_m <= 0.0:
                continue
            # Occlusion padding: continue the ray behind the visible face --
            # those cells are ALSO occupied evidence, collected the same way.
            ex2 = sx + (ranges[i] + pad_m) * cos_a[i]
            ey2 = sy + (ranges[i] + pad_m) * sin_a[i]
            c2, r2 = self.world_to_grid(ex2, ey2)
            for (c, r) in _bresenham(c1, r1, c2, r2)[1:]:
                if self.in_bounds(c, r):
                    occ_rows.append(r)
                    occ_cols.append(c)

        if occ_rows:
            rows = np.asarray(occ_rows, dtype=np.int64)
            cols = np.asarray(occ_cols, dtype=np.int64)
            flat = np.unique(rows * self.ncols + cols)
            rr = (flat // self.ncols).astype(np.int32)
            cc = (flat %  self.ncols).astype(np.int32)
            log[rr, cc] = np.minimum(log[rr, cc] + C.L_OCC, C.L_CLAMP)
            if observed is not None:
                observed[rr, cc] = True

    def _mark_free_along_ray(self, c0, r0, c1, r1, log, observed):
        """Mark cells from (c0,r0) up to (but excluding) (c1,r1) as FREE.

        Split out of _ray() so _integrate_camera_rays() can de-duplicate the
        OCCUPIED end-cell across all rays separately (see that method's
        docstring) while still marking free space per-ray as usual.
        """
        cells = _bresenham(c0, r0, c1, r1)
        for (c, r) in cells[:-1]:
            if self.in_bounds(c, r):
                log[r, c] = max(log[r, c] + C.L_FREE, -C.L_CLAMP)
                if observed is not None:
                    observed[r, c] = True

    def _ray(self, c0, r0, c1, r1, hit, log=None, observed=None):
        """Update grid cells along a single laser ray.

        All cells from (c0,r0) to the cell BEFORE the end are marked FREE.
        The end cell (c1,r1) is marked OCCUPIED if `hit` is True,
        or FREE if the beam did not hit anything.

        Args:
            c0, r0   : Start cell (sensor position).
            c1, r1   : End cell (hit or max range point).
            hit      : True if the ray struck an obstacle at (c1, r1).
            log      : log-odds array to update (default: the lidar map self.log).
                        Pass self.camera_obstacle_log to build the SEPARATE
                        camera-only obstacle map (see integrate_camera_obstacle).
            observed : "ever seen" array to update, or None to leave it alone.
                        The camera-obstacle map passes None so it does NOT touch
                        the lidar's frontier bookkeeping.
        """
        if log is None:
            log = self.log
        cells = _bresenham(c0, r0, c1, r1)
        if not cells:
            return

        # Mark all intermediate cells (between sensor and hit) as FREE.
        for (c, r) in cells[:-1]:
            if self.in_bounds(c, r):
                # Subtract L_FREE (making cell more likely free), then clamp.
                log[r, c] = max(log[r, c] + C.L_FREE, -C.L_CLAMP)
                if observed is not None:
                    observed[r, c] = True

        # Update the end cell — always a confirmed hit (inf and max-range rays
        # are skipped in integrate_scan before _ray is called).
        c, r = cells[-1]
        if self.in_bounds(c, r):
            log[r, c] = min(log[r, c] + C.L_OCC, C.L_CLAMP)
            if observed is not None:
                observed[r, c] = True

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



# ============================================================================
# 4-connected bounded flood fill
# ============================================================================
def flood_fill_4connected(seed_mask, allowed_mask, radius_cells):
    """Grow `seed_mask` outward by at most `radius_cells` 4-connected hops
    (up/down/left/right only -- no diagonals), stepping ONLY through cells
    where `allowed_mask` is True.

    Used by OccupancyGrid.reconciled_object_mask() to recover a tracked
    object's full occ_mask() footprint even when the camera only coloured
    part of it, without also reaching a nearby but physically SEPARATE
    wall (see that method's docstring for the full story).

    WHY GATE EVERY STEP ON allowed_mask -- NOT PLAIN DILATION
    -------------------------------------------------------------
    A plain bounded dilation (grow outward by a fixed radius regardless of
    what the intervening cells are) can "jump" across a gap of free space:
    if a wall happens to sit within `radius_cells` of the seed but is NOT
    physically touching it, dilation still reaches it once the radius is
    large enough, and the caller ends up believing free-standing wall
    cells are part of the object. Gating every hop on allowed_mask closes
    that hole: growth can only follow an UNBROKEN chain of allowed cells,
    so a genuinely disconnected wall across even one cell of free space
    can never be reached, no matter how large radius_cells is set. This
    is what makes it safe to use a generous radius (to recover the whole
    object) without it ever bleeding into unrelated nearby walls.

    WHY STILL BOUNDED, NOT AN OPEN-ENDED FLOOD FILL
    -----------------------------------------------------
    Maze walls are typically ONE single connected network, and (unlike a
    disconnected wall) the object's own surface IS genuinely connected to
    it wherever the object sits flush against a wall. An unbounded flood
    fill from the seed would then follow that connection and eventually
    swallow the entire wall structure. The radius cap keeps growth limited
    to "genuinely close to the seed", same intent as the old fixed-radius
    dilation this replaces -- just connectivity-aware now.

    Args:
        seed_mask    : boolean array, cells to start growing from.
        allowed_mask : boolean array, cells growth may pass through
                        (typically seed_mask itself unioned with occ_mask()
                        so the seed can always act as its own bridge).
        radius_cells : maximum number of 4-connected hops from the seed.

    Returns:
        np.ndarray bool -- seed_mask plus every allowed cell reachable
        from it within radius_cells hops.
    """
    # Normalise radius to a safe, non-negative integer hop count.
    radius_cells = int(radius_cells)
    if radius_cells < 0:
        radius_cells = 0

    out = seed_mask & allowed_mask
    if radius_cells <= 0 or not out.any():
        return out
    frontier = out.copy()
    for _ in range(radius_cells):
        grown = np.zeros_like(frontier)
        grown[:-1, :] |= frontier[1:,  :]
        grown[1:,  :] |= frontier[:-1, :]
        grown[:, :-1] |= frontier[:,  1:]
        grown[:,  1:] |= frontier[:, :-1]
        grown &= allowed_mask
        new_cells = grown & ~out
        if not new_cells.any():
            break
        out |= new_cells
        frontier = new_cells
    return out
