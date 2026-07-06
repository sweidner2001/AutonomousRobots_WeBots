"""
colored_objects.py  --  Tracking blue/yellow target objects with the RGB-D camera.
======================================================================================

WHAT DOES THIS FILE DO?
-------------------------
Somewhere in the maze there is a BLUE object and a YELLOW object the robot
must find.  This file has two jobs:

  1. ColorObjectDetector -- scans the RGB-D camera every few control steps
     for blue/yellow pixels and converts them to world (x, y) points, using
     the exact same RGB-D registration pipeline as floor_hazard.py (see that
     file's module docstring for why registration is necessary at all).

  2. TrackedObject -- a small state-holder class, ONE INSTANCE PER COLOUR,
     that remembers everything we currently know about that object:
       - world_xy   : our best estimate of where it is (or None if never seen)
       - seen       : have we ever detected this colour at all?
       - reachable  : can the planner currently find a path to it?
                      (None = not yet checked, True/False once it has been)
       - reached    : has the robot's own position come within OBJECT_REACH_TOL
                      of the object at some point?

WHY A SEPARATE CLASS FOR TRACKING STATE?
--------------------------------------------
Detection happens every few camera frames and is noisy (a single frame might
see the object from a slightly different angle, or not at all).  TrackedObject
smooths this out with a running average of all detections so far, and gives
the rest of the code (explorer.py, a future "go fetch the object" mission
phase) one simple place to ask "where is the blue object, and can I get
there?" instead of re-deriving that from raw detections every time.

HOW BLUE/YELLOW OBJECTS BECOME OBSTACLES
----------------------------------------------
Exactly like the green floor hazards: every detected point is stamped into
OccupancyGrid.mark_object_world(colour, xs, ys), which PathPlanner.build_nav_grid()
folds into `blocked` (inflated, same safety margin as walls).  A*, the flood
fill, and frontier detection therefore automatically route the robot around a
tracked object -- "we cannot drive through them" falls out of the existing
pipeline for free.

WHY NO GROUND-PLANE FILTER (UNLIKE floor_hazard.py)?
--------------------------------------------------------
floor_hazard.py only wants points that lie FLAT ON THE FLOOR (green tiles).
Blue/yellow objects are assumed to be upright items (boxes, pillars, etc.)
that can appear at any height in the image, so we scan the FULL frame
instead of only the rows below the horizon.

WHY CROSS-CHECK AGAINST THE LIDAR SCAN?
--------------------------------------------
The camera and the lidar are two INDEPENDENT sensors, each with its own
small geometric error (mount height/tilt/offset for the camera -- see
config.py's long comment on what can and can't be read live from Webots).
Occasionally the camera's computed position for a coloured point ends up
slightly FARTHER than a wall the lidar has already mapped in roughly the
same direction.  That is a physically impossible scene: a depth camera
(just like a lidar) can only ever measure the NEAREST surface along a
given ray -- nothing can be optically detected behind a solid wall.

When this mismatch happens, it is essentially always the camera's multi-
step pipeline (back-projection + registration + tilt-correction, each with
its own small calibration constant) that is slightly off, not the lidar's
single, direct range reading.  ColorObjectDetector._clamp_to_lidar() fixes
this by pulling the point inward along its own viewing direction until it
sits exactly at the lidar-confirmed distance -- i.e. the detected object
gets "snapped" onto the wall the lidar already found there, rather than
floating on the far side of it.  Points where the lidar sees nothing in
that direction (inf, out of range, or just outside the lidar's FoV) are
left untouched, since the camera may be seeing something above or below
the lidar's flat 2-D scanning plane that the lidar genuinely cannot detect.

ONLY TRUST SMALL CORRECTIONS -- DISCARD THE REST
------------------------------------------------------
A "nearby" lidar ray within the angular tolerance is not guaranteed to be
looking at the SAME physical surface as the camera point -- it could be a
coincidental angular alignment with a completely different, unrelated wall,
or genuine sensor noise.  The size of the gap is the tell: a small gap
(a few centimetres) is consistent with ordinary camera calibration drift,
so it's safe to correct.  A LARGE gap (LIDAR_CROSS_CHECK_MAX_GAP or more)
means the two readings almost certainly aren't describing the same point at
all -- forcibly relocating the camera point onto that lidar reading would
just replace one wrong position with a different, unrelated wrong position.
In that case the point is DISCARDED from this frame entirely instead of
being clamped.
"""

import math

import numpy as np

import Maze4.controllers.Controller_v1.config as C
from Maze4.controllers.Controller_v1.camera_geometry import (
    CameraIntrinsics, sample_pixel_grid, back_project,
    register_and_sample_rgb, rgb_to_hsv, tilt_correct, camera_local_to_world,
)


# ============================================================================
# TrackedObject -- one instance per colour (blue, yellow)
# ============================================================================
class TrackedObject:
    """Remembers everything currently known about one coloured target object.

    One instance of this class exists per colour (see MazeExplorer.__init__:
    self.blue_object = TrackedObject("blue"), self.yellow_object = TrackedObject("yellow")).
    """

    def __init__(self, color_name):
        self.color_name = color_name

        self.world_xy   = None   # (x, y) best-estimate position, or None if never seen
        self.seen        = False # True as soon as we detect this colour at all
        self.reachable   = False  # None = not checked yet; True/False once checked
        self.reached     = False # True once the robot has gotten close to it

        self.num_detections = 0    # how many individual points have contributed
        self.last_seen_time  = None  # simulation time (s) of the most recent detection

    # ------------------------------------------------------------------ #
    def filter_consistent(self, xs, ys, tol=None):
        """Reject a detection batch whose centroid is too far from the
        CURRENTLY ESTABLISHED position -- almost certainly odometry drift
        or a misdetection, not a genuine re-observation of the same object.

        WHY THIS IS NEEDED
        --------------------
        Every detection frame's points get stamped directly into the
        occupancy grid via OccupancyGrid.mark_object_world(), which is
        STICKY (never un-marks a cell -- see that method's docstring).
        Odometry here is pure dead-reckoning with no absolute correction,
        so over a long exploration run the robot's pose estimate drifts.
        Re-observing the SAME physical object later projects it through a
        more-drifted pose than earlier observations did, even though the
        real object hasn't moved -- and because marking never forgets,
        every one of those drifting positions accumulates permanently,
        smearing the marked footprint across an ever-growing area.

        Calling this BEFORE grid.mark_object_world() / update_detection()
        keeps the marked footprint bounded to roughly the object's real
        size: once a position is established, only detections that agree
        with it (within `tol`) are trusted going forward.

        On the VERY FIRST detection (world_xy is still None) everything is
        accepted -- there is nothing yet to compare against.

        Args:
            xs, ys : 1-D arrays of world coordinates detected THIS frame.
            tol    : consistency tolerance (m); defaults to
                      config.OBJECT_CONSISTENCY_TOL.

        Returns:
            (xs, ys) -- unchanged if consistent (or this is the first-ever
            detection), otherwise two empty arrays (batch rejected).
        """
        if len(xs) == 0 or self.world_xy is None:
            return xs, ys
        tol = C.OBJECT_CONSISTENCY_TOL if tol is None else tol
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        dist = math.hypot(cx - self.world_xy[0], cy - self.world_xy[1])
        if dist > tol:
            return np.empty(0), np.empty(0)
        return xs, ys

    # ------------------------------------------------------------------ #
    def update_detection(self, xs, ys, now):
        """Fold a new batch of detected world points into the running estimate.

        Uses a running average weighted by how many points have already
        contributed, so a single noisy frame cannot suddenly yank the
        estimated position -- each new frame's centroid nudges the estimate
        rather than replacing it outright.

        Args:
            xs, ys : 1-D NumPy arrays of world coordinates detected THIS frame
                     (may be empty -- a normal "not visible this frame" case).
            now    : current simulation time (s).
        """
        n_new = len(xs)
        if n_new == 0:
            return

        cx, cy = float(np.mean(xs)), float(np.mean(ys))

        if self.world_xy is None:
            self.world_xy = (cx, cy)
        else:
            # Weighted running average: old estimate keeps its accumulated
            # weight (num_detections), the new centroid counts as n_new points.
            n_old = self.num_detections
            total = n_old + n_new
            ox, oy = self.world_xy
            self.world_xy = (
                (ox * n_old + cx * n_new) / total,
                (oy * n_old + cy * n_new) / total,
            )

        self.num_detections += n_new
        self.seen             = True
        self.last_seen_time   = now




    # ------------------------------------------------------------------ #
    def update_reachable(self, reachable_mask, grid):
        """Refresh the `reachable` flag using the planner's flood-fill result.

        `reachable_mask` is the SAME boolean array the frontier exploration
        pipeline already computes every planning cycle (PathPlanner.build_nav_grid's
        `reachable` output, cached as Explorer._reachable_cache) -- checking a
        single cell in it is essentially free, no extra A* call needed.

        Args:
            reachable_mask : bool ndarray, True where the robot can currently
                              reach that cell (flood fill from robot position).
            grid            : OccupancyGrid, for world_to_grid conversion.
        """
        if self.world_xy is None or reachable_mask is None:
            return
        col, row = grid.world_to_grid(*self.world_xy)
        if 0 <= row < reachable_mask.shape[0] and 0 <= col < reachable_mask.shape[1]:
            self.reachable = bool(reachable_mask[row, col])



    # ------------------------------------------------------------------ #
    def update_reached(self, robot_xy, tol=None):
        """Mark this object as reached if the robot is currently close enough.

        Once reached, stays reached (sticky) -- matches the same
        "never forget a confirmed fact" philosophy as the hazard grid.

        Args:
            robot_xy : (x, y) current robot position.
            tol      : distance tolerance (m); defaults to config.OBJECT_REACH_TOL.

        Returns:
            True if this call is what newly marked it as reached (useful for
            one-shot "we just arrived!" logging), False otherwise.
        """
        if self.reached or self.world_xy is None:
            return False
        tol = C.OBJECT_REACH_TOL if tol is None else tol
        dist = math.hypot(robot_xy[0] - self.world_xy[0],
                           robot_xy[1] - self.world_xy[1])
        if dist <= tol:
            self.reached = True
            return True
        return False

    # ------------------------------------------------------------------ #
    def __repr__(self):
        pos = "None" if self.world_xy is None else (
            "(%.2f, %.2f)" % self.world_xy
        )
        return ("TrackedObject(%s: seen=%s pos=%s reachable=%s reached=%s n=%d)"
                % (self.color_name, self.seen, pos, self.reachable,
                   self.reached, self.num_detections))


# ============================================================================
# ColorObjectDetector -- finds blue/yellow pixels in one RGB-D frame
# ============================================================================
class ColorObjectDetector:
    """Finds blue and yellow pixels in one RGB-D frame and returns their
    world-frame (x, y) positions, one point cloud per colour.

    Runs the registration pipeline ONCE per frame and tests both colours
    against the same registered pixels -- cheaper than running two
    completely separate detectors.
    """

    def __init__(self, camera_width, camera_height, camera_fov,
                 depth_min_range, depth_max_range):
        """See FloorHazardDetector.__init__ -- identical intrinsics, all
        read live from the Webots device in robot.py."""
        self.intr = CameraIntrinsics(
            camera_width, camera_height, camera_fov,
            depth_min_range, depth_max_range,
        )

        # Unlike floor_hazard.py we scan the FULL frame: a standing object
        # can appear anywhere vertically in the image, not just below the
        # horizon (see module docstring).
        self._us, self._vs = sample_pixel_grid(self.intr, C.CAMERA_SAMPLE_STRIDE)

    # ------------------------------------------------------------------ #
    def detect(self, rgb_img, depth_img, pose, lidar_ranges=None, lidar_bearings=None):
        """Return world-frame points for each tracked colour this frame.

        Args:
            rgb_img        : (H, W, 3) uint8 array from Robot.read_camera_rgb().
            depth_img      : (H, W) float32 array (m) from Robot.read_camera_depth().
            pose           : (x, y, theta) robot pose from Odometry.
            lidar_ranges   : optional, the SAME control step's lidar range
                              array (Robot.read_lidar()).  When given, used
                              to sanity-check camera points against the
                              lidar scan -- see module docstring
                              ("WHY CROSS-CHECK AGAINST THE LIDAR SCAN?").
            lidar_bearings : optional, matching per-ray bearing array
                              (Robot.bearings). Required together with
                              lidar_ranges for the cross-check to run.

        Returns:
            dict {"blue": (xs, ys), "yellow": (xs, ys)} -- world coordinates
            of every matching point this frame (arrays may be empty).
        """
        empty = {"blue": (np.empty(0), np.empty(0)),
                  "yellow": (np.empty(0), np.empty(0))}

        us, vs = self._us, self._vs

        # ---- Step 1: read depth at the sampled pixels ----------------------
        depth = depth_img[vs.astype(np.int32), us.astype(np.int32)]
        valid = (
            np.isfinite(depth)
            & (depth >= self.intr.min_range)
            & (depth <= self.intr.max_range)
        )
        if not np.any(valid):
            return empty

        us, vs, depth = us[valid], vs[valid], depth[valid]

        # ---- Step 2: back-project + Step 3: register & sample colour -------
        x_cam, y_cam, z_cam = back_project(us, vs, depth, self.intr)
        r, g, b = register_and_sample_rgb(
            x_cam, y_cam, z_cam, self.intr, C.CAMERA_RGB_DEPTH_BASELINE_M, rgb_img
        )
        hue, sat, val = rgb_to_hsv(r, g, b)

        # ---- Step 4: world (x, y) for every candidate point -----------------
        # (computed once, then filtered per colour below)
        forward_lvl, _drop_lvl = tilt_correct(y_cam, z_cam, C.CAMERA_TILT_RAD)
        right_lvl = x_cam

        # ---- Step 4b: cross-check against the lidar scan --------------------
        # Pull in any point that claims to be behind a lidar-confirmed wall
        # in roughly the same direction -- see module docstring.  Points
        # whose gap to the lidar is too large to trust are dropped entirely.
        if lidar_ranges is not None and lidar_bearings is not None:
            forward_lvl, right_lvl, keep = self._clamp_to_lidar(
                forward_lvl, right_lvl, lidar_ranges, lidar_bearings
            )
            if not np.any(keep):
                return empty
            forward_lvl, right_lvl = forward_lvl[keep], right_lvl[keep]
            hue, sat, val = hue[keep], sat[keep], val[keep]

        world_xs, world_ys = camera_local_to_world(
            forward_lvl, right_lvl, pose, C.CAMERA_FORWARD_M, C.CAMERA_LATERAL_M
        )

        result = {}
        for name, (h_min, h_max, s_min, v_min) in (
            ("blue",   (C.BLUE_HUE_MIN,   C.BLUE_HUE_MAX,   C.BLUE_SAT_MIN,   C.BLUE_VAL_MIN)),
            ("yellow", (C.YELLOW_HUE_MIN, C.YELLOW_HUE_MAX, C.YELLOW_SAT_MIN, C.YELLOW_VAL_MIN)),
        ):
            match = (hue >= h_min) & (hue <= h_max) & (sat >= s_min) & (val >= v_min)
            if np.any(match):
                result[name] = (world_xs[match], world_ys[match])
            else:
                result[name] = (np.empty(0), np.empty(0))

        return result

    # ------------------------------------------------------------------ #
    # Lidar cross-check
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clamp_to_lidar(forward_lvl, right_lvl, lidar_ranges, lidar_bearings):
        """Pull camera points inward so they never claim to be BEHIND a wall
        the lidar has already confirmed in the same direction -- or, if the
        mismatch is too large to trust, discard the point entirely.

        See the module docstring ("WHY CROSS-CHECK AGAINST THE LIDAR SCAN?"
        and "ONLY TRUST SMALL CORRECTIONS -- DISCARD THE REST") for the full
        reasoning.  In short, for each candidate point:
          1. Find the nearest-bearing lidar ray(s) within LIDAR_CROSS_CHECK_ANGLE.
          2. If the lidar saw something clearly CLOSER (by more than
             LIDAR_CROSS_CHECK_MARGIN) than the camera's computed range:
               a. If the gap is small (<= LIDAR_CROSS_CHECK_MAX_GAP), trust
                  it -- rescale the point so its range exactly matches the
                  lidar's (ordinary calibration drift, safe to correct).
               b. If the gap is large, DISCARD the point -- the "nearby"
                  lidar ray is probably a different, unrelated surface, not
                  the same one the camera is looking at.
        Points with no nearby lidar reading at all (blind spot, out of
        range, or genuinely above/below the lidar's flat scan plane) are
        left completely untouched -- the camera may be seeing something the
        lidar simply cannot.

        Args:
            forward_lvl, right_lvl : camera-local (tilt-corrected) forward/
                                       right offsets for each candidate point.
            lidar_ranges            : lidar range array (metres, inf = no hit).
            lidar_bearings          : matching per-ray bearing array (rad),
                                       same sign convention as forward_lvl/
                                       right_lvl (positive right_lvl = to the
                                       robot's right = NEGATIVE bearing).

        Returns:
            (forward_lvl, right_lvl, keep) -- corrected forward/right arrays
            (same shape as input) and a boolean `keep` mask marking which
            points survived (False = discarded as unreliable).
        """
        n = len(forward_lvl)
        keep_all = np.ones(n, dtype=bool)
        if n == 0:
            return forward_lvl, right_lvl, keep_all

        # Range/bearing of each candidate point, ignoring the small camera
        # mount offset (negligible next to typical detection distances) --
        # this only needs to be "roughly in the same direction", not exact.
        body_range   = np.hypot(forward_lvl, right_lvl)
        body_bearing = np.arctan2(-right_lvl, forward_lvl)

        finite = np.isfinite(lidar_ranges)
        if not np.any(finite):
            return forward_lvl, right_lvl, keep_all
        lr = np.asarray(lidar_ranges)[finite]
        lb = np.asarray(lidar_bearings)[finite]

        # For each candidate, find lidar rays within the angular tolerance
        # and take the SHORTEST of their ranges (the nearest confirmed
        # surface in that rough direction).
        diffs  = np.abs(body_bearing[:, None] - lb[None, :])   # (n, m)
        within = diffs <= C.LIDAR_CROSS_CHECK_ANGLE
        has_match = within.any(axis=1)
        if not np.any(has_match):
            return forward_lvl, right_lvl, keep_all

        masked_ranges = np.where(within, lr[None, :], np.inf)
        nearest_lidar_range = masked_ranges.min(axis=1)
        gap = body_range - nearest_lidar_range

        needs_correction = has_match & (gap > C.LIDAR_CROSS_CHECK_MARGIN)
        if not np.any(needs_correction):
            return forward_lvl, right_lvl, keep_all

        trustworthy = gap <= C.LIDAR_CROSS_CHECK_MAX_GAP
        clamp   = needs_correction & trustworthy
        discard = needs_correction & ~trustworthy

        ratio = np.ones(n, dtype=np.float32)
        ratio[clamp] = nearest_lidar_range[clamp] / body_range[clamp]

        keep_all[discard] = False

        return forward_lvl * ratio, right_lvl * ratio, keep_all
