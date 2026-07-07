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
reads its position straight from the occupancy grid's self-correcting object
mask (via update_from_grid), so noise and false positives are smoothed and
un-done by the same log-odds machinery the lidar wall map uses.  It gives the
rest of the code (explorer.py, the "go fetch the object" mission phase) one
simple place to ask "where is the blue object, and can I get there?" instead
of re-deriving that from the grid every time.

HOW BLUE/YELLOW OBJECTS BECOME OBSTACLES
----------------------------------------------
Like the lidar wall map: every camera frame is folded into a per-colour
log-odds grid via OccupancyGrid.update_object_observation(colour, hits, frees),
which PathPlanner.build_nav_grid() thresholds and folds into `blocked`
(inflated, same safety margin as walls).  A*, the flood fill, and frontier
detection therefore automatically route the robot around a tracked object --
"we cannot drive through them" falls out of the existing pipeline for free.
Because it is log-odds (not a sticky mask), a false detection is ERASED once
the camera looks at that spot again and no longer sees the colour.

THE GROUND-PLANE FILTER (INVERSE of floor_hazard.py's)
--------------------------------------------------------
floor_hazard.py only wants points that lie FLAT ON THE FLOOR (green tiles).
Blue/yellow objects are the opposite: upright items (boxes, pillars, etc.)
that can appear at any height in the image -- so we scan the FULL frame,
but a pixel only counts as a HIT if it sits ABOVE the ground plane.
Floor-height pixels can never be object hits, only free evidence: under
some lighting the grey floor renders slightly blue-ish and would otherwise
pass the HSV test on every frame, turning the entire explored floor into
"blue object" and blocking the whole map for the planner.

THE NEAR-FIELD FALLBACK (LIDAR FILLS THE DEPTH CAMERA'S BLIND ZONE)
------------------------------------------------------------------------
The depth camera cannot measure anything closer than its minimum range
(~0.6 m).  When the robot gets close to the tracked object, the OBJECT'S
OWN surface is exactly what falls inside that dead zone -- its pixels
return inf and used to be silently dropped, leaving only the wall around
or behind it to be classified.  See detect()'s Step 1b: for those inf
pixels we look up the LIDAR's range at the same bearing (the lidar has no
such dead zone) and, if it confirms something is genuinely that close,
substitute it as the pixel's depth before running the normal registration
and colour test.  This recovers the object's true near-field pixels
instead of only ever seeing the wall behind it.
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

    One instance of this class exists per colour.
    """

    def __init__(self, color_name):
        self.color_name = color_name

        self.world_xy   = None   # (x, y) best-estimate position, or None if never seen
        self.seen        = False # True as soon as we detect this colour at all
        self.reachable   = False  # None = not checked yet; True/False once checked
        self.reached     = False # True once the robot has gotten close to it
        self.was_reachable = False # True once the robot has gotten close to it

        self.last_seen_time  = None  # simulation time (s) of the most recent detection

    # ------------------------------------------------------------------ #
    def update_from_grid(self, grid, now):
        """Refresh this object's position estimate from the occupancy grid's
        self-correcting object mask.

        The grid's per-colour log-odds map is the single source of truth for
        WHERE the object is: it gains positive evidence from matching pixels
        and negative evidence from visible non-matching pixels every frame, so
        a false detection decays away on its own (see
        OccupancyGrid.update_object_observation).  We therefore take the
        centroid of the currently-believed cells rather than keeping a
        separate running average that could never forget a bad point.

        `seen`/`reachable`/`world_xy` all follow the grid: if every cell has
        decayed back below threshold the object is no longer believed present,
        so world_xy becomes None and both flags reset -- which correctly stops
        the mission FSM from chasing (or dereferencing) a target that turned
        out to be a false positive.  `reached` stays sticky: once the robot
        has physically arrived, that fact is real regardless of later camera
        noise.

        Args:
            grid : OccupancyGrid, queried via object_centroid(color).
            now  : current simulation time (s).
        """
        centroid = grid.object_centroid(self.color_name)
        self.world_xy = centroid
        if centroid is None:
            # No cell is believed to hold this colour any more -> forget it.
            self.seen      = False
            self.reachable = False
        else:
            self.seen           = True
            self.last_seen_time = now




    # ------------------------------------------------------------------ #
    def update_reachable(self, reachable_mask, grid, tol=None):
        """Refresh the `reachable` flag using the planner's flood-fill result.

        `reachable_mask` is the SAME boolean array the frontier exploration
        pipeline already computes every planning cycle (PathPlanner.build_nav_grid's
        `reachable` output, cached as Explorer._reachable_cache) -- checking it
        is essentially free, no extra A* call needed.

        WHY A NEIGHBOURHOOD, NOT JUST THE OBJECT'S OWN CELL
        --------------------------------------------------
        An object usually sits against a wall, and the wall's inflation safety
        margin seals off the object's own cell from the reachable set -- so a
        single-cell lookup would call it "unreachable" even though the robot
        can safely park right next to it and be within OBJECT_REACH_TOL.  We
        therefore call it reachable if ANY safely-reachable cell lies within
        `tol` of the object.  That is exactly the spot GoToPoint parks at via
        planner.plan_path_near_blocked_target().

        Args:
            reachable_mask : bool ndarray, True where the robot can currently
                              reach that cell (flood fill from robot position).
            grid            : OccupancyGrid, for world_to_grid conversion.
            tol            : reach radius (m); defaults to OBJECT_REACH_TOL.
        """
        if self.world_xy is None or reachable_mask is None:
            return
        tol = C.OBJECT_REACH_TOL if tol is None else tol
        col, row = grid.world_to_grid(*self.world_xy)
        rad = max(1, int(round(tol / grid.res)))

        r0 = max(0, row - rad)
        r1 = min(reachable_mask.shape[0], row + rad + 1)
        c0 = max(0, col - rad)
        c1 = min(reachable_mask.shape[1], col + rad + 1)
        if r0 >= r1 or c0 >= c1:
            self.reachable = False   # object centre is off the grid entirely
            return
        self.reachable = bool(reachable_mask[r0:r1, c0:c1].any())

        if self.reachable and self.was_reachable is False:
            print(f"TrackedObject.update_reachable: {self.color_name} is reachable at {self.world_xy}")
            self.was_reachable = True



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
        return ("TrackedObject(%s: seen=%s pos=%s reachable=%s reached=%s)"
                % (self.color_name, self.seen, pos, self.reachable,
                   self.reached))


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
            rgb_img   : (H, W, 3) uint8 array from Robot.read_camera_rgb().
            depth_img : (H, W) float32 array (metres) from Robot.read_camera_depth().
            pose      : (x, y, theta) robot pose from Odometry.
            lidar_ranges   : optional 1-D array of the SAME control step's
                              lidar ranges (robot.read_lidar()).  Used ONLY as
                              a fallback for pixels the depth camera cannot
                              measure because the surface is closer than its
                              minimum range (see CAMERA_NEAR_FALLBACK_SLACK in
                              config.py -- the "wall gets marked instead of
                              the object" fix).  If omitted, these pixels are
                              simply dropped, same as before.
            lidar_bearings : matching 1-D array of bearings (rad, robot frame,
                              positive = left) for lidar_ranges -- robot.bearings.

        Returns:
            dict {colour: (hit_xs, hit_ys, free_xs, free_ys)} for each of
            "blue" and "yellow".  `hit_*` are the world coords of pixels that
            MATCHED the colour; `free_*` are the world coords of pixels that
            had valid depth (so were genuinely observed) but did NOT match --
            the negative evidence that lets a past false positive decay away.
            Any of the arrays may be empty.
        """
        empty = {"blue":   (np.empty(0), np.empty(0), np.empty(0), np.empty(0)),
                  "yellow": (np.empty(0), np.empty(0), np.empty(0), np.empty(0))}

        us, vs = self._us, self._vs

        # ---- Step 1: read depth at the sampled pixels ----------------------
        depth = depth_img[vs.astype(np.int32), us.astype(np.int32)]
        valid = (
            np.isfinite(depth)
            & (depth >= self.intr.min_range)
            & (depth <= self.intr.max_range)
        )

        # ---- Step 1b: near-field fallback -- recover pixels the depth ------
        # camera cannot see because the surface is CLOSER than its minimum
        # range (see config.py CAMERA_NEAR_FALLBACK_SLACK for the full story).
        # Without this, getting close to the object makes its own pixels
        # vanish entirely, leaving only the wall around/behind it to be
        # (mis-)classified.
        us_v, vs_v, depth_v = us[valid], vs[valid], depth[valid]
        if lidar_ranges is not None and lidar_bearings is not None and len(lidar_bearings) > 0:
            too_close = (~valid) & (~np.isfinite(depth) | (depth < self.intr.min_range))
            if np.any(too_close):
                us_tc, vs_tc = us[too_close], vs[too_close]
                # Bearing of each pixel is INDEPENDENT of depth (the unknown Z
                # cancels): tan(bearing) = -(u - cx) / f, same convention as
                # robot.bearings (positive = left).
                bearing_cam = np.arctan2(-(us_tc - self.intr.cx), self.intr.f)

                # Nearest lidar ray to each pixel's bearing (vectorised
                # nearest-neighbour search via a single sort + searchsorted).
                order    = np.argsort(lidar_bearings)
                sorted_b = np.asarray(lidar_bearings)[order]
                idx_s    = np.clip(np.searchsorted(sorted_b, bearing_cam), 1, len(sorted_b) - 1)
                left, right = idx_s - 1, idx_s
                use_left = np.abs(sorted_b[left] - bearing_cam) <= np.abs(sorted_b[right] - bearing_cam)
                nearest  = order[np.where(use_left, left, right)]
                r_fb     = np.asarray(lidar_ranges)[nearest]

                # Only trust the substitution where the lidar independently
                # confirms something genuinely sits inside the camera's blind
                # zone in that direction -- not just "there is SOME wall
                # somewhere in roughly that direction".
                confirmed = (
                    np.isfinite(r_fb)
                    & (r_fb >= C.LIDAR_MIN_RANGE)
                    & (r_fb <= self.intr.min_range * C.CAMERA_NEAR_FALLBACK_SLACK)
                )
                if np.any(confirmed):
                    us_v    = np.concatenate([us_v,    us_tc[confirmed]])
                    vs_v    = np.concatenate([vs_v,    vs_tc[confirmed]])
                    depth_v = np.concatenate([depth_v, r_fb[confirmed]])

        if len(us_v) == 0:
            return empty

        us, vs, depth = us_v, vs_v, depth_v

        # ---- Step 2: back-project + Step 3: register & sample colour -------
        x_cam, y_cam, z_cam = back_project(us, vs, depth, self.intr)
        r, g, b = register_and_sample_rgb(
            x_cam, y_cam, z_cam, self.intr, C.CAMERA_RGB_DEPTH_BASELINE_M, rgb_img
        )
        hue, sat, val = rgb_to_hsv(r, g, b)

        # ---- Step 4: world (x, y) for every candidate point -----------------
        # (computed once, then filtered per colour below)
        forward_lvl, drop_lvl = tilt_correct(y_cam, z_cam, C.CAMERA_TILT_RAD)
        right_lvl = x_cam
        world_xs, world_ys = camera_local_to_world(
            forward_lvl, right_lvl, pose, C.CAMERA_FORWARD_M, C.CAMERA_LATERAL_M
        )

        # ---- Step 5: hits must be ABOVE the floor plane ----------------------
        # The tracked objects are upright items; the floor is at height ~0.
        # Under some lighting the grey floor renders slightly BLUE-ish and can
        # sneak past the HSV test -- and because such pixels would count as
        # "hits", no free evidence would ever erode them: the whole explored
        # floor turns into "blue object" and the planner blocks the entire
        # map (A* can no longer reach anything).  Requiring hits to sit above
        # the ground plane kills that failure mode regardless of lighting:
        # floor-height pixels can only ever be FREE evidence.
        height_above_ground = C.CAMERA_HEIGHT_M - drop_lvl
        above_floor = height_above_ground > C.GROUND_PLANE_TOL_M

        result = {}
        for name, (h_min, h_max, s_min, v_min) in (
            ("blue",   (C.BLUE_HUE_MIN,   C.BLUE_HUE_MAX,   C.BLUE_SAT_MIN,   C.BLUE_VAL_MIN)),
            ("yellow", (C.YELLOW_HUE_MIN, C.YELLOW_HUE_MAX, C.YELLOW_SAT_MIN, C.YELLOW_VAL_MIN)),
        ):
            match = ((hue >= h_min) & (hue <= h_max)
                     & (sat >= s_min) & (val >= v_min)
                     & above_floor)
            # Every valid pixel is either a "hit" (this colour, above the
            # floor) or a "free" observation (visible, but not the object --
            # including any colour-matching pixels ON the floor).  Both are
            # returned: OccupancyGrid.update_object_observation() uses hits
            # as positive evidence and free points as negative evidence.
            result[name] = (
                world_xs[match],  world_ys[match],
                world_xs[~match], world_ys[~match],
            )

        return result
