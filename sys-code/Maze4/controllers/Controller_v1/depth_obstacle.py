"""
depth_obstacle.py  --  Finding obstacles the LIDAR can't see, using depth alone.
====================================================================================

WHY DOES THIS FILE EXIST?
----------------------------
The lidar only sweeps ONE fixed horizontal plane at a fixed height above
the floor.  Anything that sits entirely above or below that exact height
-- a low curb, a raised sill, a shelf edge, a thin rail -- is completely
invisible to it: the beam passes straight over or under it, the
occupancy grid shows FREE space there, and the robot would happily drive
straight into something it never "saw".

The RGB-D depth camera looks at a much taller vertical slice of the
world in every single frame, so it CAN catch these lidar-blind
obstacles -- we just need a way to decide "is there something solid
here?" using depth alone, since these obstacles might have no
distinctive colour at all (unlike the green floor markings or the blue/
yellow target objects, which floor_hazard.py and colored_objects.py
already find using colour).

HOW DO WE DETECT "SOMETHING SOLID" WITHOUT COLOUR?
--------------------------------------------------------
The key idea works COLUMN BY COLUMN.  An upright obstacle (even one only
a few centimetres tall) has a vertical face: every point on that face is
at roughly the SAME forward distance from the camera, so scanning DOWN a
single image column across the obstacle gives a run of near-constant
depth.  The floor does the opposite -- its depth changes smoothly row to
row because of perspective -- so a long, near-constant-depth run inside a
column is a reliable "there is an upright surface here" signal.

So for each sampled column we walk down its pixels and group them into
"runs" of near-constant depth (each pixel joins the current run while its
depth stays within DEPTH_OBSTACLE_FLAT_TOL_M of the run's top pixel).  A
run counts as an obstacle only if it is between DEPTH_OBSTACLE_MIN_RUN_PX
and DEPTH_OBSTACLE_MAX_RUN_PX pixels tall:

  - too SHORT  -> depth-image noise, not a real surface.
  - too TALL   -> a full wall / tall object, which the lidar already sees.
  - in between -> exactly the low, lidar-blind obstacle we are after
                  (a few-cm object at 0.5-1.5 m subtends ~20-80 px).

WHY EXCLUDE THE ORDINARY FLOOR?
-------------------------------------
A long constant-depth run is already very unlike the floor (whose depth
keeps changing down the column), but as a cheap safety net we still
back-project each run's representative pixel and reject it if it sits at
floor height (within GROUND_PLANE_TOL_M), exactly as floor_hazard.py
identifies the floor.  Anything else is a genuine raised obstacle.

WHY EXCLUDE "FLYING" SURFACES THE ROBOT FITS UNDER?
--------------------------------------------------------
Some hanging surfaces (e.g. the wooden beams spanning the maze) pass the
run test but are not obstacles at all: their BOTTOM edge is higher than
the robot, so the robot simply drives underneath.  For every run we
back-project its bottom pixel (the surface's lowest visible point) and
compute its height above the ground; if that height clears
ROBOT_CLEARANCE_HEIGHT_M, the run is skipped.

WHY ALSO EXCLUDE THE TRACKED BLUE/YELLOW OBJECT?
------------------------------------------------------
The near-field depth blind zone right at the tracked object's own edge is
exactly where a bad/blended depth reading produces a spurious "flying
wall" (see occupancy_grid.py for the geometric line-of-sight checks that
catch most of that). Colour is a second, independent signal that doesn't
depend on depth at all: if a candidate's registered RGB pixel is blue or
yellow, it IS (or is immediately touching) the tracked object, which
colored_objects.py already tracks properly. detect() drops those
candidates so this colour-blind detector never duplicates or interferes
with that dedicated pipeline.

OUTPUT FORMAT: WHY (range, bearing), NOT WORLD (x, y)?
------------------------------------------------------------
floor_hazard.py and colored_objects.py both return world (x, y) points
directly, because their caller just stamps a single cell per point into
a mask.  Here we want more: intermediate cells along the line from the
camera to each obstacle point should be marked FREE (exactly like a
lidar ray), not just the obstacle cell itself marked OCCUPIED.  Reducing
each candidate point to the SAME (range, bearing) polar representation
robot.bearings already uses lets OccupancyGrid.integrate_camera_obstacle()
reuse the exact same Bresenham ray-tracing + log-odds update
integrate_scan() uses for the lidar -- one shared, well-tested mechanism
for "trace a ray and update the cells along it", regardless of which
sensor the ray came from.
"""

import numpy as np

import Maze4.controllers.Controller_v1.config as C
from Maze4.controllers.Controller_v1.camera_geometry import (
    CameraIntrinsics, sample_pixel_grid, tilt_correct,
    register_and_sample_rgb, rgb_to_hsv,
)


class DepthObstacleDetector:
    """Finds upright, low, lidar-blind obstacle surfaces in one depth frame by
    scanning each image column for a run of near-constant depth, and reduces
    them to (range, bearing) pairs ready for
    OccupancyGrid.integrate_camera_obstacle()."""

    def __init__(self, camera_width, camera_height, camera_fov,
                 depth_min_range, depth_max_range):
        """See FloorHazardDetector.__init__ -- identical intrinsics, all
        read live from the Webots device in robot.py."""
        self.intr = CameraIntrinsics(
            camera_width, camera_height, camera_fov,
            depth_min_range, depth_max_range,
        )

        # Full-frame scan, same subsampling stride as the other detectors --
        # a lidar-blind obstacle could be anywhere vertically in the image.
        # sample_pixel_grid() returns a proper 2-D grid (rows x cols) so we
        # can walk each column top-to-bottom below.
        self._us, self._vs = sample_pixel_grid(self.intr, C.CAMERA_SAMPLE_STRIDE)

    # ------------------------------------------------------------------ #
    def detect(self, depth_img, rgb_img=None):
        """Return (ranges, bearings) for every detected lidar-blind obstacle
        point this frame, in the robot's own frame (same convention as
        robot.bearings: range in metres, bearing in radians, positive =
        left).

        Args:
            depth_img : (H, W) float32 array (metres) from
                         Robot.read_camera_depth().
            rgb_img   : optional (H, W, 3) uint8 array from
                         Robot.read_camera_rgb().  When given, a candidate
                         whose registered RGB pixel matches the tracked
                         blue/yellow HSV bands is dropped -- see the
                         "SAFETY CHECK" note below.

        Returns:
            (ranges, bearings) -- two 1-D NumPy float arrays (possibly
            empty if nothing qualified this frame).
        """
        empty = (np.empty(0), np.empty(0))

        us, vs = self._us, self._vs
        depth = depth_img[vs.astype(np.int32), us.astype(np.int32)]  # 2-D (rows, cols)
        valid = (
            np.isfinite(depth)
            & (depth >= self.intr.min_range)
            & (depth <= self.intr.max_range)
        )

        # ---- Step 1: find a constant-depth vertical run in each column ------
        # For every column we collect ONE representative pixel per qualifying
        # run (the run's middle row, at the run's median depth) PLUS the row
        # of the run's BOTTOM pixel -- needed for the drive-under test below.
        sel_us, sel_vs, sel_depth, sel_v_bottom = [], [], [], []
        n_cols = depth.shape[1]
        for j in range(n_cols):
            for (start, end) in self._column_runs(
                    depth[:, j], valid[:, j], vs[:, j],
                    C.DEPTH_OBSTACLE_FLAT_TOL_M,
                    C.DEPTH_OBSTACLE_MIN_RUN_PX,
                    C.DEPTH_OBSTACLE_MAX_RUN_PX):
                mid = (start + end) // 2
                sel_us.append(us[mid, j])
                sel_vs.append(vs[mid, j])
                sel_depth.append(float(np.median(depth[start:end, j])))
                # Largest row index in the run = lowest point of the surface
                # in the image = physically closest to the floor.
                sel_v_bottom.append(vs[end - 1, j])

        if not sel_us:
            return empty

        us_f       = np.asarray(sel_us,       dtype=np.float32)
        vs_f       = np.asarray(sel_vs,       dtype=np.float32)
        depth_f    = np.asarray(sel_depth,    dtype=np.float32)
        v_bottom_f = np.asarray(sel_v_bottom, dtype=np.float32)

        # ---- Step 2: back-project (pinhole model, same as camera_geometry.py) --
        right_cam = (us_f - self.intr.cx) * depth_f / self.intr.f
        down_cam  = (vs_f - self.intr.cy) * depth_f / self.intr.f
        forward_lvl, drop_lvl = tilt_correct(down_cam, depth_f, C.CAMERA_TILT_RAD)

        # ---- Step 3a: exclude the ordinary floor (safety net) ------------------
        # Same ground-plane test as floor_hazard.py, but INVERTED: we KEEP
        # everything that is NOT at floor height.
        height_above_ground = C.CAMERA_HEIGHT_M - drop_lvl
        not_floor = np.abs(height_above_ground) > C.GROUND_PLANE_TOL_M

        # ---- Step 3b: exclude "flying" surfaces the robot can drive UNDER ------
        # Back-project each run's BOTTOM pixel and compute its height above
        # the ground.  If even the surface's lowest visible point clears the
        # robot (plus margin), it is a hanging beam, not a wall -- skip it.
        down_bottom       = (v_bottom_f - self.intr.cy) * depth_f / self.intr.f
        _, drop_bottom    = tilt_correct(down_bottom, depth_f, C.CAMERA_TILT_RAD)
        bottom_height     = C.CAMERA_HEIGHT_M - drop_bottom
        can_drive_under   = bottom_height > C.ROBOT_CLEARANCE_HEIGHT_M

        keep = not_floor & ~can_drive_under

        # ---- Step 3c: SAFETY CHECK -- exclude the tracked blue/yellow -----------
        # object's OWN surface. A run's bad/blended depth right at the near-field
        # blind zone around the tracked object is exactly what produces a
        # "flying wall" there (see occupancy_grid.py's line-of-sight checks for
        # the geometric side of this fix) -- but the object's true colour is
        # independent of its depth reading, so it is a reliable extra signal:
        # if a candidate's registered RGB pixel is blue or yellow, it is (or is
        # immediately touching) the tracked object, which colored_objects.py
        # already tracks properly -- including its own near-field lidar
        # fallback and periodic clean_object_log() cleanup. There is no need
        # for this generic, colour-blind detector to ALSO claim it, and every
        # reason not to: doing so is how a bad depth reading right at the
        # object turns into a spurious camera_obstacle_log entry.
        if rgb_img is not None and np.any(keep):
            r, g, b = register_and_sample_rgb(
                right_cam, down_cam, depth_f, self.intr,
                C.CAMERA_RGB_DEPTH_BASELINE_M, rgb_img,
            )
            hue, sat, val = rgb_to_hsv(r, g, b)
            is_blue = (
                (hue >= C.BLUE_HUE_MIN) & (hue <= C.BLUE_HUE_MAX)
                & (sat >= C.BLUE_SAT_MIN) & (val >= C.BLUE_VAL_MIN)
            )
            is_yellow = (
                (hue >= C.YELLOW_HUE_MIN) & (hue <= C.YELLOW_HUE_MAX)
                & (sat >= C.YELLOW_SAT_MIN) & (val >= C.YELLOW_VAL_MIN)
            )
            keep = keep & ~is_blue & ~is_yellow

        if not np.any(keep):
            return empty

        forward_lvl = forward_lvl[keep]
        right_cam   = right_cam[keep]

        # ---- Step 4: reduce to (range, bearing) in the robot frame -------------
        # Same sign convention as robot.bearings: positive right_cam (image
        # right) is the robot's right, i.e. a NEGATIVE bearing.
        ranges   = np.hypot(forward_lvl, right_cam)
        bearings = np.arctan2(-right_cam, forward_lvl)

        return ranges, bearings

    # ------------------------------------------------------------------ #
    @staticmethod
    def _column_runs(depth_col, valid_col, rows_col, tol, min_px, max_px):
        """Yield (start, end) sampled-row index pairs (end exclusive) for each
        near-constant-depth run in one column that is between `min_px` and
        `max_px` image rows TALL.

        A pixel joins the current run while it is valid AND its depth stays
        within `tol` of the depth at the run's top pixel -- so the whole run
        spans at most ~`tol` in depth (an upright, camera-facing face).

        Args:
            depth_col : 1-D depths down one column (sampled rows).
            valid_col : 1-D bool, True where depth_col is usable.
            rows_col  : 1-D real image-row index of each sampled row (used to
                         measure a run's pixel height in FULL-resolution rows).
            tol       : max depth spread within a run (m).
            min_px    : minimum run height to accept (full-resolution rows).
            max_px    : maximum run height to accept (full-resolution rows).
        """
        n = len(depth_col)
        i = 0
        while i < n:
            if not valid_col[i]:
                i += 1
                continue
            top_depth = depth_col[i]
            j = i + 1
            while (j < n and valid_col[j]
                   and abs(depth_col[j] - top_depth) <= tol):
                j += 1
            # Run is rows [i, j).  Its height in real image rows is the gap
            # between the first and last sampled row it covers.
            run_px = float(rows_col[j - 1] - rows_col[i])
            if min_px <= run_px <= max_px:
                yield (i, j)
            i = j
