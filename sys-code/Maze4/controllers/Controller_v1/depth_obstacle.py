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
The key idea: a FLAT surface facing the camera produces near-CONSTANT
depth across a small neighbourhood of pixels, because every point on
that surface really is at roughly the same distance from the camera.
The empty space around/behind an obstacle, or a smoothly receding floor/
wall, does NOT -- depth changes noticeably from one pixel to the next.

So for every sampled pixel we compare its depth to its four immediate
neighbours (one stride-step away in each direction).  If ALL four
differences are smaller than CAMERA_FLAT_TOL_M, that pixel is sitting on
a small flat, camera-facing patch -- a real physical obstacle surface,
regardless of what colour it is.

WHY EXCLUDE THE ORDINARY FLOOR?
-------------------------------------
The floor is ALSO a flat surface, but it's not an obstacle -- it's what
the robot is supposed to drive on!  Near the horizon (far away), the
floor's depth changes very gradually from pixel to pixel due to
perspective foreshortening, so parts of it can accidentally pass the
"near-constant depth" test above.  We fix this the same way
floor_hazard.py identifies the floor: back-project each candidate point
and check its height above the ground.  A candidate whose height is
close to zero (within GROUND_PLANE_TOL_M) IS the floor -- reject it.
Anything else that passed the flatness test is a genuine raised/lowered
obstacle.

OUTPUT FORMAT: WHY (range, bearing), NOT WORLD (x, y)?
------------------------------------------------------------
floor_hazard.py and colored_objects.py both return world (x, y) points
directly, because their caller just stamps a single cell per point into
a mask.  Here we want more: intermediate cells along the line from the
camera to each obstacle point should be marked FREE (exactly like a
lidar ray), not just the obstacle cell itself marked OCCUPIED.  Reducing
each candidate point to the SAME (range, bearing) polar representation
robot.bearings already uses lets OccupancyGrid.integrate_scan_rgbd()
reuse the exact same Bresenham ray-tracing + log-odds update
integrate_scan() uses for the lidar -- one shared, well-tested mechanism
for "trace a ray and update the cells along it", regardless of which
sensor the ray came from.
"""

import numpy as np

import Maze4.controllers.Controller_v1.config as C
from Maze4.controllers.Controller_v1.camera_geometry import (
    CameraIntrinsics, sample_pixel_grid, tilt_correct,
)


class DepthObstacleDetector:
    """Finds flat, camera-facing obstacle surfaces in one depth frame that
    are NOT the ordinary floor, and reduces them to (range, bearing) pairs
    ready for OccupancyGrid.integrate_scan_rgbd()."""

    def __init__(self, camera_width, camera_height, camera_fov,
                 depth_min_range, depth_max_range):
        """See FloorHazardDetector.__init__ -- identical intrinsics, all
        read live from the Webots device in robot.py."""
        self.intr = CameraIntrinsics(
            camera_width, camera_height, camera_fov,
            depth_min_range, depth_max_range,
        )

        # Full-frame scan, same subsampling stride as the other detectors --
        # a lidar-blind obstacle could be anywhere vertically in the image,
        # so (unlike floor_hazard.py) we don't restrict to rows below the
        # horizon.  sample_pixel_grid() returns a proper 2-D grid (not
        # flattened), which we need below for the neighbour-difference test.
        self._us, self._vs = sample_pixel_grid(self.intr, C.CAMERA_SAMPLE_STRIDE)

    # ------------------------------------------------------------------ #
    def detect(self, depth_img):
        """Return (ranges, bearings) for every detected lidar-blind obstacle
        point this frame, in the robot's own frame (same convention as
        robot.bearings: range in metres, bearing in radians, positive =
        left).

        Args:
            depth_img : (H, W) float32 array (metres) from
                         Robot.read_camera_depth().

        Returns:
            (ranges, bearings) -- two 1-D NumPy float arrays (possibly
            empty if nothing qualified this frame).
        """
        empty = (np.empty(0), np.empty(0))

        us, vs = self._us, self._vs
        depth = depth_img[vs.astype(np.int32), us.astype(np.int32)]  # 2-D, same shape as us/vs

        valid = (
            np.isfinite(depth)
            & (depth >= self.intr.min_range)
            & (depth <= self.intr.max_range)
        )

        # ---- Step 1: local flatness test -------------------------------------
        # Compare each sampled pixel to its 4 neighbours IN THE SAMPLED GRID
        # (i.e. one CAMERA_SAMPLE_STRIDE step away in the real image).  Pixels
        # at the edge of the grid have no neighbour on one side; np.inf there
        # means "treat as not flat" rather than crashing on an out-of-bounds shift.
        diff_up    = np.full_like(depth, np.inf)
        diff_down  = np.full_like(depth, np.inf)
        diff_left  = np.full_like(depth, np.inf)
        diff_right = np.full_like(depth, np.inf)
        # inf - inf = nan for pixels with no valid depth on either side;
        # nan always fails the "<= tolerance" comparison below (correctly
        # treated as "not flat"), but raises a noisy RuntimeWarning unless
        # we tell NumPy this is expected.
        with np.errstate(invalid="ignore"):
            diff_up[1:, :]     = np.abs(depth[1:, :]  - depth[:-1, :])
            diff_down[:-1, :]  = np.abs(depth[:-1, :] - depth[1:, :])
            diff_left[:, 1:]   = np.abs(depth[:, 1:]  - depth[:, :-1])
            diff_right[:, :-1] = np.abs(depth[:, :-1] - depth[:, 1:])
        max_diff = np.maximum(np.maximum(diff_up, diff_down),
                               np.maximum(diff_left, diff_right))

        flat = valid & (max_diff <= C.CAMERA_FLAT_TOL_M)
        if not np.any(flat):
            return empty

        us_f    = us[flat]
        vs_f    = vs[flat]
        depth_f = depth[flat]

        # ---- Step 2: back-project (pinhole model, same as camera_geometry.py) --
        right_cam = (us_f - self.intr.cx) * depth_f / self.intr.f
        down_cam  = (vs_f - self.intr.cy) * depth_f / self.intr.f
        forward_lvl, drop_lvl = tilt_correct(down_cam, depth_f, C.CAMERA_TILT_RAD)

        # ---- Step 3: exclude the ordinary floor --------------------------------
        # Same ground-plane test as floor_hazard.py, but INVERTED: we want to
        # KEEP everything that is NOT at floor height.
        height_above_ground = C.CAMERA_HEIGHT_M - drop_lvl
        not_floor = np.abs(height_above_ground) > C.GROUND_PLANE_TOL_M

        if not np.any(not_floor):
            return empty

        forward_lvl = forward_lvl[not_floor]
        right_cam   = right_cam[not_floor]

        # ---- Step 4: reduce to (range, bearing) in the robot frame -------------
        # Same sign convention as robot.bearings: positive right_cam (image
        # right) is the robot's right, i.e. a NEGATIVE bearing.
        ranges   = np.hypot(forward_lvl, right_cam)
        bearings = np.arctan2(-right_cam, forward_lvl)

        return ranges, bearings
