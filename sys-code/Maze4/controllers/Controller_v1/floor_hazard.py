"""
floor_hazard.py  --  Green floor-marking detection using the RGB-D camera.
============================================================================

WHAT DOES THIS FILE DO?
-------------------------
During exploration the maze floor sometimes has GREEN tiles that the robot
must never drive over.  This module looks at the RGB-D camera every few
control steps, finds every camera pixel that is BOTH (a) green AND (b) part
of the floor plane, and converts those pixels into world (x, y) points.
The caller (explorer.py) then stamps those points into the occupancy grid
as permanent obstacles -- exactly like a wall -- so the existing A*
planner, flood fill, and frontier detector automatically steer around them.
No other module needs to know "green tiles" exist; they just see more
blocked cells.

THE CORE PROBLEM: RGB AND DEPTH ARE TWO DIFFERENT LENSES
-----------------------------------------------------------
An RGB-D camera like the Astra used here is really TWO cameras bolted
together: a colour lens and a depth (infra-red) lens, a few centimetres
apart (CAMERA_RGB_DEPTH_BASELINE_M in config.py).  Because they sit at
different physical positions, the SAME 3-D point in the world lands on a
DIFFERENT pixel in each image.

    RGB lens  o---\\                    IR/depth lens  o---\\
                    \\  both point            (baseline)      \\
                     \\ the same way                           \\
                      v                                          v
                 (slightly different) picture              (slightly different) picture

If we naively read rgb_image[v, u] and depth_image[v, u] at the SAME pixel
coordinates, we get the colour of a DIFFERENT 3-D point than the one the
depth value describes.  At a typical floor distance of 0.5 m the pixel
shift caused by the baseline is roughly:

    shift_px ≈ baseline * focal_length / depth
             ≈ 0.026 m  *  559 px       / 0.5 m
             ≈ 29 pixels

That is a large error (640 px wide image) -- easily enough to sample the
wrong tile's colour.  We fix this properly with "RGB-D registration":

    1. BACK-PROJECT every depth pixel to a 3-D point in the DEPTH camera's
       own local coordinate frame (pinhole camera model).
    2. SHIFT that 3-D point by the known baseline vector to express it in
       the RGB camera's local frame instead (the two lenses share the same
       orientation, so this is a pure translation -- no rotation needed).
    3. RE-PROJECT the shifted 3-D point into the RGB image plane to find
       which RGB pixel actually shows that same 3-D point.
    4. Sample the colour there.  THIS is the correctly registered colour
       for that depth pixel.

PINHOLE CAMERA MODEL (used for both back- and re-projection)
----------------------------------------------------------------
Webots' RangeFinder returns "perpendicular depth": each pixel value Z is
the distance from the camera to the object measured ALONG THE OPTICAL AXIS
(like a standard depth buffer), not the straight-line distance to the
pixel's ray.  For a pixel at column u, row v, with principal point
(cx, cy) and focal length f (in pixels), the 3-D point in the camera's own
local frame (X = right, Y = down, Z = forward) is:

    X = (u - cx) * Z / f
    Y = (v - cy) * Z / f
    Z = Z                         (given directly by the depth pixel)

This is just similar triangles: at depth Z, a pixel offset of (u-cx)
pixels corresponds to a real-world offset of (u-cx)*Z/f metres.

FINDING THE FLOOR AND CHECKING FOR GREEN
-------------------------------------------
Once we have a 3-D point in the camera's local frame we still need to
know if it is sitting ON THE GROUND.  The camera is mounted at a known
height above the floor (CAMERA_HEIGHT_M) with a known tilt
(CAMERA_TILT_RAD, 0 = perfectly level).  Rotating the local point by the
tilt and adding the mount height gives its height above the real ground;
if that height is close to zero (within GROUND_PLANE_TOL_M) the point is
on the floor.

Finally we look at the colour at the registered RGB pixel.  Converting
RGB to HSV and checking the HUE channel (not raw R/G/B) makes green
detection robust to lighting changes -- a dark-green tile in shadow and a
bright-green tile in direct light both have the same hue, just different
brightness/saturation.

COORDINATE OUTPUT
-------------------
Floor+green points are finally rotated by the robot's heading and added to
the robot's world position, giving world (x, y) metres -- the same
convention used everywhere else in this codebase (see occupancy_grid.py).
"""

import math

import numpy as np

import Maze4.controllers.Controller_v1.config as C
from Maze4.controllers.Controller_v1.camera_geometry import (
    CameraIntrinsics, sample_pixel_grid, back_project,
    register_and_sample_rgb, rgb_to_hsv, tilt_correct, camera_local_to_world,
)


class FloorHazardDetector:
    """Finds green floor-marking pixels in one RGB-D frame and returns their
    world-frame (x, y) positions."""

    def __init__(self, camera_width, camera_height, camera_fov,
                 depth_min_range, depth_max_range):
        """
        Args:
            camera_width, camera_height : pixel resolution, from
                Camera.getWidth()/getHeight() (see robot.py). Identical for
                the RGB and depth lens on this camera model.
            camera_fov      : horizontal field of view (rad), from Camera.getFov().
            depth_min_range : m, from RangeFinder.getMinRange().
            depth_max_range : m, from RangeFinder.getMaxRange().

        All five of these ARE queried live from the Webots device in
        robot.py -- they are passed in here rather than duplicated as
        config.py constants, so there is only one source of truth.
        """
        self.intr = CameraIntrinsics(
            camera_width, camera_height, camera_fov,
            depth_min_range, depth_max_range,
        )

        # --- Precompute the (subsampled) pixel grid ONCE ----------------------
        # We only need to look at rows BELOW the horizon: with a camera tilt
        # of CAMERA_TILT_RAD, the horizon (where a ray is exactly horizontal)
        # falls at row = cy + f * tan(CAMERA_TILT_RAD).  Rows above that can
        # never see the floor (they look above the horizontal), so skipping
        # them roughly halves the work for a level-mounted camera.
        horizon_row = int(self.intr.cy + self.intr.f * math.tan(C.CAMERA_TILT_RAD))
        row_start   = max(0, horizon_row)

        self._us, self._vs = sample_pixel_grid(
            self.intr, C.CAMERA_SAMPLE_STRIDE, row_start=row_start
        )

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def detect(self, rgb_img, depth_img, pose):
        """Return world-frame (xs, ys) of every detected green-floor point.

        Args:
            rgb_img   : (H, W, 3) uint8 array from Robot.read_camera_rgb().
            depth_img : (H, W) float32 array (metres) from Robot.read_camera_depth().
            pose      : (x, y, theta) robot pose from Odometry.

        Returns:
            (xs, ys) -- two 1-D NumPy float arrays of world coordinates
            (possibly empty if nothing matched this frame).
        """
        us, vs = self._us, self._vs

        # ---- Step 1: read depth at the sampled pixels ----------------------
        depth = depth_img[vs.astype(np.int32), us.astype(np.int32)]
        valid = (
            np.isfinite(depth)
            & (depth >= self.intr.min_range)
            & (depth <= self.intr.max_range)
        )
        if not np.any(valid):
            return np.empty(0), np.empty(0)

        us, vs, depth = us[valid], vs[valid], depth[valid]

        # ---- Step 2: back-project depth pixels to 3-D (depth camera frame) --
        x_cam, y_cam, z_cam = back_project(us, vs, depth, self.intr)

        # ---- Step 3: register into the RGB camera's frame + sample colour ---
        r, g, b = register_and_sample_rgb(
            x_cam, y_cam, z_cam, self.intr, C.CAMERA_RGB_DEPTH_BASELINE_M, rgb_img
        )

        # ---- Step 4: green colour test (HSV hue/sat/val thresholds) ---------
        hue, sat, val = rgb_to_hsv(r, g, b)
        green = (
            (hue >= C.GREEN_HUE_MIN) & (hue <= C.GREEN_HUE_MAX)
            & (sat >= C.GREEN_SAT_MIN)
            & (val >= C.GREEN_VAL_MIN)
        )
        if not np.any(green):
            return np.empty(0), np.empty(0)

        # ---- Step 5: ground-plane test ---------------------------------------
        # Rotate the camera-local point by the mount tilt to find how far
        # above the true ground plane it sits, then keep only points close
        # to height = 0 (the floor).
        forward_lvl, drop_lvl = tilt_correct(y_cam, z_cam, C.CAMERA_TILT_RAD)
        height_above_ground = C.CAMERA_HEIGHT_M - drop_lvl

        on_floor = np.abs(height_above_ground) <= C.GROUND_PLANE_TOL_M

        keep = green & on_floor
        if not np.any(keep):
            return np.empty(0), np.empty(0)

        forward_lvl = forward_lvl[keep]
        right_lvl   = x_cam[keep]   # left/right offset unaffected by pitch tilt

        # ---- Step 6: camera-local (forward, right) -> world (x, y) ----------
        return camera_local_to_world(
            forward_lvl, right_lvl, pose, C.CAMERA_FORWARD_M, C.CAMERA_LATERAL_M
        )
