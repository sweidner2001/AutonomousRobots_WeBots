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


class FloorHazardDetector:
    """Finds green floor-marking pixels in one RGB-D frame and returns their
    world-frame (x, y) positions."""

    def __init__(self):
        # --- Pinhole intrinsics (identical for the RGB and depth lens: same
        # resolution and FoV on this camera model) ---------------------------
        self.width  = C.CAMERA_WIDTH
        self.height = C.CAMERA_HEIGHT
        # focal length in pixels, derived from the horizontal FoV:
        #   tan(fov/2) = (width/2) / f   =>   f = (width/2) / tan(fov/2)
        self.f  = (self.width / 2.0) / math.tan(C.CAMERA_FOV / 2.0)
        self.cx = self.width  / 2.0   # principal point (image centre column)
        self.cy = self.height / 2.0   # principal point (image centre row)

        # --- Precompute the (subsampled) pixel grid ONCE ----------------------
        # We only need to look at rows BELOW the horizon: with a camera tilt
        # of CAMERA_TILT_RAD, the horizon (where a ray is exactly horizontal)
        # falls at row = cy + f * tan(CAMERA_TILT_RAD).  Rows above that can
        # never see the floor (they look above the horizontal), so skipping
        # them roughly halves the work for a level-mounted camera.
        horizon_row = int(self.cy + self.f * math.tan(C.CAMERA_TILT_RAD))
        row_start   = max(0, horizon_row)

        stride = C.CAMERA_SAMPLE_STRIDE
        self._us, self._vs = np.meshgrid(
            np.arange(0, self.width,  stride),
            np.arange(row_start, self.height, stride),
        )
        self._us = self._us.astype(np.float32)
        self._vs = self._vs.astype(np.float32)

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
            & (depth >= C.CAMERA_MIN_RANGE)
            & (depth <= C.CAMERA_MAX_RANGE)
        )
        if not np.any(valid):
            return np.empty(0), np.empty(0)

        us, vs, depth = us[valid], vs[valid], depth[valid]

        # ---- Step 2: back-project depth pixels to 3-D (depth camera frame) --
        # Camera-local axes: X = right, Y = down, Z = forward (= depth value).
        x_cam = (us - self.cx) * depth / self.f
        y_cam = (vs - self.cy) * depth / self.f
        z_cam = depth

        # ---- Step 3: register into the RGB camera's frame + sample colour ---
        r, g, b = self._sample_registered_rgb(x_cam, y_cam, z_cam, rgb_img)

        # ---- Step 4: green colour test (HSV hue/sat/val thresholds) ---------
        green = self._is_green(r, g, b)
        if not np.any(green):
            return np.empty(0), np.empty(0)

        # ---- Step 5: ground-plane test ---------------------------------------
        # Rotate the camera-local point by the mount tilt to find how far
        # above the true ground plane it sits, then keep only points close
        # to height = 0 (the floor).
        tilt = C.CAMERA_TILT_RAD
        # forward/height in the TILT-CORRECTED (level) frame:
        forward_lvl = z_cam * math.cos(tilt) - y_cam * math.sin(tilt)
        drop_lvl    = z_cam * math.sin(tilt) + y_cam * math.cos(tilt)
        height_above_ground = C.CAMERA_HEIGHT_M - drop_lvl

        on_floor = np.abs(height_above_ground) <= C.GROUND_PLANE_TOL_M

        keep = green & on_floor
        if not np.any(keep):
            return np.empty(0), np.empty(0)

        forward_lvl = forward_lvl[keep]
        right_lvl   = x_cam[keep]   # left/right offset unaffected by pitch tilt

        # ---- Step 6: camera-local (forward, right) -> world (x, y) ----------
        x, y, theta = pose
        # Camera mount offset in the robot's own frame.
        mount_fwd = C.CAMERA_FORWARD_M
        mount_lat = C.CAMERA_LATERAL_M

        # Combine mount offset with the per-point forward/right offsets,
        # then rotate the whole thing into the world frame by robot heading.
        body_fwd = mount_fwd + forward_lvl
        body_lft = mount_lat - right_lvl   # image "right" = robot "left" negated

        world_xs = x + body_fwd * np.cos(theta) - body_lft * np.sin(theta)
        world_ys = y + body_fwd * np.sin(theta) + body_lft * np.cos(theta)

        return world_xs, world_ys

    # ------------------------------------------------------------------ #
    # RGB-D registration
    # ------------------------------------------------------------------ #
    def _sample_registered_rgb(self, x_cam, y_cam, z_cam, rgb_img):
        """Sample RGB colour for 3-D points expressed in the DEPTH camera's
        local frame, correctly accounting for the RGB/depth lens baseline.

        Shifts each point by the known baseline (a pure translation, since
        both lenses share the same orientation), then re-projects into the
        RGB image using the same pinhole intrinsics (identical resolution
        and FoV for both lenses on this camera model).
        """
        # Step 2 of the module docstring: shift into the RGB camera's frame.
        # The baseline is horizontal (left-right), matching the real Astra
        # hardware -- see config.py CAMERA_RGB_DEPTH_BASELINE_M.
        x_rgb = x_cam - C.CAMERA_RGB_DEPTH_BASELINE_M
        y_rgb = y_cam
        z_rgb = z_cam

        # Step 3: re-project into the RGB image plane.
        u_rgb = self.cx + x_rgb * self.f / z_rgb
        v_rgb = self.cy + y_rgb * self.f / z_rgb

        # Round to nearest pixel and clip to valid image bounds.
        u_idx = np.clip(np.round(u_rgb).astype(np.int32), 0, self.width  - 1)
        v_idx = np.clip(np.round(v_rgb).astype(np.int32), 0, self.height - 1)

        pixels = rgb_img[v_idx, u_idx]   # (N, 3) uint8
        return pixels[:, 0], pixels[:, 1], pixels[:, 2]

    # ------------------------------------------------------------------ #
    # Colour classification
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_green(r, g, b):
        """Vectorised RGB -> HSV green threshold (no OpenCV dependency).

        Using HUE (not raw RGB) makes the test robust to shading: a tile
        lit brightly and the same tile in shadow have the same hue, just
        different value/saturation.

        Standard RGB->HSV formulas:
            V   = max(r, g, b)
            S   = (V - min) / V                (0 if V == 0)
            H   = 60 * ((g - b) / delta)        if V == r
                  60 * (2 + (b - r) / delta)    if V == g
                  60 * (4 + (r - g) / delta)    if V == b
                  (H is undefined / 0 when delta == 0, i.e. a grey pixel)
        """
        rf = r.astype(np.float32) / 255.0
        gf = g.astype(np.float32) / 255.0
        bf = b.astype(np.float32) / 255.0

        v = np.maximum(np.maximum(rf, gf), bf)
        mn = np.minimum(np.minimum(rf, gf), bf)
        delta = v - mn

        s = np.divide(delta, v, out=np.zeros_like(v), where=(v > 1e-6))

        hue = np.zeros_like(v)
        safe_delta = np.where(delta > 1e-6, delta, 1.0)  # avoid /0; masked out below

        is_r_max = (v == rf) & (delta > 1e-6)
        is_g_max = (v == gf) & (delta > 1e-6) & ~is_r_max
        is_b_max = (v == bf) & (delta > 1e-6) & ~is_r_max & ~is_g_max

        hue = np.where(is_r_max, 60.0 * (((gf - bf) / safe_delta) % 6.0), hue)
        hue = np.where(is_g_max, 60.0 * (((bf - rf) / safe_delta) + 2.0), hue)
        hue = np.where(is_b_max, 60.0 * (((rf - gf) / safe_delta) + 4.0), hue)

        return (
            (hue >= C.GREEN_HUE_MIN) & (hue <= C.GREEN_HUE_MAX)
            & (s >= C.GREEN_SAT_MIN)
            & (v >= C.GREEN_VAL_MIN)
        )
