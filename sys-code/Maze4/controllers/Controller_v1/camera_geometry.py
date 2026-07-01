"""
camera_geometry.py  --  Shared pinhole-camera math for the Astra RGB-D sensor.
================================================================================

WHY DOES THIS FILE EXIST?
---------------------------
Two different detectors need the EXACT SAME camera math:
  - floor_hazard.py    -- finds green "do not drive here" floor tiles.
  - colored_objects.py -- finds blue/yellow objects to track.

Both have to solve the identical sub-problem: "given a depth pixel, find
the 3-D point it represents, and sample its TRUE colour from the RGB
image" (see the long explanation in floor_hazard.py's module docstring for
why the RGB and depth images are NOT pixel-aligned).  Rather than copy
that math twice, it lives here once and both detectors import it.

WHAT THIS FILE PROVIDES
--------------------------
  CameraIntrinsics        -- holds width/height/focal length/principal point.
  sample_pixel_grid()     -- builds a subsampled (u, v) pixel grid to scan.
  back_project()          -- depth pixel -> 3-D point (pinhole model).
  register_and_sample_rgb() -- shift a depth-frame 3-D point into the RGB
                               camera's frame and sample its real colour.
  rgb_to_hsv()            -- vectorised RGB -> HSV (hue/saturation/value).
  tilt_correct()          -- rotate a camera-local point by the mount tilt.
  camera_local_to_world() -- camera-local (forward, right) -> world (x, y).

See floor_hazard.py for the full derivation of each formula (back-
projection via similar triangles, registration via the lens baseline,
world transform via robot heading).
"""

import math

import numpy as np


class CameraIntrinsics:
    """Pinhole camera parameters, derived once from Webots device queries."""

    def __init__(self, width, height, fov, min_range, max_range):
        self.width  = width
        self.height = height
        self.min_range = min_range
        self.max_range = max_range

        # focal length in pixels, derived from the horizontal FoV:
        #   tan(fov/2) = (width/2) / f   =>   f = (width/2) / tan(fov/2)
        self.f  = (width / 2.0) / math.tan(fov / 2.0)
        self.cx = width  / 2.0   # principal point (image centre column)
        self.cy = height / 2.0   # principal point (image centre row)


def sample_pixel_grid(intr, stride, row_start=0, row_end=None):
    """Build a subsampled (u, v) pixel grid to scan, as float32 arrays.

    Subsampling keeps the per-frame cost small: scanning every pixel of a
    640x480 image every few control steps would be far too slow for a
    pure-Python/NumPy controller.

    Args:
        intr      : CameraIntrinsics.
        stride    : only every `stride`-th pixel in each axis is sampled.
        row_start : first row to include (use this to skip rows that can
                    never contain what you're looking for -- see
                    floor_hazard.py's horizon-row optimisation).
        row_end   : last row (exclusive); defaults to the full image height.

    Returns:
        (us, vs) -- 2-D float32 meshgrid arrays of pixel coordinates.
    """
    row_end = intr.height if row_end is None else row_end
    us, vs = np.meshgrid(
        np.arange(0, intr.width, stride),
        np.arange(row_start, row_end, stride),
    )
    return us.astype(np.float32), vs.astype(np.float32)


def back_project(us, vs, depth, intr):
    """Back-project depth pixels to 3-D points in the depth camera's own frame.

    PINHOLE CAMERA MODEL
    ----------------------
    Webots' RangeFinder returns "perpendicular depth": pixel value Z is the
    distance along the optical axis (like a standard depth buffer). For a
    pixel at (u, v) with principal point (cx, cy) and focal length f, the
    3-D point in the camera's own local frame (X = right, Y = down,
    Z = forward) is:

        X = (u - cx) * Z / f
        Y = (v - cy) * Z / f
        Z = Z                (given directly by the depth pixel)

    This is similar triangles: at depth Z, a pixel offset of (u-cx) pixels
    corresponds to a real-world offset of (u-cx)*Z/f metres.

    Returns:
        (x_cam, y_cam, z_cam) -- same shape as us/vs/depth.
    """
    x_cam = (us - intr.cx) * depth / intr.f
    y_cam = (vs - intr.cy) * depth / intr.f
    z_cam = depth
    return x_cam, y_cam, z_cam


def register_and_sample_rgb(x_cam, y_cam, z_cam, intr, baseline_m, rgb_img):
    """Sample RGB colour for 3-D points expressed in the DEPTH camera's
    local frame, correctly accounting for the RGB/depth lens baseline.

    Shifts each point by the known baseline (a pure translation, since
    both lenses share the same orientation), then re-projects into the
    RGB image using the same pinhole intrinsics (identical resolution and
    FoV for both lenses on this camera model).  See floor_hazard.py's
    module docstring for the full "why registration matters" explanation.

    Args:
        x_cam, y_cam, z_cam : 3-D points in the depth camera's local frame.
        intr                : CameraIntrinsics (shared by both lenses).
        baseline_m          : horizontal distance between the two lenses (m).
        rgb_img             : (H, W, 3) uint8 RGB image.

    Returns:
        (r, g, b) -- three 1-D uint8 arrays, same length as the input points.
    """
    # Shift into the RGB camera's frame (horizontal baseline, real hardware).
    x_rgb = x_cam - baseline_m
    y_rgb = y_cam
    z_rgb = z_cam

    # Re-project into the RGB image plane.
    u_rgb = intr.cx + x_rgb * intr.f / z_rgb
    v_rgb = intr.cy + y_rgb * intr.f / z_rgb

    # Round to nearest pixel and clip to valid image bounds.
    u_idx = np.clip(np.round(u_rgb).astype(np.int32), 0, intr.width  - 1)
    v_idx = np.clip(np.round(v_rgb).astype(np.int32), 0, intr.height - 1)

    pixels = rgb_img[v_idx, u_idx]   # (N, 3) uint8
    return pixels[:, 0], pixels[:, 1], pixels[:, 2]


def rgb_to_hsv(r, g, b):
    """Vectorised RGB -> HSV conversion (no OpenCV dependency).

    Using HUE (not raw R/G/B) makes colour tests robust to shading: the
    same object lit brightly and the same object in shadow have the same
    hue, just different value/saturation.

    Standard RGB->HSV formulas:
        V   = max(r, g, b)
        S   = (V - min) / V                (0 if V == 0)
        H   = 60 * ((g - b) / delta)        if V == r
              60 * (2 + (b - r) / delta)    if V == g
              60 * (4 + (r - g) / delta)    if V == b
              (H is undefined / 0 when delta == 0, i.e. a grey pixel)

    Args:
        r, g, b : uint8 arrays (any shape), 0-255.

    Returns:
        (hue, sat, val) -- float arrays, hue in degrees [0,360), sat/val in [0,1].
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

    return hue, s, v


def tilt_correct(y_cam, z_cam, tilt_rad):
    """Rotate a camera-local point by the mount's downward tilt.

    The camera may be pitched down by `tilt_rad` relative to horizontal.
    This returns the point's coordinates in a LEVEL frame (as if the
    camera were mounted perfectly horizontal), which is what's needed to
    compute the point's true forward distance and height above the floor.

    Args:
        y_cam, z_cam : camera-local vertical (down+) and forward coordinates.
        tilt_rad     : downward pitch of the camera (rad). 0 = level.

    Returns:
        (forward_lvl, drop_lvl) -- forward distance and vertical drop
        below the camera, both in the LEVEL frame.
    """
    forward_lvl = z_cam * math.cos(tilt_rad) - y_cam * math.sin(tilt_rad)
    drop_lvl    = z_cam * math.sin(tilt_rad) + y_cam * math.cos(tilt_rad)
    return forward_lvl, drop_lvl


def camera_local_to_world(forward, right, pose, mount_fwd, mount_lat):
    """Convert camera-local (forward, right) offsets to world (x, y).

    Combines the camera's own mounting offset on the robot body with the
    per-point forward/right offsets, then rotates the whole thing into the
    world frame using the robot's heading -- the same convention used
    everywhere else in this codebase (see occupancy_grid.py).

    Args:
        forward, right : arrays of camera-local forward/right offsets (m).
        pose            : (x, y, theta) robot pose.
        mount_fwd       : camera mount offset along the robot's forward axis (m).
        mount_lat       : camera mount offset sideways, left positive (m).

    Returns:
        (world_xs, world_ys) -- arrays of world coordinates (m).
    """
    x, y, theta = pose
    body_fwd = mount_fwd + forward
    body_lft = mount_lat - right   # image "right" = robot "left" negated

    world_xs = x + body_fwd * np.cos(theta) - body_lft * np.sin(theta)
    world_ys = y + body_fwd * np.sin(theta) + body_lft * np.cos(theta)
    return world_xs, world_ys
