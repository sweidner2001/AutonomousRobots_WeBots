import numpy as np


def polar_to_cartesian(ranges, angles, min_range=0.01, max_range=12.0):
    """Convert polar LiDAR samples to Cartesian points in the sensor frame."""
    ranges_arr = np.asarray(ranges, dtype=np.float64)
    angles_arr = np.asarray(angles, dtype=np.float64)

    if ranges_arr.size == 0 or angles_arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    valid = np.isfinite(ranges_arr) & (ranges_arr > min_range) & (ranges_arr < max_range)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    r = ranges_arr[valid]
    theta = angles_arr[valid]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def compute_odometry(left_vel, right_vel, dt, wheel_radius=0.0425, wheel_base=0.235):
    """Estimate planar pose delta [dx, dy, dtheta] from differential drive wheel speeds."""
    v = wheel_radius * (left_vel + right_vel) / 2.0
    omega = wheel_radius * (right_vel - left_vel) / wheel_base

    if abs(omega) > 1e-6:
        dx = v * dt * np.cos(omega * dt / 2.0)
        dy = v * dt * np.sin(omega * dt / 2.0)
    else:
        dx = v * dt
        dy = 0.0

    dtheta = omega * dt
    return np.array([dx, dy, dtheta], dtype=np.float64)


def transform_points(points, pose):
    """Apply a rigid 2D transform pose=[x, y, theta] to an (N, 2) point set."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ rotation.T + np.array([x, y], dtype=np.float64)


def compute_relative_pose(pose1, pose2):
    """Compute relative pose delta from pose1 to pose2 in global coordinates."""
    dx = pose2[0] - pose1[0]
    dy = pose2[1] - pose1[1]
    dtheta = normalize_angle(pose2[2] - pose1[2])
    return np.array([dx, dy, dtheta], dtype=np.float64)


def normalize_angle(angle):
    """Normalize an angle to the interval [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi