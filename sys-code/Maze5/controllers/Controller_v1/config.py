"""
config.py
=========
Central place for every tunable constant used by the RosBot maze-explorer.

Coordinate convention (Webots, world here is z-up):
  * x, y  -> floor plane (the map is built in x/y)
  * theta -> heading, 0 = +x axis, CCW positive (right-handed)
"""

# --------------------------------------------------------------------------
# Device names (taken from the Rosbot / RpLidarA2 / Mpu-9250 PROTOs)
# --------------------------------------------------------------------------
LIDAR_NAME = "laser"
IMU_NAME = "imu inertial_unit"   # InertialUnit inside the Mpu-9250 ("imu")

# Wheel motors (RotationalMotor) and their PositionSensors.
MOTOR_NAMES = {
    "fl": "fl_wheel_joint",
    "fr": "fr_wheel_joint",
    "rl": "rl_wheel_joint",
    "rr": "rr_wheel_joint",
}
ENCODER_NAMES = {
    "fl": "front left wheel motor sensor",
    "fr": "front right wheel motor sensor",
    "rl": "rear left wheel motor sensor",
    "rr": "rear right wheel motor sensor",
}

# --------------------------------------------------------------------------
# Robot physical parameters (from the Rosbot PROTO)
# --------------------------------------------------------------------------
WHEEL_RADIUS = 0.085 / 2.0       # m  (wheel bounding cylinder radius)
WHEEL_BASE = 0.22          # m  (left/right wheel separation, y = +-0.110)
ROBOT_RADIUS = 0.13        # m  (used for obstacle inflation / safety)
MAX_WHEEL_SPEED = 26.0     # rad/s (motor maxVelocity)

# Lidar mounting offset relative to the robot origin (PROTO: 0.02 0 0.1).
LIDAR_OFFSET_X = 0.02      # m forward of robot centre

# --------------------------------------------------------------------------
# Lidar parameters (from the RpLidarA2 PROTO) -- read live too, these are
# only fallbacks / sanity values.
# --------------------------------------------------------------------------
LIDAR_FOV = 6.283184       # rad (360 deg)
LIDAR_RESOLUTION = 400     # rays
LIDAR_MIN_RANGE = 0.20     # m
LIDAR_MAX_RANGE = 12.0     # m
# Cap how far a "free" ray writes into the map (keeps the grid tidy & fast).
LIDAR_USE_RANGE = 6.0      # m

# Geometry of the lidar scan. If the finished map comes out MIRRORED,
# flip LIDAR_ANGLE_SIGN to -1.0. If it comes out ROTATED, adjust the offset.
# Webots orders the range image starting at +FoV/2 and sweeping to -FoV/2,
# so the bearing of ray i is:  bearing = +FoV/2 - (i + 0.5) * FoV / N
LIDAR_ANGLE_SIGN = 1.0
LIDAR_ANGLE_OFFSET = 0.0   # rad, extra rotation if the lidar were mounted turned

# --------------------------------------------------------------------------
# Robot start pose in the WORLD (must match the Rosbot translation/rotation
# in Maze5.wbt so the map is built in world coordinates).  Heading is taken
# live from the IMU, so only x/y really matter here.
# --------------------------------------------------------------------------
ROBOT_START_X = -1.19026
ROBOT_START_Y = 2.53089

# --------------------------------------------------------------------------
# Occupancy grid
# --------------------------------------------------------------------------
GRID_RESOLUTION = 0.05     # m per cell
# World rectangle the grid covers (generous box around the maze).
GRID_ORIGIN_X = -8.0       # world x of grid column 0 (left edge)
GRID_ORIGIN_Y = -8.0       # world y of grid row 0 (bottom edge)
GRID_WIDTH_M = 16.0        # m
GRID_HEIGHT_M = 16.0       # m

# Log-odds update values.
L_FREE = -0.35             # subtracted along a free ray
L_OCC = 1.00               # added at a hit cell (kept > |L_FREE| so thin
                           # walls are not eroded by grazing free beams)
L_CLAMP = 8.0              # clamp |log-odds| to this
# Probability thresholds derived from log-odds.
P_OCC_THRESH = 0.65        # >= this -> treated as wall
P_FREE_THRESH = 0.35       # <= this -> treated as free

# --------------------------------------------------------------------------
# Frontier exploration
# --------------------------------------------------------------------------
FRONTIER_MIN_CELLS = 4     # ignore frontier clusters smaller than this
FRONTIER_REACH_TOL = 0.18  # m, "arrived at frontier" distance
# Selection: cost = path_length_m - INFO_GAIN_WEIGHT * sqrt(cluster_size)
INFO_GAIN_WEIGHT = 0.25

# --------------------------------------------------------------------------
# Path planner (A*)
# --------------------------------------------------------------------------
INFLATE_RADIUS_CELLS = max(1, int(round(ROBOT_RADIUS / GRID_RESOLUTION)))
UNKNOWN_TRAVERSAL_COST = 1.6   # extra cost multiplier for crossing unknown cells

# --------------------------------------------------------------------------
# Motion / pilot
# --------------------------------------------------------------------------
CRUISE_SPEED = 0.16        # m/s nominal forward speed
MAX_TURN_SPEED = 1.8       # rad/s max angular speed
HEADING_KP = 2.2           # proportional gain on heading error
LOOKAHEAD = 0.28           # m pure-pursuit look-ahead distance
WAYPOINT_TOL = 0.12        # m distance to consider a waypoint reached

# Reactive safety (collision avoidance independent of the planner).
SAFE_FRONT_DIST = 0.35     # m, if a wall is closer than this ahead -> evade
FRONT_SECTOR = 0.52        # rad half-width of the "front" sector (~30 deg)

# Stuck detection.
STUCK_DIST = 0.05          # m, progress threshold
STUCK_TIME = 4.0           # s with < STUCK_DIST progress -> stuck

# --------------------------------------------------------------------------
# Timing / cadence (in controller steps)
# --------------------------------------------------------------------------
MAP_EVERY = 2              # integrate a scan every N steps
PLAN_PERIOD = 3.0          # s between forced replans
VIZ_EVERY = 6              # refresh the live plot every N steps
SPIN_SEED_TURN = 6.5       # rad to rotate on startup to seed the map (~1 turn)

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
SAVE_MAP_PNG = "map_final.png"   # written into the controller folder on finish
SAVE_MAP_NPY = "map_final.npy"   # raw log-odds grid for later (e.g. SLAM eval)
