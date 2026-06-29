"""
config.py
=========
Tunable constants for the RosBot maze explorer.

Map frame
---------
We do NOT know the robot's true world pose, so the robot starts at the CENTRE
of the grid map at (0, 0) and everything is measured relative to that start
pose.  The grid origin is therefore placed at (-W/2, -H/2) so that (0, 0) lands
in the middle of the map.

Hardware *device names* live in the Robot class (robot.py), not here.
Lidar resolution / FoV / max-range are read live from the sensor, also in the
Robot class.  This file only holds numbers you may want to tweak.
"""

# --------------------------------------------------------------------------
# Robot physical parameters (from the Rosbot PROTO)
# --------------------------------------------------------------------------
WHEEL_RADIUS = 0.043       # m  (wheel bounding cylinder radius)
WHEEL_BASE = 0.22          # m  (left/right wheel separation, y = +-0.110)
ROBOT_RADIUS = 0.13        # m  (used for obstacle inflation / safety)
MAX_WHEEL_SPEED = 26.0     # rad/s (motor maxVelocity)

# Lidar mounting offset relative to the robot origin (PROTO: 0.02 0 0.1).
LIDAR_OFFSET_X = 0.02      # m forward of robot centre

# --------------------------------------------------------------------------
# Lidar processing (resolution / FoV / max-range come from the sensor itself)
# --------------------------------------------------------------------------
LIDAR_MIN_RANGE = 0.20     # m, ignore returns below this (sensor min range)
LIDAR_USE_RANGE = 6.0      # m, cap how far a free ray writes into the map
# Geometry of the scan.  If the finished map is MIRRORED, flip the sign.
# Webots orders the range image +FoV/2 -> -FoV/2, so bearing(i) =
#   +FoV/2 - (i + 0.5) * FoV / N
LIDAR_ANGLE_SIGN = 1.0
LIDAR_ANGLE_OFFSET = 0.0   # rad, extra rotation if the lidar were mounted turned

# --------------------------------------------------------------------------
# Occupancy grid  (origin centred so the robot start is the map centre)
# --------------------------------------------------------------------------
GRID_RESOLUTION = 0.05     # m per cell
GRID_WIDTH_M = 16.0        # m
GRID_HEIGHT_M = 16.0       # m
GRID_ORIGIN_X = -GRID_WIDTH_M / 2.0    # world x of grid column 0
GRID_ORIGIN_Y = -GRID_HEIGHT_M / 2.0   # world y of grid row 0

# Log-odds update values.
L_FREE = -0.35             # subtracted along a free ray
L_OCC = 1.00               # added at a hit cell (kept > |L_FREE| so thin walls
                           # are not eroded by grazing free beams)
L_CLAMP = 8.0              # clamp |log-odds| to this
P_OCC_THRESH = 0.65        # >= this -> treated as wall
P_FREE_THRESH = 0.35       # <= this -> treated as free

# --------------------------------------------------------------------------
# Frontier exploration
# --------------------------------------------------------------------------
FRONTIER_MIN_CELLS = 4     # ignore frontier clusters smaller than this
FRONTIER_REACH_TOL = 0.18  # m, "arrived at frontier" distance
INFO_GAIN_WEIGHT = 0.25    # cost = path_len_m - weight * sqrt(cluster_size)

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

SAFE_FRONT_DIST = 0.35     # m, evade if a wall is closer than this ahead
FRONT_SECTOR = 0.52        # rad half-width of the "front" sector (~30 deg)

STUCK_DIST = 0.05          # m, progress threshold
STUCK_TIME = 4.0           # s with < STUCK_DIST progress -> stuck

# --------------------------------------------------------------------------
# Timing / cadence (in controller steps unless noted)
# --------------------------------------------------------------------------
MAP_EVERY = 2              # integrate a scan every N steps
PLAN_PERIOD = 3.0          # s between forced replans
VIZ_EVERY = 6              # refresh the live plot every N steps
SPIN_SEED_TURN = 6.5       # rad to rotate on startup to seed the map (~1 turn)

# --------------------------------------------------------------------------
# Mission
# --------------------------------------------------------------------------
# When True, after the map is fully explored the mission advances into the
# colour-search states (SEARCH_BLUE -> ... ).  They are scaffolding for now, so
# this stays False: the robot explores the maze, then stops (DONE).
MISSION_ENABLE_COLOR = False

# --------------------------------------------------------------------------
# Output (written into the controller folder)
# --------------------------------------------------------------------------
SAVE_MAP_PNG = "map_final.png"
SAVE_MAP_NPY = "map_final.npy"
