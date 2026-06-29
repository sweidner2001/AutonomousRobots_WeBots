"""
config.py  --  All tunable constants for the RosBot maze explorer.
==================================================================

IMPORTANT -- TWO CONFIG FILES
-------------------------------
This project has an unusual structure:
  * Maze4/config.py  is imported by  Maze5/explorer.py
  * Maze5/config.py  is imported by  Maze4/{robot, pilot, planner,
                                      occupancy_grid, frontier,
                                      mapviz, odometry}

If you want to change a motion parameter (speed, safety distance)
edit THIS file (Maze5/config.py).
If you want to change an exploration parameter (replan period, stuck
timeout, reverse time) edit Maze4/config.py.

WHY THE SPLIT?
  The Maze4 modules are the "base" implementations.  Maze5 overrides
  the top-level orchestrator (explorer.py) while keeping the base
  modules unchanged.  Both need a config, and they cross-import
  each other's config to avoid duplication.

UNITS:  metres (m), radians (rad), seconds (s).
"""

# ===========================================================================
# Robot physical dimensions  (from the Rosbot 2 PROTO file in Webots)
# ===========================================================================

WHEEL_RADIUS    = 0.043   # Radius of each wheel in metres.
WHEEL_BASE      = 0.22    # Distance left/right wheel centres (m).
ROBOT_RADIUS    = 0.13    # Approx. robot body radius (m) -- used for
                           # obstacle inflation in the path planner.
MAX_WHEEL_SPEED = 26.0    # Maximum motor speed (rad/s, from the PROTO).

# The lidar sits 0.02 m FORWARD of the robot centre (PROTO: 0.02 0 0.1).
LIDAR_OFFSET_X = 0.02     # m

# ===========================================================================
# Lidar scan processing
# ===========================================================================

LIDAR_MIN_RANGE = 0.20    # m.  Ignore returns closer than this.
LIDAR_USE_RANGE = 6.0     # m.  Cap usable ray length.

# Webots orders rays from +FoV/2 to -FoV/2.
# If the map looks mirrored left/right, set LIDAR_ANGLE_SIGN = -1.0.
LIDAR_ANGLE_SIGN   = 1.0   # +1 = normal, -1 = horizontal mirror
LIDAR_ANGLE_OFFSET = 0.0   # rad, extra offset for non-standard mounting

# ===========================================================================
# Occupancy grid (the robot's internal map)
# ===========================================================================

GRID_RESOLUTION = 0.05    # m per cell.  5 cm cells -> 320x320 grid for 16x16 m.
GRID_WIDTH_M    = 16.0    # Total map width  (m).
GRID_HEIGHT_M   = 16.0    # Total map height (m).

# Place origin at (-W/2, -H/2) so that world (0, 0) is the grid centre.
GRID_ORIGIN_X = -GRID_WIDTH_M  / 2.0
GRID_ORIGIN_Y = -GRID_HEIGHT_M / 2.0

# Log-odds Bayesian update values (see occupancy_grid.py for explanation).
L_FREE  = -0.35   # Added for each free-ray cell (decreases occupancy belief).
L_OCC   =  1.00   # Added at the hit cell (increases occupancy belief).
L_CLAMP =  8.0    # Clamp |log-odds| to this to prevent overflow.

P_OCC_THRESH  = 0.65   # Cell is "wall" if probability >= this.
P_FREE_THRESH = 0.35   # Cell is "free" if probability <= this.

# ===========================================================================
# Frontier exploration
# ===========================================================================

FRONTIER_MIN_CELLS = 4     # Ignore frontier clusters smaller than 4 cells.
FRONTIER_REACH_TOL = 0.18  # m.  Distance to frontier counted as "arrived".
INFO_GAIN_WEIGHT   = 0.25  # Target cost = path_len_m - weight*sqrt(size).

# ===========================================================================
# Path planner (A*)
# ===========================================================================

# Inflate obstacles by ROBOT_RADIUS to keep paths wall-clear.
INFLATE_RADIUS_CELLS  = max(1, int(round(ROBOT_RADIUS / GRID_RESOLUTION)))
UNKNOWN_TRAVERSAL_COST = 1.6   # A* cost multiplier for unknown cells.

# ===========================================================================
# Motion control / pilot
# ===========================================================================

CRUISE_SPEED    = 0.16   # m/s  Nominal forward speed.
MAX_TURN_SPEED  = 1.8    # rad/s  Maximum angular speed.

HEADING_KP      = 2.2    # P-gain on heading error (pure pursuit).
LOOKAHEAD       = 0.28   # m  Pure-pursuit look-ahead distance.
WAYPOINT_TOL    = 0.12   # m  Waypoint "reached" threshold.

# --- Reactive obstacle avoidance -------------------------------------------
# The pilot watches the front lidar sector every step.
# Three zones (outside to inside):
#   SLOW zone  [SLOW_FRONT_DIST, SAFE_FRONT_DIST): reduce forward speed
#   STOP zone  < SAFE_FRONT_DIST:                  stop and turn away

SLOW_FRONT_DIST = 0.55   # m  Start reducing speed at this distance.
SAFE_FRONT_DIST = 0.30   # m  Stop and turn-in-place below this distance.
                           #    (was 0.35 -- reduced to allow tighter corridors)

FRONT_SECTOR    = 0.52   # rad  Half-width of the "front" danger zone (≈ 30°).

# --- Stuck detection (kept here for pilot/explorer cross-reference) ---------
STUCK_DIST = 0.05        # m  Minimum movement to count as "not stuck".
STUCK_TIME = 3.0         # s  Stuck timeout.

# ===========================================================================
# Timing / cadence
# ===========================================================================

MAP_EVERY        = 2     # Integrate a lidar scan every N control steps.
PLAN_PERIOD      = 2.0   # s  Force a replan this often (was 3.0).
VIZ_EVERY        = 6     # Refresh live plot every N steps.
SPIN_SEED_TURN   = 6.5   # rad  Startup rotation (~1 full turn).

# ===========================================================================
# Mission
# ===========================================================================

MISSION_ENABLE_COLOR = False   # Set True to enable colour-search after mapping.

# ===========================================================================
# Output files
# ===========================================================================

SAVE_MAP_PNG = "map_final.png"
SAVE_MAP_NPY = "map_final.npy"
