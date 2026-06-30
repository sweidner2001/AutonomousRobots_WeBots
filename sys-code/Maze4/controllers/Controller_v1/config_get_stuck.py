"""
config.py  --  All tunable constants for the RosBot maze explorer.
==================================================================

WHAT IS THIS FILE FOR?
-----------------------
Every number that controls the robot's behaviour lives here so that
you can experiment without hunting through multiple files.  Think of
it as the "settings panel" of the whole program.

HOW THE COORDINATE SYSTEM WORKS
---------------------------------
The robot does NOT know where it is in the real world when it starts.
We simply define the starting position as (x=0, y=0) — our map origin.
The map grid is large enough that the robot start is always in the
middle:

           y
           ^
           |         map boundary
    -------+---------> x
           |   (0,0) = robot start

The grid array has its bottom-left corner at
  (GRID_ORIGIN_X, GRID_ORIGIN_Y) = (-W/2, -H/2)
so that grid cell [row=nrows/2, col=ncols/2] corresponds to (0, 0).

UNITS:  metres (m), radians (rad), seconds (s).
"""

# ===========================================================================
# Robot physical dimensions  (from the Rosbot 2 PROTO file in Webots)
# ===========================================================================

WHEEL_RADIUS   = 0.043   # Radius of each wheel in metres.
                          # Used to convert motor angle (rad) -> distance (m).

WHEEL_BASE     = 0.22    # Distance between the left and right wheel centres
                          # in metres.  Used for skid-steer speed calculation.

ROBOT_RADIUS   = 0.12    # Approximate radius of the robot body in metres.
                          # Used to "inflate" walls on the map so the robot
                          # keeps a safe gap and doesn't clip corners.

MAX_WHEEL_SPEED = 26.0   # Maximum motor speed in rad/s (from the PROTO).
                          # All motor commands are clamped to ±this value.

# Lidar sensor mounting offset (from the PROTO: translation 0.02 0 0.1).
# The sensor sits 0.02 m in FRONT of the robot centre along the x-axis.
LIDAR_OFFSET_X  = 0.02   # m, positive = forward of robot centre.

# ===========================================================================
# Lidar scan processing
# ===========================================================================

LIDAR_MIN_RANGE = 0.20   # m.  Ignore returns closer than this.
                          # The RPLidar A2 is unreliable below ~0.2 m.

LIDAR_USE_RANGE = 6.0    # m.  Cap how far a free-space ray is drawn into
                          # the map.  Rays that hit nothing or hit beyond
                          # this distance are treated as "no obstacle seen"
                          # (they still mark free space up to this distance).

# Webots delivers the range image ordered from +FoV/2 to -FoV/2.
# ray bearing(i) = +FoV/2 - (i + 0.5) * FoV / N
# If your finished map looks LEFT-RIGHT mirrored, flip LIDAR_ANGLE_SIGN to -1.
LIDAR_ANGLE_SIGN   = 1.0  # +1 = normal, -1 = mirror the scan horizontally
LIDAR_ANGLE_OFFSET = 0.0  # rad.  Extra rotation if the lidar is mounted
                            # at an angle.  Usually 0.

# ===========================================================================
# Occupancy grid (the robot's internal map)
# ===========================================================================
# The map is a 2-D array of cells.  Each cell stores a probability that it
# contains an obstacle.  We use "log-odds" internally (see occupancy_grid.py).
#
# Cell states:
#   UNKNOWN  -- never observed by the lidar.  Grey in the visualisation.
#   FREE     -- observed and probably empty.   White in the visualisation.
#   OCCUPIED -- probably contains a wall.      Black in the visualisation.

GRID_RESOLUTION = 0.06   # m per cell side.  Each cell is a 5 cm x 5 cm square.
                          # Smaller -> finer map, but more memory and slower.

GRID_WIDTH_M  = 15.0     # Total map width in metres.  With 0.05 m/cell
GRID_HEIGHT_M = 15.0     # and 16 m x 16 m, the grid is 320 x 320 cells.

# Place the grid origin so that world (0,0) falls exactly in the middle.
GRID_ORIGIN_X = -GRID_WIDTH_M  / 2.0   # world x of grid column 0
GRID_ORIGIN_Y = -GRID_HEIGHT_M / 2.0   # world y of grid row    0

# --- Log-odds update values (Bayesian inverse sensor model) ----------------
#
# Each lidar observation updates a cell's log-odds value:
#   - L_FREE is SUBTRACTED for every cell the beam passed through freely.
#   - L_OCC  is ADDED    for the cell where the beam hit an obstacle.
#
# We keep |L_OCC| > |L_FREE| so that a thin wall cannot be erased by a
# single grazing free beam — the robot must see many free rays before it
# believes a cell is truly empty.
#
# log-odds L and probability p are related by:  p = 1 / (1 + e^(-L))
L_FREE  = -0.35  # log-odds change per free observation (negative = less likely)
L_OCC   =  1.00  # log-odds change per hit  observation (positive = more likely)
L_CLAMP =  8.0   # Clamp log-odds to ±this value to prevent numerical overflow
                  # and to allow the map to update when the robot revisits cells.

P_OCC_THRESH  = 0.65  # A cell with p >= 0.65 is treated as "occupied / wall".
P_FREE_THRESH = 0.35  # A cell with p <= 0.35 is treated as "free".
                       # Cells between the two thresholds are "uncertain".

# ===========================================================================
# Frontier exploration
# ===========================================================================
# A "frontier" is a free cell that borders at least one unknown cell.
# Frontiers mark the edges of explored space — driving toward them reveals
# new areas.

FRONTIER_MIN_CELLS = 2   # Ignore frontier clusters smaller than 2 cells.
                          # Maze corridor openings are often only 1-2 cells wide,
                          # so a threshold of 4 would filter them out entirely.

FRONTIER_REACH_TOL = 0.18  # m.  If the robot is within this distance of the
                             # frontier target, it counts as "arrived".

INFO_GAIN_WEIGHT = 0.25  # Target selection cost = path_length_m
                          #                       - INFO_GAIN_WEIGHT * sqrt(size)
                          # Larger weight -> prefer big frontier clusters
                          # (more unexplored area) over nearby small ones.

BLACKLIST_CLEAR = 8      # After this many successful frontier plans, clear the
                          # blacklist of unreachable frontiers so that map updates
                          # get a fresh chance to make them reachable.

# ===========================================================================
# Path planner (A*)
# ===========================================================================

# Inflate all occupied cells by this many cells in every direction before
# running A*.  This keeps the planned path away from walls by at least
# ROBOT_RADIUS metres, preventing collisions at corners.
INFLATE_RADIUS_CELLS = max(1, int(round(ROBOT_RADIUS / GRID_RESOLUTION)))
# = max(1, round(0.12 / 0.05)) = max(1, 2) = 2 cells = 0.10 m clearance

UNKNOWN_TRAVERSAL_COST = 1.6  # A* cost multiplier for crossing unknown cells.
                                # The robot prefers known-free routes, but
                                # willingly crosses unknown space to reach
                                # a frontier (exploration payoff).

# ===========================================================================
# Motion control / pilot
# ===========================================================================

CRUISE_SPEED   = 0.16   # m/s  Forward speed during normal driving.
MAX_TURN_SPEED = 1.8    # rad/s  Maximum angular (turning) speed.

HEADING_KP     = 2.2    # Proportional gain on heading error for pure pursuit.
                          # Increase -> turns more aggressively toward waypoints.

LOOKAHEAD      = 0.40   # m  Pure-pursuit look-ahead distance.
                          # Larger -> smoother but cuts corners more.
                          # Smaller -> tighter tracking but may oscillate.

WAYPOINT_TOL   = 0.12   # m  A waypoint is considered "reached" when the
                          # robot is within this distance of it.

# --- Reactive safety (obstacle avoidance directly from the lidar scan) ------
# The reactive layer is independent of the path planner.  It watches the
# FRONT sector of the lidar in real time and intervenes before the robot
# hits a wall.

SLOW_FRONT_DIST = 0.55  # m  Start reducing forward speed when the nearest
                          # front obstacle is closer than this.

SAFE_FRONT_DIST = 0.12  # m  If anything is closer than this in the front
                          # sector, stop and rotate away from the obstacle.

FRONT_SECTOR   = 0.52   # rad  Half-width of the "front" danger zone.
                          # 0.52 rad ≈ 30°, so the full zone is ±30° = 60°.

# --- Stuck detection --------------------------------------------------------
STUCK_DIST     = 0.15   # m   Minimum distance to travel to be "not stuck".
STUCK_TIME     = 2.0    # s   If the robot moves less than STUCK_DIST in
                          # STUCK_TIME seconds, it is declared stuck and
                          # triggers a backup + replan.

REVERSE_TIME   = 2.0    # s   How long the robot drives backward when stuck.
                          # After this it replans to a fresh frontier target.

# ===========================================================================
# Timing / cadence
# ===========================================================================

MAP_EVERY   = 2    # Integrate a lidar scan into the map every N control steps.
                    # Step ≈ dt ms, so MAP_EVERY=2 gives one map update per 2 dt.

PLAN_PERIOD = 2.0  # s   Force a replan this often while driving.  This allows
                    # the robot to react when newly discovered walls block the
                    # current path.

VIZ_EVERY   = 6    # Refresh the matplotlib live view every N steps.
                    # Rendering is slow; updating too often stalls the controller.

SPIN_SEED_TURN = 6.5  # rad  Total rotation during the SPIN_SEED phase
                        # (≈ 1 full turn = 2π ≈ 6.28 rad, slightly more to
                        # ensure a complete 360° view before planning).

# ===========================================================================
# Mission flags
# ===========================================================================

# When True, after the maze is fully explored the mission advances into
# colour-search states (SEARCH_BLUE → ...).  These are placeholders for a
# future phase; keep this False to just map and stop.
MISSION_ENABLE_COLOR = False

# ===========================================================================
# Output files  (written into the same folder as the controller)
# ===========================================================================

SAVE_MAP_PNG = "map_final.png"   # Final map image (matplotlib figure).
SAVE_MAP_NPY = "map_final.npy"   # Raw log-odds array (NumPy binary format).
