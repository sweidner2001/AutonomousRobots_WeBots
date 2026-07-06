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

VIZALIZATION_NAV_GRID = False

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

GRID_RESOLUTION = 0.04   # m per cell side.  Each cell is a 5 cm x 5 cm square.
                          # Smaller -> finer map, but more memory and slower.

GRID_WIDTH_M  = 13.0     # Total map width in metres.  With 0.05 m/cell
GRID_HEIGHT_M = 13.0     # and 16 m x 16 m, the grid is 320 x 320 cells.

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


CORNER_SCAN_STEPS = 12   
CORNER_ANGLE_THRESHOLD = 0.45  # rad. 35  =  20° — good default
CORNER_MIN_SPEED_FACTOR = 0.2  # When the robot is turning a sharp corner, reduce the
							   # forward speed to this fraction of CRUISE_SPEED.

# ===========================================================================
# Motion control / pilot
# ===========================================================================

CRUISE_SPEED   = 0.16   # m/s  Forward speed during normal driving.
MAX_TURN_SPEED = 0.9    # rad/s  Maximum angular (turning) speed.

HEADING_KP     = 2.2    # Proportional gain on heading error for pure pursuit.
                          # Increase -> turns more aggressively toward waypoints.

LOOKAHEAD      = 0.20   # m  Pure-pursuit look-ahead distance.
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
STUCK_DIST     = 0.05   # m   Minimum distance to travel to be "not stuck".
STUCK_TIME     = 3.0    # s   If the robot moves less than STUCK_DIST in
                          # STUCK_TIME seconds, it is declared stuck and
                          # triggers a backup + replan.

REVERSE_TIME   = 2.5    # s   How long the robot drives backward when stuck.
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
# RGB-D floor hazard detection (green "do not drive here" tiles)
# ===========================================================================
# The Astra RGB-D camera is used to spot green floor markings and turn them
# into permanent obstacles on the occupancy grid, exactly like a wall.
#
# WHY REGISTRATION (ALIGNING DEPTH <-> RGB) MATTERS
# ----------------------------------------------------
# The RGB lens and the depth lens are NOT at the same physical point on the
# camera body — they sit a few centimetres apart (this is true on the real
# Orbbec Astra hardware, and the Webots model copies the real geometry).
# Because of this baseline offset, the same 3-D point projects to DIFFERENT
# pixel coordinates in the two images.  At a typical floor distance of 0.5 m
# the shift is roughly 25-30 pixels — big enough that naively reading
# rgb[u, v] and depth[u, v] at the same (u, v) samples the WRONG colour for
# many points.  floor_hazard.py fixes this with proper back-projection +
# reprojection (see that file's module docstring for the full derivation).

# NOTE ON WHAT IS / ISN'T READ LIVE FROM THE CAMERA
# ----------------------------------------------------
# Resolution, field of view, and depth range ARE queried live from the
# Webots device (Camera.getWidth()/getHeight()/getFov(),
# RangeFinder.getMinRange()/getMaxRange() in robot.py) -- no need to
# duplicate them here as constants.
#
# What CANNOT be read live: the camera's MOUNTING POSE (height above the
# floor, forward/lateral offset, downward tilt) and the physical distance
# between the RGB and depth lenses.  Webots only exposes 3-D scene-tree
# transforms (translation/rotation of a device) through the Supervisor
# API, and this robot uses a plain Robot controller (no `supervisor TRUE`
# in the world file) -- so there is no getPosition()/getOrientation() call
# available for a Camera device.  These are therefore physical constants,
# in the exact same category as WHEEL_RADIUS, WHEEL_BASE, and
# LIDAR_OFFSET_X above, which are also not queryable and hard-coded here.

# --- Camera mount pose relative to the robot body centre --------------------
# Best-effort values read from the Rosbot/Astra PROTO files.  If the detected
# floor mask looks shifted too far/near in the live debug view, tune
# CAMERA_TILT_RAD first (it has the largest effect on where the "floor band"
# falls in the image).
CAMERA_HEIGHT_M     = 0.165  # m   Camera height above the ground.
CAMERA_FORWARD_M    = -0.027 # m   Camera offset along the robot's forward axis.
# CAMERA_FORWARD_M    = -0.03 # m   Camera offset along the robot's forward axis.
# CAMERA_FORWARD_M    = -0.10 # m   Camera offset along the robot's forward axis.
CAMERA_LATERAL_M    = 0.0    # m   Camera offset sideways (left positive).
CAMERA_TILT_RAD     = 0.0    # rad Downward pitch of the camera. 0 = perfectly
                               # horizontal.  Positive = tilted down toward the floor.

# --- RGB <-> depth lens baseline (registration offset) ---------------------
# Real Astra cameras have their RGB and depth (IR) lenses offset by ~25 mm
# HORIZONTALLY (left-right), which is what causes the "slightly different
# picture" the other student mentioned.  This is used to correctly shift
# a back-projected depth point into the RGB camera's own frame before
# sampling colour (see floor_hazard.py: _sample_registered_rgb()).
CAMERA_RGB_DEPTH_BASELINE_M = 0.026   # m, horizontal offset between the two lenses.

# --- Green floor detection (HSV threshold) ----------------------------------
# HSV is far more robust to shading/lighting than raw RGB thresholds because
# hue alone identifies "green" regardless of how bright the tile looks.
GREEN_HUE_MIN = 70     # degrees (0-360).  Green hue band lower bound.
GREEN_HUE_MAX = 170    # degrees.  Upper bound (covers yellow-green to teal-green).
GREEN_SAT_MIN = 0.35   # [0,1]. Minimum colour saturation (rules out grey/white floor).
GREEN_VAL_MIN = 0.20   # [0,1]. Minimum brightness (rules out near-black shadow pixels).

# --- Ground-plane filter ----------------------------------------------------
# A back-projected 3-D point counts as "on the floor" only if its computed
# height is within this tolerance of the ground (z = 0).  This rejects green
# objects that are NOT on the floor (e.g. a green wall poster) even if their
# colour matches.
GROUND_PLANE_TOL_M = 0.05   # m

# --- Performance / cadence ---------------------------------------------------
CAMERA_EVERY        = 4   # Run floor-hazard detection every N control steps
                            # (this is a moderately expensive per-pixel operation).
CAMERA_SAMPLE_STRIDE = 6   # Only process every Nth pixel in each axis (subsampling
                            # keeps the per-frame cost small: 640x480 / 6 / 6 ≈ 8500 px).

# --- Hazard obstacle inflation ----------------------------------------------
HAZARD_INFLATE_CELLS = INFLATE_RADIUS_CELLS  # Same safety margin as walls.

# ===========================================================================
# Coloured target objects (blue / yellow) -- detection + tracking
# ===========================================================================
# Two coloured objects somewhere in the maze must be found and treated as
# obstacles (see colored_objects.py: ColorObjectDetector + TrackedObject).
# Detection reuses the same camera intrinsics and RGB-D registration
# pipeline as the green floor hazard (see camera_geometry.py); only the
# HSV colour bands differ, and objects are NOT restricted to the floor
# plane (they can appear anywhere in the frame).

BLUE_HUE_MIN = 200   # degrees.  Blue hue band lower bound.
BLUE_HUE_MAX = 250   # degrees.  Upper bound.
BLUE_SAT_MIN = 0.35  # [0,1]. Minimum saturation (rules out grey/white).
BLUE_VAL_MIN = 0.20  # [0,1]. Minimum brightness (rules out near-black shadow).

YELLOW_HUE_MIN = 45   # degrees.  Yellow hue band lower bound.
YELLOW_HUE_MAX = 69   # degrees.  Upper bound.
YELLOW_SAT_MIN = 0.35 # [0,1].
YELLOW_VAL_MIN = 0.20 # [0,1].

# Same safety margin as walls/hazards when inflating for A*.
OBJECT_INFLATE_CELLS = INFLATE_RADIUS_CELLS

# Distance within which the robot counts as having "reached" a tracked object.
OBJECT_REACH_TOL = 0.25   # m

# How close (in world metres) a lidar-detected wall cell must be to a raw
# camera colour detection before we treat it as the SAME physical object
# (see occupancy_grid.py: OccupancyGrid.reconciled_object_mask()).  The
# lidar is trusted over the camera here -- if the lidar later finds that
# cell to be free instead, it silently drops out of the reconciled mask on
# its own (no separate "undo" logic needed; see that method's docstring).
OBJECT_WALL_MATCH_DISTANCE_M = 0.16   # m (~4 cells at GRID_RESOLUTION=0.04)

# ===========================================================================
# Mission flags
# ===========================================================================

# When True, after the maze is fully explored the mission advances into
# the colour-search states (SEARCH_BLUE -> GO_BLUE -> SEARCH_YELLOW ->
# GO_YELLOW -> DONE, see explorer.py's MazeExplorer._act()).  Set False to
# just map the maze and stop without hunting for the blue/yellow objects.
MISSION_ENABLE_COLOR = True

# ===========================================================================
# Output files  (written into the same folder as the controller)
# ===========================================================================

SAVE_MAP_PNG = "map_final.png"   # Final map image (matplotlib figure).
SAVE_MAP_NPY = "map_final.npy"   # Raw log-odds array (NumPy binary format).
