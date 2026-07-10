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

LIDAR_USE_RANGE = 5.0    # m.  Cap how far a free-space ray is drawn into
                          # the map.  Rays that hit nothing or hit beyond
                          # this distance are treated as "no obstacle seen"
                          # (they still mark free space up to this distance).

# Webots delivers the range image ordered from +FoV/2 to -FoV/2.
# ray bearing(i) = +FoV/2 - (i + 0.5) * FoV / N
# If your finished map looks LEFT-RIGHT mirrored, flip LIDAR_ANGLE_SIGN to -1.
LIDAR_ANGLE_SIGN   = 1.0  # +1 = normal, -1 = mirror the scan horizontally
LIDAR_ANGLE_OFFSET = 0.0  # rad.  Extra rotation if the lidar is mounted
                            # at an angle.  Usually 0.

# --- Restrict the lidar to a narrower field of view -------------------------
# By default the full sensor FOV is used (typically 360 deg for the RPLidar
# A2 -- rays cover the whole circle around the robot).  Set LIDAR_USE_FOV_DEG
# to a smaller number to use only a WINDOW of that many degrees, centred on
# LIDAR_USE_FOV_CENTER_DEG.  Example: 180 -> a forward-facing half-circle,
# 90 deg to the left and 90 deg to the right of the centre bearing.
# Rays outside the window are treated exactly like "no return" (see
# robot.py -- read_lidar() / _compute_lidar_angle_mask()), so every existing
# consumer (map integration, frontier exploration, the safety reflex, ...)
# automatically ignores them -- no other module needs to change.
LIDAR_USE_FOV_DEG        = 230   # None = use the full sensor FOV. Degrees otherwise.
LIDAR_USE_FOV_CENTER_DEG = 0.0    # deg. Bearing at the centre of the window;
                                    # 0 = straight ahead, positive = left,
                                    # negative = right (same convention as bearings).

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

GRID_RESOLUTION = 0.03   # m per cell side.  Each cell is a 5 cm x 5 cm square.
                          # Smaller -> finer map, but more memory and slower.

GRID_WIDTH_M  = 8.0     # Total map width in metres.  With 0.05 m/cell
GRID_HEIGHT_M = 10.0     # and 16 m x 16 m, the grid is 320 x 320 cells.

# Place the grid origin so that world (0,0) falls exactly in the middle.
GRID_ORIGIN_X = -GRID_WIDTH_M  / 2.0 +2  # world x of grid column 0
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

# --- Isolated frontier filtering (noise rejection) ---------------------------
# A frontier cell almost entirely surrounded by ALREADY-OBSERVED cells is
# usually a sensor artefact, not a real "more maze beyond here" signal --
# see frontier.py: FrontierDetector.detect_cells().
# 7x7 kreis hätte 33 1er
# 5x5 kreis hätte 17 1er
FRONTIER_ISOLATION_CIRCLE_DIA       = 5    # cells. Odd size, centred on the cell.
FRONTIER_ISOLATION_MAX_UNKNOWN_CELLS = 3   # out of WINDOW*WINDOW=25 -- discard the
                                          # frontier cell if this many (or more)
                                          # neighbours are already known.

# --- Frontier target hysteresis (reduces goal flip-flopping) ----------------
# choose_target() is otherwise STATELESS: every PLAN cycle it picks purely by
# cost, with no memory of what it picked last time.  A small map update
# between cycles (a few more cells become known, shifting a path length by a
# metre) can then make a DIFFERENT, only marginally cheaper cluster win by a
# hair -- the robot commits to a frontier, immediately replans, and switches
# again, back and forth.  The cluster closest to the PREVIOUSLY chosen target
# gets a cost discount, so a competitor must be MEANINGFULLY better (not just
# infinitesimally) before the robot abandons its current frontier.
FRONTIER_STICKINESS_COUNT     = 3     # times. Number of consecutive plans the target must be chosen to be considered "sticky".
FRONTIER_STICKINESS_MATCH_TOL_M = 0.08  # m. How close a NEW cluster's centroid must be
                                          #    to the previous target to count as "the same one".

# --- Last-resort frontier recheck (before giving up a colour search) --------
# Explorer.finished (zero frontiers on the fully-inflated nav grid) doesn't
# necessarily mean the maze truly has no more space to see -- the inflation
# margin, or a single noisy/false wall pixel, can seal off a corridor that is
# physically open (see planner.py: PathPlanner.
# find_nearest_frontier_reduced_inflation() and explorer.py:
# MazeExplorer._act_search()). Before a colour search actually gives up, we
# try ONE more frontier detection pass with every inflation margin shrunk by
# this many cells, and -- if that reveals one -- drive to the nearest one
# using the exact same "get as close as safely possible" machinery GoToPoint
# already uses for a sealed-off tracked object
# (PathPlanner.plan_path_near_blocked_target()).
FRONTIER_RECHECK_INFLATE_REDUCTION = 1   # cells

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
MAX_TURN_SPEED = 0.8    # rad/s  Maximum angular (turning) speed.

HEADING_KP     = 2.2    # Proportional gain on heading error for pure pursuit.
                          # Increase -> turns more aggressively toward waypoints.

LOOKAHEAD      = 0.12   # m  Pure-pursuit look-ahead distance.
                          # Larger -> smoother but cuts corners more.
                          # Smaller -> tighter tracking but may oscillate.

WAYPOINT_TOL   = 0.08   # m  A waypoint is considered "reached" when the
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

# --- Tip-over detection (IMU roll/pitch) -------------------------------------
# The lidar only sweeps ONE fixed horizontal plane, but the RGB-D camera is
# mounted on TOP of the chassis -- it can collide with an overhang (e.g. a
# low doorway lip) that the lidar sweep passes clean under. When that
# happens the front wheels can ride UP the obstacle and lift off the
# ground, tipping the chassis. The IMU's roll/pitch (see robot.py
# read_roll_pitch()) stay near 0 while the robot sits flat, so a large
# reading is a reliable "physically stuck on something the map doesn't
# know about yet" signal -- see MazeExplorer's tip-over check.
TIP_OVER_TILT_RAD = 0.15   # rad . |roll| or |pitch| beyond this
                            # counts as tipped over.

# When tipped, the map gets a wall SEGMENT stamped ahead of the robot, not
# just its single current point -- see MazeExplorer._mark_tip_over_obstacle().
# A single point only inflates to a small disk once the planner pads it,
# and whatever the camera actually hit is unknown in EXTENT (it could be a
# beam/ledge wider than the robot); a segment gives the very next replan
# (triggered by the same tip-over signal -- see Explorer/GoToPoint._drive()'s
# `tipped` handling) a wall wide enough that it can't just angle slightly
# and try to squeeze past the same spot again.
TIP_OVER_OBSTACLE_AHEAD_M         = ROBOT_RADIUS         # m, ahead of robot centre
TIP_OVER_OBSTACLE_HALF_WIDTH_M    = ROBOT_RADIUS * 1.5   # m, either side of heading
TIP_OVER_OBSTACLE_POINT_SPACING_M = GRID_RESOLUTION      # m, between marked points

# --- Reactive safety reflex (short-range IR distance sensors) ----------------
# A last line of defence, independent of the map: if a FRONT distance sensor
# reads closer than this while the robot is driving, it immediately backs off
# and forces a fresh plan (see MazeExplorer._safety_reflex).  Catches obstacles
# the planner missed -- odometry drift, thin/low objects, dynamic surprises.
RANGE_STOP_DIST_M   = 0.05   # m   Front clearance below which the reflex fires.
RANGE_BACKUP_TIME_S = 0.7    # s   How long to reverse after a trigger.
RANGE_BACKUP_SPEED  = 0.12   # m/s Reverse speed during the backup.
RANGE_MARK_MAX_M    = 0.30   # m   Only stamp the culprit obstacle into the map
                              #     if it is at least this close (avoids marking
                              #     far, harmless readings).
RANGE_MARK_STRENGTH = 3      # How many log-odds hits to stamp per detection, so
                              #     the mark survives a few depth-detector "free"
                              #     frames long enough for the replan to avoid it.

# Mount geometry of the two FRONT sensors, straight from the Rosbot PROTO
# (translation + z-rotation).  Used to place a detected obstacle point in the
# world so the replan actually routes around it.  (fwd, lateral+left, yaw).
RANGE_FL_FWD, RANGE_FL_LAT, RANGE_FL_YAW = 0.10,  0.05,  0.13   # front-left
RANGE_FR_FWD, RANGE_FR_LAT, RANGE_FR_YAW = 0.10, -0.05, -0.13   # front-right

# ===========================================================================
# Timing / cadence
# ===========================================================================

MAP_EVERY   = 2    # Integrate a lidar scan into the map every N control steps.
                    # Step ≈ dt ms, so MAP_EVERY=2 gives one map update per 2 dt.

PLAN_PERIOD = 3.0  # s   Force a replan this often while driving.  This allows
                    # the robot to react when newly discovered walls block the
                    # current path.

VIZ_EVERY   = 6    # Refresh the matplotlib live view every N steps.
                    # Rendering is slow; updating too often stalls the controller.

SPIN_SEED_TURN = 6.5  # rad  Total rotation during the SPIN_SEED phase
                        # (≈ 1 full turn = 2π ≈ 6.28 rad, slightly more to
                        # ensure a complete 360° view before planning).

# --- Lidar motion-distortion guard -------------------------------------------
# The simulated RPLidar A2 is a REAL "rotating" sensor (type="rotating",
# 12 Hz -- one full 360 deg sweep takes ~83 ms of simulated time).  The
# WHOLE 400-point range array returned by one read_lidar() call is folded
# into the map as if every point were captured INSTANTANEOUSLY at the
# CURRENT heading -- but if the robot itself is rotating while the physical
# sweep happens, that is not true: points from one end of the array are up
# to one full sweep period "stale" relative to the current heading.  While
# driving straight this barely matters (heading is nearly constant across
# 83 ms), but while TURNING it smears/mis-places ray endpoints -- at a wall
# 2 m away, even a ~2 deg heading error during the sweep shifts the computed
# hit position by several centimetres (multiple grid cells), enough for the
# FREE-marking sweep of a mis-angled ray to pass straight through an
# already-mapped wall cell and erase it back to "free".
#
# Fix: never integrate a lidar scan while the robot's angular velocity
# (measured between control steps, from the IMU-backed heading) is above
# this threshold -- the scan is simply skipped for that control step (pose
# and mission logic continue normally; the map just isn't updated from
# unreliable data).  SPIN_SEED_SPEED below is tuned to stay comfortably
# under this so SPIN_SEED can still integrate scans WHILE it deliberately
# rotates (that is the whole point of that phase).
LIDAR_MAX_ANGULAR_VEL_FOR_MAP = 0.35   # rad/s

# SPIN_SEED's own rotation speed -- deliberately SLOWER than a generic turn
# (MAX_TURN_SPEED) so its per-sweep heading drift stays small enough that
# its scans pass the guard above and are trustworthy.  At this speed, drift
# over one 83 ms lidar sweep is ~1 deg (a few cm at 2 m range) instead of
# the ~2.3 deg (~8 cm, multiple grid cells) the old 0.6*MAX_TURN_SPEED rate
# caused.  SPIN_SEED_TURN is unchanged, so the seeding turn simply takes a
# bit longer in sim-time -- a good trade for a map that doesn't get holes
# punched in it during the very first thing the robot ever does.
SPIN_SEED_SPEED = 0.40   # rad/s

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

# Run OccupancyGrid.clean_object_log() every N control steps (see that
# method's docstring). Cheap relative to CAMERA_EVERY's per-pixel cost, but
# still not worth doing every single step -- connected-component labelling
# over the full grid twice per colour.
OBJECT_CLEAN_EVERY = 5

# During GO_BLUE/GO_YELLOW, re-centre GoToPoint's target on the tracked
# object's CURRENT centroid (TrackedObject.world_xy) and force a fresh A*
# replan every N control steps (see MazeExplorer._act_go_to()). The colour
# log-odds centroid keeps refining as the camera gets more frames of the
# object, so the position GO_BLUE/GO_YELLOW started with can drift from
# where the object actually is -- this keeps the drive target current.
GOTO_RETARGET_EVERY = 15

# --- Hazard obstacle inflation ----------------------------------------------
HAZARD_INFLATE_CELLS = INFLATE_RADIUS_CELLS  # Same safety margin as walls.
INFLATE_CAMERA_OBSTACLE_CELLS = INFLATE_RADIUS_CELLS  
# ===========================================================================
# RGB-D depth-only obstacle detection (catches obstacles the LIDAR misses)
# ===========================================================================
# The lidar only sweeps one fixed horizontal plane, so anything entirely
# above or below that height (a low curb, a raised sill, a thin rail) is
# invisible to it.  depth_obstacle.py looks for small patches of near-
# CONSTANT depth in the camera image (a flat, camera-facing surface -- a
# real obstacle) that are clearly not just the ordinary floor, and feeds
# them into OccupancyGrid.integrate_camera_obstacle() -- a SEPARATE camera-
# only log-odds map (so the lidar can't erase these low obstacles), fused
# into the planner's blocked cells like walls and hazards.

# Two neighbouring sampled pixels count as "the same flat surface" if their
# depth differs by less than this.  Too small -> real flat surfaces get
# rejected due to ordinary depth-image quantisation; too large -> the
# gently-curving floor near the horizon starts getting misclassified as
# an obstacle (see depth_obstacle.py's module docstring).
CAMERA_FLAT_TOL_M = 0.03   # m

# --- Column-run obstacle detection (depth_obstacle.py) ----------------------
# A low, lidar-blind obstacle (a few cm high) shows up in the depth image as
# a VERTICAL RUN of near-constant depth inside a single column: every pixel
# on the obstacle's upright face is at the same forward distance, differing
# only in height.  The floor does the opposite -- its depth changes smoothly
# from one row to the next -- so a long constant-depth run in a column is a
# reliable "there is an upright surface here" signal.  See depth_obstacle.py.
#
# A pixel joins a run while its depth stays within this tolerance of the
# depth at the run's TOP pixel (so the whole run spans at most ~this much).
DEPTH_OBSTACLE_FLAT_TOL_M = 0.05   # m

# A run only counts as an obstacle if it is between MIN and MAX pixels TALL
# (measured in full-resolution image rows).  These bounds are what makes the
# detector target LOW obstacles specifically: at 0.5-1.5 m a few-cm object
# subtends roughly 20-80 px, whereas taller things (walls) subtend more and
# are already handled by the lidar; noise subtends less.  Tune per world.
DEPTH_OBSTACLE_MIN_RUN_PX = 3
DEPTH_OBSTACLE_MAX_RUN_PX = 60

# "Flying" obstacles: a hanging surface (e.g. a beam spanning the maze) is
# only a real obstacle if the robot cannot fit UNDERNEATH it.  For every
# detected run we back-project its BOTTOM pixel (the surface's lowest
# visible point) and check its height above the floor: if even that lowest
# point is above this clearance, the robot simply drives under -- do NOT
# mark it as a wall.  RosBot 2 is ~0.20 m tall (lidar tower included);
# keep a few cm of safety margin on top.
ROBOT_CLEARANCE_HEIGHT_M = 0.22   # m

# The camera only ever sees an obstacle's NEAR face -- everything behind it
# is occluded, so the object's true depth is unknowable from one viewpoint.
# If we marked only the face, A* would happily plan a path through the
# (unknown) body behind it.  So every camera hit is also padded this many
# metres FURTHER along the ray, filling in a plausible body.  The padding is
# ordinary log-odds evidence, NOT sticky: if a later viewpoint sees through
# those cells (a ray passes them to a farther hit), the normal free-along-
# the-ray updates erode the wrong padding again.  Over-padding costs at most
# a small detour; under-padding risks driving through the object.
# CAMERA_OBSTACLE_DEPTH_PAD_M = 0.24   # m
CAMERA_OBSTACLE_DEPTH_PAD_M = 0.20   # m

# One sampled column only tells us an obstacle exists AT that exact
# bearing -- it says nothing about how far it extends to either side, and
# CAMERA_SAMPLE_STRIDE means neighbouring bearings often aren't
# independently tested at all that frame. So each accepted hit is marked
# as a small LATERAL spread of cells (perpendicular to that ray's own
# bearing, at the same depth) instead of a single pixel-wide sliver --
# same reasoning as MazeExplorer._mark_tip_over_obstacle()'s wall segment.
CAMERA_OBSTACLE_LATERAL_POINTS     = 3            # cells marked per hit
CAMERA_OBSTACLE_LATERAL_SPACING_M  = GRID_RESOLUTION  # m between them

# ===========================================================================
# Coloured target objects (blue / yellow) -- detection + tracking
# ===========================================================================
# Two coloured objects somewhere in the maze must be found and treated as
# obstacles (see colored_objects.py: ColorObjectDetector + TrackedObject).
# Detection reuses the same camera intrinsics and RGB-D registration
# pipeline as the green floor hazard (see camera_geometry.py); only the
# HSV colour bands differ, and objects are NOT restricted to the floor
# plane (they can appear anywhere in the frame).

BLUE_HUE_MIN = 235   # degrees.  Blue hue band lower bound.
BLUE_HUE_MAX = 245   # degrees.  Upper bound.
BLUE_SAT_MIN = 0.70  # [0,1]. Minimum saturation (rules out grey/white).
BLUE_VAL_MIN = 0.55  # [0,1]. Minimum brightness (rules out near-black shadow).
# BLUE_HUE_MIN = 220   # degrees.  Blue hue band lower bound.
# BLUE_HUE_MAX = 250   # degrees.  Upper bound.
# BLUE_SAT_MIN = 0.35  # [0,1]. Minimum saturation (rules out grey/white).
# BLUE_VAL_MIN = 0.20  # [0,1]. Minimum brightness (rules out near-black shadow).


YELLOW_HUE_MIN = 50   # degrees.  Yellow hue band lower bound.
YELLOW_HUE_MAX = 65   # degrees.  Upper bound.
YELLOW_SAT_MIN = 0.70 # [0,1].
YELLOW_VAL_MIN = 0.55 # [0,1].

# --- No-depth fallback: LIDAR fills in where the depth camera has no reading -
#
# THE PROBLEM
# ------------
# The Astra depth camera reads inf for a pixel whenever it has no valid
# measurement there -- either the surface is CLOSER than its minimum range
# (~0.6 m, see robot.py -- camera_depth_min_range), or it's simply a bad IR
# return (shiny/dark coloured plastic is a common offender, even at normal
# range).  Those pixels are dropped by the `valid` filter -- BEFORE the
# colour test even runs.  When the robot is near the tracked object, the
# OBJECT ITSELF is often exactly what produces these inf pixels: its own
# pixels vanish, and the only pixels that still have valid depth are the
# ones peeking AROUND it (usually the wall behind).  Those get stamped as
# hits instead -- the object's own true position contributes NOTHING, and
# stray colour bleed at the object's silhouette can tag the wall behind it
# as the object instead.
#
# THE FIX
# --------
# For every pixel whose depth is inf/too-close, look up the LIDAR range at
# that SAME bearing (the lidar has no dead zone down to LIDAR_MIN_RANGE).
# If the lidar confirms something is genuinely close there, substitute its
# range as this pixel's depth and run the EXACT SAME registration + colour
# pipeline on it.  This recovers real hits (or free evidence) for the object
# itself at close range instead of silently discarding those pixels.
#
# Only trust the substitution when the lidar itself reports a range within
# this multiple of the camera's own minimum range -- i.e. the lidar agrees
# "yes, something is within the camera's blind zone here", not just "there
# happens to be a wall somewhere in roughly that direction".
CAMERA_NEAR_FALLBACK_SLACK = 1.25   # multiplier on camera_depth_min_range

# Same safety margin as walls/hazards when inflating for A*.
OBJECT_INFLATE_CELLS = INFLATE_RADIUS_CELLS

# Distance within which the robot counts as having "reached" a tracked object.
OBJECT_REACH_TOL = 0.30   # m

# How long (sim seconds) to suppress reachability checks for a tracked
# object after GoToPoint's OWN pathfinding fails to reach it -- see
# colored_objects.py TrackedObject.mark_unreachable() for the full "why".
# Long enough to give SEARCH_BLUE/SEARCH_YELLOW a real chance to explore
# more map (a few PLAN_PERIOD cycles) before the mission is allowed to
# retry GO_BLUE/GO_YELLOW; too short and it retries the identical failing
# route almost immediately, too long and a route that becomes reachable
# soon after gets ignored for a while.
OBJECT_UNREACHABLE_COOLDOWN_S = 12.0   # s

# --- Colour-object log-odds (Bayesian inverse sensor model, like the lidar) --
#
# The camera's colour detection is treated EXACTLY like the lidar wall map:
# every camera frame gives POSITIVE evidence (this cell looked blue/yellow)
# AND NEGATIVE evidence (this cell was clearly visible but was NOT that
# colour).  So a false detection is erased the next time the camera looks at
# that spot and disagrees -- see occupancy_grid.py: update_object_observation().
#
# These are kept SEPARATE from the lidar's L_FREE/L_OCC because colour
# detection is noisier and you will usually want to tune it on its own.
L_OBJ_OCC    =  1.00  # log-odds added   for a cell that matched the colour.
L_OBJ_FREE   = -0.80  # log-odds added   for a visible cell that did NOT match.
L_OBJ_CLAMP  =  8.0   # clamp object log-odds to +/- this (numerical safety).
P_OBJ_THRESH =  0.60  # a cell with p >= this is treated as holding the object.
# L_OBJ_OCC    =  1.00  # log-odds added   for a cell that matched the colour.
# L_OBJ_FREE   = -0.50  # log-odds added   for a visible cell that did NOT match.
# L_OBJ_CLAMP  =  8.0   # clamp object log-odds to +/- this (numerical safety).
# P_OBJ_THRESH =  0.60  # a cell with p >= this is treated as holding the object.


# --- Cluster sanity checks for OccupancyGrid.clean_object_log() -------------
#
# See that method's docstring for the full story. In short: a real tracked
# object is a small, isolated blob; a false colour detection smeared onto a
# wall (RGB-D registration/parallax error at close range) is either a tiny
# speckle or ends up fused into the maze's much larger connected wall
# network. Two independent checks catch each failure mode, and cells that
# fail either one have their accumulated log-odds evidence permanently
# wiped -- see clean_object_log()'s docstring for why that counts as
# "forever" even though no separate sticky blacklist is kept.

# Step 1 -- denoise: an 8-connected camera-detection cluster with this many
# pixels or fewer is treated as speckle noise and dropped outright.
OBJECT_CLUSTER_MIN_PIXELS = 3   # discards 1-2 px clusters

# Step 2 -- isolation check: the connected component of occ_mask() (the
# lidar wall map) that a surviving cluster touches must be no larger than
# this many cells, or it's judged to be fused into the wall network rather
# than a standalone object and gets wiped too.
OBJECT_ISOLATED_BLOB_MAX_CELLS = 50   # ~0.08 m^2 at GRID_RESOLUTION=0.04




# How close (in world metres) a lidar-detected wall cell must be to a raw
# camera colour detection before we treat it as the SAME physical object
# (see occupancy_grid.py: OccupancyGrid.reconciled_object_mask()).  The
# lidar is trusted over the camera here -- if the lidar later finds that
# cell to be free instead, it silently drops out of the reconciled mask on
# its own (no separate "undo" logic needed; see that method's docstring).
OBJECT_WALL_MATCH_DISTANCE_M = 0.24   # m (~6 cells at GRID_RESOLUTION=0.04)



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
