"""
config.py
=========
Every tunable number for the Graph-SLAM maze mapper lives here, grouped by
topic.  Think of it as the project's "settings panel": change behaviour from
this one file instead of hunting through the code.

UNITS:  metres (m), radians (rad), seconds (s).

COORDINATE FRAMES
-----------------
We never use the Webots supervisor, so the robot does NOT know its true world
position.  Instead we define the SLAM frame as:

        the robot's STARTING pose = origin (x=0, y=0, theta=0).

`theta = 0` means "the direction the robot faced at start-up".  Everything
(odometry, the pose graph, the map) is expressed in this robot-centric frame.
The occupancy grid is centred on the origin so the robot begins in the middle.
"""

# ===========================================================================
# 1. Robot physical dimensions  (Husarion RosBot 2 / Webots Rosbot PROTO)
# ===========================================================================
# These are facts about the hardware, read off the PROTO geometry.

WHEEL_RADIUS    = 0.043   # m   wheel radius -> converts encoder rad to metres
WHEEL_BASE      = 0.22    # m   effective left<->right track width (skid-steer;
                          #     slightly larger than nominal to absorb slip)
ROBOT_RADIUS    = 0.13    # m   body radius (used by phase-2 path inflation)
MAX_WHEEL_SPEED = 26.0    # rad/s  motor limit; all wheel commands clamp to this

# The 2-D lidar sits a little in FRONT of the robot centre (PROTO: ~0.02 m).
LIDAR_OFFSET_X  = 0.02    # m   positive = forward of the robot's centre

# ===========================================================================
# 2. Lidar scan processing  (RPLidar A2)
# ===========================================================================
# Resolution, field-of-view and max range are read LIVE from the sensor in
# robot.py, so the code adapts automatically if the PROTO changes.

LIDAR_MIN_RANGE = 0.20    # m   ignore returns nearer than this (sensor blind
                          #     zone + the robot seeing its own body)
LIDAR_MAX_RANGE = 6.0     # m   ignore returns farther than this for mapping
                          #     (long rays are noisier; a maze is small anyway)

# Webots delivers the range image as one distance per ray, ordered across the
# field of view.  We turn ray index -> bearing angle with:
#     bearing(i) = ANGLE_SIGN * (FoV/2 - (i + 0.5) * FoV/N) + ANGLE_OFFSET
# bearing = 0 is straight ahead, > 0 is to the LEFT (counter-clockwise).
# If the finished map comes out left-right MIRRORED, flip LIDAR_ANGLE_SIGN.
LIDAR_ANGLE_SIGN   = 1.0
LIDAR_ANGLE_OFFSET = 0.0  # rad  extra rotation if the sensor is mounted turned

# Scan matching and mapping do not need every ray.  Keep at most this many
# points per scan (evenly strided) to keep ICP fast.
SCAN_MAX_POINTS = 360

# ===========================================================================
# 3. Occupancy grid  (the digital map)
# ===========================================================================
# A 2-D array of cells; each stores the log-odds that it holds a wall.
#   p ~ 1  -> occupied (wall, drawn black)
#   p ~ .5 -> unknown  (never seen, drawn grey)
#   p ~ 0  -> free     (empty,  drawn white)

GRID_RESOLUTION = 0.05    # m per cell (5 cm squares)
GRID_WIDTH_M    = 16.0    # m   -> 320 cells wide
GRID_HEIGHT_M   = 16.0    # m   -> 320 cells tall
# Origin placed so that world (0,0) = the robot start = the grid centre.
GRID_ORIGIN_X   = -GRID_WIDTH_M  / 2.0
GRID_ORIGIN_Y   = -GRID_HEIGHT_M / 2.0

# --- Log-odds inverse-sensor-model values ---------------------------------
# Each ray subtracts L_FREE from cells it passes through and adds L_OCC to the
# cell it hits.  |L_OCC| > |L_FREE| so a thin wall is not erased by one stray
# free beam.  Clamp keeps values bounded and lets the map change over time.
#     p = 1 / (1 + e^(-L))
L_FREE  = -0.35
L_OCC   =  0.95
L_CLAMP =  8.0
P_OCC_THRESH  = 0.65      # p >= this -> treated as wall  (phase-2 planning)
P_FREE_THRESH = 0.35      # p <= this AND observed -> treated as free

# ===========================================================================
# 4. Wheel + IMU odometry  (the SLAM motion prior)
# ===========================================================================
# Odometry only seeds the SLAM front-end; lidar scan-matching corrects it.
# Heading is taken straight from the IMU (reliable on a skid-steer base);
# forward distance comes from averaging the wheel encoders.
# (No tunables yet -- listed here so the section exists for future use.)

# ===========================================================================
# 5. Scan matcher  (ICP -- iterative closest point)
# ===========================================================================
ICP_MAX_ITERS      = 30     # max refinement iterations
ICP_MAX_CORR_DIST  = 0.40   # m   reject point pairs farther apart than this
ICP_CONVERGE_EPS   = 1e-4   # stop when the pose update is smaller than this
ICP_MIN_POINTS     = 25     # need at least this many matched points to trust it

# ===========================================================================
# 6. Pose-graph SLAM  (keyframes, loop closure, optimisation)
# ===========================================================================
# A new keyframe (graph node) is added once the robot has moved far enough or
# turned enough since the previous one.
KEYFRAME_DIST   = 0.25    # m
KEYFRAME_ANGLE  = 0.30    # rad  (~17 deg)

# Loop closure: when a NEW keyframe is near an OLD one, try to scan-match them.
LOOP_SEARCH_RADIUS = 1.50  # m   only consider old keyframes within this range
LOOP_MIN_GAP       = 12    # don't loop-close to the last N keyframes (too recent)
LOOP_FITNESS_MIN   = 0.55  # accept a closure only if >= this fraction of points
                           # match (inlier ratio from ICP)
LOOP_RESIDUAL_MAX  = 0.10  # m   ...and the mean inlier distance is below this

# Edge information (inverse covariance) = how much we trust each constraint.
# Larger = more trusted.  Loop closures from good scan-matches are trusted a
# little more than raw odometry.  Stored as diagonal (x, y, theta) weights.
INFO_ODOM_XY     = 200.0   # 1/sigma^2 with sigma ~ 0.07 m
INFO_ODOM_THETA  = 400.0   # 1/sigma^2 with sigma ~ 0.05 rad
INFO_LOOP_XY     = 600.0   # tighter: ICP on real geometry is confident
INFO_LOOP_THETA  = 800.0

GRAPH_OPT_ITERS  = 20      # Gauss-Newton iterations per optimisation call

# ===========================================================================
# 7. Tele-operation  (manual driving while SLAM maps)
# ===========================================================================
# Webots keyboard input -- read through the normal Robot API, no supervisor.
TELEOP_LINEAR  = 0.20     # m/s   forward / reverse speed
TELEOP_ANGULAR = 1.20     # rad/s turn speed

# ===========================================================================
# 8. Timing / cadence  (how often each subsystem runs, in control steps)
# ===========================================================================
SLAM_EVERY = 1   # run the SLAM front-end every N steps
MAP_EVERY  = 4   # fuse the live scan into the grid every N steps (live view)
VIZ_EVERY  = 8   # refresh the matplotlib window every N steps (rendering is slow)

# ===========================================================================
# 9. Output files  (written next to this controller)
# ===========================================================================
SAVE_MAP_PNG = "map_final.png"   # rendered map image
SAVE_MAP_NPY = "map_final.npy"   # raw log-odds array (NumPy binary)
