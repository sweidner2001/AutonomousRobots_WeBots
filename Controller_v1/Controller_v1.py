"""Controller_v1 — No-ROS autonomous maze navigation for Webots Rosbot.

High-level goal
---------------
Navigate an unknown maze, reach a blue cylinder first, then a yellow one.
Avoid walls at all times.

Sensor setup
------------
- RpLidarA2    : 2-D laser scanner, used for mapping and pose tracking.
- Astra RGB    : colour camera, used only for blue/yellow blob detection.
- Wheel encoders: provide short-term dead-reckoning pose between scans.

Algorithmic pipeline (executed every simulation tick)
------------------------------------------------------
1. _read_odometry()     : update robot pose from wheel encoder deltas.
2. _read_lidar()        : fetch new laser scan from the sensor.
3. _update_map()        : insert scan into occupancy grid
                          (or run full BreezySLAM if enabled).
4. _update_mission()    : detect coloured targets, advance mission state.
5. _plan_if_needed()    : run A* on inflated map to get a waypoint path.
6. _compute_cmd()       : track next waypoint; override if obstacle near.
7. _set_velocity()      : send resulting wheel speeds to motors.

BreezySLAM integration
-----------------------
Set USE_BREEZYSLAM = True in _setup_breezyslam_hook() after installing:
    pip install breezyslam
When enabled, BreezySLAM replaces steps 1-3 with a proper scan-matching
SLAM back-end that produces a corrected pose and an occupancy map.
The rest of the pipeline (mission, planning, control) is identical.
"""

from collections import deque
import heapq
import math

from controller import Robot


def clamp(value, low, high):
    """Clamp 'value' so it never goes below 'low' or above 'high'.

    Used everywhere a physical quantity must stay within safe limits,
    e.g. wheel speed, sensor range, pixel index.
    """
    return max(low, min(high, value))


def wrap_angle(angle):
    """Normalise any angle (radians) into the range [-pi, +pi].

    Without this, heading errors like 'I need to turn 359 degrees right'
    would be computed instead of the correct '1 degree left', which would
    cause the robot to spin endlessly.
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class Pose2D:
    """Holds the robot's estimated position and heading in the 2-D world frame.

    Attributes
    ----------
    x     : float  — east/west position in metres (positive = east).
    y     : float  — north/south position in metres (positive = north).
    theta : float  — heading angle in radians, measured counter-clockwise
                     from the positive x-axis.
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta


class OccupancyGrid:
    """2-D grid map of the environment used for path planning and exploration.

    The world is divided into square cells of 'resolution' metres each side.
    Every cell stores one of three states:
        unknown (-1) : the robot has never observed this area.
        free    ( 0) : a lidar ray passed through here → drivable space.
        occupied( 1) : a lidar ray ended here → wall or obstacle.

    This grid is the shared data structure that connects:
        - the mapper (fills cells from lidar rays)
        - BreezySLAM (imports its output bytes here)
        - the path planner (A* reads occupied/free cells)
        - the frontier explorer (looks for unknown cells next to free ones)
    """

    def __init__(self, size_m=12.0, resolution=0.05):
        """Allocate a square grid covering 'size_m' x 'size_m' metres.

        Parameters
        ----------
        size_m     : total side length of the map in metres.
        resolution : side length of one cell in metres (e.g. 0.05 = 5 cm).

        The origin (0, 0) of the world frame is placed at the map centre.
        """
        self.resolution = resolution
        self.width = int(size_m / resolution)   # number of columns
        self.height = int(size_m / resolution)  # number of rows
        self.origin_x = -0.5 * size_m           # world-x of the left edge
        self.origin_y = -0.5 * size_m           # world-y of the bottom edge
        # 2-D list: self.grid[row][col], initialised to 'free' (0).
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.unknown = -1   # cell state: not yet seen
        self.free = 0       # cell state: laser ray passed through
        self.occ = 1        # cell state: laser ray ended here (obstacle)

    def world_to_grid(self, x, y):
        """Convert a real-world position (metres) to grid cell indices.

        Example: x=0.0, y=0.0 → the centre cell of the map.
        Returns (column_index, row_index).  No bounds check here; call
        in_bounds() afterwards when safety matters.
        """
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Convert grid cell indices back to the centre point in metres.

        The '+0.5' offsets the result to the cell centre rather than the
        cell corner, which gives smoother waypoint navigation.
        """
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return x, y

    def in_bounds(self, gx, gy):
        """Return True if (gx, gy) is a valid cell index inside the grid.

        Always call this before reading or writing a cell to avoid
        IndexError when lidar rays point outside the mapped area.
        """
        return 0 <= gx < self.width and 0 <= gy < self.height

    def set_free(self, gx, gy):
        """Mark a cell as free (drivable), but never overwrite an occupied cell.

        A cell that was already seen as occupied (a wall) should not be
        erased by a passing ray — the wall was seen first and is trusted.
        """
        if self.in_bounds(gx, gy):
            if self.grid[gy][gx] <= 0:   # only overwrite unknown or already-free
                self.grid[gy][gx] = self.free

    def set_occ(self, gx, gy):
        """Mark a cell as occupied (obstacle/wall).

        Called for the endpoint of a lidar ray that hit something before
        reaching the sensor's maximum range.
        """
        if self.in_bounds(gx, gy):
            self.grid[gy][gx] = self.occ

    def is_occ(self, gx, gy):
        """Return True if the cell contains a known obstacle.

        Out-of-bounds cells are treated as occupied so the planner
        never plans a path outside the map boundary.
        """
        if not self.in_bounds(gx, gy):
            return True
        return self.grid[gy][gx] == self.occ

    def is_free(self, gx, gy):
        """Return True if the cell is confirmed free (laser ray passed through).

        Unknown cells return False — the robot treats unseen areas as
        'not yet confirmed safe', not as guaranteed free space.
        """
        if not self.in_bounds(gx, gy):
            return False
        return self.grid[gy][gx] == self.free

    def bresenham(self, x0, y0, x1, y1):
        """Return all grid cells on the straight line from (x0,y0) to (x1,y1).

        This is Bresenham's line algorithm — an efficient way to find which
        discrete grid cells a straight lidar ray passes through.

        Used by insert_scan() to:
          - mark all intermediate cells as FREE (the ray passed through them)
          - mark the final cell as OCCUPIED (the ray hit something there)

        Returns a list of (col, row) tuples in order from start to end.
        """
        # Integer line rasterization.
        points = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points

    def insert_scan(self, pose, ranges, angle_min, angle_inc, range_max):
        """Update the occupancy grid using one full lidar scan.

        For each lidar beam:
          1. Compute the absolute direction of the beam in the world frame
             (robot heading + relative beam angle).
          2. Compute the endpoint in world coordinates from the measured range.
          3. Convert start and end to grid cells.
          4. Trace a line between them with Bresenham → mark all intermediate
             cells FREE (the laser physically passed through them).
          5. If the range is below the sensor maximum, the beam hit an object:
             mark the endpoint cell OCCUPIED.

        Parameters
        ----------
        pose      : Pose2D — current robot position and heading.
        ranges    : list of floats — measured distances for each beam (metres).
        angle_min : float — angle of the first beam relative to robot forward.
        angle_inc : float — angular step between consecutive beams.
        range_max : float — sensor maximum range; beams at this value are ignored.
        """
        # Insert one full lidar scan into the occupancy map:
        robot_gx, robot_gy = self.world_to_grid(pose.x, pose.y)
        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            r = clamp(r, 0.0, range_max)
            beam = pose.theta + angle_min + i * angle_inc
            ex = pose.x + r * math.cos(beam)
            ey = pose.y + r * math.sin(beam)
            end_gx, end_gy = self.world_to_grid(ex, ey)
            ray = self.bresenham(robot_gx, robot_gy, end_gx, end_gy)
            if len(ray) < 2:
                continue
            for px, py in ray[:-1]:
                self.set_free(px, py)
            # Treat near-max returns as no-hit so we only carve free space.
            if r < 0.98 * range_max:
                self.set_occ(ray[-1][0], ray[-1][1])

    def inflated_occupancy(self, radius_cells):
        """Return a boolean 2-D grid where obstacles are expanded outward.

        Why this is necessary:
          The robot has a physical body (~26 cm wide).  If A* plans a path
          right next to a wall, the robot body will collide even though the
          centre-point path is technically clear.
          By 'inflating' every obstacle by radius_cells before planning,
          we force the planner to keep a safe margin from all walls.

        Parameters
        ----------
        radius_cells : int — number of cells to expand each obstacle outward.
                       e.g. INFLATION_RADIUS_M / MAP_RESOLUTION_M = 0.12/0.05 = 2

        Returns
        -------
        A list-of-lists of booleans: True = blocked, False = passable.
        This format is passed directly to astar().
        """
        # Inflate obstacles so planner keeps a safety distance from walls.
        inflated = [[self.is_occ(x, y) for x in range(self.width)] for y in range(self.height)]
        if radius_cells <= 0:
            return inflated
        occ_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if inflated[y][x]:
                    occ_cells.append((x, y))
        for ox, oy in occ_cells:
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy > radius_cells * radius_cells:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if self.in_bounds(nx, ny):
                        inflated[ny][nx] = True
        return inflated

    def nearest_frontier(self, start):
        """Find the nearest exploration frontier to 'start' using BFS.

        A 'frontier' is a free cell that has at least one unknown neighbour.
        It represents the boundary between explored and unexplored space.

        Exploration strategy: always drive to the nearest frontier.
        This gradually expands the explored area until the whole maze is known.

        Parameters
        ----------
        start : (col, row) — current robot cell.

        Returns
        -------
        (col, row) of the nearest frontier, or None if no frontier exists
        (meaning the entire reachable area has been explored).
        """
        # Frontier = free cell next to unknown cell.
        sx, sy = start
        if not self.in_bounds(sx, sy):
            return None

        visited = set()
        q = deque([(sx, sy)])
        visited.add((sx, sy))

        neigh4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            x, y = q.popleft()
            if self.is_frontier(x, y):
                return (x, y)
            for dx, dy in neigh4:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited or not self.in_bounds(nx, ny):
                    continue
                if not self.is_free(nx, ny):
                    continue
                visited.add((nx, ny))
                q.append((nx, ny))
        return None

    def is_frontier(self, x, y):
        """Return True if cell (x, y) qualifies as an exploration frontier.

        Conditions:
          1. The cell itself must be FREE (robot can physically be there).
          2. At least one of the 8 surrounding cells must be UNKNOWN
             (meaning there is unexplored space nearby worth visiting).
        """
        if not self.is_free(x, y):
            return False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny) and self.grid[ny][nx] == self.unknown:
                    return True
        return False

    def import_from_breezyslam_bytes(self, slam_bytes, slam_size_pixels):
        # -----------------------------------------------------------------
        # THIS IS THE CONNECTION BETWEEN BREEZYSLAM AND OUR GRID.
        #
        # BreezySLAM stores its map as a flat byte array (length = pixels^2).
        # Each byte is a value 0-255:
        #   0   = unknown / not observed
        #   < 127 = likely occupied (wall)
        #   >= 127 = likely free (drivable space)
        #
        # We read every byte, threshold it, and write the result into our
        # OccupancyGrid so the planner (A*, frontier search) can use it.
        # Without this method the two worlds (SLAM and planner) are disconnected.
        # -----------------------------------------------------------------

        # BreezySLAM pixel count may differ from our grid size.
        # We compute a scale factor so both grids map to the same real-world area.
        scale = slam_size_pixels / self.width  # e.g. 1.0 if they match

        for gy in range(self.height):
            for gx in range(self.width):
                # Find the corresponding pixel in the SLAM byte array.
                sx = int(gx * scale)
                sy = int(gy * scale)
                sx = clamp(sx, 0, slam_size_pixels - 1)
                sy = clamp(sy, 0, slam_size_pixels - 1)

                byte_val = slam_bytes[sy * slam_size_pixels + sx]

                # Translate BreezySLAM byte value into our three-state grid.
                if byte_val == 0:
                    # BreezySLAM uses 0 for "not yet observed".
                    self.grid[gy][gx] = self.unknown
                elif byte_val < 127:
                    # Low value = high obstacle probability → mark occupied.
                    self.grid[gy][gx] = self.occ
                else:
                    # High value = free space the robot can drive through.
                    self.grid[gy][gx] = self.free


def astar(start, goal, blocked):
    """Find the shortest path from 'start' to 'goal' on a 2-D grid.

    This is the A* search algorithm.  It is like Dijkstra's shortest-path
    algorithm, but accelerated by a heuristic that estimates the remaining
    distance to the goal (Manhattan distance here).

    The algorithm maintains a priority queue of cells sorted by
    f = g + h  where:
        g = actual cost to reach this cell so far
        h = heuristic estimate of cost still remaining (Manhattan distance)

    It expands the cheapest cell first and stops as soon as the goal is
    popped, guaranteeing the optimal path.

    Parameters
    ----------
    start   : (col, row) — starting grid cell.
    goal    : (col, row) — target grid cell.
    blocked : 2-D list of booleans [row][col] — True = obstacle (inflated map).

    Returns
    -------
    List of (col, row) tuples from start to goal (inclusive),
    or an empty list [] if no path exists.
    """
    # Classic A* path planning on a 2D grid.
    h = len(blocked)
    w = len(blocked[0]) if h else 0

    def in_bounds(x, y):
        return 0 <= x < w and 0 <= y < h

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    neigh8 = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, 1.414),
        (-1, 1, 1.414),
        (1, -1, 1.414),
        (-1, -1, 1.414),
    ]

    visited = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy, cost in neigh8:
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny) or blocked[ny][nx]:
                continue
            tentative = g_score[current] + cost
            ncell = (nx, ny)
            if tentative < g_score.get(ncell, float("inf")):
                came_from[ncell] = current
                g_score[ncell] = tentative
                f = tentative + heuristic(ncell, goal)
                heapq.heappush(open_set, (f, ncell))
    return []


class Mission:
    """Named constants for the mission state machine.

    The robot progresses through these states in strict order:
        SEARCH_BLUE  : Explore the maze while scanning camera for blue.
        GO_BLUE      : Blue detected and reachable — navigate to it.
        SEARCH_YELLOW: Blue reached. Explore while scanning for yellow.
        GO_YELLOW    : Yellow detected and reachable — navigate to it.
        DONE         : Yellow reached. Stop motors, mission complete.
    """
    SEARCH_BLUE = "SEARCH_BLUE"
    GO_BLUE = "GO_BLUE"
    SEARCH_YELLOW = "SEARCH_YELLOW"
    GO_YELLOW = "GO_YELLOW"
    DONE = "DONE"


class ControllerV1:
    """Top-level controller: owns all sensors, the map, and the mission logic.

    Class-level constants are the tuning knobs you will most likely adjust:
        WHEEL_RADIUS       : physical radius of the drive wheels (metres).
        AXLE_TRACK         : distance between left and right wheels (metres).
        MAX_WHEEL_SPEED    : safety cap on wheel angular velocity (rad/s).
        SAFE_FRONT_DIST    : emergency-stop distance ahead (metres).
        GOAL_REACHED_DIST  : distance at which a goal is declared 'reached'.
        PLAN_PERIOD_STEPS  : how many ticks between full A* replans.
        MAP_SIZE_M         : side length of the square map in metres.
        MAP_RESOLUTION_M   : cell size in metres (smaller = finer, slower).
        INFLATION_RADIUS_M : how much to grow obstacles for safety margin.
    """

    # Kinematics (Rosbot-like defaults). Tune if your model differs.
    WHEEL_RADIUS = 0.043
    AXLE_TRACK = 0.265
    MAX_WHEEL_SPEED = 26.0

    # Navigation tuning.
    SAFE_FRONT_DIST = 0.35
    GOAL_REACHED_DIST = 0.22
    PLAN_PERIOD_STEPS = 7

    # Map/planning tuning.
    MAP_SIZE_M = 12.0
    MAP_RESOLUTION_M = 0.05
    INFLATION_RADIUS_M = 0.12

    def __init__(self):
        """Initialise robot, sensors, map, and mission state.

        Called once when the controller starts.  Sets up:
          - Webots robot handle and simulation timestep.
          - All sensor and motor device handles (_init_devices).
          - The shared occupancy grid map.
          - The starting mission state (SEARCH_BLUE).
          - Optional BreezySLAM back-end (_setup_breezyslam_hook).
        """
        # Main runtime object: devices, map, mission state and planner buffers.
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self._init_devices()
        self.pose = Pose2D(0.0, 0.0, 0.0)
        self.prev_left = None
        self.prev_right = None

        self.map = OccupancyGrid(self.MAP_SIZE_M, self.MAP_RESOLUTION_M)

        self.state = Mission.SEARCH_BLUE
        self.blue_goal = None
        self.yellow_goal = None
        self.active_goal = None
        self.current_path = []
        self.path_index = 0
        self.step_count = 0

        self.use_breezyslam = False
        self._setup_breezyslam_hook()

    def _init_devices(self):
        """Get handles to all hardware devices and enable sensors.

        Webots requires calling robot.getDevice() to get a handle before
        reading from or writing to any sensor or actuator.

        Motors are set to velocity-control mode by setting position = infinity.
        Initial velocity is 0 so the robot starts stationary.

        All sensors must be enabled with an update period (milliseconds)
        before they start returning data.
        """
        # Initialize wheel motors in velocity mode.
        self.front_left_motor = self.robot.getDevice("fl_wheel_joint")
        self.front_right_motor = self.robot.getDevice("fr_wheel_joint")
        self.rear_left_motor = self.robot.getDevice("rl_wheel_joint")
        self.rear_right_motor = self.robot.getDevice("rr_wheel_joint")

        for m in (
            self.front_left_motor,
            self.front_right_motor,
            self.rear_left_motor,
            self.rear_right_motor,
        ):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # Wheel encoders are used for odometry (dead-reckoning).
        self.fl_enc = self.robot.getDevice("front left wheel motor sensor")
        self.fr_enc = self.robot.getDevice("front right wheel motor sensor")
        self.rl_enc = self.robot.getDevice("rear left wheel motor sensor")
        self.rr_enc = self.robot.getDevice("rear right wheel motor sensor")
        for s in (self.fl_enc, self.fr_enc, self.rl_enc, self.rr_enc):
            s.enable(self.timestep)

        # RGB camera is used only for color-based target detection.
        self.camera = self.robot.getDevice("camera rgb")
        self.camera.enable(self.timestep)

        # 2D lidar is the main mapping/localization source.
        self.lidar = self.robot.getDevice("laser")
        self.lidar.enable(self.timestep)
        self.lidar_resolution = self.lidar.getHorizontalResolution()
        self.lidar_fov = self.lidar.getFov()
        self.lidar_max = self.lidar.getMaxRange()

    def _setup_breezyslam_hook(self):
        # -----------------------------------------------------------------
        # BREEZYSLAM INITIALIZATION
        #
        # BreezySLAM needs two things to start:
        #   1. A Laser object that describes the physical sensor properties
        #      (how many beams, field of view, min/max range, scan rate).
        #   2. The map dimensions in pixels and real-world meters so it can
        #      allocate its internal occupancy map buffer.
        #
        # After this, every call to slam.update() will:
        #   - compare the new scan against the stored map
        #   - correct the pose estimate (localization)
        #   - update the internal map with new observations (mapping)
        #
        # Set USE_BREEZYSLAM = True here once the package is installed.
        # -----------------------------------------------------------------
        USE_BREEZYSLAM = False  # ← flip to True after: pip install breezyslam

        self.slam = None
        self.slam_map_pixels = int(self.MAP_SIZE_M / self.MAP_RESOLUTION_M)  # must match OccupancyGrid width
        self.slam_map_bytes = bytearray(self.slam_map_pixels * self.slam_map_pixels)

        if not USE_BREEZYSLAM:
            self.use_breezyslam = False
            return

        try:
            from breezyslam.algorithms import RMHC_SLAM  # pylint: disable=import-outside-toplevel
            from breezyslam.sensors import Laser         # pylint: disable=import-outside-toplevel

            # Describe the RpLidarA2 sensor geometry.
            # Arguments: num_beams, scan_rate_hz, detection_angle_degrees,
            #            distance_no_detection_mm, detection_margin, offset_mm
            lidar_sensor = Laser(
                self.lidar_resolution,   # number of beams in one scan
                10,                      # typical RPLidar A2 scan rate in Hz
                360,                     # full 360-degree scan
                12000,                   # max detection range in mm (12 m)
                0,                       # no detection margin
                0,                       # no physical offset
            )

            # Create the SLAM object.
            # MAP_SIZE_M is the real-world size of the square map.
            self.slam = RMHC_SLAM(lidar_sensor, self.slam_map_pixels, self.MAP_SIZE_M)
            self.use_breezyslam = True
            print("[SLAM] BreezySLAM initialized successfully.")

        except Exception as e:
            print("[SLAM] BreezySLAM not available, using fallback mapper. Reason:", e)
            self.use_breezyslam = False

    def _read_odometry(self):
        """Estimate robot pose from wheel encoder increments (dead-reckoning).

        How it works:
          1. Read current cumulative encoder angle for left and right wheels.
          2. Compute how many radians each wheel turned since last call.
          3. Multiply by wheel radius to get distance each wheel travelled.
          4. Average left+right distances → forward distance (ds).
          5. Difference right-left, divided by axle width → heading change (dθ).
          6. Update x, y, theta using standard differential-drive kinematics.

        Limitation:
          Odometry accumulates error over time (wheel slip, timing jitter).
          With BreezySLAM enabled, the SLAM-corrected pose overwrites this
          estimate after each scan update.
        """
        # Differential-drive odometry from wheel encoder increments.
        left = 0.5 * (self.fl_enc.getValue() + self.rl_enc.getValue())
        right = 0.5 * (self.fr_enc.getValue() + self.rr_enc.getValue())

        if self.prev_left is None:
            self.prev_left = left
            self.prev_right = right
            return

        d_left = (left - self.prev_left) * self.WHEEL_RADIUS
        d_right = (right - self.prev_right) * self.WHEEL_RADIUS
        self.prev_left = left
        self.prev_right = right

        ds = 0.5 * (d_left + d_right)
        dtheta = (d_right - d_left) / self.AXLE_TRACK
        heading = self.pose.theta + 0.5 * dtheta
        self.pose.x += ds * math.cos(heading)
        self.pose.y += ds * math.sin(heading)
        self.pose.theta = wrap_angle(self.pose.theta + dtheta)

    def _read_lidar(self):
        """Fetch the latest lidar scan and compute per-beam angle parameters.

        Returns
        -------
        ranges    : tuple of floats — one range measurement per beam (metres).
        angle_min : float — angle of beam 0 relative to robot forward (radians).
        angle_inc : float — angular increment between consecutive beams (radians).

        The angle of beam i in the robot frame is: angle_min + i * angle_inc.
        Adding self.pose.theta converts this to world-frame bearing.
        """
        # Read lidar ranges and precompute angular calibration.
        ranges = self.lidar.getRangeImage()
        angle_min = -0.5 * self.lidar_fov
        angle_inc = self.lidar_fov / max(1, self.lidar_resolution - 1)
        return ranges, angle_min, angle_inc

    def _update_map(self, ranges, angle_min, angle_inc):
        # -----------------------------------------------------------------
        # STEP 2 OF THE MAIN LOOP: UPDATE POSE AND MAP
        #
        # There are two paths through this function:
        #
        # PATH A — BreezySLAM enabled (USE_BREEZYSLAM = True):
        #   1. Convert lidar ranges from meters to millimeters (BreezySLAM unit).
        #   2. Call slam.update(scan_mm) — BreezySLAM now:
        #        a) matches new scan against its stored map
        #        b) corrects the pose estimate (this is the LOCALIZATION part)
        #        c) updates its internal occupancy map (this is the MAPPING part)
        #   3. Read the corrected pose back from BreezySLAM into self.pose.
        #        → This replaces the raw odometry pose with a better one.
        #   4. Call slam.getmap(byte_array) to copy the SLAM map bytes out.
        #   5. Call self.map.import_from_breezyslam_bytes() to translate those
        #        bytes into our OccupancyGrid → planner can now use the SLAM map.
        #
        # PATH B — Fallback (no BreezySLAM):
        #   Uses odometry pose (already updated in _read_odometry) and
        #   manually casts lidar rays into our OccupancyGrid (Bresenham lines).
        #   This is less accurate but requires zero extra libraries.
        # -----------------------------------------------------------------

        if self.use_breezyslam and self.slam is not None:
            # --- PATH A: BreezySLAM ---

            # BreezySLAM expects distances in millimetres as integers.
            scan_mm = [int(r * 1000) for r in ranges]

            # update() does the full SLAM cycle internally.
            self.slam.update(scan_mm)

            # Read the SLAM-corrected pose.
            # BreezySLAM returns position in mm measured from the map corner,
            # not from the robot start.  We convert to meters and re-center.
            x_mm, y_mm, yaw_deg = self.slam.getpos()
            half_map = 0.5 * self.MAP_SIZE_M
            self.pose.x = (x_mm / 1000.0) - half_map
            self.pose.y = (y_mm / 1000.0) - half_map
            self.pose.theta = wrap_angle(math.radians(yaw_deg))

            # Copy the SLAM occupancy map into our OccupancyGrid.
            # After this call, self.map contains the SLAM-built world model.
            self.slam.getmap(self.slam_map_bytes)
            self.map.import_from_breezyslam_bytes(self.slam_map_bytes, self.slam_map_pixels)

        else:
            # --- PATH B: Fallback — manual ray-casting into our own grid ---
            # Uses current odometry pose (set by _read_odometry) + lidar scan.
            self.map.insert_scan(self.pose, ranges, angle_min, angle_inc, self.lidar_max)

    def _detect_color_blob(self, target):
        """Detect a coloured object in the camera image and estimate its position.

        Step 1 — Colour segmentation:
          Scan every 4th pixel of the image (stride=4 for speed).
          For each pixel test if it matches the target colour:
            blue   : blue channel dominant and significantly above red and green.
            yellow : red and green channels both high, blue channel low.

        Step 2 — Find centroid:
          Average the x-coordinates of all matching pixels to find the
          horizontal centre of the colour blob in the image.

        Step 3 — Compute bearing:
          Convert pixel centroid position to a horizontal angle using the
          camera's field of view.  Centre of image = 0 rad (straight ahead).

        Step 4 — Estimate range via lidar:
          Look up the lidar beam closest to the camera bearing.
          This gives a distance estimate to whatever is in that direction.
          This is approximate but good enough to seed the map-goal position.

        Step 5 — Project to world frame:
          Combine robot pose + bearing + range → world (x, y) of the object.

        Parameters
        ----------
        target : str — 'blue' or 'yellow'.

        Returns
        -------
        (world_x, world_y) in metres, or None if no blob detected.
        """
        # Lightweight color segmentation for blue/yellow cylinders.
        image = self.camera.getImage()
        if image is None:
            return None

        width = self.camera.getWidth()
        height = self.camera.getHeight()
        step = 4
        count = 0
        sum_x = 0

        for y in range(0, height, step):
            for x in range(0, width, step):
                r = self.camera.imageGetRed(image, width, x, y)
                g = self.camera.imageGetGreen(image, width, x, y)
                b = self.camera.imageGetBlue(image, width, x, y)

                if target == "blue":
                    ok = b > 120 and b > 1.4 * r and b > 1.4 * g
                else:
                    ok = r > 120 and g > 120 and b < 110

                if ok:
                    count += 1
                    sum_x += x

        if count < 25:
            return None

        centroid_x = sum_x / float(count)
        fov = self.camera.getFov()
        bearing = ((centroid_x / width) - 0.5) * fov

        # Estimate range using nearest lidar beam near the camera bearing.
        ranges, angle_min, angle_inc = self._read_lidar()
        idx = int((bearing - angle_min) / max(angle_inc, 1e-6))
        idx = int(clamp(idx, 0, len(ranges) - 1))
        obs_range = ranges[idx]
        if not math.isfinite(obs_range):
            return None
        obs_range = clamp(obs_range, 0.2, self.lidar_max)

        gx = self.pose.x + obs_range * math.cos(self.pose.theta + bearing)
        gy = self.pose.y + obs_range * math.sin(self.pose.theta + bearing)
        return (gx, gy)

    def _is_reachable(self, goal_world):
        """Check whether a world-frame goal can actually be reached by A*.

        This is the key safeguard against the 'window trap' problem:
          The robot might see the blue cylinder through a window in a wall.
          The camera says 'object is at (x, y)' and bearing looks clear,
          but the only path to it goes all the way around the wall.

        How it works:
          Run a full A* search from robot cell to goal cell on the inflated
          map.  If A* finds no path (returns empty list), the goal is
          not yet reachable — we continue exploring instead of driving
          straight into a dead end.

        Parameters
        ----------
        goal_world : (x, y) in metres.

        Returns
        -------
        True if a path exists, False otherwise.
        """
        # Important check for "window trap" situations:
        start = self.map.world_to_grid(self.pose.x, self.pose.y)
        goal = self.map.world_to_grid(goal_world[0], goal_world[1])
        if not self.map.in_bounds(goal[0], goal[1]):
            return False

        inflate_cells = int(self.INFLATION_RADIUS_M / self.MAP_RESOLUTION_M)
        blocked = self.map.inflated_occupancy(inflate_cells)
        path = astar(start, goal, blocked)
        return len(path) > 0

    def _update_mission(self):
        """Advance the mission state machine based on what the camera currently sees.

        Called once per control tick.  Checks current camera for colour blobs,
        verifies reachability, and transitions between mission states.

        State transitions:
          SEARCH_BLUE  → GO_BLUE      : blue blob seen AND map path exists to it.
          GO_BLUE      → SEARCH_YELLOW: robot is within GOAL_REACHED_DIST of blue.
          SEARCH_YELLOW→ GO_YELLOW    : yellow blob seen AND map path exists to it.
          GO_YELLOW    → DONE         : robot is within GOAL_REACHED_DIST of yellow.

        Important: yellow is only searched AFTER blue is reached, enforcing order.
        """
        # Mission state machine:
        blue_obs = self._detect_color_blob("blue")
        yellow_obs = self._detect_color_blob("yellow")

        if self.state == Mission.SEARCH_BLUE:
            if blue_obs is not None and self._is_reachable(blue_obs):
                self.blue_goal = blue_obs
                self.active_goal = self.blue_goal
                self.state = Mission.GO_BLUE

        elif self.state == Mission.GO_BLUE:
            if self.blue_goal is not None:
                if math.hypot(self.pose.x - self.blue_goal[0], self.pose.y - self.blue_goal[1]) < self.GOAL_REACHED_DIST:
                    self.active_goal = None
                    self.state = Mission.SEARCH_YELLOW

        elif self.state == Mission.SEARCH_YELLOW:
            if yellow_obs is not None and self._is_reachable(yellow_obs):
                self.yellow_goal = yellow_obs
                self.active_goal = self.yellow_goal
                self.state = Mission.GO_YELLOW

        elif self.state == Mission.GO_YELLOW:
            if self.yellow_goal is not None:
                if math.hypot(self.pose.x - self.yellow_goal[0], self.pose.y - self.yellow_goal[1]) < self.GOAL_REACHED_DIST:
                    self.active_goal = None
                    self.state = Mission.DONE

    def _choose_goal(self):
        """Decide where the robot should go next.

        Priority order:
          1. If the mission has an active target (blue or yellow object position),
             return that world position as the goal.
          2. Otherwise, find the nearest frontier on the map and return it.
             This drives autonomous exploration of unknown areas until a
             target is found.

        Returns
        -------
        (world_x, world_y) of the chosen goal, or None if no goal available
        (e.g. map fully explored and no target found yet).
        """
        # Priority:
        if self.active_goal is not None:
            return self.active_goal

        # Exploration fallback: nearest frontier cell in current map.
        start = self.map.world_to_grid(self.pose.x, self.pose.y)
        frontier = self.map.nearest_frontier(start)
        if frontier is None:
            return None
        return self.map.grid_to_world(frontier[0], frontier[1])

    def _plan_if_needed(self):
        """Run A* path planning when the path is stale or missing.

        Replanning is triggered when:
          - No current path exists yet.
          - PLAN_PERIOD_STEPS ticks have passed since the last plan
            (periodically adapts to newly discovered walls and openings).

        Steps:
          1. Ask _choose_goal() for the current target world position.
          2. Convert world position to grid cell.
          3. Build the inflated obstacle grid from the current map.
          4. Run A* from robot cell to goal cell.
          5. Store the resulting path in self.current_path.

        If A* finds no path (obstacle-blocked or unexplored), current_path
        is cleared and the robot will spin-in-place (_compute_cmd fallback).
        """
        # Replan periodically to adapt to newly discovered walls/openings.
        if self.step_count % self.PLAN_PERIOD_STEPS != 0 and self.current_path:
            return

        goal_world = self._choose_goal()
        if goal_world is None:
            self.current_path = []
            self.path_index = 0
            return

        start = self.map.world_to_grid(self.pose.x, self.pose.y)
        goal = self.map.world_to_grid(goal_world[0], goal_world[1])
        if not self.map.in_bounds(goal[0], goal[1]):
            self.current_path = []
            self.path_index = 0
            return

        inflate_cells = int(self.INFLATION_RADIUS_M / self.MAP_RESOLUTION_M)
        blocked = self.map.inflated_occupancy(inflate_cells)
        path = astar(start, goal, blocked)
        self.current_path = path
        self.path_index = 0

    def _front_sector_clearance(self, ranges, angle_min, angle_inc):
        """Return the minimum lidar distance inside the forward-facing cone.

        Scans all beams within ±25 degrees of straight ahead and returns
        the nearest obstacle distance.  This value is used in _compute_cmd()
        as a reactive safety check: if something is closer than
        SAFE_FRONT_DIST, the robot stops and turns away regardless of
        what the planner says.

        Parameters
        ----------
        ranges, angle_min, angle_inc : scan data from _read_lidar().

        Returns
        -------
        float — nearest obstacle distance in the front cone (metres).
        """
        # Find nearest obstacle in a forward cone.
        min_front = self.lidar_max
        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            a = angle_min + i * angle_inc
            if abs(a) < math.radians(25.0):
                min_front = min(min_front, r)
        return min_front

    def _compute_cmd(self, ranges, angle_min, angle_inc):
        """Compute the desired linear velocity (v) and angular velocity (omega).

        Normal operation — path following:
          1. Skip any waypoints that are already behind the robot
             (path_index advances automatically).
          2. Compute the heading angle to the next waypoint.
          3. Compute heading error = desired_heading - current_heading.
          4. Proportional angular controller: omega = 2.2 * heading_error.
             (Positive error = must turn left; negative = turn right.)
          5. Forward speed is reduced when heading error is large so the
             robot slows down and turns before driving forward.

        Safety override (reactive layer):
          If the minimum distance in the forward cone is less than
          SAFE_FRONT_DIST, forward speed is forced to zero and the robot
          turns away from the obstacle.  This fires regardless of the plan.

        Fallback (no path):
          Rotate in place so the map can update and a new plan can form.

        Returns
        -------
        (v, omega) : linear speed (m/s) and angular rate (rad/s).
        """
        # Path tracking controller (go-to-waypoint with heading correction)
        if self.state == Mission.DONE:
            return 0.0, 0.0

        if not self.current_path:
            return 0.0, 0.8

        # Skip waypoints that are already close.
        while self.path_index < len(self.current_path):
            wx, wy = self.map.grid_to_world(
                self.current_path[self.path_index][0],
                self.current_path[self.path_index][1],
            )
            if math.hypot(self.pose.x - wx, self.pose.y - wy) > 0.15:
                break
            self.path_index += 1

        if self.path_index >= len(self.current_path):
            return 0.0, 0.0

        wx, wy = self.map.grid_to_world(
            self.current_path[self.path_index][0],
            self.current_path[self.path_index][1],
        )
        heading_target = math.atan2(wy - self.pose.y, wx - self.pose.x)
        heading_err = wrap_angle(heading_target - self.pose.theta)

        omega = clamp(2.2 * heading_err, -1.6, 1.6)
        v = 0.35 * (1.0 - min(1.0, abs(heading_err) / math.pi))

        min_front = self._front_sector_clearance(ranges, angle_min, angle_inc)
        if min_front < self.SAFE_FRONT_DIST:
            # Reactive safety override.
            v = 0.0
            omega = 1.2 if heading_err >= 0.0 else -1.2

        return v, omega

    def _set_velocity(self, linear, angular):
        """Convert (linear, angular) robot velocity into left/right wheel speeds.

        Differential-drive inverse kinematics:
            v_left  = (2*v - omega * track) / (2 * wheel_radius)
            v_right = (2*v + omega * track) / (2 * wheel_radius)

        Where:
          v      : desired forward speed of the robot centre (m/s).
          omega  : desired rotation rate (rad/s), positive = counter-clockwise.
          track  : distance between the two wheel contact points (metres).
          radius : wheel radius (metres).

        Results are clamped to MAX_WHEEL_SPEED and sent to all four motors
        (Rosbot has two motors per side, front and rear).
        """
        # Convert desired (v, omega) into left/right wheel angular speeds.
        left = (2.0 * linear - angular * self.AXLE_TRACK) / (2.0 * self.WHEEL_RADIUS)
        right = (2.0 * linear + angular * self.AXLE_TRACK) / (2.0 * self.WHEEL_RADIUS)
        left = clamp(left, -self.MAX_WHEEL_SPEED, self.MAX_WHEEL_SPEED)
        right = clamp(right, -self.MAX_WHEEL_SPEED, self.MAX_WHEEL_SPEED)

        self.front_left_motor.setVelocity(left)
        self.rear_left_motor.setVelocity(left)
        self.front_right_motor.setVelocity(right)
        self.rear_right_motor.setVelocity(right)

    def run(self):
        """Main control loop — executed repeatedly until Webots stops the sim.

        Each call to robot.step(timestep) advances the simulation by one tick
        and returns the sensor data for that instant.  Returning -1 means the
        simulation has ended.

        Execution order per tick:
          1. _read_odometry()   : update pose from wheel encoders.
          2. _read_lidar()      : fetch new laser scan.
          3. _update_map()      : insert scan (or run BreezySLAM).
          4. _update_mission()  : check camera, advance state machine.
          5. _plan_if_needed()  : replan global path when stale.
          6. _compute_cmd()     : choose (v, omega) for this tick.
          7. _set_velocity()    : apply wheel speeds.
        """
        # Main control loop (executed each simulation step).
        while self.robot.step(self.timestep) != -1:
            self.step_count += 1

            self._read_odometry()
            ranges, angle_min, angle_inc = self._read_lidar()
            self._update_map(ranges, angle_min, angle_inc)

            self._update_mission()
            self._plan_if_needed()

            linear, angular = self._compute_cmd(ranges, angle_min, angle_inc)
            self._set_velocity(linear, angular)


if __name__ == "__main__":
    ControllerV1().run()
