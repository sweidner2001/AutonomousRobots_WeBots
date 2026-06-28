from Maze1.controllers.Controller_v1.Helper import HelperMethods
from Maze1.controllers.Controller_v1.DataWrapper import SensorData, Pose2D
from Maze1.controllers.Controller_v1.GridWorld import GridWorld
from Maze1.controllers.Controller_v1.Mission import Mission
from Maze1.controllers.Controller_v1.PlanPath import PlanPath
from controller import Robot

import math


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
    WHEEL_RADIUS = 0.085 / 2.0
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

        self.map = GridWorld(self.MAP_SIZE_M, self.MAP_RESOLUTION_M)

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
        self.camera_rgb = self.robot.getDevice("camera rgb")
        self.camera_rgb.enable(self.timestep)
        self.camera_depth = self.robot.getDevice("camera depth")
        self.camera_depth.enable(self.timestep)

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
        USE_BREEZYSLAM = True  # ← flip to True after: pip install breezyslam

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
        self.pose.theta = HelperMethods.wrap_angle(self.pose.theta + dtheta)



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
            self.pose.theta = HelperMethods.wrap_angle(math.radians(yaw_deg))

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
        image = self.camera_rgb.getImage()
        if image is None:
            return None

        width = self.camera_rgb.getWidth()
        height = self.camera_rgb.getHeight()
        step = 4
        count = 0
        sum_x = 0

        for y in range(0, height, step):
            for x in range(0, width, step):
                r = self.camera_rgb.imageGetRed(image, width, x, y)
                g = self.camera_rgb.imageGetGreen(image, width, x, y)
                b = self.camera_rgb.imageGetBlue(image, width, x, y)

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
        fov = self.camera_rgb.getFov()
        bearing = ((centroid_x / width) - 0.5) * fov

        # Estimate range using nearest lidar beam near the camera bearing.
        ranges, angle_min, angle_inc = self._read_lidar()
        idx = int((bearing - angle_min) / max(angle_inc, 1e-6))
        idx = int(HelperMethods.clamp(idx, 0, len(ranges) - 1))
        obs_range = ranges[idx]
        if not math.isfinite(obs_range):
            return None
        obs_range = HelperMethods.clamp(obs_range, 0.2, self.lidar_max)

        gx = self.pose.x + obs_range * math.cos(self.pose.theta + bearing)
        gy = self.pose.y + obs_range * math.sin(self.pose.theta + bearing)
        return (gx, gy)



    # def _detect_color_blob(self, target_color):
    #     """
    #     Completes the missing vision system using OpenCV.
    #     Returns the (x, y) world coordinates of the target if found, else None.
    #     """
    #     import cv2
    #     import numpy as np

    #     # 1. Get image from Webots and convert to OpenCV format (BGRA to BGR)
    #     img_array = np.frombuffer(self.camera.getImage(), dtype=np.uint8)
    #     img_array = img_array.reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
    #     frame = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #     # 2. Define color masks (Tune these HSV values if lighting is weird)
    #     if target_color == "BLUE":
    #         lower_bound = np.array([100, 150, 0])
    #         upper_bound = np.array([140, 255, 255])
    #     elif target_color == "YELLOW":
    #         lower_bound = np.array([20, 100, 100])
    #         upper_bound = np.array([30, 255, 255])
    #     else:
    #         return None

    #     # 3. Find blobs
    #     mask = cv2.inRange(hsv, lower_bound, upper_bound)
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     if not contours:
    #         return None

    #     # 4. Find the largest contour (ignore tiny noise pixels)
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     if cv2.contourArea(largest_contour) < 100: # Minimum pixel area
    #         return None

    #     # 5. Calculate Centroid and Bearing
    #     M = cv2.moments(largest_contour)
    #     if M["m00"] == 0: return None
    #     cx = int(M["m10"] / M["m00"])
        
    #     # Calculate horizontal angle offset from the center of the camera
    #     img_width = self.camera.getWidth()
    #     fov = self.camera.getFov()
    #     angle_offset = ((cx / img_width) - 0.5) * fov
        
    #     # 6. Project to World Coordinates using LiDAR mapping
    #     target_bearing = wrap_angle(self.pose.theta - angle_offset)
        
    #     # Find the LiDAR distance at this bearing.
    #     # We add 0.5 meters to push the target "through" the window glass on the map.
    #     ranges = self.lidar.getRangeImage()
    #     # Approximate the index based on the bearing
    #     # (Assuming Lidar 0 is front. You may need to tweak this index math based on your specific Lidar mounting)
    #     center_index = len(ranges) // 2 
    #     index_offset = int(-angle_offset / (self.lidar_fov / self.lidar_resolution))
    #     lidar_idx = clamp(center_index + index_offset, 0, len(ranges)-1)
        
    #     distance = ranges[lidar_idx]
    #     if math.isinf(distance) or distance > self.lidar_max:
    #         distance = 3.0 # Guess a distance if Lidar misses it
            
    #     distance += 0.5 # Window Labyrinth offset trick

    #     target_x = self.pose.x + distance * math.cos(target_bearing)
    #     target_y = self.pose.y + distance * math.sin(target_bearing)

    #     return (target_x, target_y)
    


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
        return False
    
        # # Important check for "window trap" situations:
        # start = self.map.world_to_grid(self.pose.x, self.pose.y)
        # goal = self.map.world_to_grid(goal_world[0], goal_world[1])
        # if not self.map.in_bounds(goal[0], goal[1]):
        #     return False

        # inflate_cells = int(self.INFLATION_RADIUS_M / self.MAP_RESOLUTION_M)
        # blocked = self.map.inflated_occupancy(inflate_cells)
        # path = astar(start, goal, blocked)
        # return len(path) > 0



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
            else:
                # Explore: Plan path to nearest unknown frontier
                if self.step_count % self.PLAN_PERIOD_STEPS == 0:
                    robot_cell = self.map.world_to_grid(self.pose.x, self.pose.y)
                    frontier = self.map.nearest_frontier(robot_cell)
                    if frontier:
                        blocked = self.map.inflated_occupancy(int(self.INFLATION_RADIUS_M / self.MAP_RESOLUTION_M))
                        # self.current_path = astar(robot_cell, frontier, blocked)
                        self.path_index = 0



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
        heading_err = HelperMethods.wrap_angle(heading_target - self.pose.theta)

        omega = HelperMethods.clamp(2.2 * heading_err, -1.6, 1.6)
        v = 0.35 * (1.0 - min(1.0, abs(heading_err) / math.pi))

        min_front = self._front_sector_clearance(ranges, angle_min, angle_inc)
        if min_front < self.SAFE_FRONT_DIST:
            # Reactive safety override.
            v = 0.0
            omega = 1.2 if heading_err >= 0.0 else -1.2

        return v, omega



def _compute_cmd(self):
        """
        Follows the A* path (self.current_path) using a simple Pure Pursuit controller.
        Returns (left_speed, right_speed)
        """
        if not self.current_path or self.path_index >= len(self.current_path):
            return 0.0, 0.0 # Stop

        # Get next waypoint from grid
        gx, gy = self.current_path[self.path_index]
        target_x, target_y = self.map.grid_to_world(gx, gy)

        # Calculate distance and heading error
        dx = target_x - self.pose.x
        dy = target_y - self.pose.y
        distance = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        heading_error = HelperMethods.wrap_angle(target_heading - self.pose.theta)

        # If close enough to waypoint, move to the next one
        if distance < self.MAP_RESOLUTION_M * 2:
            self.path_index += 1
            return self._compute_cmd()

        # Simple P-Controller for steering
        base_speed = 5.0
        kp_steer = 10.0
        
        # If the turn is too sharp, spin in place first
        if abs(heading_error) > 0.5: 
            base_speed = 0.0

        left_speed = base_speed - (kp_steer * heading_error)
        right_speed = base_speed + (kp_steer * heading_error)

        return HelperMethods.clamp(left_speed, -self.MAX_WHEEL_SPEED, self.MAX_WHEEL_SPEED), \
               HelperMethods.clamp(right_speed, -self.MAX_WHEEL_SPEED, self.MAX_WHEEL_SPEED)



    # def _compute_cmd(self, ranges, angle_min, angle_inc):
    #     """Reactive obstacle avoidance: drive forward, turn away from walls.

    #     No path planning involved. The robot drives straight until something
    #     appears within SAFE_FRONT_DIST ahead, then turns toward the side
    #     with more free space.

    #     Sector definitions (robot-frame angles):
    #       front : beams within ±30°  of 0
    #       left  : beams between +30° and +90°
    #       right : beams between -90° and -30°

    #     Returns
    #     -------
    #     (v, omega) : linear speed (m/s) and angular rate (rad/s).
    #     """
    #     front_min = self.lidar_max
    #     left_sum = 0.0
    #     right_sum = 0.0
    #     left_count = 0
    #     right_count = 0

    #     for i, r in enumerate(ranges):
    #         if not math.isfinite(r):
    #             continue
    #         a = angle_min + i * angle_inc

    #         if abs(a) < math.radians(30.0):
    #             front_min = min(front_min, r)
    #         elif math.radians(30.0) <= a < math.radians(90.0):
    #             left_sum += r
    #             left_count += 1
    #         elif math.radians(-90.0) < a <= math.radians(-30.0):
    #             right_sum += r
    #             right_count += 1

    #     left_avg = (left_sum / left_count) if left_count > 0 else self.lidar_max
    #     right_avg = (right_sum / right_count) if right_count > 0 else self.lidar_max

    #     if front_min < self.SAFE_FRONT_DIST:
    #         # Obstacle ahead: stop and turn toward the side with more space.
    #         v = 0.0
    #         omega = 1.2 if left_avg >= right_avg else -1.2
    #     else:
    #         # Path is clear: drive forward at full speed.
    #         v = 0.35
    #         omega = 0.0

    #     return v, omega




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
        left = HelperMethods.clamp(left, -self.MAX_WHEEL_SPEED, self.MAX_WHEEL_SPEED)
        right = HelperMethods.clamp(right, -self.MAX_WHEEL_SPEED, self.MAX_WHEEL_SPEED)

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
            # self._update_map(ranges, angle_min, angle_inc)

            # self._update_mission()
            # self._plan_if_needed()

            linear, angular = self._compute_cmd(ranges, angle_min, angle_inc)
            self._set_velocity(linear, angular)





if __name__ == "__main__":
    ControllerV1().run()