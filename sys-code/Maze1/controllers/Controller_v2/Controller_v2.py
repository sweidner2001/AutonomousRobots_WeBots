from controller import Robot
from Maze1.controllers.Controller_v1.GridWorld import GridWorld
from Maze1.controllers.Controller_v1.PlanPath import PlanPath
from Maze1.controllers.Controller_v1.Helper import HelperMethods
from Maze1.controllers.Controller_v1.DataWrapper import Pose2D

import math


class ExplorerController:
    """Autonomous maze explorer: builds an occupancy map while driving.

    The robot has no prior knowledge of the environment.  It uses:
      - Wheel encoders  : dead-reckoning odometry (fallback pose estimate).
      - RPLidar A2      : 360-degree scan for mapping and obstacle avoidance.
      - BreezySLAM      : scan-matching to correct odometry drift.
      - GridWorld       : 2-D occupancy grid shared by mapper and planner.
      - A* + frontiers  : plan paths toward unexplored areas of the map.

    The control loop runs at every Webots simulation step:
      1. Read odometry → update dead-reckoning pose.
      2. Read lidar    → get current scan.
      3. Update map    → BreezySLAM or manual ray-casting.
      4. Replan        → find nearest frontier, run A* to it.
      5. Compute cmd   → follow path, with reactive safety override.
      6. Set velocity  → send wheel speeds to motors.
    """

    # --- Robot kinematics ---
    WHEEL_RADIUS = 0.085 / 2.0   # metres
    AXLE_TRACK   = 0.265          # metres (left-right wheel distance)
    MAX_WHEEL_SPEED = 26.0        # rad/s

    # --- Navigation ---
    SAFE_FRONT_DIST   = 0.35   # metres: emergency stop threshold
    WAYPOINT_REACH_M  = 0.15   # metres: distance to consider a waypoint reached

    # --- Map ---
    # MAP_SIZE_M must be >= 2 * maze_diagonal so the robot never reaches the
    # map edge regardless of where it starts.  For a 10x10m maze the diagonal
    # is ~14.1m, so 30m gives a safe margin in every direction.
    # Resolution 0.10 m → 300×300 = 90 000 cells (vs 600×600 at 0.05 m).
    MAP_SIZE_M       = 30.0   # side length of square map (metres)
    MAP_RESOLUTION_M = 0.10   # cell size (metres) — coarser = much faster
    INFLATION_RADIUS_M = 0.20 # obstacle inflation (metres)

    # --- Timing ---
    PLAN_PERIOD_STEPS = 20    # simulation ticks between full replans

    def __init__(self):
        self.robot    = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self._init_devices()

        # Pose estimate (updated by odometry, corrected by SLAM).
        self.pose      = Pose2D(0.0, 0.0, 0.0)
        self.prev_left  = None
        self.prev_right = None

        # Occupancy grid map of the environment.
        self.map = GridWorld(self.MAP_SIZE_M, self.MAP_RESOLUTION_M)

        # Path planner.
        self.planner      = PlanPath(self.map)
        self.current_path = []
        self.path_index   = 0
        self.step_count   = 0

        # BreezySLAM setup (graceful fallback if not installed).
        self._setup_slam()

    # ------------------------------------------------------------------
    # Device initialisation
    # ------------------------------------------------------------------

    def _init_devices(self):
        """Get handles for all motors and sensors and enable them."""

        # Wheel motors in velocity-control mode.
        self.fl_motor = self.robot.getDevice("fl_wheel_joint")
        self.fr_motor = self.robot.getDevice("fr_wheel_joint")
        self.rl_motor = self.robot.getDevice("rl_wheel_joint")
        self.rr_motor = self.robot.getDevice("rr_wheel_joint")
        for m in (self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # Wheel encoders for odometry.
        self.fl_enc = self.robot.getDevice("front left wheel motor sensor")
        self.fr_enc = self.robot.getDevice("front right wheel motor sensor")
        self.rl_enc = self.robot.getDevice("rear left wheel motor sensor")
        self.rr_enc = self.robot.getDevice("rear right wheel motor sensor")
        for s in (self.fl_enc, self.fr_enc, self.rl_enc, self.rr_enc):
            s.enable(self.timestep)

        # 2-D lidar: primary sensor for mapping and avoidance.
        self.lidar = self.robot.getDevice("laser")
        self.lidar.enable(self.timestep)
        self.lidar_resolution = self.lidar.getHorizontalResolution()
        self.lidar_fov        = self.lidar.getFov()
        self.lidar_max        = self.lidar.getMaxRange()

    # ------------------------------------------------------------------
    # SLAM setup
    # ------------------------------------------------------------------

    def _setup_slam(self):
        """Try to initialise BreezySLAM; fall back to manual ray-casting."""
        self.slam            = None
        self.use_breezyslam  = False
        self.slam_map_pixels = int(self.MAP_SIZE_M / self.MAP_RESOLUTION_M)
        self.slam_map_bytes  = bytearray(self.slam_map_pixels ** 2)

        try:
            from breezyslam.algorithms import RMHC_SLAM
            from breezyslam.sensors import Laser

            laser_model = Laser(
                self.lidar_resolution,  # number of beams
                10,                     # scan rate Hz (RPLidar A2 typical)
                360,                    # full-circle scan
                12000,                  # max range mm
                0, 0,
            )
            self.slam = RMHC_SLAM(laser_model, self.slam_map_pixels, self.MAP_SIZE_M)
            self.use_breezyslam = True
            print("[SLAM] BreezySLAM ready.")
        except Exception as e:
            print("[SLAM] Fallback mapper active. Reason:", e)

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def _read_odometry(self):
        """Update self.pose from wheel encoder increments (dead-reckoning)."""
        left  = 0.5 * (self.fl_enc.getValue() + self.rl_enc.getValue())
        right = 0.5 * (self.fr_enc.getValue() + self.rr_enc.getValue())

        if self.prev_left is None:
            self.prev_left  = left
            self.prev_right = right
            return

        d_left  = (left  - self.prev_left)  * self.WHEEL_RADIUS
        d_right = (right - self.prev_right) * self.WHEEL_RADIUS
        self.prev_left  = left
        self.prev_right = right

        ds     = 0.5 * (d_left + d_right)
        dtheta = (d_right - d_left) / self.AXLE_TRACK
        mid    = self.pose.theta + 0.5 * dtheta

        self.pose.x     += ds * math.cos(mid)
        self.pose.y     += ds * math.sin(mid)
        self.pose.theta  = HelperMethods.wrap_angle(self.pose.theta + dtheta)

    def _read_lidar(self):
        """Return (ranges, angle_min, angle_inc) for the current scan."""
        ranges    = self.lidar.getRangeImage()
        angle_min = -0.5 * self.lidar_fov
        angle_inc = self.lidar_fov / max(1, self.lidar_resolution - 1)
        return ranges, angle_min, angle_inc

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def _update_map(self, ranges, angle_min, angle_inc):
        """Insert the latest lidar scan into the occupancy grid.

        PATH A (BreezySLAM): scan-matching corrects pose drift, then the
        SLAM map bytes are imported into GridWorld so A* can use them.

        PATH B (fallback): raw odometry pose + Bresenham ray-casting.
        """
        if self.use_breezyslam and self.slam is not None:
            # BreezySLAM expects integer mm values; 0 means "no detection".
            # Passing float inf (Webots no-hit value) silently breaks SLAM.
            scan_mm = [
                int(r * 1000) if math.isfinite(r) else 0
                for r in ranges
            ]
            self.slam.update(scan_mm)

            # Overwrite odometry pose with the SLAM-corrected estimate.
            x_mm, y_mm, yaw_deg = self.slam.getpos()
            half = 0.5 * self.MAP_SIZE_M
            self.pose.x     = (x_mm / 1000.0) - half
            self.pose.y     = (y_mm / 1000.0) - half
            self.pose.theta = HelperMethods.wrap_angle(math.radians(yaw_deg))

            # Also ray-cast into GridWorld using the corrected pose so the
            # occupancy grid is populated immediately (SLAM bytes take several
            # scans before they contain usable free/occupied data).
            self.map.insert_scan(self.pose, ranges, angle_min, angle_inc, self.lidar_max)
        else:
            self.map.insert_scan(self.pose, ranges, angle_min, angle_inc, self.lidar_max)

    # ------------------------------------------------------------------
    # Exploration planning
    # ------------------------------------------------------------------

    def _replan(self):
        """Find the nearest unexplored frontier and compute an A* path to it.

        Called every PLAN_PERIOD_STEPS ticks or when the current path is empty.
        If no frontier exists (map fully explored), current_path stays empty
        and the robot stops.
        """
        if self.step_count % self.PLAN_PERIOD_STEPS != 0 and self.current_path:
            return

        start_cell = self.map.world_to_grid(self.pose.x, self.pose.y)
        frontier   = self.map.nearest_frontier(start_cell)

        if frontier is None:
            # Entire reachable area explored — stop.
            self.current_path = []
            self.path_index   = 0
            print("[Explorer] Map fully explored. Stopping.")
            return

        goal_cell = frontier
        if not self.map.in_bounds(goal_cell[0], goal_cell[1]):
            self.current_path = []
            self.path_index   = 0
            return

        inflate   = int(self.INFLATION_RADIUS_M / self.MAP_RESOLUTION_M)
        blocked   = self.map.inflated_occupancy(inflate)
        path      = self.planner.astar(start_cell, goal_cell, blocked)

        self.current_path = path
        self.path_index   = 0

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _front_clearance(self, ranges, angle_min, angle_inc):
        """Minimum lidar distance in the ±30-degree forward cone."""
        min_dist = self.lidar_max
        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            a = angle_min + i * angle_inc
            if abs(a) < math.radians(30.0):
                min_dist = min(min_dist, r)
        return min_dist

    def _compute_cmd(self, ranges, angle_min, angle_inc):
        """Compute (linear_speed, angular_speed) for this control tick.

        Normal operation — waypoint following:
          Steer toward the next A* waypoint with a proportional heading
          controller.  Forward speed drops when the heading error is large
          so the robot turns before it drives.

        Safety override — reactive avoidance:
          If an obstacle enters the ±30-degree front cone within
          SAFE_FRONT_DIST, forward motion stops and the robot turns toward
          whichever side has more free space (left vs. right average distance).

        Fallback — no path:
          Rotate slowly in place so new lidar scans can update the map and
          a fresh frontier can be found.
        """
        # No path available: spin slowly to gather new map data.
        if not self.current_path:
            return 0.0, 0.5

        # Advance path index past already-reached waypoints.
        while self.path_index < len(self.current_path):
            wx, wy = self.map.grid_to_world(
                self.current_path[self.path_index][0],
                self.current_path[self.path_index][1],
            )
            if math.hypot(self.pose.x - wx, self.pose.y - wy) > self.WAYPOINT_REACH_M:
                break
            self.path_index += 1

        if self.path_index >= len(self.current_path):
            # Path fully traversed: trigger a new plan next tick.
            self.current_path = []
            return 0.0, 0.0

        # Target the next waypoint.
        wx, wy = self.map.grid_to_world(
            self.current_path[self.path_index][0],
            self.current_path[self.path_index][1],
        )
        heading_target = math.atan2(wy - self.pose.y, wx - self.pose.x)
        heading_err    = HelperMethods.wrap_angle(heading_target - self.pose.theta)

        # Proportional angular controller; forward speed reduces with error.
        omega = HelperMethods.clamp(2.5 * heading_err, -1.8, 1.8)
        v     = 0.4 * (1.0 - min(1.0, abs(heading_err) / math.radians(90.0)))

        # Reactive safety override.
        if self._front_clearance(ranges, angle_min, angle_inc) < self.SAFE_FRONT_DIST:
            # Measure average space on each side to decide which way to turn.
            left_sum, left_n   = 0.0, 0
            right_sum, right_n = 0.0, 0
            for i, r in enumerate(ranges):
                if not math.isfinite(r):
                    continue
                a = angle_min + i * angle_inc
                if math.radians(30.0) <= a < math.radians(90.0):
                    left_sum += r;  left_n  += 1
                elif math.radians(-90.0) < a <= math.radians(-30.0):
                    right_sum += r; right_n += 1
            left_avg  = (left_sum  / left_n)  if left_n  else self.lidar_max
            right_avg = (right_sum / right_n) if right_n else self.lidar_max

            v     = 0.0
            omega = 1.2 if left_avg >= right_avg else -1.2

        return v, omega

    def _set_velocity(self, linear, angular):
        """Convert (v, omega) into individual wheel angular speeds and apply."""
        left  = (2.0 * linear - angular * self.AXLE_TRACK) / (2.0 * self.WHEEL_RADIUS)
        right = (2.0 * linear + angular * self.AXLE_TRACK) / (2.0 * self.WHEEL_RADIUS)
        left  = HelperMethods.clamp(left,  -self.MAX_WHEEL_SPEED, self.MAX_WHEEL_SPEED)
        right = HelperMethods.clamp(right, -self.MAX_WHEEL_SPEED, self.MAX_WHEEL_SPEED)

        self.fl_motor.setVelocity(left)
        self.rl_motor.setVelocity(left)
        self.fr_motor.setVelocity(right)
        self.rr_motor.setVelocity(right)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _visualize(self):
        """Show the GridWorld map and (if active) the raw BreezySLAM map.

        Renders two OpenCV windows side by side:
          'GridWorld'  : the occupancy grid used by A* and frontier search.
                         Gray=unknown, White=free, Black=wall,
                         Orange=planned path, Red dot=robot.
          'BreezySLAM' : the raw internal SLAM byte map (only when active).
                         Black=occupied, Gray=unknown, White=free.

        Call cv2.waitKey(1) to pump the GUI event loop — 1 ms is enough
        for non-blocking real-time display.
        """
        import cv2
        import numpy as np

        # --- GridWorld window ---
        robot_gx, robot_gy = self.map.world_to_grid(self.pose.x, self.pose.y)
        gw_img = self.map.render(
            robot_gx=robot_gx,
            robot_gy=robot_gy,
            path=self.current_path,
            scale=1,
        )
        # Resize to a fixed display size so the window stays manageable.
        display_size = 600
        gw_img = cv2.resize(
            gw_img,
            (display_size, display_size),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("GridWorld", gw_img)

        # --- BreezySLAM window (only when SLAM is active) ---
        if self.use_breezyslam and self.slam is not None:
            slam_img = np.array(self.slam_map_bytes, dtype=np.uint8).reshape(
                self.slam_map_pixels, self.slam_map_pixels
            )
            slam_img = cv2.resize(
                slam_img,
                (display_size, display_size),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("BreezySLAM raw", slam_img)

        cv2.waitKey(1)   # pump GUI event loop (non-blocking)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Run the exploration loop until the simulation ends."""
        print("[Explorer] Starting maze exploration.")

        while self.robot.step(self.timestep) != -1:
            self.step_count += 1
            print(f"step_count {self.step_count}")

            # 1. Pose estimate from wheel encoders.
            self._read_odometry()

            # 2. Fetch lidar scan.
            ranges, angle_min, angle_inc = self._read_lidar()

            # 3. Update occupancy map (SLAM or ray-casting).
            self._update_map(ranges, angle_min, angle_inc)

            # 4. Replan path toward nearest unexplored frontier.
            self._replan()

            # 5. Compute wheel commands.
            linear, angular = self._compute_cmd(ranges, angle_min, angle_inc)

            # 6. Apply to motors.
            self._set_velocity(linear, angular)

            # 7. Refresh visualisation every 5 ticks.
            # if self.step_count % 5 == 0:
            #     self._visualize()


if __name__ == "__main__":
    ExplorerController().run()
