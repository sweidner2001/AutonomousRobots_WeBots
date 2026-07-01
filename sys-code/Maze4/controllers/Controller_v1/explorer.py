"""
explorer.py  --  Exploration FSM and top-level mission orchestrator.
=====================================================================

TWO CLASSES IN THIS FILE
-------------------------
  1. Explorer  -- the frontier-exploration sub-state-machine
  2. MazeExplorer  -- the top-level orchestrator (owns everything, main loop)

HOW THE WHOLE SYSTEM FITS TOGETHER
------------------------------------
The MazeExplorer is the "conductor" that owns all subsystems:

  ┌──────────────────────────────────────────────────────┐
  │                    MazeExplorer                      │
  │                                                      │
  │  Robot        -- talks to Webots hardware            │
  │  OccupancyGrid -- the probabilistic map              │
  │  Odometry     -- estimates (x, y, heading)           │
  │  FrontierDetector -- finds unexplored boundaries     │
  │  PathPlanner  -- A* path to best frontier            │
  │  Pilot        -- converts path into motor commands   │
  │  MapViz       -- live matplotlib visualisation       │
  │  FloorHazardDetector -- finds green "no-go" tiles    │
  │  Explorer     -- decides what to do next (FSM below) │
  └──────────────────────────────────────────────────────┘

Every simulation step the MazeExplorer does three things:
  _perceive() -- read sensors, update pose, update map
  _act()      -- decide motor command (drives via Explorer FSM)
  _render()   -- refresh the map visualisation

EXPLORER STATE MACHINE
-----------------------
The Explorer has 5 phases:

  SPIN_SEED ─┐
  Rotate ~1  │ (enough rotation accumulated)
  full turn  │
  to seed    ▼
  the map.  PLAN ──────────────────────────────────────────────┐
             │  Detect frontiers, pick best one, run A*,       │
             │  give path to Pilot.                            │
             ▼ (path found)                             (no reachable frontier)
            DRIVE ──────────────────────────────────────── DONE
             │  Follow path.  Three exit conditions:
             │   a. Path traversed -> back to PLAN
             │   b. PLAN_PERIOD expired -> back to PLAN (adapt to new walls)
             │   c. Robot stuck -> REVERSE
             ▼
           REVERSE ──> PLAN
             Back up briefly, then replan.

FRONTIER BLACKLISTING
----------------------
If the robot tries multiple times to reach a frontier but gets stuck
or the path is blocked each time, we add that frontier's grid cell
to a blacklist.  Future PLAN calls skip blacklisted frontiers.
This prevents the robot from looping forever on an unreachable target.
The blacklist is automatically cleared after BLACKLIST_CLEAR successful
plans so that updated-map data gets a fresh chance.
"""

import math
import os

import numpy as np

import Maze4.controllers.Controller_v1.config as C
from Maze4.controllers.Controller_v1.robot          import Robot
from Maze4.controllers.Controller_v1.occupancy_grid import OccupancyGrid
from Maze4.controllers.Controller_v1.odometry       import Odometry
from Maze4.controllers.Controller_v1.frontier       import FrontierDetector
from Maze4.controllers.Controller_v1.planner        import PathPlanner
from Maze4.controllers.Controller_v1.pilot_2          import Pilot
from Maze4.controllers.Controller_v1.mapviz         import MapViz
from Maze4.controllers.Controller_v1.mission        import Mission
from Maze4.controllers.Controller_v1.floor_hazard   import FloorHazardDetector


class Explorer:
    """Frontier-exploration finite state machine (FSM).

    The Explorer decides WHAT the robot should do next based on the
    current map.  It does NOT talk to hardware directly — it only
    produces (v, w) motor commands and delegates execution to Pilot.
    """

    # Phase name constants (string tags make debug prints readable).
    SPIN_SEED = "SPIN_SEED"   # initial rotation to seed the map
    PLAN      = "PLAN"        # pick a frontier and compute a path
    DRIVE     = "DRIVE"       # follow the path with the Pilot
    REVERSE   = "REVERSE"     # back up after getting stuck
    DONE      = "DONE"        # exploration complete

    def __init__(self, grid, frontier, planner, pilot):
        """
        Args:
            grid     : OccupancyGrid  -- shared map reference (read only)
            frontier : FrontierDetector
            planner  : PathPlanner
            pilot    : Pilot
        """
        self.grid     = grid
        self.frontier = frontier
        self.planner  = planner
        self.pilot    = pilot

        # Start in SPIN_SEED phase.
        self.phase    = self.SPIN_SEED
        self.finished = False          # set to True when DONE is reached

        # ---- SPIN_SEED bookkeeping ----------------------------------------
        self._spin_accum = 0.0    # accumulated rotation so far (rad)
        self._prev_yaw   = None   # yaw from last step (to compute delta)

        # ---- PLAN / DRIVE bookkeeping ------------------------------------
        self._last_plan_time      = -1e9   # simulation time of last replan
        self._fail_count          = 0      # consecutive planning failures
        self._last_progress_xy    = (0.0, 0.0)
        self._last_progress_time  = 0.0    # when the robot last moved > STUCK_DIST

        # Path currently being followed (also read by the visualiser).
        self._path_rc   = None   # path as grid (row, col) cells
        self.world_path = None   # path as world (x, y) metres
        self.target_xy  = None   # frontier target world position

        # Cached blocked mask to avoid rebuilding it every step.
        self._blocked_cache = None

        # ---- REVERSE bookkeeping -----------------------------------------
        self._reverse_end_time = 0.0  # simulation time when backup ends

        # ---- Frontier blacklist ------------------------------------------
        # Set of (row, col) centroids that are known-unreachable.
        # Cleared automatically after BLACKLIST_CLEAR successful plans.
        self._blacklisted   = set()
        self._plan_count    = 0    # number of successful plans so far
        # Track the current target centroid to decide when to blacklist.
        self._current_target_centroid = None

    # ---------------------------------------------------------------------- #
    # Main dispatch
    # ---------------------------------------------------------------------- #
    def update(self, pose, ranges, bearings, now, scan_similarity=1.0, previous_speed_command=0.0):
        """Return (v, w) wheel command for this control step.

        Called once per simulation step by MazeExplorer._act().

        Args:
            pose     : (x, y, theta) from Odometry.
            ranges   : lidar range array from Robot.
            bearings : per-ray bearing array (precomputed by Robot).
            now      : current simulation time (seconds).

        Returns:
            (v, w) -- forward speed (m/s) and angular speed (rad/s).
        """
        if self.phase == self.SPIN_SEED:
            return self._spin(pose)
        if self.phase == self.PLAN:
            return self._plan(pose, now)
        if self.phase == self.DRIVE:
            return self._drive(pose, ranges, bearings, now, scan_similarity, previous_speed_command)
        if self.phase == self.REVERSE:
            return self._reverse(pose, now)
        return 0.0, 0.0   # DONE: stand still

    # ---------------------------------------------------------------------- #
    # SPIN_SEED phase
    # ---------------------------------------------------------------------- #
    def _spin(self, pose):
        """Rotate ~one full turn in place to seed the map with a 360° view.

        WHY DO WE DO THIS?
        After boot, the map is completely unknown.  Before we can detect
        any frontiers (which require free cells adjacent to unknown cells),
        we need to observe at least SOME free space around us.

        By spinning a full revolution, we mark all nearby cells as free
        and create a ring of frontier cells at the lidar's maximum range.
        The planner can then immediately find a valid frontier to drive to.

        The accumulated rotation is measured from the IMU yaw so it is
        accurate regardless of wheel slip.
        """
        yaw = pose[2]

        # Initialise the reference yaw on the first call.
        if self._prev_yaw is None:
            self._prev_yaw = yaw

        # Compute the absolute angle change since last step.
        # atan2(sin(delta), cos(delta)) correctly wraps to [-π, π].
        delta = abs(math.atan2(
            math.sin(yaw - self._prev_yaw),
            math.cos(yaw - self._prev_yaw)
        ))
        self._spin_accum += delta
        self._prev_yaw    = yaw

        # Check if we have rotated enough.
        if self._spin_accum >= C.SPIN_SEED_TURN:
            print("[explorer] spin complete (%.2f rad) -> PLAN" % self._spin_accum)
            self.phase = self.PLAN
            return 0.0, 0.0   # stop for one step while transitioning

        # Keep spinning counterclockwise at 60% of max turn speed.
        return 0.0, C.MAX_TURN_SPEED * 0.6

    # ---------------------------------------------------------------------- #
    # PLAN phase
    # ---------------------------------------------------------------------- #
    def _plan(self, pose, now):
        """Detect frontiers, choose the best one, plan an A* path to it.

        PIPELINE (4 steps):
          1. build_nav_grid  -- flood fill reachable area + inflate obstacles.
                                Produces a 3-value grid: 1.0=free, 0.5=unknown, 0.0=blocked.
          2. detect_cells    -- frontier = reachable cell with an unexplored neighbour.
          3. cluster         -- group adjacent frontier cells, filter tiny ones.
          4. choose_target   -- A* to nearest/largest frontier; give path to Pilot.
        """
        # --- Step 1: navigation grid -----------------------------------------
        # Flood fill from the robot through observed, non-blocked cells.
        # Also inflates obstacles by INFLATE_RADIUS_CELLS for safety.
        nav, reachable, blocked = self.planner.build_nav_grid(
            self.grid, (pose[0], pose[1])
        )
        self._blocked_cache = blocked   # used in DRIVE phase to detect path blockage

        # --- Step 2: frontier detection ---------------------------------------
        # A frontier is any reachable cell adjacent to an unexplored cell (nav=0.5).
        fmask = self.frontier.detect_cells(nav, reachable)
        if not fmask.any():
            print("[explorer] no frontiers -> exploration complete.")
            self.phase    = self.DONE
            self.finished = True
            return 0.0, 0.0

        # --- Step 3: clustering ----------------------------------------------
        clusters = self.frontier.cluster(fmask)
        print("[explorer] PLAN: %d frontier cells -> %d clusters, robot=(%.2f,%.2f)"
              % (int(fmask.sum()), len(clusters), pose[0], pose[1]))

        # Skip clusters whose centroid was previously blacklisted as unreachable.
        candidates = [cl for cl in clusters
                      if cl["centroid"] not in self._blacklisted]
        if not candidates:
            print("[explorer] all frontiers blacklisted -> clearing blacklist.")
            self._blacklisted.clear()
            candidates = clusters

        # --- Step 4: A* path planning ----------------------------------------
        # unknown mask: cells with nav=0.5 cost more to cross (exploration penalty).
        unknown = (nav == 0.5)
        path_rc, target = self.planner.choose_target(
            self.grid, candidates, (pose[0], pose[1]), blocked, unknown
        )

        if path_rc is None or target is None:
            self._fail_count += 1
            print("[explorer] no reachable frontier (fail %d/6) — "
                  "%d clusters, robot=(%.2f,%.2f)."
                  % (self._fail_count, len(candidates), pose[0], pose[1]))
            if self._fail_count >= 6:
                print("[explorer] too many failures -> exploration done.")
                self.phase    = self.DONE
                self.finished = True
                return 0.0, 0.0
            # Spin slightly so the lidar sees new cells and may open a path.
            return 0.0, C.MAX_TURN_SPEED * 0.5

        # --- SUCCESS: path found ---------------------------------------------
        self._fail_count = 0
        self._plan_count += 1
        if self._plan_count >= C.BLACKLIST_CLEAR:
            self._blacklisted.clear()
            self._plan_count = 0

        # Store path data (also read by the visualiser).
        self._path_rc   = path_rc
        self.world_path = self.planner.path_to_world(self.grid, path_rc)
        gr, gc          = target["centroid"]
        self.target_xy  = self.grid.grid_to_world(gc, gr)
        self._current_target_centroid = target["centroid"]

        self.pilot.set_path(self.world_path)
        self._last_plan_time     = now
        self._last_progress_xy   = (pose[0], pose[1])
        self._last_progress_time = now

        self.phase = self.DRIVE
        return 0.0, 0.0

    # ---------------------------------------------------------------------- #
    # DRIVE phase
    # ---------------------------------------------------------------------- #
    def _drive(self, pose, ranges, bearings, now, scan_similarity=0.0, previous_speed_command=0.0):
        """Follow the current path and detect when replanning is needed.

        Delegates actual steering to pilot.compute().
        Checks three conditions that trigger a replan:
          a) Path is fully traversed (Pilot signals done).
          b) Time since last plan >= PLAN_PERIOD (force periodic replan).
          c) Current path is blocked by newly-discovered walls.
          d) Robot has not moved enough for too long (stuck).
        """
        # Ask the pilot for the next (v, w) command.
        stuck = False
        v, w, done = self.pilot.compute(pose, ranges, bearings)

        # Check if the robot is close enough to the frontier target
        # to count as "arrived" even if the path isn't finished.
        if self.target_xy is not None:
            dist_to_target = math.hypot(
                pose[0] - self.target_xy[0],
                pose[1] - self.target_xy[1]
            )
            if dist_to_target < C.FRONTIER_REACH_TOL:
                done = True   # close enough -> count as reached

        # --- Check replan triggers ----------------------------------------
        if done:
            # Path traversed or target reached -> immediately replan.
            self._trigger_replan()
            return 0.0, 0.0

        if (now - self._last_plan_time) >= C.PLAN_PERIOD:
            # Periodic replan: the map may have changed, new walls may have
            # appeared, and a better frontier may now be available.
            self._trigger_replan()
            return 0.0, 0.0

        if self._path_rc and self._blocked_cache is not None:
            if self.planner.path_blocked(self._path_rc, self._blocked_cache):
                # A cell on our planned path has become blocked (new wall found).
                self._trigger_replan()
                return 0.0, 0.0

        # --- Stuck detection -----------------------------------------------
        # Track how far the robot has moved since we last recorded progress.
        moved = math.hypot(
            pose[0] - self._last_progress_xy[0],
            pose[1] - self._last_progress_xy[1]
        )
        if scan_similarity > 0.995 and previous_speed_command > 0.1:
            # scans are almost identical → robot hasn't moved
            print("[explorer] scan similarity %.3f → robot is stuck!" % scan_similarity)
            stuck = True
        if moved > C.STUCK_DIST and stuck is False:
            # Robot is making progress -- update the reference point.
            self._last_progress_xy   = (pose[0], pose[1])
            self._last_progress_time = now
        elif (now - self._last_progress_time) > C.STUCK_TIME:
            # Robot hasn't moved far enough in STUCK_TIME seconds -> stuck.
            print("[explorer] stuck at (%.2f, %.2f) -> backing up. Scan similarity: %.3f" % (pose[0], pose[1], scan_similarity))
            # Blacklist the current frontier so we don't loop back to it.
            if self._current_target_centroid is not None:
                self._blacklisted.add(self._current_target_centroid)
                print("[explorer] blacklisted frontier %s."
                      % str(self._current_target_centroid))
            self._trigger_replan()            # discard path
            self.phase = self.REVERSE         # override -> backup first
            self._reverse_end_time = now + C.REVERSE_TIME
            return -C.CRUISE_SPEED, 0.0  # first step: reverse

        return v, w

    def _trigger_replan(self):
        """Reset path state so the next update() enters PLAN phase."""
        self.pilot.clear()
        self._path_rc   = None
        self.world_path = None
        self.target_xy  = None
        self._current_target_centroid = None
        self.phase      = self.PLAN

    # ---------------------------------------------------------------------- #
    # REVERSE phase
    # ---------------------------------------------------------------------- #
    def _reverse(self, pose, now):
        """Drive backward for REVERSE_TIME seconds, then replan.

        This backup manoeuvre helps the robot escape corners and tight
        spots where the path follower got it stuck.  After reversing,
        the robot replans from the new position.
        """
        if now >= self._reverse_end_time:
            # Backup complete -- now go back to PLAN.
            print("[explorer] backup complete -> PLAN.")
            self.phase = self.PLAN
            return 0.0, 0.0

        # Drive backward at half cruise speed.
        return -C.CRUISE_SPEED, 0.0


# ============================================================================
# MazeExplorer -- top-level orchestrator
# ============================================================================
class MazeExplorer:
    """Owns all components and runs the main simulation loop.

    This is the class that Webots instantiates:
      Controller_v1.py  ->  MazeExplorer().run()

    The run() loop does three things every simulation step:
      perceive  -- read sensors, update pose, integrate lidar into map
      act       -- decide motor command via the mission + Explorer FSM
      render    -- refresh the matplotlib visualisation
    """

    def __init__(self):
        # Create all subsystems.
        self.robot    = Robot()
        self.grid     = OccupancyGrid()
        self.odom     = Odometry()
        self.frontier = FrontierDetector()
        self.planner  = PathPlanner()
        self.pilot    = Pilot()
        self.viz      = MapViz(self.grid)
        self.hazard_detector = FloorHazardDetector()
        self.explorer = Explorer(
            self.grid, self.frontier, self.planner, self.pilot
        )

        # Top-level mission state.
        self.mission  = Mission.EXPLORE_MAP

        # Per-step counters / state.
        self.step_i   = 0          # step counter (incremented every step)
        self.now      = 0.0        # current simulation time (seconds)
        self.pose     = (0.0, 0.0, 0.0)  # (x, y, theta) from odometry
        self.ranges   = None       # latest lidar scan
        self.previous_ranges   = None       # previous lidar scan

        self._saved   = False      # prevent saving the map more than once
        self.out_dir  = os.path.dirname(os.path.abspath(__file__))

    # ---------------------------------------------------------------------- #
    # Main simulation loop
    # ---------------------------------------------------------------------- #
    def run(self):
        """Run the simulation until Webots stops.

        The loop structure is intentionally simple:
          _startup()   -- one-time initialisation
          loop:
            robot.step()  -- advance physics by one timestep
            _perceive()   -- read sensors
            _act()        -- decide motors
            _render()     -- update visualisation
          _shutdown()  -- save the final map
        """
        self._startup()
        while self.robot.step():
            self._perceive()
            self._act()
            self._render()
        self._shutdown()

    # ---------------------------------------------------------------------- #
    # Startup
    # ---------------------------------------------------------------------- #
    def _startup(self):
        """One-time initialisation AFTER the first Webots step.

        We must call robot.step() once before reading sensors because
        Webots sensors return garbage (0 or NaN) until the first physics
        tick has been processed.
        """
        self.robot.step()  # advance one step so sensors have valid data
        self.odom.initialise(self.robot.read_encoders(), self.robot.read_yaw())
        print("[mission] start: dt=%.3f s, lidar rays=%d, map %dx%d cells" % (
            self.robot.dt,
            len(self.robot.bearings),
            self.grid.ncols,
            self.grid.nrows,
        ))

    # ---------------------------------------------------------------------- #
    # Per-step perception
    # ---------------------------------------------------------------------- #
    def _perceive(self):
        """Read sensors and update the map.

        Called every simulation step BEFORE _act().

        Actions:
          - Increment step counter.
          - Read simulation time.
          - Update odometry pose (x, y, theta) from encoders + IMU.
          - Read the latest lidar scan.
          - Every MAP_EVERY steps: fuse the scan into the occupancy grid.
        """
        self.step_i += 1
        self.now     = self.robot.get_time()
        self.pose    = self.odom.update(
            self.robot.read_encoders(), self.robot.read_yaw()
        )
        self.previous_ranges = self.ranges
        self.ranges  = self.robot.read_lidar()

        # Fuse the lidar scan into the map (not every single step to save CPU).
        if self.step_i % C.MAP_EVERY == 0:
            self.grid.integrate_scan(
                self.pose[0], self.pose[1], self.pose[2],
                self.ranges, self.robot.bearings
            )

        # Look for green floor hazards with the RGB-D camera (also not every
        # step -- back-projecting + registering pixels has a real CPU cost).
        if self.step_i % C.CAMERA_EVERY == 0:
            rgb_img   = self.robot.read_camera_rgb()
            depth_img = self.robot.read_camera_depth()
            xs, ys = self.hazard_detector.detect(rgb_img, depth_img, self.pose)
            self.grid.mark_hazard_world(xs, ys)

    def get_scan_similarity_to_previous(self):
        """Compute the similarity between the current and previous lidar scans.

        Returns:
            similarity : float in [-1, 1].  Values near 1.0 mean the scan
            is almost identical to the last one (robot likely not moving).
        """
        if self.previous_ranges is None or self.ranges is None:
            return 1.0

        prev = np.asarray(self.previous_ranges, dtype=np.float32)
        curr = np.asarray(self.ranges,          dtype=np.float32)

        # Replace inf (no-return rays) with lidar_max.
        # inf means "beam reached the sensor's physical limit without hitting
        # anything" -- lidar_max is the correct semantic value and keeps the
        # array the same shape for corrcoef.
        lidar_max = float(self.robot.lidar_max)
        prev = np.where(np.isfinite(prev), prev, lidar_max + 1)
        curr = np.where(np.isfinite(curr), curr, lidar_max + 1)

        if prev.std() < 1e-9 or curr.std() < 1e-9:   # zero-variance -> undefined
            return 1.0

        return float(np.corrcoef(prev, curr)[0, 1])
    

    # ---------------------------------------------------------------------- #
    # Per-step action
    # ---------------------------------------------------------------------- #
    def _act(self):
        """Decide the motor command for this step and send it to the motors.

        The top-level mission FSM selects which behaviour runs:
          EXPLORE_MAP  -> Explorer FSM (spin, plan, drive, reverse)
          DONE         -> stop motors, save map once
          colour states -> placeholder stub (not implemented)
        """
        if self.mission == Mission.EXPLORE_MAP:
            v, w = self.explorer.update(
                self.pose, self.ranges, self.robot.bearings, self.now, 
                # scan_similarity=self.get_scan_similarity_to_previous(), 
                # previous_speed_command=self.robot.previous_v
            )
            self.robot.set_velocity(v, w)
            if self.explorer.finished:
                self._advance_from_explore()

        elif self.mission == Mission.DONE:
            self.robot.stop()
            self._save_once()

        else:
            # SEARCH_BLUE / GO_BLUE / SEARCH_YELLOW / GO_YELLOW are not
            # implemented yet -- fall through to DONE.
            self._color_stub()

    # ---------------------------------------------------------------------- #
    # Per-step visualisation
    # ---------------------------------------------------------------------- #
    def _render(self):
        """Refresh the matplotlib map visualisation.

        Only called every VIZ_EVERY steps to keep the simulation fast.
        Rendering every step would stall the controller.
        """
        if self.step_i % C.VIZ_EVERY == 0:
            self.viz.update(
                self.pose,
                scan_xy    = self._scan_world_points(),
                world_path = self.explorer.world_path,
                target_xy  = self.explorer.target_xy,
            )

    # ---------------------------------------------------------------------- #
    # Mission transitions
    # ---------------------------------------------------------------------- #
    def _advance_from_explore(self):
        """Called when Explorer.finished becomes True.

        If colour search is enabled, move to SEARCH_BLUE.
        Otherwise, go straight to DONE.
        """
        print("[mission] EXPLORE_MAP complete.")
        if C.MISSION_ENABLE_COLOR:
            self.mission = Mission.SEARCH_BLUE
            print("[mission] -> SEARCH_BLUE (not yet implemented).")
        else:
            self.mission = Mission.DONE

    def _color_stub(self):
        """Placeholder for colour-detection mission phases.

        Not implemented yet.  Just stops the robot and falls to DONE.
        """
        self.robot.stop()
        print("[mission] state %s is a placeholder -> DONE." % self.mission)
        self.mission = Mission.DONE

    # ---------------------------------------------------------------------- #
    # Helper methods
    # ---------------------------------------------------------------------- #
    def _scan_world_points(self):
        """Convert the current lidar scan to world-frame (x, y) hit points.

        Used by the visualiser to draw the blue "laser dot cloud".

        Returns:
            (xs, ys) -- two NumPy arrays of x and y coordinates (metres).
                        Only finite hits within sensor range are included.
        """
        x, y, theta = self.pose

        # The lidar is mounted LIDAR_OFFSET_X ahead of the robot centre.
        sx = x + C.LIDAR_OFFSET_X * math.cos(theta)
        sy = y + C.LIDAR_OFFSET_X * math.sin(theta)

        r = self.ranges
        # Keep only valid hits: finite, above min range, below max range.
        finite = (np.isfinite(r)
                  & (r > C.LIDAR_MIN_RANGE)
                  & (r < self.robot.lidar_max * 0.999))

        # Convert (range, bearing) in robot frame to (x, y) in world frame.
        a  = theta + self.robot.bearings[finite]
        rr = r[finite]
        return sx + rr * np.cos(a), sy + rr * np.sin(a)

    def _save_once(self):
        """Save the map to disk exactly once (called repeatedly in DONE state)."""
        if self._saved:
            return
        self._save()
        self._saved = True

    def _save(self):
        """Write the final map PNG and raw NumPy array to disk."""
        try:
            self.viz.save(os.path.join(self.out_dir, C.SAVE_MAP_PNG))
            self.grid.save(os.path.join(self.out_dir, C.SAVE_MAP_NPY))
            print("[mission] map saved to %s" % self.out_dir)
        except Exception as e:
            print("[mission] save error: %s" % e)

    def _shutdown(self):
        """Called when Webots stops the simulation.  Save the final map."""
        self._save()
