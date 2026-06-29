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
from Maze4.controllers.Controller_v1.pilot          import Pilot
from Maze4.controllers.Controller_v1.mapviz         import MapViz
from Maze4.controllers.Controller_v1.mission        import Mission


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
    def update(self, pose, ranges, bearings, now):
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
            return self._drive(pose, ranges, bearings, now)
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

        Steps:
          1. Build the inflated obstacle map (blocked) and unknown mask.
          2. Find all frontier cells (free cells adjacent to unknown).
          3. Cluster them into groups.
          4. Filter out blacklisted clusters.
          5. Run A* to pick the best reachable cluster (nearest + largest).
          6. Hand the path to the Pilot and switch to DRIVE phase.

        If no reachable frontier is found, we count the failure and
        either rotate in place (to gather new data) or declare DONE.
        """
        # Build the cost layers used by A*.
        blocked, unknown = self.planner.build_cost_layers(self.grid)
        self._blocked_cache = blocked   # cache for DRIVE phase path-check

        # Detect all frontier cells in the current map.
        fmask = self.frontier.detect_cells(self.grid)
        if not fmask.any():
            # No frontier cells at all -> the entire reachable area is explored.
            print("[explorer] no frontiers left -> exploration complete.")
            self.phase    = self.DONE
            self.finished = True
            return 0.0, 0.0

        # Group frontier cells into clusters.
        clusters = self.frontier.cluster(fmask)

        # Filter out blacklisted clusters (ones we failed to reach before).
        reachable = [cl for cl in clusters
                     if cl["centroid"] not in self._blacklisted]
        if not reachable:
            # All clusters blacklisted -- clear the list and retry.
            print("[explorer] all frontiers blacklisted -> clearing blacklist.")
            self._blacklisted.clear()
            reachable = clusters

        # Ask the planner to choose the best reachable frontier.
        path_rc, target = self.planner.choose_target(
            self.grid, reachable, (pose[0], pose[1]), blocked, unknown
        )

        if path_rc is None or target is None:
            # No path found to any frontier (may be temporarily blocked).
            self._fail_count += 1
            print("[explorer] no reachable frontier (fail %d/6)." % self._fail_count)
            if self._fail_count >= 6:
                print("[explorer] too many failures -> exploration done.")
                self.phase    = self.DONE
                self.finished = True
                return 0.0, 0.0
            # Nudge the robot to a slightly different position so the lidar
            # might see new cells and open up a new path.
            return 0.0, C.MAX_TURN_SPEED * 0.5

        # SUCCESS: we have a valid path.
        self._fail_count = 0
        self._plan_count += 1
        if self._plan_count >= C.BLACKLIST_CLEAR:
            # Periodically clear the blacklist so stale entries don't
            # permanently block valid targets that the map has updated around.
            self._blacklisted.clear()
            self._plan_count = 0

        # Store path data (shared with the visualiser).
        self._path_rc   = path_rc
        self.world_path = self.planner.path_to_world(self.grid, path_rc)
        gr, gc          = target["centroid"]
        self.target_xy  = self.grid.grid_to_world(gc, gr)
        self._current_target_centroid = target["centroid"]

        # Give the path to the Pilot.
        self.pilot.set_path(self.world_path)

        # Record timing and starting position for stuck detection.
        self._last_plan_time     = now
        self._last_progress_xy   = (pose[0], pose[1])
        self._last_progress_time = now

        self.phase = self.DRIVE
        return 0.0, 0.0   # stand still for this one step while transitioning

    # ---------------------------------------------------------------------- #
    # DRIVE phase
    # ---------------------------------------------------------------------- #
    def _drive(self, pose, ranges, bearings, now):
        """Follow the current path and detect when replanning is needed.

        Delegates actual steering to pilot.compute().
        Checks three conditions that trigger a replan:
          a) Path is fully traversed (Pilot signals done).
          b) Time since last plan >= PLAN_PERIOD (force periodic replan).
          c) Current path is blocked by newly-discovered walls.
          d) Robot has not moved enough for too long (stuck).
        """
        # Ask the pilot for the next (v, w) command.
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
        if moved > C.STUCK_DIST:
            # Robot is making progress -- update the reference point.
            self._last_progress_xy   = (pose[0], pose[1])
            self._last_progress_time = now
        elif (now - self._last_progress_time) > C.STUCK_TIME:
            # Robot hasn't moved far enough in STUCK_TIME seconds -> stuck.
            print("[explorer] stuck at (%.2f, %.2f) -> backing up."
                  % (pose[0], pose[1]))
            # Blacklist the current frontier so we don't loop back to it.
            if self._current_target_centroid is not None:
                self._blacklisted.add(self._current_target_centroid)
                print("[explorer] blacklisted frontier %s."
                      % str(self._current_target_centroid))
            self._trigger_replan()            # discard path
            self.phase = self.REVERSE         # override -> backup first
            self._reverse_end_time = now + C.REVERSE_TIME
            return -C.CRUISE_SPEED * 0.5, 0.0  # first step: reverse

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
        return -C.CRUISE_SPEED * 0.5, 0.0


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
        self.ranges  = self.robot.read_lidar()

        # Fuse the lidar scan into the map (not every single step to save CPU).
        if self.step_i % C.MAP_EVERY == 0:
            self.grid.integrate_scan(
                self.pose[0], self.pose[1], self.pose[2],
                self.ranges, self.robot.bearings
            )

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
                self.pose, self.ranges, self.robot.bearings, self.now
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
