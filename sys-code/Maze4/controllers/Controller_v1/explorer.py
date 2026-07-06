"""
explorer.py  --  Exploration FSM and top-level mission orchestrator.
=====================================================================

THREE CLASSES IN THIS FILE
----------------------------
  1. Explorer     -- the frontier-exploration sub-state-machine.
  2. GoToPoint    -- drives to ONE known (x, y) target (used to fetch a
                      tracked blue/yellow object once it's been located).
  3. MazeExplorer -- the top-level orchestrator (owns everything, main loop).

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
  │  PathPlanner  -- A* path to best frontier / a point  │
  │  Pilot        -- converts path into motor commands   │
  │  MapViz       -- live matplotlib visualisation       │
  │  FloorHazardDetector -- finds green "no-go" tiles    │
  │  ColorObjectDetector -- finds blue/yellow objects    │
  │  Explorer     -- frontier-exploration FSM            │
  │  GoToPoint    -- "drive to one known point" FSM      │
  └──────────────────────────────────────────────────────┘

Every simulation step the MazeExplorer does three things, REGARDLESS of
which mission phase is active -- perception never stops, even while the
robot is driving toward a tracked object:
  _perceive() -- read sensors, update pose, update map, look for hazards/objects
  _act()      -- decide motor command (dispatches on Mission state, see below)
  _render()   -- refresh the map visualisation

TOP-LEVEL MISSION FLOW  (see mission.py for the Mission constants)
------------------------------------------------------------------------
  EXPLORE_MAP ──(fully mapped)──► SEARCH_BLUE ──(blue seen+reachable)──► GO_BLUE
                                                                            │
                       ┌────────────────────────────────────────(arrived)──┘
                       ▼
                 SEARCH_YELLOW ──(yellow seen+reachable)──► GO_YELLOW ──(arrived)──► DONE

  SEARCH_BLUE / SEARCH_YELLOW keep running the SAME frontier-exploration
  Explorer used in EXPLORE_MAP -- the robot keeps mapping unexplored area
  while watching TrackedObject.seen/.reachable for the colour it's after.
  As soon as both are True, the mission switches to GO_BLUE/GO_YELLOW and
  GoToPoint takes over, driving straight to the object instead of the next
  frontier.  If GoToPoint ever fails repeatedly (e.g. the object turns out
  to be unreachable after all), the mission falls back to searching again.

EXPLORER STATE MACHINE  (used during EXPLORE_MAP / SEARCH_BLUE / SEARCH_YELLOW)
------------------------------------------------------------------------------------
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

GOTOPOINT STATE MACHINE  (used during GO_BLUE / GO_YELLOW)
-------------------------------------------------------------
Same PLAN / DRIVE / REVERSE shape as Explorer, but with no frontier logic
at all -- the target is already known.  See the GoToPoint class docstring.
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
from Maze4.controllers.Controller_v1.colored_objects import ColorObjectDetector, TrackedObject
from Maze4.controllers.Controller_v1.depth_obstacle import DepthObstacleDetector


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
        self._nav_cache = None
        self._reachable_cache = None
        self._fmask_cache = None
        self._fcluster_cache = None

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
    def resume(self):
        """Resume active frontier exploration.

        Needed whenever MazeExplorer switches INTO SEARCH_BLUE/SEARCH_YELLOW:
          - After EXPLORE_MAP finished (self.finished was True, self.phase
            was DONE) -- update() would otherwise just return (0, 0) forever,
            since no phase handler exists for DONE.
          - After a failed GO_BLUE/GO_YELLOW attempt falls back to searching
            again -- the FSM may have been sitting idle in that same DONE
            state, or in a stale DRIVE/REVERSE phase pointed at wherever it
            was heading before GoToPoint took over.

        Does NOT reset the frontier blacklist or plan counters -- those
        remain valid, only the phase/finished flags need clearing.
        """
        self.finished = False
        self.phase    = self.PLAN
        self.pilot.clear()

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
            self.grid, (pose[0], pose[1]), is_for_frontier=True
        )
        self._nav_cache = nav
        self._reachable_cache = reachable
        self._blocked_cache = blocked   # used in DRIVE phase to detect path blockage

        # --- Step 2: frontier detection ---------------------------------------
        # A frontier is any reachable cell adjacent to an unexplored cell (nav=0.5).
        fmask = self.frontier.detect_cells(nav, reachable)
        self._fmask_cache = fmask 
        if not fmask.any():
            print("[explorer] no frontiers -> exploration complete.")
            self.phase    = self.DONE
            self.finished = True
            return 0.0, 0.0

        # --- Step 3: clustering ----------------------------------------------
        clusters = self.frontier.cluster(fmask)
        self._fcluster_cache = clusters
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


    def is_spin_seed_phase(self):
        return self.phase == self.SPIN_SEED

# ============================================================================
# GoToPoint -- drive to ONE known (x, y) target (used for GO_BLUE / GO_YELLOW)
# ============================================================================
class GoToPoint:
    """Drives the robot to a single fixed world (x, y) point.

    This is deliberately much simpler than Explorer: there is no frontier
    detection, clustering, or blacklisting -- the target is already known
    (a tracked blue/yellow object's estimated position). It reuses the
    exact same building blocks as Explorer (PathPlanner.build_nav_grid,
    A*, Pilot, stuck/backup handling) so driving BEHAVES identically --
    only "how the target is chosen" differs.

    PHASES
    -------
      PLAN    -- run A* from the robot to the target.
      DRIVE   -- follow the path; replan periodically or if it gets blocked
                 by newly-discovered walls; detect stuck and back up.
      REVERSE -- back up briefly after getting stuck, then replan.

    OUTCOMES (checked by the caller every step)
    -----------------------------------------------
      self.arrived : True once the robot is within OBJECT_REACH_TOL of the target.
      self.failed  : True after too many consecutive failed plan attempts
                     (e.g. the target turned out to be unreachable after all).
                     The caller should fall back to searching/exploring again.
    """

    PLAN    = "PLAN"
    DRIVE   = "DRIVE"
    REVERSE = "REVERSE"

    def __init__(self, grid, planner, pilot):
        self.grid    = grid
        self.planner = planner
        self.pilot   = pilot

        self.target_xy = None
        self.phase      = self.PLAN
        self.arrived    = False
        self.failed     = False

        self._fail_count         = 0
        self._last_plan_time     = -1e9
        self._last_progress_xy   = (0.0, 0.0)
        self._last_progress_time = 0.0
        self._reverse_end_time   = 0.0
        self._path_rc            = None
        self.world_path          = None
        self._blocked_cache      = None
        self._nav_cache          = None
        self._reachable_cache    = None

    # ---------------------------------------------------------------------- #
    def start(self, target_xy):
        """Begin driving toward a new target, resetting all state.

        Called once by MazeExplorer when the mission switches into
        GO_BLUE/GO_YELLOW -- NOT every step.
        """
        self.target_xy = target_xy
        self.phase      = self.PLAN
        self.arrived    = False
        self.failed     = False
        self._fail_count = 0
        self.pilot.clear()

    # ---------------------------------------------------------------------- #
    def update(self, pose, ranges, bearings, now):
        """Return (v, w) wheel command for this control step."""
        if self.phase == self.PLAN:
            return self._plan(pose, now)
        if self.phase == self.DRIVE:
            return self._drive(pose, ranges, bearings, now)
        if self.phase == self.REVERSE:
            return self._reverse(pose, now)
        return 0.0, 0.0

    # ---------------------------------------------------------------------- #
    # PLAN phase
    # ---------------------------------------------------------------------- #
    def _plan(self, pose, now):
        """Build the nav grid and run A* straight to the target."""
        nav, _reachable, blocked = self.planner.build_nav_grid(
            self.grid, (pose[0], pose[1]), is_for_frontier=False
        )
        self._blocked_cache = blocked
        self._nav_cache = nav
        self._reachable_cache = _reachable
        unknown = (nav == 0.5)

        path_rc = self.planner.plan_path_to(
            self.grid, blocked, unknown, (pose[0], pose[1]), self.target_xy
        )

        if path_rc is None:
            # The direct (inflated) plan failed -- but a physical route to the
            # object always exists; the inflation margin just sealed the gap.
            # Fall back to parking as close to the object as is safely
            # possible (see planner.plan_path_near_blocked_target()).
            path_rc = self.planner.plan_path_near_blocked_target(
                self.grid, (pose[0], pose[1]), self.target_xy
            )
            print("[goto] direct path to target blocked by safety margin; 'falling back' plan is applied.")
            if path_rc is not None:
                print("[goto] direct path sealed by safety margin; "
                      "approaching object as close as safely possible.")

        if path_rc is None:
            self._fail_count += 1
            print("[goto] no path to target (fail %d/6)." % self._fail_count)
            if self._fail_count >= 6:
                self.failed = True
            return 0.0, C.MAX_TURN_SPEED * 0.5   # nudge around while retrying

        self._fail_count = 0
        self._path_rc    = path_rc
        self.world_path  = self.planner.path_to_world(self.grid, path_rc)
        self.pilot.set_path(self.world_path)

        self._last_plan_time     = now
        self._last_progress_xy   = (pose[0], pose[1])
        self._last_progress_time = now

        self.phase = self.DRIVE
        return 0.0, 0.0

    # ---------------------------------------------------------------------- #
    # DRIVE phase
    # ---------------------------------------------------------------------- #
    def _drive(self, pose, ranges, bearings, now):
        """Follow the path; replan/stuck-handling mirrors Explorer._drive()."""
        v, w, done = self.pilot.compute(pose, ranges, bearings)

        dist_to_target = math.hypot(
            pose[0] - self.target_xy[0], pose[1] - self.target_xy[1]
        )
        if dist_to_target < C.OBJECT_REACH_TOL:
            self.arrived = True
            return 0.0, 0.0

        if done:
            self._trigger_replan()
            return 0.0, 0.0

        if (now - self._last_plan_time) >= C.PLAN_PERIOD:
            self._trigger_replan()
            return 0.0, 0.0

        if self._path_rc and self._blocked_cache is not None:
            if self.planner.path_blocked(self._path_rc, self._blocked_cache):
                self._trigger_replan()
                return 0.0, 0.0

        moved = math.hypot(
            pose[0] - self._last_progress_xy[0],
            pose[1] - self._last_progress_xy[1]
        )
        if moved > C.STUCK_DIST:
            self._last_progress_xy   = (pose[0], pose[1])
            self._last_progress_time = now
        elif (now - self._last_progress_time) > C.STUCK_TIME:
            print("[goto] stuck at (%.2f, %.2f) -> backing up." % (pose[0], pose[1]))
            self._trigger_replan()
            self.phase = self.REVERSE
            self._reverse_end_time = now + C.REVERSE_TIME
            return -C.CRUISE_SPEED, 0.0

        return v, w

    def _trigger_replan(self):
        self.pilot.clear()
        self._path_rc   = None
        self.world_path = None
        self.phase      = self.PLAN

    # ---------------------------------------------------------------------- #
    # REVERSE phase
    # ---------------------------------------------------------------------- #
    def _reverse(self, pose, now):
        if now >= self._reverse_end_time:
            self.phase = self.PLAN
            return 0.0, 0.0
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
        # Camera resolution/FoV/range are all read live from the Webots
        # device inside Robot.__init__ (see robot.py) -- passed straight
        # through here rather than duplicated as config.py constants.
        self.hazard_detector = FloorHazardDetector(
            self.robot.camera_width,
            self.robot.camera_height,
            self.robot.camera_fov,
            self.robot.camera_depth_min_range,
            self.robot.camera_depth_max_range,
        )
        self.color_detector = ColorObjectDetector(
            self.robot.camera_width,
            self.robot.camera_height,
            self.robot.camera_fov,
            self.robot.camera_depth_min_range,
            self.robot.camera_depth_max_range,
        )
        # Finds obstacles the lidar's fixed-height sweep can't see at all
        # (see depth_obstacle.py) -- colour-agnostic, depth-only detection.
        self.depth_obstacle_detector = DepthObstacleDetector(
            self.robot.camera_width,
            self.robot.camera_height,
            self.robot.camera_fov,
            self.robot.camera_depth_min_range,
            self.robot.camera_depth_max_range,
        )
        # One TrackedObject instance per colour -- see colored_objects.py for
        # what each field (seen / reachable / reached / world_xy) means.
        self.blue_object   = TrackedObject("blue")
        self.yellow_object = TrackedObject("yellow")

        self.explorer = Explorer(
            self.grid, self.frontier, self.planner, self.pilot
        )
        # GoToPoint drives to ONE known point (used for GO_BLUE/GO_YELLOW).
        # Shares the SAME planner/pilot as Explorer -- safe because only one
        # of the two FSMs is ever "in control" at a time (see _act()).
        self.goto = GoToPoint(self.grid, self.planner, self.pilot)

        # Top-level mission state.
        self.mission  = Mission.EXPLORE_MAP
        # self.mission  = Mission.SEARCH_BLUE

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

        # Look for green floor hazards AND blue/yellow tracked objects with
        # the RGB-D camera (also not every step -- back-projecting +
        # registering pixels has a real CPU cost).  Both detectors reuse the
        # SAME camera frames -- no need to read the camera twice.
        if self.step_i % C.CAMERA_EVERY == 0:
            rgb_img   = self.robot.read_camera_rgb()
            depth_img = self.robot.read_camera_depth()

            xs, ys = self.hazard_detector.detect(rgb_img, depth_img, self.pose)
            self.grid.mark_hazard_world(xs, ys)

            # Pass the SAME control step's lidar scan so the detector can
            # reject any coloured point that would sit behind a wall the
            # lidar has already confirmed in the same direction (see
            # colored_objects.py -- ColorObjectDetector._clamp_to_lidar()).
            points_by_color = self.color_detector.detect(
                rgb_img, depth_img, self.pose)
            for color, obj in (("blue", self.blue_object), ("yellow", self.yellow_object)):
                hit_xs, hit_ys, free_xs, free_ys = points_by_color[color]
                # Fold this frame into the colour's log-odds map, EXACTLY like
                # the lidar: matching pixels are positive evidence, visible
                # non-matching pixels are negative evidence -- so a false
                # detection here is un-marked the next time the camera sees
                # that spot and disagrees (see occupancy_grid.py --
                # update_object_observation()).
                self.grid.update_object_observation(
                    color, hit_xs, hit_ys, free_xs, free_ys)
                # Re-derive the object's position/seen flags FROM the grid we
                # just updated -- the grid is the single self-correcting source
                # of truth (see colored_objects.py -- update_from_grid()).
                obj.update_from_grid(self.grid, self.now)

            # Depth-only obstacle detection -- catches obstacles the lidar's
            # fixed-height sweep can't see at all (see depth_obstacle.py).
            # Reuses the SAME depth_img already read above; feeds straight
            # into the SAME log-odds map the lidar uses, via integrate_scan_rgbd().
            # obstacle_ranges, obstacle_bearings = self.depth_obstacle_detector.detect(depth_img)
            # self.grid.integrate_scan_rgbd(
            #     self.pose[0], self.pose[1], self.pose[2],
            #     obstacle_ranges, obstacle_bearings
            # )

        # Update "reached" every step -- cheap distance check, no reason to
        # wait for the next camera frame.
        for obj in (self.blue_object, self.yellow_object):
            if obj.update_reached(self.pose[:2]):
                print("[objects] reached the %s object at (%.2f, %.2f)!"
                      % (obj.color_name, obj.world_xy[0], obj.world_xy[1]))



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

        The top-level Mission state (see mission.py) selects which
        behaviour runs:
          EXPLORE_MAP    -> Explorer FSM: pure frontier exploration.
          SEARCH_BLUE    -> Explorer FSM keeps exploring, ALSO watches for
                             the blue object; switches to GO_BLUE once it
                             is seen and confirmed reachable.
          GO_BLUE        -> GoToPoint FSM drives straight to the blue object.
          SEARCH_YELLOW  -> same as SEARCH_BLUE, but for yellow.
          GO_YELLOW      -> same as GO_BLUE, but for yellow.
          DONE           -> stop motors, save map once.

        Note: perception (_perceive(), called just before _act() every
        step -- see run()) is completely independent of `self.mission`.
        The map keeps building and the camera keeps looking for hazards
        and tracked objects no matter which phase is active below.
        """
        if self.mission == Mission.EXPLORE_MAP:
            self._act_explore_map()

        elif self.mission == Mission.SEARCH_BLUE:
            self._act_search(self.blue_object, Mission.GO_BLUE, Mission.SEARCH_YELLOW)

        elif self.mission == Mission.GO_BLUE:
            self._act_go_to(self.blue_object, Mission.SEARCH_YELLOW, Mission.SEARCH_BLUE)

        elif self.mission == Mission.SEARCH_YELLOW:
            self._act_search(self.yellow_object, Mission.GO_YELLOW, Mission.DONE)

        elif self.mission == Mission.GO_YELLOW:
            self._act_go_to(self.yellow_object, Mission.DONE, Mission.SEARCH_YELLOW)

        elif self.mission == Mission.DONE:
            self.robot.stop()
            self._save_once()

    # ---------------------------------------------------------------------- #
    # Mission phase handlers
    # ---------------------------------------------------------------------- #
    def _act_explore_map(self):
        """EXPLORE_MAP: pure frontier exploration, no object-seeking yet."""
        v, w = self.explorer.update(self.pose, self.ranges, self.robot.bearings, self.now,
                                    # scan_similarity=self.get_scan_similarity_to_previous(),
                                    # previous_speed_command=self.robot.previous_v
                                    )
        self.robot.set_velocity(v, w)
        self._refresh_object_reachability()
        if self.explorer.finished or self.explorer.is_spin_seed_phase() is False:
            self._advance_from_explore()

    def _act_search(self, target_obj, go_mission, exhausted_mission):
        """SEARCH_BLUE / SEARCH_YELLOW: keep frontier-exploring (identical to
        EXPLORE_MAP) while watching `target_obj`.  As soon as it has been
        seen AND the flood-fill confirms it is currently reachable, switch
        the mission to `go_mission` (GO_BLUE/GO_YELLOW) and hand the target
        to GoToPoint.

        If the whole map gets fully explored first without finding the
        object, move on to `exhausted_mission` instead of stalling forever.
        """
        v, w = self.explorer.update(self.pose, self.ranges, self.robot.bearings, self.now)
        self.robot.set_velocity(v, w)
        self._refresh_object_reachability()

        if target_obj.seen and target_obj.reachable:
            print("[mission] %s object found and reachable at (%.2f, %.2f) -> %s"
                  % (target_obj.color_name, target_obj.world_xy[0],
                     target_obj.world_xy[1], go_mission))
            self.goto.start(target_obj.world_xy)
            self.mission = go_mission
            return

        if self.explorer.finished:
            print("[mission] map fully explored, %s object never found/reachable -> %s"
                  % (target_obj.color_name, exhausted_mission))
            self.explorer.resume()   # in case exhausted_mission is another SEARCH_* state
            self.mission = exhausted_mission

    def _act_go_to(self, target_obj, arrived_mission, fallback_mission):
        """GO_BLUE / GO_YELLOW: drive straight to a known object position.

        On arrival, moves on to `arrived_mission`.  If GoToPoint gives up
        (the target turned out to be unreachable after all -- e.g. a wall
        was discovered mid-drive that blocks the only path), falls back to
        `fallback_mission` so the robot resumes searching/exploring instead
        of getting stuck.
        """
        v, w = self.goto.update(self.pose, self.ranges, self.robot.bearings, self.now)
        self.robot.set_velocity(v, w)

        if self.goto.arrived:
            print("[mission] arrived at the %s object -> %s"
                  % (target_obj.color_name, arrived_mission))
            self.mission = arrived_mission

        elif self.goto.failed:
            print("[mission] could not reach the %s object after all -> back to %s"
                  % (target_obj.color_name, fallback_mission))
            target_obj.reachable = False   # force a fresh reachability check
            self.explorer.resume()
            self.mission = fallback_mission

    def _refresh_object_reachability(self):
        """Refresh both tracked objects' `reachable` flag from the flood-fill
        mask the frontier planner already computed this PLAN cycle -- a
        single cell lookup per object, effectively free (see
        TrackedObject.update_reachable)."""
        if self.blue_object.world_xy is None and self.yellow_object.world_xy is None:
            return
    
        reachable = self.planner.get_target_reachablity_mask(self.grid, self.pose[:2])
        if reachable is not None:
            self.blue_object.update_reachable(reachable, self.grid)
            self.yellow_object.update_reachable(reachable, self.grid)

    # ---------------------------------------------------------------------- #
    # Per-step visualisation
    # ---------------------------------------------------------------------- #
    def _render(self):
        """Refresh the matplotlib map visualisation.

        Only called every VIZ_EVERY steps to keep the simulation fast.
        Rendering every step would stall the controller.
        """
        if self.step_i % C.VIZ_EVERY == 0:
            world_path, target_xy = self._current_path_and_target()
            self.viz.update(
                self.pose,
                scan_xy    = self._scan_world_points(),
                world_path = world_path,
                target_xy  = target_xy,
            )


            self.viz.update_drive_map(
                pose=self.pose,
                # self.explorer._blocked_cache,    # image 1
                # self.explorer._reachable_cache,  # image 2
                nav = self.explorer._nav_cache,         # image 3 + robot pose
                fmask = self.explorer._fmask_cache,      # image 4
                fclusters = self.explorer._fcluster_cache,   # image 5 (list of cluster dicts)
            )



    def get_visualisation_data_drive_map(self):
        """Return the data used to render the images in the "drive map" tab.

        Returns:
            dict with keys:
                nav       : 2D NumPy array of navigation grid values (1.0=free, 0.5=unknown, 0.0=blocked)
                fmask     : 2D boolean NumPy array of frontier cells
                fclusters : list of dicts, each with keys 'centroid' and 'size'
        """
        if self.mission == Mission.GO_BLUE or self.mission == Mission.GO_YELLOW:
            # In GO_BLUE/GO_YELLOW, the GoToPoint FSM is active, but we still want to show the last Explorer data.
            return {
                "nav": self.goto._nav_cache,
                "fmask": None,      # GoToPoint does not compute frontiers
                "fclusters": None,  # GoToPoint does not compute frontiers
            }
        else:
            return {
                "nav": self.explorer._nav_cache,
                "fmask": self.explorer._fmask_cache,
                "fclusters": self.explorer._fcluster_cache,
            }

    def _current_path_and_target(self):
        """Pick which FSM's path/target to display, based on the active mission.

        Explorer and GoToPoint are never active at the same time (see
        _act()), so exactly one of them holds the currently-relevant path.
        """
        if self.mission in (Mission.GO_BLUE, Mission.GO_YELLOW):
            return self.goto.world_path, self.goto.target_xy
        return self.explorer.world_path, self.explorer.target_xy

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
            self.explorer.resume()   # Explorer.finished is True -> needs resetting
            self.mission = Mission.SEARCH_BLUE
            print("[mission] -> SEARCH_BLUE.")
        else:
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
