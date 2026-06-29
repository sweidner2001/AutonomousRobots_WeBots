"""
explorer.py
===========
Two classes:

  * Explorer      -- the exploration sub-state-machine (SPIN_SEED -> PLAN ->
                     DRIVE -> DONE) that turns perception into wheel commands.
  * MazeExplorer  -- the top-level orchestrator: owns every component, runs the
                     mission state machine, and keeps the main loop short.

The main loop is deliberately three calls:  perceive -> act -> render.
"""

import math
import os

import numpy as np

import Maze5.controllers.Controller_v1.config as C
from Maze5.controllers.Controller_v1.robot import Robot
from Maze5.controllers.Controller_v1.occupancy_grid import OccupancyGrid
from Maze5.controllers.Controller_v1.odometry import Odometry
from Maze5.controllers.Controller_v1.frontier import FrontierDetector
from Maze5.controllers.Controller_v1.planner import PathPlanner
from Maze5.controllers.Controller_v1.pilot import Pilot
from Maze5.controllers.Controller_v1.mapviz import MapViz
from Maze5.controllers.Controller_v1.mission import Mission


class Explorer:
    """Frontier-exploration finite state machine.

    Phases:
        SPIN_SEED : rotate ~one turn in place to seed the map with a 360 view
        PLAN      : detect frontiers, choose the best reachable one, plan a path
        DRIVE     : follow the path (pure pursuit + reactive lidar safety)
        DONE      : nothing reachable left -> finished
    """
    SPIN_SEED = "SPIN_SEED"
    PLAN = "PLAN"
    DRIVE = "DRIVE"
    DONE = "DONE"

    def __init__(self, grid, frontier, planner, pilot):
        self.grid = grid
        self.frontier = frontier
        self.planner = planner
        self.pilot = pilot

        self.phase = self.SPIN_SEED
        self.finished = False

        # spin-seed bookkeeping
        self._spin_accum = 0.0
        self._prev_yaw = None
        # planning / driving bookkeeping
        self._last_plan_time = -1e9
        self._fail_count = 0
        self._last_progress_xy = (0.0, 0.0)
        self._last_progress_time = 0.0
        # current plan (also read by the visualizer)
        self._path_rc = None
        self.world_path = None
        self.target_xy = None

    # ------------------------------------------------------------------ #
    def update(self, pose, ranges, bearings, now):
        """Return (v, w) wheel command for this tick."""
        if self.phase == self.SPIN_SEED:
            return self._spin(pose)
        if self.phase == self.PLAN:
            return self._plan(pose, now)
        if self.phase == self.DRIVE:
            return self._drive(pose, ranges, bearings, now)
        return 0.0, 0.0  # DONE

    # ------------------------------------------------------------------ #
    def _spin(self, pose):
        yaw = pose[2]
        if self._prev_yaw is None:
            self._prev_yaw = yaw
        d = abs(math.atan2(math.sin(yaw - self._prev_yaw),
                           math.cos(yaw - self._prev_yaw)))
        self._spin_accum += d
        self._prev_yaw = yaw
        if self._spin_accum >= C.SPIN_SEED_TURN:
            self.phase = self.PLAN
            return 0.0, 0.0
        return 0.0, C.MAX_TURN_SPEED * 0.6

    # ------------------------------------------------------------------ #
    def _plan(self, pose, now):
        blocked, unknown = self.planner.build_cost_layers(self.grid)
        fmask = self.frontier.detect_cells(self.grid)
        if not fmask.any():
            print("[explorer] no frontiers left -> exploration complete.")
            self.phase = self.DONE
            self.finished = True
            return 0.0, 0.0

        clusters = self.frontier.cluster(fmask)
        path_rc, target = self.planner.choose_target(
            self.grid, clusters, (pose[0], pose[1]), blocked, unknown)

        if path_rc is None or target is None:
            self._fail_count += 1
            print("[explorer] no reachable frontier (%d)." % self._fail_count)
            if self._fail_count >= 6:
                print("[explorer] giving up on frontiers -> done.")
                self.phase = self.DONE
                self.finished = True
                return 0.0, 0.0
            return 0.0, C.MAX_TURN_SPEED * 0.5  # nudge to refresh the view

        self._fail_count = 0
        self._path_rc = path_rc
        self.world_path = self.planner.path_to_world(self.grid, path_rc)
        gr, gc = target["centroid"]
        self.target_xy = self.grid.grid_to_world(gc, gr)
        self.pilot.set_path(self.world_path)
        self._last_plan_time = now
        self._last_progress_xy = (pose[0], pose[1])
        self._last_progress_time = now
        self.phase = self.DRIVE
        return 0.0, 0.0

    # ------------------------------------------------------------------ #
    def _drive(self, pose, ranges, bearings, now):
        v, w, done = self.pilot.compute(pose, ranges, bearings)
        if self._needs_replan(pose, done, now):
            self.pilot.clear()
            self._path_rc = None
            self.world_path = None
            self.target_xy = None
            self.phase = self.PLAN
            return 0.0, 0.0
        return v, w

    def _needs_replan(self, pose, done, now):
        if done:
            return True
        if (now - self._last_plan_time) >= C.PLAN_PERIOD:
            return True
        # path invalidated by newly-seen walls?
        blocked, _ = self.planner.build_cost_layers(self.grid)
        if self._path_rc and self.planner.path_blocked(self._path_rc, blocked):
            return True
        # stuck?
        moved = math.hypot(pose[0] - self._last_progress_xy[0],
                           pose[1] - self._last_progress_xy[1])
        if moved > C.STUCK_DIST:
            self._last_progress_xy = (pose[0], pose[1])
            self._last_progress_time = now
        elif (now - self._last_progress_time) > C.STUCK_TIME:
            print("[explorer] stuck -> replanning.")
            return True
        return False


class MazeExplorer:
    """Owns all components and runs the mission state machine."""

    def __init__(self):
        self.robot = Robot()
        self.grid = OccupancyGrid()
        self.odom = Odometry()
        self.frontier = FrontierDetector()
        self.planner = PathPlanner()
        self.pilot = Pilot()
        self.viz = MapViz(self.grid)
        self.explorer = Explorer(self.grid, self.frontier, self.planner,
                                 self.pilot)

        self.mission = Mission.EXPLORE_MAP
        self.step_i = 0
        self.now = 0.0
        self.pose = (0.0, 0.0, 0.0)
        self.ranges = None
        self._saved = False
        self.out_dir = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------ #
    # Main loop -- intentionally tiny
    # ------------------------------------------------------------------ #
    def run(self):
        self._startup()
        while self.robot.step():
            self._perceive()
            self._act()
            self._render()
        self._shutdown()

    # ------------------------------------------------------------------ #
    def _startup(self):
        # One step so the sensors deliver real values, then seed odometry.
        self.robot.step()
        self.odom.initialise(self.robot.read_encoders(), self.robot.read_yaw())
        print("[mission] start: dt=%.3fs, lidar rays=%d, map %dx%d cells" %
              (self.robot.dt, len(self.robot.bearings),
               self.grid.ncols, self.grid.nrows))

    def _perceive(self):
        self.step_i += 1
        self.now = self.robot.get_time()
        self.pose = self.odom.update(self.robot.read_encoders(),
                                     self.robot.read_yaw())
        self.ranges = self.robot.read_lidar()
        if self.step_i % C.MAP_EVERY == 0:
            self.grid.integrate_scan(self.pose[0], self.pose[1], self.pose[2],
                                     self.ranges, self.robot.bearings)

    def _act(self):
        if self.mission == Mission.EXPLORE_MAP:
            v, w = self.explorer.update(self.pose, self.ranges,
                                        self.robot.bearings, self.now)
            self.robot.set_velocity(v, w)
            if self.explorer.finished:
                self._advance_from_explore()
        elif self.mission == Mission.DONE:
            self.robot.stop()
            self._save_once()
        else:
            self._color_stub()  # SEARCH_BLUE / GO_BLUE / SEARCH_YELLOW / GO_YELLOW

    def _render(self):
        if self.step_i % C.VIZ_EVERY == 0:
            self.viz.update(self.pose,
                            scan_xy=self._scan_world_points(),
                            world_path=self.explorer.world_path,
                            target_xy=self.explorer.target_xy)

    # ------------------------------------------------------------------ #
    # Mission transitions
    # ------------------------------------------------------------------ #
    def _advance_from_explore(self):
        print("[mission] EXPLORE_MAP complete.")
        if C.MISSION_ENABLE_COLOR:
            self.mission = Mission.SEARCH_BLUE
            print("[mission] -> SEARCH_BLUE (scaffold, not yet implemented).")
        else:
            self.mission = Mission.DONE

    def _color_stub(self):
        # TODO: implement camera-based blue/yellow search & approach.
        self.robot.stop()
        print("[mission] state %s is scaffolding only -> DONE." % self.mission)
        self.mission = Mission.DONE

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _scan_world_points(self):
        """World (x, y) of finite lidar hits, for plotting."""
        x, y, theta = self.pose
        sx = x + C.LIDAR_OFFSET_X * math.cos(theta)
        sy = y + C.LIDAR_OFFSET_X * math.sin(theta)
        r = self.ranges
        finite = np.isfinite(r) & (r > C.LIDAR_MIN_RANGE) & (r < self.robot.lidar_max * 0.999)
        a = theta + self.robot.bearings[finite]
        rr = r[finite]
        return sx + rr * np.cos(a), sy + rr * np.sin(a)

    def _save_once(self):
        if self._saved:
            return
        self._save()
        self._saved = True

    def _save(self):
        try:
            self.viz.save(os.path.join(self.out_dir, C.SAVE_MAP_PNG))
            self.grid.save(os.path.join(self.out_dir, C.SAVE_MAP_NPY))
            print("[mission] map saved to %s" % self.out_dir)
        except Exception as e:                       # pragma: no cover
            print("[mission] save error: %s" % e)

    def _shutdown(self):
        self._save()
