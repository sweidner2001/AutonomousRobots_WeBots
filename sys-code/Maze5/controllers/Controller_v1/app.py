"""
app.py
======
The top-level application: ``SlamApp`` owns every subsystem and runs the main
control loop.  It is the "conductor" that wires the hardware, the SLAM
library, the map and the visualiser together.

THE LOOP (once per simulation step)
-----------------------------------
    perceive  -- read encoders/IMU/lidar, update odometry, run the SLAM update
    map       -- fuse the scan into the occupancy grid (rebuild on loop close)
    act       -- read the keyboard and drive the robot (tele-op)
    render    -- refresh the live map view

SUBSYSTEMS
----------
    Robot          hardware abstraction (sensors + motors + keyboard)
    Odometry       wheel+IMU motion prior  -> a raw Pose2D each step
    ScanMatcher    ICP front-end           (inside GraphSlam)
    GraphSlam      pose graph + loop closure + optimisation
    OccupancyGrid  the digital map (log-odds)
    Teleop         keyboard -> (v, w)
    MapViz         live matplotlib view
"""

import os

import Maze5.controllers.Controller_v1.config as C
from Maze5.controllers.Controller_v1.robot import Robot
from Maze5.controllers.Controller_v1.odometry import Odometry
from Maze5.controllers.Controller_v1.occupancy_map import OccupancyGrid
from Maze5.controllers.Controller_v1.teleop import Teleop
from Maze5.controllers.Controller_v1.mapviz import MapViz
from Maze5.controllers.Controller_v1.slam import Scan, ScanMatcher, GraphSlam


class SlamApp:
    """Owns all components and runs the perceive/map/act/render loop."""

    def __init__(self):
        # --- hardware + map ------------------------------------------------
        self.robot = Robot()
        self.odom = Odometry()
        self.grid = OccupancyGrid()
        self.teleop = Teleop()
        self.viz = MapViz(self.grid)

        # --- SLAM (front-end + back-end), configured from config.py --------
        matcher = ScanMatcher(
            max_iters=C.ICP_MAX_ITERS,
            max_corr_dist=C.ICP_MAX_CORR_DIST,
            converge_eps=C.ICP_CONVERGE_EPS,
            min_points=C.ICP_MIN_POINTS,
        )
        self.slam = GraphSlam(
            matcher,
            keyframe_dist=C.KEYFRAME_DIST,
            keyframe_angle=C.KEYFRAME_ANGLE,
            loop_search_radius=C.LOOP_SEARCH_RADIUS,
            loop_min_gap=C.LOOP_MIN_GAP,
            loop_fitness_min=C.LOOP_FITNESS_MIN,
            loop_residual_max=C.LOOP_RESIDUAL_MAX,
            info_odom=(C.INFO_ODOM_XY, C.INFO_ODOM_THETA),
            info_loop=(C.INFO_LOOP_XY, C.INFO_LOOP_THETA),
            opt_iters=C.GRAPH_OPT_ITERS,
        )

        # --- per-step state ------------------------------------------------
        self.step_i = 0
        self.now = 0.0
        self.current_pose = None
        self.scan = None

        self.out_dir = os.path.dirname(os.path.abspath(__file__))

    # ---------------------------------------------------------------------- #
    def run(self):
        """Run until Webots stops, then save the final map."""
        self._startup()
        while self.robot.step():
            self._perceive()
            self._map()
            self._act()
            self._render()
        self._shutdown()

    # ---------------------------------------------------------------------- #
    def _startup(self):
        """One-time init after the first step (sensors need one tick first)."""
        self.robot.step()
        self.odom.initialise(self.robot.read_encoders(), self.robot.read_yaw())
        print("[app] start: dt=%.3f s, lidar rays=%d, grid %dx%d cells" % (
            self.robot.dt, len(self.robot.bearings),
            self.grid.ncols, self.grid.nrows))

    # ---------------------------------------------------------------------- #
    def _perceive(self):
        """Read sensors, update odometry, run the SLAM front/back-end."""
        self.step_i += 1
        self.now = self.robot.get_time()

        odom_pose = self.odom.update(self.robot.read_encoders(),
                                     self.robot.read_yaw())
        # Cap just below the sensor's true max range so that "no-return" rays
        # (which Webots may report as the max value rather than inf) are
        # filtered out instead of mapped as phantom walls at maximum range.
        max_range = min(C.LIDAR_MAX_RANGE, 0.99 * self.robot.lidar_max)
        self.scan = Scan(self.robot.read_lidar(), self.robot.bearings,
                         min_range=C.LIDAR_MIN_RANGE,
                         max_range=max_range,
                         max_points=C.SCAN_MAX_POINTS)

        if self.step_i % C.SLAM_EVERY == 0:
            self.current_pose = self.slam.update(odom_pose, self.scan)

    # ---------------------------------------------------------------------- #
    def _map(self):
        """Fuse the live scan; rebuild fully after a loop-closure optimisation."""
        if self.current_pose is None:
            return
        if self.step_i % C.MAP_EVERY == 0:
            self.grid.integrate(self.current_pose, self.scan)
        if self.slam.take_map_dirty():
            # Poses just shifted -> redraw every keyframe from its new pose.
            self.grid.rebuild(self.slam.keyframe_poses())

    # ---------------------------------------------------------------------- #
    def _act(self):
        """Drive the robot from the keyboard."""
        keys = self.robot.read_keys()
        v, w, save_requested = self.teleop.command(keys)
        self.robot.set_velocity(v, w)
        if save_requested:
            self._save()

    # ---------------------------------------------------------------------- #
    def _render(self):
        """Refresh the live map view every few steps."""
        if self.current_pose is None or self.step_i % C.VIZ_EVERY != 0:
            return
        scan_world = self.current_pose.transform_points(self.scan.points)
        node_xy, _odom_pairs, loop_pairs = self.slam.edges_for_view()
        self.viz.update(self.current_pose, scan_world, node_xy, loop_pairs)

    # ---------------------------------------------------------------------- #
    def _shutdown(self):
        """Webots stopped: park the robot and save the final map."""
        self.robot.stop()
        self._save()

    def _save(self):
        """Write the map image (PNG) and the raw log-odds array (NPY)."""
        self.viz.save(os.path.join(self.out_dir, C.SAVE_MAP_PNG))
        self.grid.save(os.path.join(self.out_dir, C.SAVE_MAP_NPY))
        print("[app] map saved (%d keyframes) -> %s"
              % (len(self.slam.keyframes), self.out_dir))
