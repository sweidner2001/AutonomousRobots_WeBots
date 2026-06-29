"""
Controller_v1.py  --  RosBot 2 autonomous maze explorer (base solution)
=======================================================================

Pipeline (no ROS, no SLAM yet):

      IMU yaw + wheel encoders  ->  Odometry  ->  pose (x, y, theta)
                                                     |
      RpLidar A2 (360 deg scan) -------------------> OccupancyGrid (log-odds)
                                                     |
                          Frontier detection  --->  A* planner  --->  Pilot
                                                     |                  |
                                            live matplotlib map     wheel speeds

State machine:
    SPIN_SEED : rotate ~one turn in place to seed the map with a 360 view
    PLAN      : detect frontiers, choose the best reachable one, plan a path
    DRIVE     : follow the path (pure pursuit + reactive lidar safety)
    DONE      : no reachable frontiers left -> stop and save the map

The pose source (Odometry) is deliberately isolated so a Graph-SLAM module can
replace it later without touching mapping, planning or control.

NOTE: assign this controller to the Rosbot in Webots (Scene tree -> Rosbot ->
controller field -> "Controller_v1").
"""

import math
import os
import sys

import numpy as np

# Make sibling modules importable no matter how Webots launches us.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import Robot                      # Webots

import Maze5.controllers.Controller_v1.config as C
from Maze5.controllers.Controller_v1.occupancy_grid import OccupancyGrid
from Maze5.controllers.Controller_v1.odometry import Odometry
import Maze5.controllers.Controller_v1.frontier as F
import Maze5.controllers.Controller_v1.planner as P
from Maze5.controllers.Controller_v1.pilot import Pilot
from Maze5.controllers.Controller_v1.mapviz import MapViz


# ---------------------------------------------------------------------- #
# Device setup
# ---------------------------------------------------------------------- #
def setup(robot, timestep):
    dev = {}

    lidar = robot.getDevice(C.LIDAR_NAME)
    lidar.enable(timestep)
    try:
        lidar.enablePointCloud()
    except Exception:
        pass
    dev["lidar"] = lidar

    imu = robot.getDevice(C.IMU_NAME)
    imu.enable(timestep)
    dev["imu"] = imu

    motors = {}
    for key, name in C.MOTOR_NAMES.items():
        m = robot.getDevice(name)
        m.setPosition(float("inf"))     # velocity-control mode
        m.setVelocity(0.0)
        motors[key] = m
    dev["motors"] = motors

    encoders = {}
    for key, name in C.ENCODER_NAMES.items():
        e = robot.getDevice(name)
        e.enable(timestep)
        encoders[key] = e
    dev["encoders"] = encoders

    return dev


def read_encoders(encoders):
    return {k: e.getValue() for k, e in encoders.items()}


def lidar_bearings(lidar):
    """Per-ray bearing in the robot frame (rad)."""
    n = lidar.getHorizontalResolution()
    fov = lidar.getFov()
    i = np.arange(n)
    # Webots orders the range image +FoV/2 -> -FoV/2.
    bearings = (fov / 2.0) - (i + 0.5) * (fov / n)
    bearings = C.LIDAR_ANGLE_SIGN * bearings + C.LIDAR_ANGLE_OFFSET
    return bearings


def read_scan(lidar):
    r = np.array(lidar.getRangeImage(), dtype=np.float32)
    return r


def set_drive(motors, v, w):
    """Convert (v, w) to skid-steer wheel velocities and apply."""
    left_lin = v - w * C.WHEEL_BASE / 2.0
    right_lin = v + w * C.WHEEL_BASE / 2.0
    left_w = left_lin / C.WHEEL_RADIUS
    right_w = right_lin / C.WHEEL_RADIUS
    lim = C.MAX_WHEEL_SPEED
    left_w = max(-lim, min(lim, left_w))
    right_w = max(-lim, min(lim, right_w))
    motors["fl"].setVelocity(left_w)
    motors["rl"].setVelocity(left_w)
    motors["fr"].setVelocity(right_w)
    motors["rr"].setVelocity(right_w)


def scan_world_points(pose, ranges, bearings):
    """World (x, y) of finite lidar hits, for plotting."""
    x, y, theta = pose
    sx = x + C.LIDAR_OFFSET_X * math.cos(theta)
    sy = y + C.LIDAR_OFFSET_X * math.sin(theta)
    finite = np.isfinite(ranges) & (ranges < C.LIDAR_MAX_RANGE * 0.999) \
        & (ranges > C.LIDAR_MIN_RANGE)
    a = theta + bearings[finite]
    rr = ranges[finite]
    xs = sx + rr * np.cos(a)
    ys = sy + rr * np.sin(a)
    return xs, ys


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    dev = setup(robot, timestep)
    motors = dev["motors"]
    lidar = dev["lidar"]
    imu = dev["imu"]
    encoders = dev["encoders"]

    bearings = lidar_bearings(lidar)

    grid = OccupancyGrid()
    odom = Odometry(C.ROBOT_START_X, C.ROBOT_START_Y)
    pilot = Pilot()
    viz = MapViz(grid)

    # Let one step pass so sensors deliver real values.
    robot.step(timestep)
    enc0 = read_encoders(encoders)
    yaw0 = imu.getRollPitchYaw()[2]
    odom.initialise(enc0, yaw0)

    state = "SPIN_SEED"
    spin_accum = 0.0
    prev_yaw = yaw0
    last_plan_time = -1e9
    fail_count = 0

    # stuck tracking
    last_progress_xy = (odom.x, odom.y)
    last_progress_time = robot.getTime()

    cur_path_rc = None
    cur_world_path = None
    cur_target_xy = None

    step_i = 0
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print("[controller] start, dt = %.3f s, lidar rays = %d" %
          (dt, len(bearings)))

    while robot.step(timestep) != -1:
        step_i += 1
        now = robot.getTime()

        # ---- sensing & pose ------------------------------------------
        enc = read_encoders(encoders)
        yaw = imu.getRollPitchYaw()[2]
        pose = odom.update(enc, yaw)
        ranges = read_scan(lidar)

        # ---- mapping (on cadence) ------------------------------------
        if step_i % C.MAP_EVERY == 0:
            grid.integrate_scan(pose[0], pose[1], pose[2], ranges, bearings)

        # ---- behaviour ------------------------------------------------
        v, w = 0.0, 0.0

        if state == "SPIN_SEED":
            d = abs(math.atan2(math.sin(yaw - prev_yaw),
                               math.cos(yaw - prev_yaw)))
            spin_accum += d
            prev_yaw = yaw
            v, w = 0.0, C.MAX_TURN_SPEED * 0.6
            if spin_accum >= C.SPIN_SEED_TURN:
                state = "PLAN"
                set_drive(motors, 0.0, 0.0)

        elif state == "PLAN":
            blocked, unknown = P.build_cost_layers(grid)
            fmask = F.detect_frontier_cells(grid)
            if not fmask.any():
                print("[controller] no frontiers left -> exploration complete.")
                state = "DONE"
            else:
                clusters = F.cluster_frontiers(fmask)
                path_rc, target = P.choose_target(
                    grid, clusters, (pose[0], pose[1]), blocked, unknown)
                if path_rc is None or target is None:
                    fail_count += 1
                    print("[controller] no reachable frontier (%d)." % fail_count)
                    # Nudge: rotate a bit to refresh the view, then retry.
                    v, w = 0.0, C.MAX_TURN_SPEED * 0.5
                    if fail_count >= 6:
                        print("[controller] giving up on frontiers -> DONE.")
                        state = "DONE"
                else:
                    fail_count = 0
                    cur_path_rc = path_rc
                    cur_world_path = P.path_to_world(grid, path_rc)
                    gr, gc = target["centroid"]
                    cur_target_xy = grid.grid_to_world(gc, gr)
                    pilot.set_path(cur_world_path)
                    last_plan_time = now
                    last_progress_xy = (pose[0], pose[1])
                    last_progress_time = now
                    state = "DRIVE"

        elif state == "DRIVE":
            v, w, done = pilot.compute(pose, ranges, bearings)

            # replan triggers
            need_replan = False
            if done:
                need_replan = True
            elif (now - last_plan_time) >= C.PLAN_PERIOD:
                need_replan = True
            else:
                blocked, _ = P.build_cost_layers(grid)
                if cur_path_rc and P.path_blocked(grid, cur_path_rc, blocked):
                    need_replan = True

            # stuck detection
            moved = math.hypot(pose[0] - last_progress_xy[0],
                               pose[1] - last_progress_xy[1])
            if moved > C.STUCK_DIST:
                last_progress_xy = (pose[0], pose[1])
                last_progress_time = now
            elif (now - last_progress_time) > C.STUCK_TIME:
                print("[controller] stuck -> replanning.")
                need_replan = True

            if need_replan:
                pilot.clear()
                cur_path_rc = None
                cur_world_path = None
                cur_target_xy = None
                state = "PLAN"
                v, w = 0.0, 0.0

        elif state == "DONE":
            v, w = 0.0, 0.0

        set_drive(motors, v, w)

        # ---- visualization -------------------------------------------
        if step_i % C.VIZ_EVERY == 0:
            sxy = scan_world_points(pose, ranges, bearings)
            viz.update(pose, scan_xy=sxy,
                       world_path=cur_world_path, target_xy=cur_target_xy)

        # ---- save once when finished ---------------------------------
        if state == "DONE" and step_i % 50 == 0:
            try:
                viz.save(os.path.join(out_dir, C.SAVE_MAP_PNG))
                grid.save(os.path.join(out_dir, C.SAVE_MAP_NPY))
            except Exception as e:
                print("[controller] save error: %s" % e)

    # On Webots stop: final save.
    try:
        viz.save(os.path.join(out_dir, C.SAVE_MAP_PNG))
        grid.save(os.path.join(out_dir, C.SAVE_MAP_NPY))
        print("[controller] map saved to %s" % out_dir)
    except Exception as e:
        print("[controller] final save error: %s" % e)


if __name__ == "__main__":
    main()
