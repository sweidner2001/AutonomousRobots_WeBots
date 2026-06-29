"""
robot.py
========
Hardware abstraction for the RosBot 2.

The Robot class owns the Webots `Robot` handle and every device, and exposes a
small, intention-revealing API to the rest of the code:

    step()            advance the simulation one tick (False when Webots stops)
    read_lidar()      -> np.array of ranges (m)
    read_encoders()   -> dict {'fl','fr','rl','rr'} of wheel angles (rad)
    read_yaw()        -> IMU heading (rad)
    set_velocity(v,w) command a forward / angular velocity (skid-steer)
    stop()            set all wheels to zero
    bearings          per-ray angle (rad, robot frame), computed once

Device *names* live here (not in config); lidar resolution / FoV / max-range are
read live from the sensor so the code adapts if the PROTO changes.
"""

import numpy as np

from controller import Robot as WebotsRobot

import Maze5.controllers.Controller_v1.config as C


class Robot:
    def __init__(self):
        self.robot = WebotsRobot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt = self.timestep / 1000.0

        self._init_devices()
        self.bearings = self._compute_bearings()

    # ------------------------------------------------------------------ #
    # Device setup
    # ------------------------------------------------------------------ #
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

        # IMU (InertialUnit) gives an absolute heading, used for odometry.
        self.imu = self.robot.getDevice("imu inertial_unit")
        self.imu.enable(self.timestep)

        # RGB-D camera (Astra) -- initialised now, used later by the colour
        # mission (SEARCH_BLUE / SEARCH_YELLOW). Not read during exploration.
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

    def _compute_bearings(self):
        """Per-ray bearing in the robot frame (rad).

        Webots orders the range image from +FoV/2 to -FoV/2.
        """
        n = self.lidar_resolution
        fov = self.lidar_fov
        i = np.arange(n)
        bearings = (fov / 2.0) - (i + 0.5) * (fov / n)
        return C.LIDAR_ANGLE_SIGN * bearings + C.LIDAR_ANGLE_OFFSET

    # ------------------------------------------------------------------ #
    # Simulation stepping
    # ------------------------------------------------------------------ #
    def step(self):
        """Advance one tick. Returns False when Webots asks us to quit."""
        return self.robot.step(self.timestep) != -1

    def get_time(self):
        return self.robot.getTime()

    # ------------------------------------------------------------------ #
    # Sensors
    # ------------------------------------------------------------------ #
    def read_lidar(self):
        """360 deg scan as a numpy array of ranges (m). inf = no return."""
        return np.array(self.lidar.getRangeImage(), dtype=np.float32)

    def read_encoders(self):
        """Wheel angles (rad) keyed by wheel position."""
        return {
            "fl": self.fl_enc.getValue(),
            "fr": self.fr_enc.getValue(),
            "rl": self.rl_enc.getValue(),
            "rr": self.rr_enc.getValue(),
        }

    def read_yaw(self):
        """IMU heading (rad), rotation about the world up-axis."""
        return self.imu.getRollPitchYaw()[2]

    # ------------------------------------------------------------------ #
    # Actuation
    # ------------------------------------------------------------------ #
    def set_velocity(self, v, w):
        """Drive the skid-steer base at forward speed v (m/s), turn rate w (rad/s)."""
        left_lin = v - w * C.WHEEL_BASE / 2.0
        right_lin = v + w * C.WHEEL_BASE / 2.0
        left_w = left_lin / C.WHEEL_RADIUS
        right_w = right_lin / C.WHEEL_RADIUS

        lim = C.MAX_WHEEL_SPEED
        left_w = max(-lim, min(lim, left_w))
        right_w = max(-lim, min(lim, right_w))

        self.front_left_motor.setVelocity(left_w)
        self.rear_left_motor.setVelocity(left_w)
        self.front_right_motor.setVelocity(right_w)
        self.rear_right_motor.setVelocity(right_w)

    def stop(self):
        self.set_velocity(0.0, 0.0)
