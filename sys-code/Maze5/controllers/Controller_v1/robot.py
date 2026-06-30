"""
robot.py
========
Hardware Abstraction Layer (HAL) for the RosBot 2.

Every other module talks to the robot through THIS class instead of calling
Webots functions directly.  If a device name or the sensor layout changes,
only this file needs editing.

DEVICES (standard Webots ``Rosbot`` PROTO)
------------------------------------------
  Wheels   : 4-wheel skid-steer (fl, fr, rl, rr), driven in velocity mode.
  Encoders : one per wheel; cumulative rotation angle (rad) -> odometry.
  IMU      : an InertialUnit; we read its YAW as the robot's heading.
  Lidar    : RPLidar A2 ("laser"), a 360 deg range scanner -> mapping.
  Camera   : RGB-D (Astra); enabled but unused here, reserved for phase 2.
  Keyboard : Webots keyboard input -> manual tele-operation.  This is part of
             the normal Robot API (NOT the supervisor).

PUBLIC API
----------
  step()             advance one simulation tick; False when Webots stops
  get_time()         simulation time (s)
  read_lidar()       1-D array of ranges (m); inf = no return
  read_encoders()    dict {'fl','fr','rl','rr'} of wheel angles (rad)
  read_yaw()         IMU heading (rad)
  read_keys()        set of currently pressed key codes
  set_velocity(v,w)  drive forward v (m/s) and turn w (rad/s)
  stop()             all wheels to zero
  bearings           per-ray angle array (rad, robot frame), computed once
  dt                 duration of one step (s)
"""

import numpy as np

from controller import Robot as WebotsRobot

import Maze5.controllers.Controller_v1.config as C


class Robot:
    """Owns all Webots device handles and exposes a small, clean API."""

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
        """Get handles to all devices and enable the sensors.

        Motors run in velocity mode: setting position = infinity turns a
        position-controlled joint into a speed-controlled one.  Sensors must
        be ``enable``d with a sampling period (we use the simulation
        timestep, so they refresh every step).
        """
        # Wheel motors (skid-steer): velocity mode, start stopped.
        self.front_left_motor = self.robot.getDevice("fl_wheel_joint")
        self.front_right_motor = self.robot.getDevice("fr_wheel_joint")
        self.rear_left_motor = self.robot.getDevice("rl_wheel_joint")
        self.rear_right_motor = self.robot.getDevice("rr_wheel_joint")
        for m in (self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # Wheel encoders -> odometry (distance travelled).
        self.fl_enc = self.robot.getDevice("front left wheel motor sensor")
        self.fr_enc = self.robot.getDevice("front right wheel motor sensor")
        self.rl_enc = self.robot.getDevice("rear left wheel motor sensor")
        self.rr_enc = self.robot.getDevice("rear right wheel motor sensor")
        for s in (self.fl_enc, self.fr_enc, self.rl_enc, self.rr_enc):
            s.enable(self.timestep)

        # IMU (InertialUnit) -> heading.  Reliable on a skid-steer base where
        # wheel-derived heading drifts badly.
        self.imu = self.robot.getDevice("imu inertial_unit")
        self.imu.enable(self.timestep)

        # RGB-D camera (Astra): enabled so Webots buffers frames; not used
        # during SLAM/teleop, reserved for a later colour-search phase.
        self.camera_rgb = self.robot.getDevice("camera rgb")
        self.camera_rgb.enable(self.timestep)
        self.camera_depth = self.robot.getDevice("camera depth")
        self.camera_depth.enable(self.timestep)

        # 2-D lidar (RPLidar A2): the main mapping/localisation sensor.
        # Resolution / FoV / range are read live so the code adapts to the PROTO.
        self.lidar = self.robot.getDevice("laser")
        self.lidar.enable(self.timestep)
        self.lidar_resolution = self.lidar.getHorizontalResolution()
        self.lidar_fov = self.lidar.getFov()
        self.lidar_max = self.lidar.getMaxRange()

        # Keyboard for tele-operation (normal Robot API, not the supervisor).
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.timestep)

    def _compute_bearings(self):
        """Per-ray bearing angle in the robot frame, computed once.

        Webots returns the range image as one distance per ray, ordered
        across the field of view.  The bearing of ray ``i`` is::

            bearing(i) = SIGN * (FoV/2 - (i + 0.5) * FoV/N) + OFFSET

        bearing = 0 is straight ahead, > 0 is to the LEFT.  The ``+0.5``
        centres each ray inside its angular bin.  ``SIGN``/``OFFSET`` (config)
        let us fix a mirrored or rotated mounting without touching the maths.
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
        """Advance one timestep.  Returns False when Webots asks us to stop."""
        return self.robot.step(self.timestep) != -1

    def get_time(self):
        """Current simulation time in seconds."""
        return self.robot.getTime()

    # ------------------------------------------------------------------ #
    # Sensors
    # ------------------------------------------------------------------ #
    def read_lidar(self):
        """360 deg scan as a float array of ranges (m); inf = no return."""
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
        """IMU heading (rad).  getRollPitchYaw() -> (roll, pitch, yaw)."""
        return self.imu.getRollPitchYaw()[2]

    def read_keys(self):
        """Return the set of key codes currently held down.

        Webots reports one key per ``getKey()`` call and returns -1 when no
        more are pending, so we drain the queue into a set.  This lets the
        tele-op layer handle several keys at once (e.g. forward + turn).
        """
        keys = set()
        while True:
            k = self.keyboard.getKey()
            if k == -1:
                break
            keys.add(k)
        return keys

    # ------------------------------------------------------------------ #
    # Actuation
    # ------------------------------------------------------------------ #
    def set_velocity(self, v, w):
        """Drive the skid-steer base at forward speed v (m/s), turn rate w (rad/s).

        Differential-drive kinematics: the two sides run at different linear
        speeds; the difference produces rotation.
            left  = v - w * track/2 ,   right = v + w * track/2
        Linear speed -> wheel angular speed via the wheel radius, clamped to
        the motor limit.
        """
        half = C.WHEEL_BASE / 2.0
        left = (v - w * half) / C.WHEEL_RADIUS
        right = (v + w * half) / C.WHEEL_RADIUS

        lim = C.MAX_WHEEL_SPEED
        left = max(-lim, min(lim, left))
        right = max(-lim, min(lim, right))

        self.front_left_motor.setVelocity(left)
        self.rear_left_motor.setVelocity(left)
        self.front_right_motor.setVelocity(right)
        self.rear_right_motor.setVelocity(right)

    def stop(self):
        """Set all wheels to zero."""
        self.set_velocity(0.0, 0.0)
