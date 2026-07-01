"""
robot.py  --  Hardware abstraction layer for the RosBot 2.
==========================================================

WHAT IS THIS FILE FOR?
-----------------------
This file wraps the Webots robot API into a simpler, robot-specific
interface.  All other modules talk to THIS file — they never call
Webots functions directly.

This pattern is called a "Hardware Abstraction Layer" (HAL).  Its
benefit:  if the robot hardware changes (different motors, different
sensor names), you only update THIS file, not every module that reads
sensors or drives motors.

THE ROSBOT 2 HARDWARE (short summary)
---------------------------------------
  Wheels   : 4 wheels in a skid-steer layout.
             Left pair   (fl, rl)  and  right pair (fr, rr).
             All four are independently driven.
             To turn LEFT:  right wheels faster than left.
             To turn RIGHT: left wheels faster than right.

  Encoders : Each wheel has a position sensor (encoder) that measures
             the cumulative angle the wheel has rotated (in radians).
             Odometry converts these angles into distances travelled.

  IMU      : An InertialUnit that gives roll, pitch, YAW.
             We only use the yaw (heading, rotation around vertical axis).

  Lidar    : RPLidar A2.  Emits 360° laser scan up to ~8 m range.
             Returns a flat list of distances (range image).

  Camera   : An RGB-D camera (Astra). Not used during mapping.

PUBLIC API
----------
  step()              -- advance one simulation tick; returns False when done
  get_time()          -- current simulation time (seconds)
  read_lidar()        -- 1-D array of ranges (m); inf = no return
  read_encoders()     -- dict {'fl','fr','rl','rr'} of wheel angles (rad)
  read_yaw()          -- IMU heading (rad)
  read_camera_rgb()   -- (H,W,3) uint8 RGB image
  read_camera_depth() -- (H,W) float32 depth image (m), perpendicular Z-depth
  set_velocity(v, w)  -- drive at v m/s forward, w rad/s turning
  stop()              -- set all motors to zero
  bearings            -- precomputed per-ray angle array (rad, robot frame)
  dt                  -- duration of one simulation step (seconds)
"""

import numpy as np

# Webots Python API -- the "controller" module is provided by Webots itself.
# It gives us access to the simulated robot's sensors and actuators.
from controller import Robot as WebotsRobot

import Maze4.controllers.Controller_v1.config as C


class Robot:
    """Abstraction layer: owns all Webots device handles and exposes a
    clean API to the rest of the system."""

    def __init__(self):
        # Create the main Webots robot object.
        # This MUST be the first thing a Webots controller does.
        self.robot = WebotsRobot()

        # getBasicTimeStep() returns the simulation timestep in milliseconds.
        # We store it both as an integer (for step() calls) and as seconds (dt).
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt = self.timestep / 1000.0  # convert ms -> seconds

        self._init_devices()                  # get handles + enable sensors
        self.bearings = self._compute_bearings()  # precompute ray angles once
        self.previous_v = 0.0                   # last commanded forward speed (m/s)

    # ---------------------------------------------------------------------- #
    # Device initialisation
    # ---------------------------------------------------------------------- #
    def _init_devices(self):
        """Get handles to every hardware device and enable sensors.

        In Webots you must call robot.getDevice('name') to obtain a handle
        before you can read from or command a device.

        MOTORS:
          We use velocity control, not position control.
          setPosition(float('inf')) = "never stop rotating" (velocity mode).
          setVelocity(0.0)          = start at rest.

        SENSORS:
          All sensors must be "enabled" with a sampling period in ms.
          Here we use the simulation timestep so the sensor updates every step.
        """

        # ---- Wheel motors (4-wheel skid-steer) ----------------------------
        # Device names come from the Webots PROTO definition for the Rosbot.
        self.front_left_motor  = self.robot.getDevice("fl_wheel_joint")
        self.front_right_motor = self.robot.getDevice("fr_wheel_joint")
        self.rear_left_motor   = self.robot.getDevice("rl_wheel_joint")
        self.rear_right_motor  = self.robot.getDevice("rr_wheel_joint")

        for motor in (self.front_left_motor,  self.front_right_motor,
                      self.rear_left_motor,   self.rear_right_motor):
            motor.setPosition(float("inf"))  # switch to velocity-control mode
            motor.setVelocity(0.0)           # start stopped

        # ---- Wheel encoders (position sensors) ----------------------------
        # These measure how far each wheel has rotated since the start.
        # We use the accumulated angle to estimate distance driven (odometry).
        self.fl_enc = self.robot.getDevice("front left wheel motor sensor")
        self.fr_enc = self.robot.getDevice("front right wheel motor sensor")
        self.rl_enc = self.robot.getDevice("rear left wheel motor sensor")
        self.rr_enc = self.robot.getDevice("rear right wheel motor sensor")
        for encoder in (self.fl_enc, self.fr_enc, self.rl_enc, self.rr_enc):
            encoder.enable(self.timestep)  # update every simulation step

        # ---- IMU (InertialUnit) -------------------------------------------
        # Gives roll / pitch / yaw angles.  We use YAW (index 2) as the
        # robot's compass heading.  The IMU is much more reliable than trying
        # to derive heading from the wheel encoders on a skid-steer robot.
        self.imu = self.robot.getDevice("imu inertial_unit")
        self.imu.enable(self.timestep)

        # ---- RGB-D camera (Astra) ----------------------------------------
        # Used for green floor-hazard detection (see floor_hazard.py).
        # RGB and depth are two PHYSICALLY SEPARATE lenses a few centimetres
        # apart, so their images are not pixel-aligned -- floor_hazard.py
        # performs proper registration before comparing colour and depth.
        self.camera_rgb   = self.robot.getDevice("camera rgb")
        self.camera_rgb.enable(self.timestep)
        self.camera_depth = self.robot.getDevice("camera depth")
        self.camera_depth.enable(self.timestep)

        # Cache intrinsics once (identical resolution/FoV for both lenses
        # on this camera model, read live so the code adapts to the PROTO).
        self.camera_width  = self.camera_rgb.getWidth()
        self.camera_height = self.camera_rgb.getHeight()
        self.camera_fov    = self.camera_rgb.getFov()   # horizontal FoV, rad

        # ---- 2-D Lidar (RPLidar A2) --------------------------------------
        # The lidar spins 360° and measures distance to obstacles.
        # We read the raw "range image" — a flat list of distances.
        # The sensor parameters (resolution, FoV, max range) are read live
        # from the device so the code adapts if the PROTO changes.
        self.lidar = self.robot.getDevice("laser")
        self.lidar.enable(self.timestep)
        self.lidar_resolution = self.lidar.getHorizontalResolution()  # number of rays
        self.lidar_fov        = self.lidar.getFov()                   # full FoV in rad
        self.lidar_max        = self.lidar.getMaxRange()              # m

    def _compute_bearings(self):
        """Precompute the angle of each lidar ray in the robot's own frame.

        Webots orders the range image from the LEFT edge (+FoV/2) sweeping
        clockwise to the RIGHT edge (-FoV/2), i.e.:

            ray 0   -> bearing = +FoV/2          (left-most ray)
            ray N-1 -> bearing = -FoV/2 + delta  (right-most ray)

        In our robot frame:
            bearing = 0      -> directly ahead
            bearing > 0      -> to the left  (counterclockwise)
            bearing < 0      -> to the right (clockwise)

        Formula: bearing(i) = +FoV/2 - (i + 0.5) * (FoV / N)
        The "+0.5" centres each ray within its angular bin.
        """
        n   = self.lidar_resolution
        fov = self.lidar_fov
        i   = np.arange(n)
        # Raw bearings from the Webots ordering.
        bearings = (fov / 2.0) - (i + 0.5) * (fov / n)
        # Apply sign flip / offset from config (in case the sensor is mounted
        # at an angle or the scan appears mirrored).
        return C.LIDAR_ANGLE_SIGN * bearings + C.LIDAR_ANGLE_OFFSET

    # ---------------------------------------------------------------------- #
    # Simulation stepping
    # ---------------------------------------------------------------------- #
    def step(self):
        """Advance the simulation by one timestep.

        Webots uses a synchronous step model: the simulation pauses while
        the controller runs, then advances by exactly one timestep when we
        call robot.step().

        Returns:
            True  -- simulation is still running; sensors have new data.
            False -- Webots asked the controller to stop (user pressed Stop,
                     simulation ended, etc.).  The main loop should exit.
        """
        return self.robot.step(self.timestep) != -1

    def get_time(self):
        """Current simulation time in seconds."""
        return self.robot.getTime()

    # ---------------------------------------------------------------------- #
    # Sensor readings
    # ---------------------------------------------------------------------- #
    def read_lidar(self):
        """Return the latest 360° lidar scan as a NumPy float32 array.

        Each element is the measured distance (in metres) for one ray.
        No-return rays (beam missed everything) are represented as inf.
        Array length == lidar_resolution (e.g. 720 rays for the RPLidar A2).
        """
        return np.array(self.lidar.getRangeImage(), dtype=np.float32)

    def read_encoders(self):
        """Return current wheel angles (rad) as a dict.

        Keys: 'fl' (front-left), 'fr' (front-right),
              'rl' (rear-left),  'rr' (rear-right).

        The angles increase continuously; each 2π means one full wheel
        revolution.  Odometry computes the CHANGE between two readings.
        """
        return {
            "fl": self.fl_enc.getValue(),
            "fr": self.fr_enc.getValue(),
            "rl": self.rl_enc.getValue(),
            "rr": self.rr_enc.getValue(),
        }

    def read_yaw(self):
        """Return the IMU yaw angle (heading) in radians.

        The IMU returns (roll, pitch, yaw).  We take index 2 = yaw.
        Yaw = 0 means the robot points along the world +x axis.
        Yaw increases counterclockwise (standard mathematical convention).
        """
        return self.imu.getRollPitchYaw()[2]

    def read_camera_rgb(self):
        """Return the latest RGB camera frame as a (H, W, 3) uint8 array.

        Webots' Camera.getImage() returns a raw byte buffer in BGRA order
        (4 bytes per pixel).  We reshape it with NumPy (fast, no per-pixel
        Python loop) and drop the alpha channel, then reorder BGR -> RGB.
        """
        raw = self.camera_rgb.getImage()
        img = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.camera_height, self.camera_width, 4)
        )
        return img[:, :, [2, 1, 0]]   # BGRA -> RGB (drop alpha)

    def read_camera_depth(self):
        """Return the latest depth frame as a (H, W) float32 array, in metres.

        Each value is the PERPENDICULAR distance (Z-depth along the optical
        axis) from the camera to the surface at that pixel -- NOT the
        straight-line/radial distance.  This matches Webots' own definition
        of RangeFinder.getRangeImage() and is exactly what the pinhole
        back-projection formulas in floor_hazard.py expect.
        """
        flat = self.camera_depth.getRangeImage()
        return np.array(flat, dtype=np.float32).reshape(
            (self.camera_height, self.camera_width)
        )

    # ---------------------------------------------------------------------- #
    # Actuation
    # ---------------------------------------------------------------------- #
    def set_velocity(self, v, w):
        """Command the skid-steer drive.

        Args:
            v (float): Forward speed in m/s.  Positive = forward, negative = back.
            w (float): Angular (turning) speed in rad/s.
                       Positive = counterclockwise (left turn).
                       Negative = clockwise (right turn).

        SKID-STEER KINEMATICS
        ----------------------
        In skid-steer, both wheels on one side spin at the same speed.
        Turning is achieved by spinning the two sides at DIFFERENT speeds.

          left_linear  = v - w * WHEEL_BASE / 2
          right_linear = v + w * WHEEL_BASE / 2

        Converting linear speed to wheel angular speed (rad/s):
          wheel_angular = linear_speed / WHEEL_RADIUS

        Finally all values are clamped to ±MAX_WHEEL_SPEED.
        """
        half_base = C.WHEEL_BASE / 2.0
        left_linear  = v - w * half_base
        right_linear = v + w * half_base

        left_rad_s  = left_linear  / C.WHEEL_RADIUS
        right_rad_s = right_linear / C.WHEEL_RADIUS

        # Clamp to safe motor limits.
        lim = C.MAX_WHEEL_SPEED
        left_rad_s  = max(-lim, min(lim, left_rad_s))
        right_rad_s = max(-lim, min(lim, right_rad_s))

        # Both wheels on the same side always spin at the same speed.
        self.front_left_motor.setVelocity(left_rad_s)
        self.rear_left_motor.setVelocity(left_rad_s)
        self.front_right_motor.setVelocity(right_rad_s)
        self.rear_right_motor.setVelocity(right_rad_s)

        self.previous_v = v

    def stop(self):
        """Set all motors to zero — robot stands still."""
        self.set_velocity(0.0, 0.0)
        self.previous_v = 0.0
