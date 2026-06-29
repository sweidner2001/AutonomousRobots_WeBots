"""
odometry.py  --  Pose estimation from wheel encoders + IMU.
===========================================================

WHAT IS "ODOMETRY"?
--------------------
Odometry (from Greek: hodos = path) estimates WHERE the robot is by
integrating (accumulating) its motion over time.  Think of it like a
step counter that also tracks direction: each small movement is added
to the previous known position.

WHY DO WE COMBINE ENCODERS WITH THE IMU?
------------------------------------------
The RosBot 2 is a SKID-STEER robot — it turns by spinning the two
sides at different speeds and SKIDDING (slipping) the wheels sideways.

Problem: wheel slip during turns makes the HEADING derived from encoder
differences very inaccurate.  After several turns the cumulative
heading error can be large (tens of degrees).

Solution: use the IMU (InertialUnit / gyroscope) for heading because
the IMU measures actual rotation, not wheel rotation.  Use the
encoders only for forward DISTANCE (which is reliable since the wheels
roll without slip in the forward direction).

HOW THE MATH WORKS
--------------------
At each timestep we:
  1. Measure new wheel encoder angles.
  2. Subtract the previous angles -> delta left, delta right wheels (rad).
  3. Convert angle change to linear distance:  d = delta_angle * WHEEL_RADIUS
  4. Average left and right: d_centre = (d_left + d_right) / 2
  5. Get heading from IMU directly (no integration needed).
  6. Integrate position:
       x += d_centre * cos(mid_theta)
       y += d_centre * sin(mid_theta)
     where mid_theta is the average heading between the previous and
     current orientation (trapezoidal rule).

This is called "encoder odometry with IMU heading fusion".

LIMITATION
-----------
Over long distances the x, y position still accumulates error because
encoder slippage and the offset between the encoder and IMU origins.
A full SLAM (Simultaneous Localisation And Mapping) system would
correct this.  For this project the drift is small enough that the
robot can explore a room-sized maze reliably.

OUTPUT
------
pose = (x, y, theta)
  x, y  : position in metres, relative to the start point (0, 0).
  theta  : heading in radians.
            0   = pointing along +x axis
            π/2 = pointing along +y axis (left)
"""

import math

import Maze4.controllers.Controller_v1.config as C


class Odometry:
    """Estimates robot pose (x, y, theta) from encoders + IMU yaw."""

    def __init__(self):
        # Current estimated pose, starting at the map origin.
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0  # heading (rad)

        # Previous wheel positions for computing delta each step.
        self._prev_left  = None
        self._prev_right = None

        # Flag to know whether initialise() has been called yet.
        self._initialised = False

    # ---------------------------------------------------------------------- #
    def initialise(self, enc, yaw):
        """Seed the encoder baseline and initial heading.

        Call ONCE after the first valid sensor readings are available.
        Webots sensors do not return meaningful data until at least one
        robot.step() has completed, so Controller_v1 calls this after
        the first step.

        Args:
            enc (dict): Encoder readings from robot.read_encoders().
            yaw (float): IMU heading from robot.read_yaw() (radians).
        """
        self._prev_left, self._prev_right = self._wheel_means(enc)
        self.theta = yaw  # start with the real IMU heading
        self._initialised = True

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _wheel_means(enc):
        """Average encoder reading for each side of the robot.

        We have two wheels per side (front + rear).  We average them to
        get a single "left wheel position" and a single "right wheel
        position".  This smooths out tiny differences between front and
        rear encoders on the same side.

        Returns:
            (left_avg, right_avg) in radians.
        """
        left  = 0.5 * (enc["fl"] + enc["rl"])
        right = 0.5 * (enc["fr"] + enc["rr"])
        return left, right

    # ---------------------------------------------------------------------- #
    def update(self, enc, yaw):
        """Advance the pose estimate from new sensor readings.

        This is called every control step with the CURRENT encoder and
        IMU readings.

        Args:
            enc (dict): Latest encoder readings (robot.read_encoders()).
            yaw (float): Latest IMU heading (robot.read_yaw()) in radians.

        Returns:
            (x, y, theta) -- the new estimated pose.
        """
        # If called before initialise(), auto-initialise on first call.
        if not self._initialised:
            self.initialise(enc, yaw)
            return self.pose()

        # --- Step 1: compute distance each side travelled this step --------
        left, right = self._wheel_means(enc)
        # Delta angle (rad) since last step, converted to linear distance (m).
        d_left  = (left  - self._prev_left)  * C.WHEEL_RADIUS
        d_right = (right - self._prev_right) * C.WHEEL_RADIUS
        # Save current readings for next step.
        self._prev_left, self._prev_right = left, right

        # --- Step 2: net forward distance (average of both sides) ----------
        d_centre = 0.5 * (d_left + d_right)

        # --- Step 3: heading from IMU (trust it fully over encoder-derived) -
        new_theta = yaw

        # --- Step 4: integrate position using the AVERAGE heading ----------
        # Using the midpoint heading (average of old and new) gives slightly
        # better accuracy than using either endpoint alone (trapezoidal rule).
        mid_theta = math.atan2(
            math.sin(self.theta) + math.sin(new_theta),
            math.cos(self.theta) + math.cos(new_theta),
        )
        self.x     += d_centre * math.cos(mid_theta)
        self.y     += d_centre * math.sin(mid_theta)
        self.theta  = new_theta

        return self.pose()

    # ---------------------------------------------------------------------- #
    def pose(self):
        """Return the current pose as a tuple (x, y, theta)."""
        return self.x, self.y, self.theta
