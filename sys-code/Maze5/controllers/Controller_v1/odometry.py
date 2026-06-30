"""
odometry.py
===========
Wheel + IMU odometry -- the SLAM system's MOTION PRIOR.

WHAT IT DOES
------------
Estimates where the robot is by accumulating its small movements over time
(dead reckoning).  It produces a ``Pose2D`` in the robot-start frame.

WHY FUSE ENCODERS WITH THE IMU?
-------------------------------
The RosBot 2 is SKID-STEER: it turns by spinning its left and right wheels at
different speeds and letting the wheels SKID sideways.  That skidding makes a
heading estimated from wheel-speed differences drift badly.

So we split the job by trusting each sensor where it is good:
  * FORWARD DISTANCE comes from the wheel encoders (wheels roll cleanly
    forward, so distance is reliable).
  * HEADING comes straight from the IMU (it measures real rotation, immune to
    wheel slip).

IMPORTANT: this is only the PRIOR.  It still drifts over time.  The SLAM
front-end (lidar scan matching) and the pose-graph back-end (loop closure)
are what actually correct that drift -- odometry just gives them a good
starting guess each step.

START FRAME
-----------
We never use the supervisor, so we cannot know the true world pose.  We
define the robot's start as the origin: position (0, 0) and heading 0.  The
IMU's first yaw reading is stored as a reference and subtracted from every
later reading, so ``theta`` is measured relative to "where the robot first
faced".
"""

import math

import Maze5.controllers.Controller_v1.config as C
from Maze5.controllers.Controller_v1.slam.geometry import Pose2D, wrap_angle


class Odometry:
    """Tracks the robot pose from encoder distance + IMU heading."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self._yaw0 = 0.0          # IMU yaw at start-up (our heading reference)
        self._prev_left = None    # previous averaged left-wheel angle (rad)
        self._prev_right = None   # previous averaged right-wheel angle (rad)
        self._initialised = False

    # ---------------------------------------------------------------------- #
    def initialise(self, enc, yaw):
        """Seed the encoder baseline and the heading reference.

        Call once, after the first valid sensor readings (Webots sensors only
        return real data after the first ``robot.step()``).
        """
        self._prev_left, self._prev_right = self._wheel_means(enc)
        self._yaw0 = yaw
        self.x = self.y = self.theta = 0.0
        self._initialised = True

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _wheel_means(enc):
        """Average the two encoders on each side into one value per side."""
        left = 0.5 * (enc["fl"] + enc["rl"])
        right = 0.5 * (enc["fr"] + enc["rr"])
        return left, right

    # ---------------------------------------------------------------------- #
    def update(self, enc, yaw):
        """Advance the pose from new encoder + IMU readings; return a Pose2D."""
        if not self._initialised:
            self.initialise(enc, yaw)
            return self.pose()

        # Distance each side rolled since last step (encoder delta -> metres).
        left, right = self._wheel_means(enc)
        d_left = (left - self._prev_left) * C.WHEEL_RADIUS
        d_right = (right - self._prev_right) * C.WHEEL_RADIUS
        self._prev_left, self._prev_right = left, right

        # Net forward distance is the average of the two sides.
        d_centre = 0.5 * (d_left + d_right)

        # Heading comes straight from the IMU, relative to the start heading.
        new_theta = wrap_angle(yaw - self._yaw0)

        # Integrate position using the heading AVERAGED over the step
        # (trapezoidal rule -- a touch more accurate than using either end).
        mid = math.atan2(math.sin(self.theta) + math.sin(new_theta),
                         math.cos(self.theta) + math.cos(new_theta))
        self.x += d_centre * math.cos(mid)
        self.y += d_centre * math.sin(mid)
        self.theta = new_theta

        return self.pose()

    # ---------------------------------------------------------------------- #
    def pose(self):
        """Current odometry estimate as a Pose2D."""
        return Pose2D(self.x, self.y, self.theta)
