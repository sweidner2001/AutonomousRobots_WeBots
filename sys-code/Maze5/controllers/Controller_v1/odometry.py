"""
odometry.py
===========
Pose estimation that fuses two sources:

  * POSITION  comes from the wheel encoders (distance travelled).
  * HEADING   comes from the IMU (InertialUnit yaw).

Why split them?  The RosBot is a 4-wheel skid-steer platform.  Its wheels slip
heavily during turns, so encoder-derived heading drifts badly.  The IMU gives a
clean yaw, so we trust it for orientation and only use the encoders for how far
the robot rolled forward.  This keeps the base map sharp without any SLAM, and
leaves a clean seam to drop in Graph-SLAM later (just replace this class).
"""

import math

import Maze5.controllers.Controller_v1.config as C


class Odometry:
    def __init__(self, start_x=0.0, start_y=0.0):
        self.x = start_x
        self.y = start_y
        self.theta = 0.0
        self._prev_left = None
        self._prev_right = None
        self._initialised = False

    def initialise(self, enc, yaw):
        """Seed encoder baseline and initial heading (call once devices are up)."""
        left, right = self._wheel_means(enc)
        self._prev_left = left
        self._prev_right = right
        self.theta = yaw
        self._initialised = True

    @staticmethod
    def _wheel_means(enc):
        """enc: dict of the four encoder readings (rad)."""
        left = 0.5 * (enc["fl"] + enc["rl"])
        right = 0.5 * (enc["fr"] + enc["rr"])
        return left, right

    def update(self, enc, yaw):
        """
        enc : dict {'fl','fr','rl','rr'} of position-sensor values (rad)
        yaw : IMU yaw (rad)
        Returns the updated (x, y, theta).
        """
        if not self._initialised:
            self.initialise(enc, yaw)
            return self.x, self.y, self.theta

        left, right = self._wheel_means(enc)
        d_left = (left - self._prev_left) * C.WHEEL_RADIUS
        d_right = (right - self._prev_right) * C.WHEEL_RADIUS
        self._prev_left = left
        self._prev_right = right

        d_center = 0.5 * (d_left + d_right)

        # Heading: trust the IMU absolutely.
        new_theta = yaw
        # Integrate position using the average heading over the step
        # (midpoint rule reduces error on curved motion).
        mid_theta = math.atan2(
            math.sin(self.theta) + math.sin(new_theta),
            math.cos(self.theta) + math.cos(new_theta),
        )
        self.x += d_center * math.cos(mid_theta)
        self.y += d_center * math.sin(mid_theta)
        self.theta = new_theta
        return self.x, self.y, self.theta

    def pose(self):
        return self.x, self.y, self.theta
