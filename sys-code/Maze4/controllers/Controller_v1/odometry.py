"""
odometry.py
===========
Pose estimation that fuses two sources:

  * POSITION  comes from the wheel encoders (distance travelled).
  * HEADING   comes from the IMU (InertialUnit yaw).

The RosBot is a 4-wheel skid-steer platform whose wheels slip heavily in turns,
so encoder-derived heading drifts badly.  The IMU yaw is clean, so we trust it
for orientation and use the encoders only for travelled distance.

The robot's true world pose is unknown, so we start at the map centre (0, 0).
The starting heading is taken from the IMU at initialise().  Replacing this
class with a SLAM front-end is all that's needed for the next stage.
"""

import math

import Maze5.controllers.Controller_v1.config as C


class Odometry:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._prev_left = None
        self._prev_right = None
        self._initialised = False

    def initialise(self, enc, yaw):
        """Seed the encoder baseline and starting heading (call once)."""
        self._prev_left, self._prev_right = self._wheel_means(enc)
        self.theta = yaw
        self._initialised = True

    @staticmethod
    def _wheel_means(enc):
        left = 0.5 * (enc["fl"] + enc["rl"])
        right = 0.5 * (enc["fr"] + enc["rr"])
        return left, right

    def update(self, enc, yaw):
        """Advance the pose from new encoder values and IMU yaw."""
        if not self._initialised:
            self.initialise(enc, yaw)
            return self.pose()

        left, right = self._wheel_means(enc)
        d_left = (left - self._prev_left) * C.WHEEL_RADIUS
        d_right = (right - self._prev_right) * C.WHEEL_RADIUS
        self._prev_left, self._prev_right = left, right
        d_center = 0.5 * (d_left + d_right)

        new_theta = yaw  # trust the IMU absolutely
        # Integrate position with the average heading over the step.
        mid_theta = math.atan2(
            math.sin(self.theta) + math.sin(new_theta),
            math.cos(self.theta) + math.cos(new_theta),
        )
        self.x += d_center * math.cos(mid_theta)
        self.y += d_center * math.sin(mid_theta)
        self.theta = new_theta
        return self.pose()

    def pose(self):
        return self.x, self.y, self.theta
