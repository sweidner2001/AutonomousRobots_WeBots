"""
pilot.py
========
Turns a planned path into wheel commands.

Two layers:
  1. PURE PURSUIT path follower -- picks a look-ahead point on the path and
     steers toward it, scaling forward speed down while turning.
  2. REACTIVE SAFETY -- independent of the planner.  If the lidar sees a wall
     too close ahead, it overrides the follower (stop & rotate away).  This
     guards against collisions in the gaps between replans.

The pilot outputs (v, w) = (forward m/s, angular rad/s); the main controller
converts that to left/right wheel speeds for the skid-steer drive.
"""

import math

import numpy as np

import Maze5.controllers.Controller_v1.config as C


def _ang_diff(a, b):
    """Smallest signed difference a-b wrapped to [-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


class Pilot:
    def __init__(self):
        self.path = []          # list of (x, y) world waypoints
        self.idx = 0            # index of current target waypoint

    def set_path(self, world_path):
        self.path = list(world_path)
        self.idx = 0

    def clear(self):
        self.path = []
        self.idx = 0

    def has_path(self):
        return self.idx < len(self.path)

    # ------------------------------------------------------------------ #
    def _lookahead_point(self, x, y):
        """Advance past reached waypoints, return the look-ahead target."""
        # Drop waypoints we are already on top of.
        while self.idx < len(self.path):
            wx, wy = self.path[self.idx]
            if math.hypot(wx - x, wy - y) < C.WAYPOINT_TOL:
                self.idx += 1
            else:
                break
        if self.idx >= len(self.path):
            return None
        # Walk forward until we are at least LOOKAHEAD away.
        j = self.idx
        while j < len(self.path) - 1:
            wx, wy = self.path[j]
            if math.hypot(wx - x, wy - y) >= C.LOOKAHEAD:
                break
            j += 1
        return self.path[j]

    # ------------------------------------------------------------------ #
    def compute(self, pose, scan_ranges, scan_bearings):
        """
        Returns (v, w, done).
          done == True  -> the path has been fully traversed.
        """
        x, y, theta = pose

        # ---- reactive safety first -------------------------------------
        evade = self._reactive(scan_ranges, scan_bearings)
        if evade is not None:
            return evade[0], evade[1], False

        # ---- pure pursuit ----------------------------------------------
        if not self.has_path():
            return 0.0, 0.0, True
        target = self._lookahead_point(x, y)
        if target is None:
            return 0.0, 0.0, True

        tx, ty = target
        desired = math.atan2(ty - y, tx - x)
        err = _ang_diff(desired, theta)

        w = max(-C.MAX_TURN_SPEED, min(C.MAX_TURN_SPEED, C.HEADING_KP * err))
        # Slow forward speed as heading error grows; basically turn-in-place
        # when badly misaligned.
        align = max(0.0, 1.0 - abs(err) / 1.2)
        v = C.CRUISE_SPEED * align
        return v, w, False

    # ------------------------------------------------------------------ #
    def _reactive(self, ranges, bearings):
        """
        If a wall is closer than SAFE_FRONT_DIST within the front sector,
        return (v, w) that rotates away from the nearer side.  Else None.
        """
        if ranges is None or len(ranges) == 0:
            return None
        finite = np.isfinite(ranges)
        front = (np.abs(bearings) < C.FRONT_SECTOR) & finite
        if not np.any(front):
            return None
        front_ranges = ranges[front]
        if front_ranges.min() >= C.SAFE_FRONT_DIST:
            return None

        # Decide turn direction: compare nearest obstacle on left vs right.
        left = front & (bearings > 0)
        right = front & (bearings < 0)
        left_min = ranges[left].min() if np.any(left) else math.inf
        right_min = ranges[right].min() if np.any(right) else math.inf
        # Turn toward the more open side.
        turn = +1.0 if left_min > right_min else -1.0
        return (0.0, turn * C.MAX_TURN_SPEED)
