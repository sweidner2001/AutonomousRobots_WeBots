"""
pilot.py  --  Path follower and reactive obstacle avoidance.
=============================================================

WHAT DOES THIS FILE DO?
------------------------
The Pilot converts an abstract planned path (a list of waypoints) into
actual motor commands (forward speed v, angular speed w) every step.

TWO LAYERS — in priority order:
  1. REACTIVE SAFETY (highest priority)
     Watches the lidar directly, every step.
     If a wall is dangerously close AHEAD, it slows or stops and
     rotates away — regardless of what the path says.
     This is the last line of defence against collisions.

  2. PURE PURSUIT PATH FOLLOWER
     Steers the robot along the planned waypoint path.
     Works when reactive safety is not triggered.

WHY PURE PURSUIT?
------------------
Pure pursuit is a classic path-following algorithm (Coulter, 1992).
Instead of trying to reach the next waypoint exactly, we pick a
"look-ahead point" some distance L ahead on the path and steer
directly toward it.

Benefits:
  - Smooth steering (doesn't overshoot sharp waypoints)
  - Speed naturally scales with heading error (robot slows in turns)
  - Simple to implement and tune (only one parameter: LOOKAHEAD)

HOW IT WORKS STEP BY STEP:
  1. Advance past any waypoints already within WAYPOINT_TOL of the robot.
  2. Walk forward along the remaining path until at least LOOKAHEAD metres
     away from the robot -> that point is the look-ahead target.
  3. Compute the desired heading to the look-ahead target.
  4. Compute heading error = desired - current.
  5. Angular speed  w = HEADING_KP * heading_error   (P-controller)
  6. Forward speed  v = CRUISE_SPEED * max(0, 1 - |error| / 1.2)
     This makes v go to zero when the heading error is large (≥ 1.2 rad
     ≈ 69°), so the robot turns in place when badly misaligned.

REACTIVE SAFETY LOGIC:
  - Measure the minimum lidar distance in the FRONT sector (±FRONT_SECTOR rad).
  - If this distance is below SAFE_FRONT_DIST: turn in place away from the
    nearest obstacle.
  - If it is between SAFE_FRONT_DIST and SLOW_FRONT_DIST: reduce speed
    proportionally (slow zone), still let pure pursuit steer.
  - Otherwise: let pure pursuit run at full CRUISE_SPEED.
"""

import math

import numpy as np

import Maze4.controllers.Controller_v1.config as C


def _ang_diff(a, b):
    """Smallest signed angle difference (a - b) wrapped to [-π, π].

    Normal subtraction can give values outside [-π, π].
    Using atan2(sin, cos) correctly wraps the result.

    Example: a=0.1, b=6.0  ->  0.1 - 6.0 = -5.9  (wrong, > π)
             _ang_diff gives: ≈ 0.38  (correct, going the short way round)
    """
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


class Pilot:
    """Pure-pursuit path follower with reactive lidar-based safety."""

    def __init__(self):
        self.path = []   # list of (x, y) world-frame waypoints
        self.idx  = 0    # index of the first waypoint we haven't passed yet

    def set_path(self, world_path):
        """Load a new path and reset the waypoint index."""
        self.path = list(world_path)
        self.idx  = 0

    def clear(self):
        """Discard the current path (e.g. before replanning)."""
        self.path = []
        self.idx  = 0

    def has_path(self):
        """True if there are still unvisited waypoints in the path."""
        return self.idx < len(self.path)

    # ---------------------------------------------------------------------- #
    # Pure pursuit internals
    # ---------------------------------------------------------------------- #
    def _lookahead_point(self, x, y):
        """Return the look-ahead target point for pure pursuit.

        Algorithm:
          Step A: Skip over any waypoints that are already within
                  WAYPOINT_TOL of the robot (we've passed them).
          Step B: Walk forward through the remaining waypoints until
                  we find one at least LOOKAHEAD metres away.
                  That is our look-ahead target.

        Returns (wx, wy) of the look-ahead point, or None if the path
        is exhausted (done).
        """
        # Step A: advance past nearby waypoints.
        while self.idx < len(self.path):
            wx, wy = self.path[self.idx]
            if math.hypot(wx - x, wy - y) < C.WAYPOINT_TOL:
                self.idx += 1   # this waypoint is close enough — skip it
            else:
                break

        if self.idx >= len(self.path):
            return None  # all waypoints done

        # Step B: find the look-ahead point at least LOOKAHEAD metres away.
        j = self.idx
        while j < len(self.path) - 1:
            wx, wy = self.path[j]
            if math.hypot(wx - x, wy - y) >= C.LOOKAHEAD:
                break   # found a point far enough ahead
            j += 1
        return self.path[j]

    # ---------------------------------------------------------------------- #
    # Main output function
    # ---------------------------------------------------------------------- #
    def compute(self, pose, scan_ranges, scan_bearings):
        """Compute wheel commands for the current control step.

        Args:
            pose          : (x, y, theta) from odometry.
            scan_ranges   : lidar range array (m), from robot.read_lidar().
            scan_bearings : per-ray bearing array (rad), robot.bearings.

        Returns:
            (v, w, done)
              v    : forward speed  (m/s)
              w    : angular speed  (rad/s)
              done : True when the path has been fully traversed.
        """
        x, y, theta = pose

        # ---- Layer 1: reactive safety (highest priority) ------------------
        evade = self._reactive(scan_ranges, scan_bearings)
        if evade is not None:
            # An imminent collision is detected; use the escape command.
            return evade[0], evade[1], False

        # ---- Layer 2: pure pursuit ----------------------------------------
        if not self.has_path():
            return 0.0, 0.0, True   # no path -> path traversal complete

        target = self._lookahead_point(x, y)
        if target is None:
            return 0.0, 0.0, True   # all waypoints visited -> done

        tx, ty = target
        desired_heading = math.atan2(ty - y, tx - x)
        heading_error   = _ang_diff(desired_heading, theta)

        # Angular speed: proportional controller on heading error.
        w = max(-C.MAX_TURN_SPEED,
                min(C.MAX_TURN_SPEED, C.HEADING_KP * heading_error))

        # Forward speed: scale down when heading error is large.
        # align = 1.0 when perfectly aligned, 0.0 when error >= 1.2 rad.
        align = max(0.0, 1.0 - abs(heading_error) / 1.2)
        v = C.CRUISE_SPEED * align

        # ---- Speed reduction zone: wall ahead but not yet dangerous -------
        v = v * self._front_speed_factor(scan_ranges, scan_bearings)

        return v, w, False

    # ---------------------------------------------------------------------- #
    # Reactive safety helpers
    # ---------------------------------------------------------------------- #
    def _reactive(self, ranges, bearings):
        """Return an (v, w) escape command if a collision is imminent.

        Only activates when the nearest obstacle in the FRONT sector is
        closer than SAFE_FRONT_DIST.  Returns None otherwise (meaning
        "pure pursuit is in charge").

        Turn direction: rotate TOWARD the more open side (away from
        the nearest obstacle).
          bearings > 0  = left  side  -> turn w > 0 (counterclockwise = left)
          bearings < 0  = right side  -> turn w < 0 (clockwise = right)
        """
        if ranges is None or len(ranges) == 0:
            return None

        finite = np.isfinite(ranges)
        # Only look at rays within the front sector (±FRONT_SECTOR).
        front = (np.abs(bearings) < C.FRONT_SECTOR) & finite
        if not np.any(front):
            return None

        min_dist = ranges[front].min()
        if min_dist >= C.SAFE_FRONT_DIST:
            return None  # safe; let pure pursuit handle everything

        # Too close — find which side is more open and turn that way.
        left  = front & (bearings > 0)   # rays toward the left
        right = front & (bearings < 0)   # rays toward the right
        left_min  = ranges[left].min()  if np.any(left)  else math.inf
        right_min = ranges[right].min() if np.any(right) else math.inf

        # Turn toward the more open side (larger minimum distance).
        turn_sign = +1.0 if left_min > right_min else -1.0
        return (0.0, turn_sign * C.MAX_TURN_SPEED)

    def _front_speed_factor(self, ranges, bearings):
        """Return a [0, 1] speed multiplier based on the closest front obstacle.

        This creates a smooth SLOW ZONE between SLOW_FRONT_DIST and
        SAFE_FRONT_DIST.  Inside that zone the robot slows down
        proportionally so it doesn't crash into the safety boundary
        at full speed.

        Timeline as the robot approaches a wall:
          dist > SLOW_FRONT_DIST  ->  factor = 1.0  (full cruise speed)
          dist = mid-point        ->  factor ≈ 0.5  (half speed)
          dist = SAFE_FRONT_DIST  ->  factor ≈ 0.0  (already handled by reactive)
        """
        if ranges is None or len(ranges) == 0:
            return 1.0

        finite = np.isfinite(ranges)
        front  = (np.abs(bearings) < C.FRONT_SECTOR) & finite
        if not np.any(front):
            return 1.0

        min_dist = ranges[front].min()

        if min_dist >= C.SLOW_FRONT_DIST:
            return 1.0   # far from any wall: full speed

        if min_dist < C.SAFE_FRONT_DIST:
            return 0.0   # in danger zone: already handled by _reactive

        # Interpolate linearly between 0 and 1 in the slow zone.
        # t = 1.0 at SLOW_FRONT_DIST (far end), t = 0.0 at SAFE_FRONT_DIST.
        t = ((min_dist - C.SAFE_FRONT_DIST)
             / (C.SLOW_FRONT_DIST - C.SAFE_FRONT_DIST))
        return max(0.2, t)   # always keep at least 20% speed to avoid stalling
