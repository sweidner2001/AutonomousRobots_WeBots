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

        Step A: Skip waypoints already within WAYPOINT_TOL of the robot.
        Step B: Walk forward until a point at least LOOKAHEAD metres away
                is found — that is the steering target.

        The lookahead distance is intentionally FIXED here.  Speed at corners
        is handled separately by _corner_speed_factor(), which slows the robot
        down BEFORE a sharp bend so it arrives there at a safe speed.

        Returns (wx, wy) of the look-ahead point, or None if exhausted.
        """
        # Step A: advance past nearby waypoints.
        while self.idx < len(self.path):
            wx, wy = self.path[self.idx]
            if math.hypot(wx - x, wy - y) < C.WAYPOINT_TOL:
                self.idx += 1
            else:
                break

        if self.idx >= len(self.path):
            return None  # all waypoints done

        # Step B: find the look-ahead point at the standard distance.
        j = self.idx
        while j < len(self.path) - 1:
            wx, wy = self.path[j]
            if math.hypot(wx - x, wy - y) >= C.LOOKAHEAD:
                break
            j += 1
        return self.path[j]

    def _corner_speed_factor(self, x, y):
        """Return a [0.4, 1.0] speed multiplier based on upcoming path curvature.

        WHY THIS — NOT AN ADAPTIVE LOOKAHEAD?
        ---------------------------------------
        A LARGER lookahead at corners does NOT help — it makes things worse.
        With a large lookahead the target point jumps to the far side of the
        corner.  The robot then tries to steer diagonally straight toward it,
        cutting directly through the inner corner wall.

        The correct fix is to slow down BEFORE the corner so the robot
        arrives there at a speed it can actually handle.  The existing
        heading-error term (`align`) already reduces speed once the robot
        IS turning hard.  This method adds predictive braking: it looks
        ahead along the path, finds the sharpest upcoming bend within
        CORNER_SCAN_STEPS waypoints, and reduces speed proportionally —
        BEFORE the turn begins.

        Example — 90° corner 0.3 m ahead:
          _corner_speed_factor returns ~0.4
          → robot slows to 40 % cruise before reaching the bend
          → arrives at low speed, turns cleanly, no wall contact

        Returns:
            float in [CORNER_MIN_SPEED_FACTOR, 1.0]
        """
        n = len(self.path)
        if n - self.idx < 3:
            return 1.0

        max_turn = 0.0
        scan_end = min(self.idx + C.CORNER_SCAN_STEPS, n - 1)

        for k in range(self.idx + 1, scan_end):
            wx, wy = self.path[k]
            # Only care about corners within a reasonable look-ahead window.
            if math.hypot(wx - x, wy - y) > C.LOOKAHEAD * 3.0:
                break
            if k < 1 or k >= n - 1:
                continue
            ax, ay = self.path[k - 1]
            bx, by = self.path[k]
            cx, cy = self.path[k + 1]
            in_dx,  in_dy  = bx - ax,  by - ay
            out_dx, out_dy = cx - bx,  cy - by
            in_len  = math.hypot(in_dx,  in_dy)
            out_len = math.hypot(out_dx, out_dy)
            if in_len < 1e-6 or out_len < 1e-6:
                continue
            cos_t    = (in_dx * out_dx + in_dy * out_dy) / (in_len * out_len)
            turn_rad = math.acos(max(-1.0, min(1.0, cos_t)))
            if turn_rad > max_turn:
                max_turn = turn_rad

        if max_turn < C.CORNER_ANGLE_THRESHOLD:
            return 1.0   # no significant corner ahead — full speed

        # Linear ramp from 1.0 at threshold down to CORNER_MIN_SPEED_FACTOR
        # at a 90° (π/2) corner.
        t = min(1.0, (max_turn - C.CORNER_ANGLE_THRESHOLD)
                     / (math.pi / 2.0 - C.CORNER_ANGLE_THRESHOLD + 1e-6))
        return max(C.CORNER_MIN_SPEED_FACTOR, 1.0 - (1.0 - C.CORNER_MIN_SPEED_FACTOR) * t)

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

        # Predictive corner braking: slow down BEFORE a sharp bend is reached.
        # v = v * self._corner_speed_factor(x, y)

        # Reactive wall braking: slow down when a wall is close ahead.
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
