"""
teleop.py
=========
Manual keyboard driving (tele-operation).

While SLAM builds the map, YOU steer the robot.  This keeps the first phase
focused on the mapping core: the quality of the map depends only on the SLAM
code, not on any autonomous-driving logic.

Webots keyboard input is read through the normal Robot API (``getKeyboard``),
so this uses NO supervisor.  IMPORTANT: the Webots 3-D view must have focus
(click it once) for key presses to reach the controller.

CONTROLS
--------
    Up    / W : forward            Down  / S : backward
    Left  / A : turn left          Right / D : turn right
    Space     : stop
    M         : save the map now

Several keys work together, so "Up + Left" drives a left-hand arc.
"""

from controller import Keyboard

import Maze5.controllers.Controller_v1.config as C


class Teleop:
    """Translates the set of pressed keys into a ``(v, w)`` drive command."""

    def __init__(self, linear=None, angular=None):
        self.linear = C.TELEOP_LINEAR if linear is None else linear
        self.angular = C.TELEOP_ANGULAR if angular is None else angular
        self._print_help()

    @staticmethod
    def _print_help():
        print("[teleop] click the 3-D view, then drive:")
        print("[teleop]   Up/W forward  Down/S back  Left/A + Right/D turn")
        print("[teleop]   Space stop    M save map")

    # ---------------------------------------------------------------------- #
    def command(self, keys):
        """Map a set of key codes to ``(v, w, save_requested)``.

        Args:
            keys: set of key codes currently held (from ``Robot.read_keys()``).

        Returns:
            (v, w, save_requested):
                v  -- forward speed (m/s)
                w  -- turn rate (rad/s, positive = left)
                save_requested -- True if the save key (M) is held.
        """
        v = 0.0
        w = 0.0

        if _any(keys, Keyboard.UP, ord("W")):
            v += self.linear
        if _any(keys, Keyboard.DOWN, ord("S")):
            v -= self.linear
        if _any(keys, Keyboard.LEFT, ord("A")):
            w += self.angular
        if _any(keys, Keyboard.RIGHT, ord("D")):
            w -= self.angular

        if ord(" ") in keys:        # explicit stop overrides everything
            v = 0.0
            w = 0.0

        save_requested = ord("M") in keys
        return v, w, save_requested


def _any(keys, *codes):
    """True if any of ``codes`` is in the pressed-keys set."""
    return any(c in keys for c in codes)
