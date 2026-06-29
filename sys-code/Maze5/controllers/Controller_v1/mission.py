"""
mission.py
==========
Named constants for the high-level mission state machine.
"""


class Mission:
    """Named constants for the mission state machine.

    The robot progresses through these states in order:
        EXPLORE_MAP  : Drive frontier-based exploration until the maze is mapped.
        SEARCH_BLUE  : Explore the maze while scanning the camera for blue.
        GO_BLUE      : Blue detected and reachable -- navigate to it.
        SEARCH_YELLOW: Blue reached. Explore while scanning for yellow.
        GO_YELLOW    : Yellow detected and reachable -- navigate to it.
        DONE         : Mission complete. Stop motors.

    Only EXPLORE_MAP and DONE are implemented so far; the colour states are
    scaffolding for the next stage (config.MISSION_ENABLE_COLOR gates them).
    """
    EXPLORE_MAP = "EXPLORE_MAP"
    SEARCH_BLUE = "SEARCH_BLUE"
    GO_BLUE = "GO_BLUE"
    SEARCH_YELLOW = "SEARCH_YELLOW"
    GO_YELLOW = "GO_YELLOW"
    DONE = "DONE"
