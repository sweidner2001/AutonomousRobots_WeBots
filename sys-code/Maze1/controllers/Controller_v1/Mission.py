class Mission:
    """Named constants for the mission state machine.

    The robot progresses through these states in strict order:
        SEARCH_BLUE  : Explore the maze while scanning camera for blue.
        GO_BLUE      : Blue detected and reachable — navigate to it.
        SEARCH_YELLOW: Blue reached. Explore while scanning for yellow.
        GO_YELLOW    : Yellow detected and reachable — navigate to it.
        DONE         : Yellow reached. Stop motors, mission complete.
    """
    SEARCH_BLUE = "SEARCH_BLUE"
    GO_BLUE = "GO_BLUE"
    SEARCH_YELLOW = "SEARCH_YELLOW"
    GO_YELLOW = "GO_YELLOW"
    DONE = "DONE"