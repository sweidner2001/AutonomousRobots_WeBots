"""
mission.py  --  High-level mission state constants.
====================================================

WHAT IS A STATE MACHINE?
--------------------------
A state machine (or finite-state machine, FSM) models a system that can
be in exactly ONE "state" at a time and switches between states when
certain events occur.

For our robot the top-level states are the PHASES of its mission:

  ┌─────────────┐     map complete      ┌──────────────┐
  │ EXPLORE_MAP │ ─────────────────────>│  DONE        │
  └─────────────┘                       └──────────────┘
        │           (if color enabled)        ^
        │        ┌──────────────┐             │
        └──────> │ SEARCH_BLUE  │─── found -->│
                 └──────────────┘             │
                        │                     │
                        v                     │
                 ┌──────────────┐             │
                 │  GO_BLUE     │             │
                 └──────────────┘             │
                        │                     │
                        v                     │
                 ┌──────────────┐             │
                 │ SEARCH_YELLOW│─── found -->│
                 └──────────────┘             │
                        │                     │
                        v                     │
                 ┌──────────────┐             │
                 │  GO_YELLOW   │─────────────┘
                 └──────────────┘

Only EXPLORE_MAP and DONE are implemented in this project.
The colour-search states are placeholders guarded by MISSION_ENABLE_COLOR.
"""


class Mission:
    """String constants for the mission state machine.

    Using named string constants instead of plain strings prevents typos
    and makes the code easier to read:
        if mission == Mission.DONE:   <-- clear intent
        if mission == "done":         <-- typo-prone
    """

    EXPLORE_MAP   = "EXPLORE_MAP"   # Drive frontier exploration until fully mapped.
    SEARCH_BLUE   = "SEARCH_BLUE"   # Explore while looking for a blue object.
    GO_BLUE       = "GO_BLUE"       # Blue found -- navigate to it.
    SEARCH_YELLOW = "SEARCH_YELLOW" # Blue reached -- now look for yellow.
    GO_YELLOW     = "GO_YELLOW"     # Yellow found -- navigate to it.
    DONE          = "DONE"          # Mission complete -- stop all motors.

    # One-shot last resort: SEARCH_BLUE/SEARCH_YELLOW think the map is fully
    # explored (no frontiers) without ever finding the target.  Before really
    # giving up, drive toward a frontier that only appears once the
    # inflation margin is shrunk by a cell -- see MazeExplorer._act_search()
    # / _act_recheck_frontier().  Afterward, control returns to whichever
    # SEARCH_* state entered this one.
    RECHECK_FRONTIER = "RECHECK_FRONTIER"
