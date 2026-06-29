"""
Controller_v1.py  --  Webots entry point for the RosBot maze explorer.
======================================================================

This file is intentionally tiny.  All behaviour lives in classes:

    Robot        (robot.py)            -- hardware abstraction
    OccupancyGrid(occupancy_grid.py)   -- log-odds mapping
    Odometry     (odometry.py)         -- pose from encoders + IMU
    FrontierDetector (frontier.py)     -- frontier detection
    PathPlanner  (planner.py)          -- inflation + A* + target choice
    Pilot        (pilot.py)            -- path following + reactive safety
    MapViz       (mapviz.py)           -- live matplotlib map
    Mission      (mission.py)          -- mission state constants
    Explorer / MazeExplorer (explorer.py) -- exploration FSM + orchestrator

Assign this controller to the Rosbot in Webots
(Scene tree -> Rosbot -> controller -> "Controller_v1").
"""

import os
import sys

# The modules use absolute package imports (Maze5.controllers.Controller_v1.*),
# so the repository root (the parent of the "Maze5" folder) must be importable.
# Webots launches this file directly, so we add that root to sys.path here.
# _HERE = os.path.dirname(os.path.abspath(__file__))
# _REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
# if _REPO_ROOT not in sys.path:
#     sys.path.insert(0, _REPO_ROOT)

from Maze5.controllers.Controller_v1.explorer import MazeExplorer


def main():
    MazeExplorer().run()


if __name__ == "__main__":
    main()
