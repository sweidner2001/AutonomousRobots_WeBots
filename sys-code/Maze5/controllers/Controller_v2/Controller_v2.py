"""
Controller_v1.py  --  Webots entry point for the RosBot maze explorer.
=======================================================================

THIS IS THE FILE WEBOTS LAUNCHES.

HOW WEBOTS FINDS AND RUNS A CONTROLLER
-----------------------------------------
In the Webots world editor:
  Scene tree -> Rosbot node -> controller field -> "Controller_v1"

Webots looks for a Python file named "Controller_v1.py" in the
controllers/Controller_v1/ folder and runs it as a Python script.

WHY IS THIS FILE SO SHORT?
----------------------------
All the interesting behaviour lives in separate, well-named classes.
This file is intentionally minimal: it just adds the project root to
Python's import path and calls MazeExplorer().run().

This is a common software engineering pattern:  keep the "entry point"
as simple as possible and delegate all logic to well-tested classes.

THE MODULE LAYOUT
------------------
    robot.py            -- Hardware abstraction (motors, sensors)
    odometry.py         -- Pose estimation from encoders + IMU
    occupancy_grid.py   -- Probabilistic map (log-odds grid)
    frontier.py         -- Frontier detection (explore edges)
    planner.py          -- Obstacle inflation + A* path planning
    pilot.py            -- Pure-pursuit path following + reactive safety
    mapviz.py           -- Live matplotlib visualisation
    mission.py          -- High-level mission state constants
    explorer.py (Maze5) -- Exploration FSM + top-level orchestrator

HOW PYTHON IMPORTS WORK HERE
------------------------------
Webots launches this file from inside the controllers/ folder.
The "Maze4" and "Maze5" packages must be importable, so we need their
parent directory (the repository root) to be in sys.path.

We compute the repository root as:
  __file__        = .../sys-code/Maze4/controllers/Controller_v1/Controller_v1.py
  go up 3 levels  = .../sys-code/
  -> that is the folder that contains both "Maze4" and "Maze5"

We insert that path into sys.path[0] so Python finds the packages.
"""

import os
import sys



# The orchestrator (MazeExplorer) lives in Maze5's explorer module.
# It imports all implementation modules from Maze4.
from Maze4.controllers.Controller_v1.explorer import MazeExplorer

class DebugHelper:
    """
    Helper class for debugging and visualizing the maze exploration process.

    This class provides methods to visualize the navigable and unknown areas
    of the maze using matplotlib. It can be used to debug the exploration
    algorithm by showing the current state of the maze.
    """

    @staticmethod
    def debug_show(navigable, unknown):
        """
        Display the navigable and unknown areas of the maze.

        Parameters:
        - navigable: A 2D numpy array representing the navigable cells (True for navigable, False otherwise).
        - unknown: A 2D numpy array representing the unknown cells (True for unknown, False otherwise).
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, data, title in zip(axes, [navigable, unknown], ["navigable", "unknown"]):
            ax.imshow(data, cmap="gray", origin="lower", interpolation="nearest")
            ax.set_title(title)
        plt.tight_layout()
        plt.pause(0.001)


def main():
    """Create the explorer and run the simulation loop."""
    MazeExplorer().run()


if __name__ == "__main__":
    main()
