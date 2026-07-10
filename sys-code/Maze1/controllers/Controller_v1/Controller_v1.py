"""
Controller_v1.py  --  Webots entry point for the RosBot maze explorer (Maze1).
================================================================================

THIS IS THE FILE WEBOTS LAUNCHES.

HOW WEBOTS FINDS AND RUNS A CONTROLLER
-----------------------------------------
In the Webots world editor:
  Scene tree -> Rosbot node -> controller field -> "Controller_v1"

Webots looks for a Python file named "Controller_v1.py" in the
controllers/Controller_v1/ folder and runs it as a Python script.

WHY IS THIS FILE SO SHORT?
----------------------------
All the interesting behaviour lives in one SHARED codebase (Maze4's
controllers/Controller_v1/ package -- explorer.py, occupancy_grid.py,
planner.py, ...). Every maze (Maze1, Maze2, ...) reuses that exact same
code; this file's only two jobs are:
  1. Apply THIS maze's own config.py as an override on top of the shared
     defaults (see "PER-MAZE CONFIG" below) -- BEFORE anything else gets
     imported.
  2. Call MazeExplorer().run().

HOW PYTHON IMPORTS WORK HERE
------------------------------
Webots launches this file from inside the controllers/ folder. The
"Maze4" package must be importable, so we need its parent directory (the
repository root, "sys-code/") in sys.path. We compute that as:
  __file__        = .../sys-code/Maze1/controllers/Controller_v1/Controller_v1.py
  go up 3 levels  = .../sys-code/
  -> the folder that contains Maze1, Maze4, ...
and insert it into sys.path[0] so Python finds the shared package.

PER-MAZE CONFIG
------------------
config.py (right next to this file) contains ONLY the constants THIS
maze needs to differ from Maze4's defaults (e.g. a different
GRID_WIDTH_M/GRID_HEIGHT_M for a differently-sized maze layout) -- it
does not need to be a full copy of every constant. See
Maze4/controllers/Controller_v1/config.py's "PER-MAZE OVERRIDES" section
and load_and_apply_overrides() for the full mechanism: because Python
caches modules by name, patching attributes on the shared config module
HERE makes every other shared module's `import
Maze4.controllers.Controller_v1.config as C` see the patched values too,
with no changes needed to any of them.
"""

import os
import sys


# --- Apply THIS maze's own config.py as an override, BEFORE importing -------
# anything else -- explorer.py and everything it pulls in read config
# values the moment they're imported, so the override has to land first.
import Maze4.controllers.Controller_v1.config as C
_applied = C.load_and_apply_overrides(__file__)
print("[Controller_v1] Maze1 config.py override applied." if _applied
      else "[Controller_v1] no Maze1 config.py override found -- using Maze4 defaults.")

from Maze4.controllers.Controller_v1.explorer import MazeExplorer


def main():
    """Create the explorer and run the simulation loop."""
    MazeExplorer().run()


if __name__ == "__main__":
    main()
