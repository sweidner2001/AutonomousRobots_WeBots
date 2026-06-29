# RosBot 2 — Maze Exploration (class-based design)

Autonomous frontier exploration + live occupancy mapping for the RosBot 2 with
the RpLidar A2, in Webots, pure Python (no ROS). The pose source is isolated so
a Graph-SLAM layer can replace it later without touching mapping or control.

## How to run

1. Open `worlds/Maze5.wbt` in Webots (R2025a).
2. Select **Rosbot → `controller`** field and set it to **`Controller_v1`**.
3. Press **Play**. A matplotlib window shows the map being built:
   white = free, black = wall, grey = unknown, red = robot + heading arrow,
   blue dots = current LIDAR scan, green line = planned path, magenta star =
   the frontier being driven to.
4. When no reachable frontier is left, the robot stops and writes
   `map_final.png` and `map_final.npy` into this folder.

## Class design

```
MazeExplorer (explorer.py)            orchestrator; short perceive/act/render loop
 ├─ Robot           (robot.py)        hardware: devices, read_lidar, set_velocity
 ├─ Odometry        (odometry.py)     pose from wheel distance + IMU heading
 ├─ OccupancyGrid   (occupancy_grid.py) log-odds map, Bresenham ray tracing
 ├─ FrontierDetector(frontier.py)     free/unknown boundary detection + clustering
 ├─ PathPlanner     (planner.py)      obstacle inflation, A*, frontier selection
 ├─ Pilot           (pilot.py)        pure-pursuit follow + reactive LIDAR safety
 ├─ MapViz          (mapviz.py)       live matplotlib map (+ PNG fallback)
 ├─ Explorer        (explorer.py)     exploration FSM: SPIN_SEED→PLAN→DRIVE→DONE
 └─ Mission         (mission.py)      mission state constants
```

`Controller_v1.py` is a tiny entry point: it puts the repo root on `sys.path`
(so the absolute package imports resolve when Webots launches the file directly)
and calls `MazeExplorer().run()`.

The main loop is intentionally three calls:

```python
while self.robot.step():
    self._perceive()   # read sensors, update pose, integrate scan
    self._act()        # mission state machine -> wheel command
    self._render()     # refresh the live map
```

## Map frame (robot-centric)

The robot's true world pose is unknown, so it **starts at the centre of the
grid map at (0, 0)**; the grid origin is placed at `(-W/2, -H/2)`. Everything —
distances, walls, the map — is measured relative to that start pose. Starting
heading is taken from the IMU. There is no `ROBOT_START_X/Y` to configure.

## Mission state machine

`Mission` defines `EXPLORE_MAP → SEARCH_BLUE → GO_BLUE → SEARCH_YELLOW →
GO_YELLOW → DONE`. Only **EXPLORE_MAP** (frontier exploration) and **DONE** are
implemented. The colour-search states are scaffolding: the Astra RGB-D camera
(`camera rgb`, `camera depth`) is initialised in `Robot` but not yet read.
`config.MISSION_ENABLE_COLOR` (default `False`) gates whether the mission
advances into them after the map is explored; for now it goes straight to DONE.

## Tuning notes
- All knobs live in `config.py`. Device names and the live LIDAR resolution /
  FoV / max-range live in the `Robot` class.
- **If the live map looks mirrored**, set `LIDAR_ANGLE_SIGN = -1.0`. If rotated,
  adjust `LIDAR_ANGLE_OFFSET`. (Webots' LIDAR index→angle convention is the one
  thing that can't be checked outside the simulator; everything else was
  validated headless against a synthetic maze.)

## Imports
Modules use absolute package imports
(`import Maze5.controllers.Controller_v1.config as C`). This requires the three
`__init__.py` files (present) and the repo root on `sys.path` — which
`Controller_v1.py` adds automatically at startup.

## Toward Graph-SLAM (next stage)
Replace `Odometry` with a SLAM front-end (scan-matching) + back-end (pose-graph
optimisation); grid/frontier/planner/pilot stay unchanged. `map_final.npy` (raw
log-odds) is saved each run for evaluation.

## Requirements
`numpy`, `matplotlib` (both ship with a standard Webots Python). The controller
degrades to PNG snapshots if no display is available.
