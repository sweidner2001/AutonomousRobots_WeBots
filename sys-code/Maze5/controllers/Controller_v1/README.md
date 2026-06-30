# Controller_v1 — Graph-SLAM maze mapper (RosBot 2 / Webots)

Phase 1 of the navigation project: the robot builds a digital map of the maze
from its **RPLidar A2** using **pose-graph SLAM**, while you drive it by
keyboard.  No Webots supervisor is used; the robot and world are untouched.

## How to run

1. Open `worlds/Maze5.wbt` in Webots and press ▶ (the Rosbot's `controller`
   field is already set to `Controller_v1`).
2. **Click the 3-D view** so it has keyboard focus.
3. Drive the robot:

   | Key            | Action        |
   |----------------|---------------|
   | `↑` / `W`      | forward       |
   | `↓` / `S`      | backward      |
   | `←` / `A`      | turn left     |
   | `→` / `D`      | turn right    |
   | `Space`        | stop          |
   | `M`            | save the map  |

4. A matplotlib window shows the live map (white = free, black = wall,
   grey = unknown), the current lidar scan (blue), the pose-graph trajectory
   (green) and loop-closure links (red).  Drive a loop back over an explored
   area and watch the map **snap straight** when a loop closes (a
   `[slam] loop closure ...` line prints).
5. On `M` or when you stop the simulation, the map is written next to the
   controller as `map_final.png` and `map_final.npy`.

## How it works

```
encoders + IMU yaw ─► Odometry ─► motion prior ┐
                                                ├─► GraphSlam.update ─► corrected pose + keyframes
LIDAR ranges ─► Scan ───────────────────────────┘     │  ICP refine, keyframe, loop closure
                                                       └─► PoseGraph.optimize (on loop closure)
                                                                    │
                                          OccupancyGrid ◄───────────┘  (rebuilt from optimised poses)
keyboard ─► Teleop ─► (v, w) ─► Robot
```

* **Front-end** (`slam/scan_matcher.py`): ICP aligns each scan to the last
  keyframe, seeded by odometry, to measure the true motion.
* **Back-end** (`slam/pose_graph.py`): a pose graph of keyframe nodes and
  relative-motion edges, optimised with Gauss-Newton on SE(2) so all
  constraints agree — this corrects accumulated drift on loop closure.
* **Map** (`occupancy_map.py`): a log-odds occupancy grid, rebuilt from the
  optimised keyframe poses whenever the graph is re-optimised.

## File layout

| File / package        | Responsibility                                        |
|-----------------------|-------------------------------------------------------|
| `Controller_v1.py`    | Webots entry point (thin) → `SlamApp().run()`         |
| `app.py`              | main loop: perceive → map → act → render              |
| `config.py`           | all tunable constants                                 |
| `robot.py`            | hardware abstraction (sensors, motors, keyboard)      |
| `odometry.py`         | wheel + IMU motion prior                              |
| `occupancy_map.py`    | log-odds occupancy grid                               |
| `teleop.py`           | keyboard → velocity command                           |
| `mapviz.py`           | live matplotlib visualisation                         |
| `slam/geometry.py`    | `Pose2D` SE(2) algebra                                 |
| `slam/lidar_scan.py`  | `Scan` (ranges → point cloud)                          |
| `slam/scan_matcher.py`| ICP front-end                                         |
| `slam/pose_graph.py`  | pose graph + Gauss-Newton optimiser (back-end)        |
| `slam/graph_slam.py`  | keyframes, loop closure, glue                          |
| `mission.py`          | mission-state constants (reserved for phase 2)        |
| `tests/`              | offline SLAM self-tests (run without Webots)          |

The `slam/` package is pure Python + NumPy (no Webots imports), so it can be
unit-tested offline:

```
python tests/test_slam_core.py
```

## Next phase

Autonomous exploration replaces teleop: frontier detection + **A\*** path
planning + path following, driving the robot to map the maze on its own.
`mission.py` is the scaffolding for that state machine.
```
