# RosBot 2 — Maze Exploration (base solution)

Autonomous frontier exploration + live occupancy mapping for the RosBot 2 with
the RpLidar A2, in Webots, pure Python (no ROS). Designed so a Graph-SLAM layer
can be dropped in later by replacing only the pose module.

## How to run

1. Open `worlds/Maze5.wbt` in Webots (R2025a).
2. In the Scene tree select **Rosbot → `controller`** field and set it to
   **`Controller_v1`**. *(The world currently has no controller assigned; this
   is the only manual step. I did not edit the world file as requested.)*
3. Press **Play**. A matplotlib window opens showing the map being built:
   white = free, black = wall, grey = unknown, red = robot + heading arrow,
   blue dots = current LIDAR scan, green line = planned path, magenta star =
   the frontier the robot is driving to.
4. When no reachable frontier is left, the robot stops and writes
   `map_final.png` and `map_final.npy` into this folder.

## How it works

```
IMU yaw + wheel encoders ─▶ Odometry ─▶ pose (x, y, θ)
                                          │
RpLidar A2 (360°, 400 rays) ───────────▶ OccupancyGrid (log-odds, Bresenham)
                                          │
            Frontier detection ─▶ A* planner ─▶ Pilot (pure pursuit + safety)
                                          │                     │
                                  live matplotlib map      wheel speeds
```

State machine: **SPIN_SEED** (one in-place turn to seed a 360° view) →
**PLAN** (detect frontiers, pick the best reachable one, A* a path) →
**DRIVE** (pure-pursuit follow + reactive LIDAR collision avoidance, replanning
on arrival / blockage / timeout / stuck) → **DONE** (no frontiers left).

### Files
| file | role |
|------|------|
| `config.py` | all tunable constants (one place to tweak) |
| `odometry.py` | pose: encoder distance + IMU heading |
| `occupancy_grid.py` | log-odds grid, ray tracing, masks |
| `frontier.py` | frontier detection + clustering |
| `planner.py` | obstacle inflation, A*, frontier selection |
| `pilot.py` | pure-pursuit follower + reactive safety |
| `mapviz.py` | live matplotlib visualization (+ PNG fallback) |
| `Controller_v1.py` | device setup + main loop / state machine |

## Tuning notes
- All knobs live in `config.py` (speeds, grid resolution, log-odds, frontier
  size, look-ahead, safety distance, etc.).
- **If the live map looks mirrored**, set `LIDAR_ANGLE_SIGN = -1.0`. If it looks
  rotated, adjust `LIDAR_ANGLE_OFFSET`. (Webots' LIDAR index→angle convention is
  the one thing that can't be checked outside the simulator; everything else was
  validated headless against a synthetic maze.)
- `ROBOT_START_X/Y` must match the Rosbot translation in the world so the map is
  built in world coordinates. Heading is taken live from the IMU.

## Toward Graph-SLAM (next stage)
Pose is fully isolated in `odometry.py`. To add SLAM, keep the grid/frontier/
planner/pilot untouched and replace the pose source with a SLAM front-end
(scan-matching for the odometry constraint) + back-end (pose-graph optimisation).
`map_final.npy` (raw log-odds) is saved each run for evaluation.

## Requirements
Python packages: `numpy`, `matplotlib` (both ship with a standard Webots
Python). The controller degrades gracefully to PNG snapshots if no display is
available.
