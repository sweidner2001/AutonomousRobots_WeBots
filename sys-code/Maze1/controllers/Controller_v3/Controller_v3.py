"""Controller_v3 — SLAM-based autonomous maze explorer.

Hardware assumed (ROSbot 2, Webots):
  Motors   : fl_wheel_joint, fr_wheel_joint, rl_wheel_joint, rr_wheel_joint
  Encoders : front/rear left/right wheel motor sensor
  Lidar    : laser  (RPLidar A2, 360°, 12 m range)

Algorithm:
  1. Dead-reckoning odometry from wheel encoders.
  2. BreezySLAM (RMHC) corrects pose every tick via scan-matching.
  3. Bresenham ray-casting builds a GridWorld occupancy grid immediately.
  4. BFS finds the nearest unexplored frontier cell.
  5. A* plans a collision-free path to that frontier.
  6. Proportional heading controller follows the path.
  7. Reactive layer stops the robot and turns away if a wall is too close.
"""

import heapq
import math
from collections import deque

import cv2
import numpy as np
from controller import Robot


# ── Tuning constants ────────────────────────────────────────────────────────

# Robot kinematics (ROSbot 2)
WHEEL_RADIUS    = 0.085 / 2.0   # metres
AXLE_TRACK      = 0.265          # metres
MAX_WHEEL_SPEED = 26.0           # rad/s

# Safety
SAFE_FRONT_DIST  = 0.35   # metres – reactive stop distance
WAYPOINT_REACH_M = 0.18   # metres – waypoint considered reached

# Map
MAP_SIZE_M         = 30.0   # square map side length (metres)
MAP_RESOLUTION_M   = 0.10   # cell size (metres)  → 300×300 grid
INFLATION_RADIUS_M = 0.20   # obstacle buffer for path planning

# Control timing
PLAN_PERIOD_STEPS  = 20     # simulation ticks between full replans
VIS_PERIOD_STEPS   = 10     # ticks between visualisation refreshes


# ── Occupancy grid ───────────────────────────────────────────────────────────

class OccupancyGrid:
    """2-D numpy occupancy grid.

    Cell values:
        -1  unknown  (never observed)
         0  free     (laser ray passed through)
         1  occupied (laser ray endpoint / wall)
    """

    UNKNOWN  = -1
    FREE     =  0
    OCCUPIED =  1

    def __init__(self, size_m: float, resolution: float):
        self.resolution = resolution
        n = int(size_m / resolution)
        self.width  = n
        self.height = n
        self.origin = -0.5 * size_m          # world coordinate of cell (0,0) corner
        self.grid   = np.full((n, n), self.FREE, dtype=np.int8)

    # ── Coordinate conversion ──────────────────────────────────────────────

    def world_to_cell(self, wx: float, wy: float):
        gx = int((wx - self.origin) / self.resolution)
        gy = int((wy - self.origin) / self.resolution)
        return gx, gy

    def cell_to_world(self, gx: int, gy: int):
        wx = self.origin + (gx + 0.5) * self.resolution
        wy = self.origin + (gy + 0.5) * self.resolution
        return wx, wy

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height

    # ── Cell setters ───────────────────────────────────────────────────────

    def mark_free(self, gx: int, gy: int):
        if self.in_bounds(gx, gy) and self.grid[gy, gx] != self.OCCUPIED:
            self.grid[gy, gx] = self.FREE

    def mark_occupied(self, gx: int, gy: int):
        if self.in_bounds(gx, gy):
            self.grid[gy, gx] = self.OCCUPIED

    def mark_unknown(self, gx: int, gy: int):
        if self.in_bounds(gx, gy) and self.grid[gy, gx] == self.FREE:
            self.grid[gy, gx] = self.UNKNOWN

    # ── Lidar scan insertion ───────────────────────────────────────────────

    def insert_scan(self, px, py, theta, ranges, angle_min, angle_inc, range_max):
        """Bresenham ray-casting: carve free space, mark endpoints occupied."""
        rgx, rgy = self.world_to_cell(px, py)
        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            r = min(r, range_max)
            bearing = theta + angle_min + i * angle_inc
            ex = px + r * math.cos(bearing)
            ey = py + r * math.sin(bearing)
            egx, egy = self.world_to_cell(ex, ey)

            # Walk the ray with Bresenham
            for cx, cy in self._bresenham(rgx, rgy, egx, egy)[:-1]:
                self.mark_free(cx, cy)

            if r < 0.98 * range_max:
                self.mark_occupied(egx, egy)
            else:
                self.mark_free(egx, egy)

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        pts, dx, sx = [], abs(x1 - x0), 1 if x0 < x1 else -1
        dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            pts.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy; x += sx
            if e2 <= dx:
                err += dx; y += sy
        return pts

    # ── Planning helpers ───────────────────────────────────────────────────

    def inflated_blocked(self, radius_cells: int) -> np.ndarray:
        """Boolean array (True = blocked) with obstacles expanded outward."""
        occ = (self.grid == self.OCCUPIED).astype(np.uint8)
        if radius_cells > 0:
            k = 2 * radius_cells + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            occ = cv2.dilate(occ, kernel)
        return occ.astype(bool)

    def nearest_frontier(self, start_gx: int, start_gy: int):
        """BFS from robot cell → first free cell with an unknown neighbour."""
        if not self.in_bounds(start_gx, start_gy):
            return None
        visited = {(start_gx, start_gy)}
        q = deque([(start_gx, start_gy)])
        while q:
            x, y = q.popleft()
            if self.grid[y, x] == self.FREE and self._is_frontier(x, y):
                return x, y
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited or not self.in_bounds(nx, ny):
                    continue
                if self.grid[ny, nx] == self.FREE:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return None

    def _is_frontier(self, x: int, y: int) -> bool:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny) and self.grid[ny, nx] == self.UNKNOWN:
                    return True
        return False

    # ── Visualisation ──────────────────────────────────────────────────────

    def render(self, robot_gx=None, robot_gy=None, path=None) -> np.ndarray:
        """Return a BGR image of the grid (unknown=gray, free=white, occ=black)."""
        img = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        img[self.grid == self.FREE]     = (255, 255, 255)
        img[self.grid == self.OCCUPIED] = (0,   0,   0)
        if path:
            for gx, gy in path:
                if self.in_bounds(gx, gy):
                    img[gy, gx] = (0, 165, 255)      # orange
        if robot_gx is not None:
            r = max(2, self.width // 150)
            img[max(0,robot_gy-r):robot_gy+r+1,
                max(0,robot_gx-r):robot_gx+r+1] = (0, 0, 255)  # red
        return img


# ── A* path planner ──────────────────────────────────────────────────────────

def astar(start, goal, blocked: np.ndarray):
    """A* on a 2-D grid.  blocked[row][col] == True means impassable."""
    h, w = blocked.shape

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_set = [(0.0, start)]
    came_from, g = {}, {start: 0.0}
    visited = set()
    moves = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
             (1,1,1.414),(-1,1,1.414),(1,-1,1.414),(-1,-1,1.414)]

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur); cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path
        cx, cy = cur
        for dx, dy, cost in moves:
            nx, ny = cx+dx, cy+dy
            if not (0 <= nx < w and 0 <= ny < h) or blocked[ny, nx]:
                continue
            ng = g[cur] + cost
            nb = (nx, ny)
            if ng < g.get(nb, 1e18):
                came_from[nb] = cur
                g[nb] = ng
                heapq.heappush(open_set, (ng + heuristic(nb, goal), nb))
    return []


# ── Main controller ───────────────────────────────────────────────────────────

class ExplorerV3:

    def __init__(self):
        self.robot    = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self._init_devices()

        # Pose (updated by odometry, corrected by SLAM)
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self._prev_left = self._prev_right = None

        # Map and planner state
        self.map          = OccupancyGrid(MAP_SIZE_M, MAP_RESOLUTION_M)
        self.inflate_cells = int(INFLATION_RADIUS_M / MAP_RESOLUTION_M)
        self.path         = []
        self.path_idx     = 0
        self.step         = 0

        # BreezySLAM
        self.slam            = None
        self.use_slam        = False
        self.slam_pixels     = int(MAP_SIZE_M / MAP_RESOLUTION_M)
        self._setup_slam()

    # ── Devices ────────────────────────────────────────────────────────────

    def _init_devices(self):
        motors = ["fl_wheel_joint", "fr_wheel_joint",
                  "rl_wheel_joint", "rr_wheel_joint"]
        self._fl, self._fr, self._rl, self._rr = (
            self.robot.getDevice(n) for n in motors
        )
        for m in (self._fl, self._fr, self._rl, self._rr):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        enc_names = ["front left wheel motor sensor",
                     "front right wheel motor sensor",
                     "rear left wheel motor sensor",
                     "rear right wheel motor sensor"]
        self._fl_enc, self._fr_enc, self._rl_enc, self._rr_enc = (
            self.robot.getDevice(n) for n in enc_names
        )
        for s in (self._fl_enc, self._fr_enc, self._rl_enc, self._rr_enc):
            s.enable(self.timestep)

        self._lidar = self.robot.getDevice("laser")
        self._lidar.enable(self.timestep)
        self._lidar_res = self._lidar.getHorizontalResolution()
        self._lidar_fov = self._lidar.getFov()
        self._lidar_max = self._lidar.getMaxRange()

    # ── SLAM setup ─────────────────────────────────────────────────────────

    def _setup_slam(self):
        try:
            from breezyslam.algorithms import RMHC_SLAM
            from breezyslam.sensors   import Laser
            sensor = Laser(
                self._lidar_res,        # beams per scan
                10,                     # scan rate Hz  (RPLidar A2)
                360,                    # field of view degrees
                int(self._lidar_max * 1000),  # max range mm
                0, 0,
            )
            self.slam     = RMHC_SLAM(sensor, self.slam_pixels, MAP_SIZE_M)
            self.use_slam = True
            print("[SLAM] BreezySLAM ready.")
        except Exception as e:
            print("[SLAM] Fallback (ray-casting only). Reason:", e)

    # ── Sensing ────────────────────────────────────────────────────────────

    def _update_odometry(self):
        left  = 0.5 * (self._fl_enc.getValue() + self._rl_enc.getValue())
        right = 0.5 * (self._fr_enc.getValue() + self._rr_enc.getValue())
        if self._prev_left is None:
            self._prev_left, self._prev_right = left, right
            return
        dl = (left  - self._prev_left)  * WHEEL_RADIUS
        dr = (right - self._prev_right) * WHEEL_RADIUS
        self._prev_left, self._prev_right = left, right
        ds, dtheta = 0.5*(dl+dr), (dr-dl)/AXLE_TRACK
        mid = self.theta + 0.5 * dtheta
        self.x     += ds * math.cos(mid)
        self.y     += ds * math.sin(mid)
        self.theta  = _wrap(self.theta + dtheta)

    def _get_scan(self):
        ranges    = self._lidar.getRangeImage()
        angle_min = -0.5 * self._lidar_fov
        angle_inc = self._lidar_fov / max(1, self._lidar_res - 1)
        return ranges, angle_min, angle_inc

    # ── Mapping ────────────────────────────────────────────────────────────

    def _update_map(self, ranges, angle_min, angle_inc):
        if self.use_slam and self.slam:
            # 0 = "no detection" in BreezySLAM (never pass float inf)
            scan_mm = [
                int(r * 1000) if math.isfinite(r) else 0
                for r in ranges
            ]
            self.slam.update(scan_mm)
            # Overwrite odometry with SLAM-corrected pose
            xm, ym, ydeg = self.slam.getpos()
            half = 0.5 * MAP_SIZE_M
            self.x     = (xm / 1000.0) - half
            self.y     = (ym / 1000.0) - half
            self.theta = _wrap(math.radians(ydeg))

        # Always ray-cast so GridWorld is populated from the first scan
        self.map.insert_scan(
            self.x, self.y, self.theta,
            ranges, angle_min, angle_inc, self._lidar_max,
        )

    # ── Planning ───────────────────────────────────────────────────────────

    def _replan(self):
        """Find nearest frontier and run A* to it."""
        if self.step % PLAN_PERIOD_STEPS != 0 and self.path:
            return

        rgx, rgy = self.map.world_to_cell(self.x, self.y)
        frontier  = self.map.nearest_frontier(rgx, rgy)

        if frontier is None:
            self.path, self.path_idx = [], 0
            print("[Explorer] No frontier found – map may be fully explored.")
            return

        blocked = self.map.inflated_blocked(self.inflate_cells)
        self.path     = astar((rgx, rgy), frontier, blocked)
        self.path_idx = 0

    # ── Control ────────────────────────────────────────────────────────────

    def _front_clearance(self, ranges, angle_min, angle_inc) -> float:
        min_r = self._lidar_max
        for i, r in enumerate(ranges):
            if math.isfinite(r) and abs(angle_min + i*angle_inc) < math.radians(30):
                min_r = min(min_r, r)
        return min_r

    def _compute_cmd(self, ranges, angle_min, angle_inc):
        # No path: spin slowly so new lidar data fills the map
        if not self.path:
            return 0.0, 0.5

        # Advance past already-reached waypoints
        while self.path_idx < len(self.path):
            wx, wy = self.map.cell_to_world(*self.path[self.path_idx])
            if math.hypot(self.x-wx, self.y-wy) > WAYPOINT_REACH_M:
                break
            self.path_idx += 1

        if self.path_idx >= len(self.path):
            self.path = []
            return 0.0, 0.0

        wx, wy    = self.map.cell_to_world(*self.path[self.path_idx])
        target    = math.atan2(wy - self.y, wx - self.x)
        err       = _wrap(target - self.theta)
        omega     = _clamp(2.5 * err, -1.8, 1.8)
        v         = 0.4 * (1.0 - min(1.0, abs(err) / math.radians(90.0)))

        # Reactive safety: obstacle too close → stop and turn away
        if self._front_clearance(ranges, angle_min, angle_inc) < SAFE_FRONT_DIST:
            ls, ln, rs, rn = 0.0, 0, 0.0, 0
            for i, r in enumerate(ranges):
                if not math.isfinite(r):
                    continue
                a = angle_min + i * angle_inc
                if math.radians(30) <= a < math.radians(90):
                    ls += r; ln += 1
                elif math.radians(-90) < a <= math.radians(-30):
                    rs += r; rn += 1
            left_avg  = ls/ln if ln else self._lidar_max
            right_avg = rs/rn if rn else self._lidar_max
            v, omega = 0.0, (1.2 if left_avg >= right_avg else -1.2)

        return v, omega

    def _set_velocity(self, v, omega):
        l = _clamp((2*v - omega*AXLE_TRACK) / (2*WHEEL_RADIUS),
                   -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        r = _clamp((2*v + omega*AXLE_TRACK) / (2*WHEEL_RADIUS),
                   -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        self._fl.setVelocity(l); self._rl.setVelocity(l)
        self._fr.setVelocity(r); self._rr.setVelocity(r)

    # ── Visualisation ──────────────────────────────────────────────────────

    def _visualize(self):
        rgx, rgy = self.map.world_to_cell(self.x, self.y)
        img = self.map.render(robot_gx=rgx, robot_gy=rgy, path=self.path)
        img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img,
                    f"x={self.x:.1f} y={self.y:.1f} th={math.degrees(self.theta):.0f}deg",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.imshow("Explorer V3 – Occupancy Map", img)
        cv2.waitKey(1)

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self):
        print("[Explorer V3] Starting.")
        while self.robot.step(self.timestep) != -1:
            self.step += 1

            self._update_odometry()
            ranges, angle_min, angle_inc = self._get_scan()
            self._update_map(ranges, angle_min, angle_inc)
            self._replan()

            v, omega = self._compute_cmd(ranges, angle_min, angle_inc)
            self._set_velocity(v, omega)

            if self.step % VIS_PERIOD_STEPS == 0:
                self._visualize()


# ── Utility functions ────────────────────────────────────────────────────────

def _wrap(a: float) -> float:
    """Wrap angle to [-π, π]."""
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ExplorerV3().run()

