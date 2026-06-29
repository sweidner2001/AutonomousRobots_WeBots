"""
Controller_v4 – lean SLAM-only exploration
==========================================
No GridWorld / ray-casting / A* / BFS.
Pipeline every tick:
  lidar ranges  →  BreezySLAM.update()  →  reactive motor command

Navigation strategy (state machine):
  FORWARD  – drive ahead until obstacle detected in front cone
  TURN     – rotate toward the side with more free space, then back to FORWARD

BreezySLAM accumulates the map internally; every VIS_PERIOD steps its raw
byte-map is rendered to an OpenCV window so you can watch exploration progress.
"""

import math
import struct

from controller import Robot

try:
    from breezyslam.algorithms import RMHC_SLAM
    from breezyslam.sensors import Laser
    SLAM_OK = True
except ImportError:
    SLAM_OK = False
    print("[WARN] breezyslam not found – running reactive-only, no map.")

try:
    import numpy as np
    import cv2
    VIS_OK = True
except ImportError:
    VIS_OK = False
    print("[WARN] numpy/cv2 not found – visualization disabled.")

# ── Timing ───────────────────────────────────────────────────────────────────
TIMESTEP          = 64          # ms

# ── Robot geometry ───────────────────────────────────────────────────────────
WHEEL_RADIUS      = 0.085 / 2.0      # m
AXLE_TRACK        = 0.265       # m

# ── Motion ───────────────────────────────────────────────────────────────────
MAX_SPEED         = 6.0         # rad/s  (keep moderate for clean SLAM)
TURN_SPEED        = 4.0         # rad/s  (in-place turn)
SAFE_FRONT_DIST   = 0.35        # m  – enter TURN when obstacle is this close
TURN_CLEAR_DIST   = 0.50        # m  – exit TURN only when front is THIS clear (hysteresis)
FRONT_HALF_DEG    = 35          # degrees either side of forward heading
MAX_TURN_TICKS    = 80          # ~5 s max spinning; if still blocked → BACKUP
BACKUP_TICKS      = 20          # ticks to drive in reverse before re-attempting turn

# ── BreezySLAM map ────────────────────────────────────────────────────────────
# Internal resolution: MAP_SIZE_PX × MAP_SIZE_PX pixels, MAP_SIZE_M metres square
# One pixel = MAP_SIZE_M / MAP_SIZE_PX  (here 30/300 = 0.10 m)
MAP_SIZE_M        = 30.0
MAP_SIZE_PX       = 600

# ── Visualization ─────────────────────────────────────────────────────────────
VIS_PERIOD        = 10          # render every N ticks


# ─────────────────────────────────────────────────────────────────────────────
class ExplorerV4:
    """Reactive explorer with BreezySLAM for map building (no GridWorld)."""

    def __init__(self):
        self._robot = Robot()
        self._init_devices()
        self._setup_slam()

        # Navigation state
        self._state       = "forward"
        self._turn_dir    = 1.0   # +1 = CCW (left), -1 = CW (right)
        self._turn_budget = 0     # remaining ticks in TURN state

        # Odometry bookkeeping
        self._prev_enc    = None

        # BreezySLAM map buffer (flat bytes, row-major)
        self._map_bytes   = bytearray(MAP_SIZE_PX * MAP_SIZE_PX)

    # ── Device initialisation ─────────────────────────────────────────────────
    def _init_devices(self):
        ts = TIMESTEP

        # Motors (velocity control)
        motor_names = [
            "fl_wheel_joint", "fr_wheel_joint",
            "rl_wheel_joint", "rr_wheel_joint",
        ]
        self._motors = []
        for name in motor_names:
            m = self._robot.getDevice(name)
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
            self._motors.append(m)

        # Wheel encoders
        encoder_names = [
            "front left wheel motor sensor",
            "front right wheel motor sensor",
            "rear left wheel motor sensor",
            "rear right wheel motor sensor",
        ]
        self._encoders = []
        for name in encoder_names:
            e = self._robot.getDevice(name)
            e.enable(ts)
            self._encoders.append(e)

        # 2-D Lidar (RPLidar A2, 360°)
        self._lidar = self._robot.getDevice("laser")
        self._lidar.enable(ts)
        self._lidar.enablePointCloud()

        self._n_beams   = self._lidar.getNumberOfPoints()
        self._lidar_resolution = self._lidar.getHorizontalResolution()
        self._fov_rad   = self._lidar.getFov()            # full FOV in radians
        self._max_range = self._lidar.getMaxRange()       # metres

    # ── BreezySLAM setup ──────────────────────────────────────────────────────
    def _setup_slam(self):
        if not SLAM_OK:
            self._slam = None
            return

        fov_deg = math.degrees(self._fov_rad)
        laser = Laser(
            scan_size                  = self._lidar_resolution,
            scan_rate_hz               = 10,
            detection_angle_degrees    = fov_deg,
            distance_no_detection_mm   = int(self._max_range * 1000),
            detection_margin           = 0,
            offset_mm                  = 0,
        )
        self._slam = RMHC_SLAM(laser, MAP_SIZE_PX, MAP_SIZE_M)



    # ── Odometry ──────────────────────────────────────────────────────────────
    def _odometry(self):
        """Return (d_mm, d_deg) since last call."""
        enc = [e.getValue() for e in self._encoders]
        if self._prev_enc is None:
            self._prev_enc = enc
            return 0.0, 0.0

        # average left / right wheel travel (metres)
        d_left  = ((enc[0] - self._prev_enc[0]) + (enc[2] - self._prev_enc[2])) \
                  / 2.0 * WHEEL_RADIUS
        d_right = ((enc[1] - self._prev_enc[1]) + (enc[3] - self._prev_enc[3])) \
                  / 2.0 * WHEEL_RADIUS
        self._prev_enc = enc

        d_trans_mm = (d_left + d_right) / 2.0 * 1000.0
        d_rot_deg  = math.degrees((d_right - d_left) / AXLE_TRACK)
        return d_trans_mm, d_rot_deg



    # ── Lidar ─────────────────────────────────────────────────────────────────
    def _get_scan(self):
        """Return (ranges_m, scan_mm).
        scan_mm uses 0 for no-hit beams (BreezySLAM convention)."""
        ranges  = self._lidar.getRangeImage()
        scan_mm = [int(r * 1000) if math.isfinite(r) else 0 for r in ranges]
        return ranges, scan_mm



    # ── Reactive navigation command ───────────────────────────────────────────
    def _reactive_cmd(self, ranges):
        """Return (left_vel, right_vel) based on raw lidar ranges.

        State machine:
          FORWARD – drive ahead
          TURN    – spin until front is clear (exit on TURN_CLEAR_DIST)
                    if stuck > MAX_TURN_TICKS → BACKUP
          BACKUP  – reverse briefly, then re-enter TURN with flipped direction
        """
        n             = len(ranges)
        beams_per_rad = n / self._fov_rad
        front_half    = int(math.radians(FRONT_HALF_DEG) * beams_per_rad)

        # Minimum distance inside the front cone
        front_slice = list(ranges[:front_half]) + list(ranges[n - front_half:])
        valid_front = [r for r in front_slice if math.isfinite(r)]
        min_front   = min(valid_front) if valid_front else self._max_range

        # ── FORWARD ───────────────────────────────────────────────────────────
        if self._state == "forward":
            if min_front < SAFE_FRONT_DIST:
                mid         = n // 2
                left_slice  = [r for r in ranges[front_half:mid]     if math.isfinite(r)]
                right_slice = [r for r in ranges[mid:n - front_half] if math.isfinite(r)]
                avg_left    = sum(left_slice)  / len(left_slice)  if left_slice  else self._max_range
                avg_right   = sum(right_slice) / len(right_slice) if right_slice else self._max_range
                self._turn_dir    = 1.0 if avg_left >= avg_right else -1.0
                self._turn_budget = MAX_TURN_TICKS
                self._state       = "turn"
            else:
                return MAX_SPEED, MAX_SPEED

        # ── TURN ──────────────────────────────────────────────────────────────
        if self._state == "turn":
            if min_front >= TURN_CLEAR_DIST:
                # Front is clear – leave TURN state
                self._state = "forward"
                return MAX_SPEED, MAX_SPEED
            self._turn_budget -= 1
            if self._turn_budget <= 0:
                # Still blocked after MAX_TURN_TICKS → back up, flip direction
                self._turn_dir   *= -1.0
                self._turn_budget = BACKUP_TICKS
                self._state       = "backup"
            return (-TURN_SPEED * self._turn_dir, TURN_SPEED * self._turn_dir)

        # ── BACKUP ────────────────────────────────────────────────────────────
        if self._state == "backup":
            self._turn_budget -= 1
            if self._turn_budget <= 0:
                self._turn_budget = MAX_TURN_TICKS
                self._state       = "turn"
            return (-MAX_SPEED * 0.5, -MAX_SPEED * 0.5)

        return MAX_SPEED, MAX_SPEED   # fallback



    # ── Motor helper ──────────────────────────────────────────────────────────
    def _set_velocity(self, left, right):
        left  = max(-MAX_SPEED, min(MAX_SPEED, left))
        right = max(-MAX_SPEED, min(MAX_SPEED, right))
        for i, v in enumerate([left, right, left, right]):
            self._motors[i].setVelocity(v)




    # ── Visualisation (BreezySLAM raw map bytes) ───────────────────────────────
    def _visualize(self, pose):
        if not (VIS_OK and self._slam):
            return

        self._slam.getmap(self._map_bytes)

        # Convert to numpy uint8 image
        img = np.frombuffer(self._map_bytes, dtype=np.uint8).reshape(
            MAP_SIZE_PX, MAP_SIZE_PX
        ).copy()

        # BreezySLAM encoding: 0 = unknown (gray), 1-127 = occupied (dark), 128-255 = free (bright)
        display = np.zeros((MAP_SIZE_PX, MAP_SIZE_PX, 3), dtype=np.uint8)
        unknown  = img == 0
        occupied = (img > 0)   & (img <= 127)
        free     = img > 127

        display[unknown]  = (128, 128, 128)   # gray
        display[free]     = (240, 240, 240)   # light
        display[occupied] = (30,  30,  30)    # dark

        # Robot position (pose in mm from bottom-left of SLAM map)
        if pose:
            px = int(pose[0] / 1000.0 / MAP_SIZE_M * MAP_SIZE_PX)
            py = MAP_SIZE_PX - int(pose[1] / 1000.0 / MAP_SIZE_M * MAP_SIZE_PX)
            if 0 <= px < MAP_SIZE_PX and 0 <= py < MAP_SIZE_PX:
                cv2.circle(display, (px, py), 4, (0, 0, 220), -1)

        cv2.imshow("BreezySLAM map", display)
        cv2.waitKey(1)



    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        step  = 0
        pose  = None

        while self._robot.step(TIMESTEP) != -1:
            ranges, scan_mm = self._get_scan()
            d_mm, d_deg     = self._odometry()

            # ── SLAM update (pose correction + internal map) ───────────────────
            if self._slam:
                self._slam.update(scan_mm, pose_change=(d_mm, d_deg, 0.0))
                x, y, theta = self._slam.getpos()
                pose = (x, y, theta)   # mm from SLAM map origin, degrees

            # ── Navigation ────────────────────────────────────────────────────
            left, right = self._reactive_cmd(ranges)
            self._set_velocity(left, right)

            # ── Visualisation ─────────────────────────────────────────────────
            if step % VIS_PERIOD == 0:
                self._visualize(pose)

            step += 1


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ExplorerV4().run()
