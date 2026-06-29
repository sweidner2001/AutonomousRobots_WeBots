"""
occupancy_grid.py
=================
A 2-D probabilistic occupancy grid stored as log-odds.

  * Each lidar scan is fused with an inverse sensor model:
      - cells along a beam (robot -> hit) get L_FREE added  (become free)
      - the hit cell gets L_OCC added                       (becomes wall)
  * An `observed` mask records which cells have ever been touched.  This is
    what frontier detection uses to tell "free" apart from "never seen".

The grid is indexed [row, col] = [y, x].  Its origin is centred (config places
GRID_ORIGIN at -W/2, -H/2) so the robot's start pose (0, 0) sits in the middle.
"""

import numpy as np

import Maze5.controllers.Controller_v1.config as C


class OccupancyGrid:
    def __init__(self):
        self.res = C.GRID_RESOLUTION
        self.ox = C.GRID_ORIGIN_X
        self.oy = C.GRID_ORIGIN_Y
        self.ncols = int(round(C.GRID_WIDTH_M / self.res))
        self.nrows = int(round(C.GRID_HEIGHT_M / self.res))
        # log-odds, 0 == unknown (prob 0.5)
        self.log = np.zeros((self.nrows, self.ncols), dtype=np.float32)
        # has this cell ever been measured?
        self.observed = np.zeros((self.nrows, self.ncols), dtype=bool)

    # ------------------------------------------------------------------ #
    # Coordinate helpers
    # ------------------------------------------------------------------ #
    def world_to_grid(self, x, y):
        col = int((x - self.ox) / self.res)
        row = int((y - self.oy) / self.res)
        return col, row

    def grid_to_world(self, col, row):
        x = self.ox + (col + 0.5) * self.res
        y = self.oy + (row + 0.5) * self.res
        return x, y

    def in_bounds(self, col, row):
        return 0 <= col < self.ncols and 0 <= row < self.nrows

    # ------------------------------------------------------------------ #
    # Probability views
    # ------------------------------------------------------------------ #
    def prob(self):
        """Full probability-of-occupied array in [0, 1]."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log))

    def occ_mask(self):
        return self.prob() >= C.P_OCC_THRESH

    def free_mask(self):
        return (self.prob() <= C.P_FREE_THRESH) & self.observed

    def unknown_mask(self):
        return ~self.observed

    # ------------------------------------------------------------------ #
    # Scan integration
    # ------------------------------------------------------------------ #
    def integrate_scan(self, x, y, theta, ranges, bearings):
        """
        Fuse one lidar scan.

        x, y, theta : robot pose (lidar origin is offset forward by
                      LIDAR_OFFSET_X, applied here).
        ranges      : np.array of measured distances (m). inf / >max -> no hit.
        bearings    : np.array of beam bearings in the ROBOT frame (rad).
        """
        # Lidar sensor origin in world coordinates.
        sx = x + C.LIDAR_OFFSET_X * np.cos(theta)
        sy = y + C.LIDAR_OFFSET_X * np.sin(theta)
        c0, r0 = self.world_to_grid(sx, sy)

        world_ang = theta + bearings
        cos_a = np.cos(world_ang)
        sin_a = np.sin(world_ang)

        max_use = C.LIDAR_USE_RANGE
        max_range = C.LIDAR_USE_RANGE  # rays at/over this are treated as "no hit"
        for i in range(len(ranges)):
            rng = ranges[i]
            hit = True
            if not np.isfinite(rng) or rng >= max_range:
                rng = max_use
                hit = False
            elif rng < C.LIDAR_MIN_RANGE:
                continue  # below min range -> unreliable, skip
            else:
                rng = min(rng, max_use)
                if rng >= max_use:
                    hit = False

            ex = sx + rng * cos_a[i]
            ey = sy + rng * sin_a[i]
            c1, r1 = self.world_to_grid(ex, ey)
            self._ray(c0, r0, c1, r1, hit)

    def _ray(self, c0, r0, c1, r1, hit):
        """Bresenham from (c0,r0) to (c1,r1): free along the way, occ at end."""
        cells = _bresenham(c0, r0, c1, r1)
        if not cells:
            return
        # All but the last cell are free space.
        for (c, r) in cells[:-1]:
            if self.in_bounds(c, r):
                v = self.log[r, c] + C.L_FREE
                self.log[r, c] = max(v, -C.L_CLAMP)
                self.observed[r, c] = True
        # Endpoint.
        c, r = cells[-1]
        if self.in_bounds(c, r):
            if hit:
                v = self.log[r, c] + C.L_OCC
                self.log[r, c] = min(v, C.L_CLAMP)
            else:
                v = self.log[r, c] + C.L_FREE
                self.log[r, c] = max(v, -C.L_CLAMP)
            self.observed[r, c] = True

    def save(self, npy_path):
        np.save(npy_path, self.log)


def _bresenham(x0, y0, x1, y1):
    """Integer Bresenham line. Returns list of (col, row) incl. both ends."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    # Guard against runaway loops far outside the grid.
    max_steps = dx + dy + 2
    steps = 0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        steps += 1
        if steps > max_steps:
            break
    return points
