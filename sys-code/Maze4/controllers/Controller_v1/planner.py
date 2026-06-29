"""
planner.py
==========
Grid path planning (A*) plus obstacle inflation and frontier-target selection.

Layers derived from the occupancy grid:
  * occupied (inflated by the robot radius) -> blocked
  * free                                    -> cheap to cross
  * unknown                                 -> traversable but penalised
    (so the robot prefers known-free routes but will push into the unknown to
     actually reach a frontier).
"""

import heapq
import math

import numpy as np

import Maze5.controllers.Controller_v1.config as C


class PathPlanner:
    _NEIGH = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, 1.41421), (-1, 1, 1.41421),
              (1, -1, 1.41421), (1, 1, 1.41421)]

    # ------------------------------------------------------------------ #
    # Cost layers
    # ------------------------------------------------------------------ #
    def build_cost_layers(self, grid):
        """Return (blocked, unknown) boolean arrays for the planner."""
        blocked = self._inflate(grid.occ_mask(), C.INFLATE_RADIUS_CELLS)
        unknown = grid.unknown_mask()
        return blocked, unknown

    @staticmethod
    def _inflate(occ_mask, radius_cells):
        """Binary dilation of occ_mask by radius_cells (Chebyshev), numpy only."""
        out = occ_mask.copy()
        for _ in range(radius_cells):
            s = out.copy()
            s[:-1, :] |= out[1:, :]
            s[1:, :] |= out[:-1, :]
            s[:, :-1] |= out[:, 1:]
            s[:, 1:] |= out[:, :-1]
            s[:-1, :-1] |= out[1:, 1:]
            s[1:, 1:] |= out[:-1, :-1]
            s[:-1, 1:] |= out[1:, :-1]
            s[1:, :-1] |= out[:-1, 1:]
            out = s
        return out

    # ------------------------------------------------------------------ #
    # A*
    # ------------------------------------------------------------------ #
    def astar(self, blocked, unknown, start_rc, goal_rc):
        """A* on the grid. start/goal are (row, col). Returns path or None."""
        nrows, ncols = blocked.shape
        sr, sc = start_rc
        gr, gc = goal_rc
        if not (0 <= sr < nrows and 0 <= sc < ncols):
            return None
        if not (0 <= gr < nrows and 0 <= gc < ncols):
            return None
        if blocked[gr, gc]:
            return None

        def h(r, c):
            dr, dc = abs(r - gr), abs(c - gc)
            return (dr + dc) + (1.41421 - 2) * min(dr, dc)

        open_heap = [(h(sr, sc), 0.0, (sr, sc))]
        came_from = {}
        g_score = {(sr, sc): 0.0}
        closed = set()

        while open_heap:
            _, g_cur, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == (gr, gc):
                return self._reconstruct(came_from, cur)

            r, c = cur
            for dr, dc, step in self._NEIGH:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < nrows and 0 <= cc < ncols):
                    continue
                if blocked[rr, cc] or (rr, cc) in closed:
                    continue
                cost = step * (C.UNKNOWN_TRAVERSAL_COST if unknown[rr, cc] else 1.0)
                tentative = g_cur + cost
                if tentative < g_score.get((rr, cc), math.inf):
                    g_score[(rr, cc)] = tentative
                    came_from[(rr, cc)] = cur
                    heapq.heappush(open_heap,
                                   (tentative + h(rr, cc), tentative, (rr, cc)))
        return None

    @staticmethod
    def _reconstruct(came_from, cur):
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    # ------------------------------------------------------------------ #
    # Frontier target selection
    # ------------------------------------------------------------------ #
    def choose_target(self, grid, clusters, robot_xy, blocked, unknown):
        """Pick the best reachable frontier cluster.

        Returns (path_rc, cluster) or (None, None).
        Cost = path_length_m - INFO_GAIN_WEIGHT * sqrt(cluster_size).
        Clusters are tried nearest-first so we usually stop after a few A* calls.
        """
        rx, ry = robot_xy
        sc0, sr0 = grid.world_to_grid(rx, ry)
        start_rc = (sr0, sc0)

        def euclid(cl):
            gr, gc = cl["centroid"]
            wx, wy = grid.grid_to_world(gc, gr)
            return math.hypot(wx - rx, wy - ry)

        clusters = sorted(clusters, key=euclid)

        best = None
        best_path = None
        best_cost = math.inf
        tried = 0
        for cl in clusters:
            goal_rc = self._nearest_free(blocked, cl["centroid"])
            if goal_rc is None:
                continue
            path = self.astar(blocked, unknown, start_rc, goal_rc)
            tried += 1
            if path is None:
                if tried > 12:
                    break
                continue
            length_m = (len(path) - 1) * grid.res
            cost = length_m - C.INFO_GAIN_WEIGHT * math.sqrt(cl["size"])
            if cost < best_cost:
                best_cost, best, best_path = cost, cl, path
            if best is not None and length_m < 1.0:
                break
        return best_path, best

    @staticmethod
    def _nearest_free(blocked, rc, max_r=4):
        """Spiral outward from rc to find a non-blocked cell."""
        r0, c0 = rc
        nrows, ncols = blocked.shape
        for rad in range(max_r + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < nrows and 0 <= c < ncols and not blocked[r, c]:
                        return (r, c)
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def path_to_world(grid, path_rc):
        return [grid.grid_to_world(c, r) for (r, c) in path_rc]

    @staticmethod
    def path_blocked(path_rc, blocked):
        return any(blocked[r, c] for (r, c) in path_rc)
