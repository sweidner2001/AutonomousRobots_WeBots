"""
planner.py
==========
Grid path planning (A*) plus obstacle inflation, and frontier-target selection.

The planner operates on three derived layers of the occupancy grid:
  * occupied (inflated by the robot radius) -> blocked
  * free                                    -> cheap to cross
  * unknown                                 -> traversable but penalised
    (so the robot prefers known-free routes but will push into the unknown
     to actually reach a frontier).
"""

import heapq
import math

import numpy as np

import Maze5.controllers.Controller_v1.config as C


# ---------------------------------------------------------------------- #
# Obstacle inflation
# ---------------------------------------------------------------------- #
def inflate(occ_mask, radius_cells):
    """Binary dilation of occ_mask by `radius_cells` (Chebyshev), numpy only."""
    out = occ_mask.copy()
    for _ in range(radius_cells):
        shifted = out.copy()
        shifted[:-1, :] |= out[1:, :]
        shifted[1:, :] |= out[:-1, :]
        shifted[:, :-1] |= out[:, 1:]
        shifted[:, 1:] |= out[:, :-1]
        # diagonals
        shifted[:-1, :-1] |= out[1:, 1:]
        shifted[1:, 1:] |= out[:-1, :-1]
        shifted[:-1, 1:] |= out[1:, :-1]
        shifted[1:, :-1] |= out[:-1, 1:]
        out = shifted
    return out


def build_cost_layers(grid):
    """
    Returns (blocked, unknown) boolean arrays for the planner.
      blocked : cells the robot must not enter (inflated walls)
      unknown : never-observed cells (extra traversal cost)
    """
    blocked = inflate(grid.occ_mask(), C.INFLATE_RADIUS_CELLS)
    unknown = grid.unknown_mask()
    return blocked, unknown


# ---------------------------------------------------------------------- #
# A*
# ---------------------------------------------------------------------- #
_NEIGH = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, 1.41421), (-1, 1, 1.41421),
          (1, -1, 1.41421), (1, 1, 1.41421)]


def astar(blocked, unknown, start_rc, goal_rc):
    """
    A* on the grid.  start_rc / goal_rc are (row, col).
    Returns a list of (row, col) from start to goal, or None if unreachable.
    """
    nrows, ncols = blocked.shape
    sr, sc = start_rc
    gr, gc = goal_rc
    if not (0 <= sr < nrows and 0 <= sc < ncols):
        return None
    if not (0 <= gr < nrows and 0 <= gc < ncols):
        return None
    if blocked[gr, gc]:
        return None  # goal sits inside a wall

    def h(r, c):
        # octile distance
        dr = abs(r - gr)
        dc = abs(c - gc)
        return (dr + dc) + (1.41421 - 2) * min(dr, dc)

    open_heap = [(h(sr, sc), 0.0, (sr, sc))]
    came_from = {}
    g_score = {(sr, sc): 0.0}
    closed = set()

    while open_heap:
        _, gc_cur, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == (gr, gc):
            return _reconstruct(came_from, cur)

        r, c = cur
        for dr, dc, step in _NEIGH:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < nrows and 0 <= cc < ncols):
                continue
            if blocked[rr, cc]:
                continue
            if (rr, cc) in closed:
                continue
            cost = step
            if unknown[rr, cc]:
                cost *= C.UNKNOWN_TRAVERSAL_COST
            tentative = gc_cur + cost
            if tentative < g_score.get((rr, cc), math.inf):
                g_score[(rr, cc)] = tentative
                came_from[(rr, cc)] = cur
                heapq.heappush(open_heap, (tentative + h(rr, cc), tentative, (rr, cc)))
    return None


def _reconstruct(came_from, cur):
    path = [cur]
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


# ---------------------------------------------------------------------- #
# Frontier target selection
# ---------------------------------------------------------------------- #
def choose_target(grid, clusters, robot_xy, blocked, unknown):
    """
    Pick the best reachable frontier cluster and return (path_rc, target_cluster).
    path_rc is the A* path in (row,col); None if nothing reachable.

    Cost = path_length_m - INFO_GAIN_WEIGHT * sqrt(cluster_size)
    (shorter & bigger frontiers preferred).  Clusters are tried nearest-first
    by Euclidean distance so we usually stop after a few A* calls.
    """
    rx, ry = robot_xy
    start_rc = (grid.world_to_grid(rx, ry)[1], grid.world_to_grid(rx, ry)[0])

    # Pre-sort clusters by straight-line distance to the robot.
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
        gr, gc = cl["centroid"]
        # Find a nearby non-blocked goal cell (centroid is free, but be safe).
        goal_rc = _nearest_free(blocked, (gr, gc))
        if goal_rc is None:
            continue
        path = astar(blocked, unknown, start_rc, goal_rc)
        tried += 1
        if path is None:
            # Limit how many unreachable far clusters we probe.
            if tried > 12:
                break
            continue
        length_m = (len(path) - 1) * grid.res
        cost = length_m - C.INFO_GAIN_WEIGHT * math.sqrt(cl["size"])
        if cost < best_cost:
            best_cost = cost
            best = cl
            best_path = path
        # Because clusters are nearest-first, once we have a cheap reachable
        # one we can stop probing distant clusters.
        if best is not None and length_m < 1.0:
            break
    return best_path, best


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


def path_to_world(grid, path_rc):
    """Convert an (row,col) path to a list of (x,y) world waypoints."""
    pts = []
    for (r, c) in path_rc:
        x, y = grid.grid_to_world(c, r)
        pts.append((x, y))
    return pts


def path_blocked(grid, path_rc, blocked):
    """True if any cell of the current path is now blocked (needs replanning)."""
    for (r, c) in path_rc:
        if blocked[r, c]:
            return True
    return False
