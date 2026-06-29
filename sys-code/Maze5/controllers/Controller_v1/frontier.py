"""
frontier.py
===========
Frontier detection for autonomous exploration.

A *frontier cell* is a known-free cell that touches at least one unknown cell.
Frontier cells are grouped into connected clusters; each cluster is a candidate
exploration target.  The planner then decides which one to drive to.

Everything works on the OccupancyGrid masks (free / unknown).
"""

from collections import deque

import numpy as np

import Maze5.controllers.Controller_v1.config as C


def detect_frontier_cells(grid):
    """Return a boolean array marking free cells adjacent to unknown space."""
    free = grid.free_mask()
    unknown = grid.unknown_mask()

    # A free cell is a frontier if any 4-neighbour is unknown.
    # Shift the unknown mask in the four directions and AND with free.
    up = np.zeros_like(unknown)
    down = np.zeros_like(unknown)
    left = np.zeros_like(unknown)
    right = np.zeros_like(unknown)
    up[:-1, :] = unknown[1:, :]
    down[1:, :] = unknown[:-1, :]
    left[:, :-1] = unknown[:, 1:]
    right[:, 1:] = unknown[:, :-1]
    neighbour_unknown = up | down | left | right
    return free & neighbour_unknown


def cluster_frontiers(frontier_mask):
    """
    Group frontier cells into 8-connected clusters.

    Returns a list of clusters, each a dict:
        {'cells': [(row,col), ...], 'centroid': (row,col), 'size': n}
    """
    visited = np.zeros_like(frontier_mask, dtype=bool)
    nrows, ncols = frontier_mask.shape
    clusters = []
    neigh = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),          (0, 1),
             (1, -1),  (1, 0),  (1, 1)]

    rows, cols = np.nonzero(frontier_mask)
    for r0, c0 in zip(rows, cols):
        if visited[r0, c0]:
            continue
        # BFS flood fill.
        cells = []
        q = deque()
        q.append((r0, c0))
        visited[r0, c0] = True
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in neigh:
                rr, cc = r + dr, c + dc
                if 0 <= rr < nrows and 0 <= cc < ncols \
                        and frontier_mask[rr, cc] and not visited[rr, cc]:
                    visited[rr, cc] = True
                    q.append((rr, cc))

        if len(cells) < C.FRONTIER_MIN_CELLS:
            continue
        arr = np.array(cells)
        cr = int(round(arr[:, 0].mean()))
        cc = int(round(arr[:, 1].mean()))
        # Snap centroid to the nearest actual frontier cell (centroid may
        # land on a wall or unknown cell).
        d2 = (arr[:, 0] - cr) ** 2 + (arr[:, 1] - cc) ** 2
        cr, cc = arr[int(np.argmin(d2))]
        clusters.append({
            "cells": cells,
            "centroid": (int(cr), int(cc)),
            "size": len(cells),
        })
    return clusters


def has_frontiers(grid):
    return bool(detect_frontier_cells(grid).any())
