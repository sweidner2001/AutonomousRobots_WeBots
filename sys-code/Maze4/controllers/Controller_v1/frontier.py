"""
frontier.py
===========
Frontier detection for autonomous exploration.

A *frontier cell* is a known-free cell that touches at least one unknown cell.
Frontier cells are grouped into connected clusters; each cluster is a candidate
exploration target.  The planner then decides which one to drive to.
"""

from collections import deque

import numpy as np

import Maze5.controllers.Controller_v1.config as C


class FrontierDetector:
    """Stateless detector; operates on an OccupancyGrid's masks."""

    _NEIGH8 = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    def detect_cells(self, grid):
        """Boolean array marking free cells adjacent to unknown space."""
        free = grid.free_mask()
        unknown = grid.unknown_mask()

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

    def has_frontiers(self, grid):
        return bool(self.detect_cells(grid).any())

    def cluster(self, frontier_mask):
        """Group frontier cells into 8-connected clusters.

        Returns a list of dicts:
            {'cells': [(row,col),...], 'centroid': (row,col), 'size': n}
        """
        visited = np.zeros_like(frontier_mask, dtype=bool)
        nrows, ncols = frontier_mask.shape
        clusters = []

        rows, cols = np.nonzero(frontier_mask)
        for r0, c0 in zip(rows, cols):
            if visited[r0, c0]:
                continue
            cells = self._flood(frontier_mask, visited, r0, c0, nrows, ncols)
            if len(cells) < C.FRONTIER_MIN_CELLS:
                continue
            clusters.append(self._summarise(cells))
        return clusters

    def _flood(self, mask, visited, r0, c0, nrows, ncols):
        cells = []
        q = deque([(r0, c0)])
        visited[r0, c0] = True
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in self._NEIGH8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < nrows and 0 <= cc < ncols \
                        and mask[rr, cc] and not visited[rr, cc]:
                    visited[rr, cc] = True
                    q.append((rr, cc))
        return cells

    @staticmethod
    def _summarise(cells):
        arr = np.array(cells)
        cr = arr[:, 0].mean()
        cc = arr[:, 1].mean()
        # Snap the centroid to the nearest actual frontier cell.
        d2 = (arr[:, 0] - cr) ** 2 + (arr[:, 1] - cc) ** 2
        sr, sc = arr[int(np.argmin(d2))]
        return {"cells": cells, "centroid": (int(sr), int(sc)), "size": len(cells)}
