"""
frontier.py  --  Frontier detection for autonomous exploration.
===============================================================

WHAT IS A "FRONTIER"?
----------------------
A frontier cell is a cell that is:
  1. KNOWN FREE  -- the robot has seen it and it's probably empty, AND
  2. ADJACENT TO UNKNOWN -- it borders at least one cell never observed.

Frontiers form the boundary between explored and unexplored space.
By driving toward a frontier, the robot moves into new territory and
expands its map.

This idea is the foundation of "frontier-based exploration" (Yamauchi,
1997), one of the most widely used autonomous exploration algorithms.

VISUAL EXAMPLE
--------------
  U U U U U U      U = unknown (grey on map)
  U F F F F U      F = frontier = known-free adjacent to unknown
  U F . . F U      . = free interior (not a frontier)
  U F . R F U      R = robot position
  U F . . F U
  U F F F F U
  U U U U U U

The robot starts in the middle where everything was unknown.  After
scanning, the interior cells become free.  The outer ring of free cells
that touch the unknown "wall" of unobserved space are the frontiers.

The robot drives to one frontier, scans more, those cells become free,
new frontiers appear further out, and so on — until the whole maze is
explored.

IMPLEMENTATION
--------------
detect_cells() uses NumPy array operations (shifts) to check all
8 neighbours of every free cell at once.  This is much faster than
looping over every cell individually.

cluster() groups adjacent frontier cells into "clusters" using
Breadth-First Search (BFS).  Each cluster is one continuous stretch of
frontier.  The CENTROID (average position) of a cluster is the target
the planner tries to reach.
"""

from collections import deque

import numpy as np

import Maze4.controllers.Controller_v1.config as C


class FrontierDetector:
    """Stateless detector; all methods operate directly on an OccupancyGrid."""

    # 8-connected neighbourhood offsets: all cells touching a given cell.


    # ---------------------------------------------------------------------- #
    def detect_cells(self, nav, reachable):
        """Return a boolean mask of all frontier cells.

        Uses the 3-value navigation grid from PathPlanner.build_nav_grid().

        A frontier cell is simply:
          - IN the reachable flood-fill area (reachable == True), AND
          - HAS AT LEAST ONE 4-connected neighbour that is UNEXPLORED (nav == 0.5)

        WHY THIS IS CLEAN AND ROBUST
        --------------------------------
        No raw probability thresholds are needed here.  The nav grid encodes
        all the information this function needs:

          nav == 1.0  ->  reachable free space (the robot can stand here)
          nav == 0.5  ->  unexplored  (lidar has never touched this cell)
          nav == 0.0  ->  blocked     (wall or inflation safety buffer)

        The frontier is precisely the boundary between the reachable area
        (1.0) and unexplored space (0.5).  Driving toward any frontier cell
        will reveal new area beyond it.

        The flood fill in build_nav_grid() also removes "orphan" free islands
        — cells that look free but can only be reached by crossing walls.
        So every frontier returned here is actually reachable by the robot.

        Args:
            nav      : float32 array (1.0 / 0.5 / 0.0) from build_nav_grid().
            reachable: bool mask, True at cells the robot can reach.

        Returns:
            np.ndarray bool -- True at every frontier cell.
        """
        unexplored = (nav == 0.5)   # cells the lidar has never observed

        # Shift the unexplored mask in each axis-aligned direction to check
        # whether any 4-connected neighbour of a cell is unexplored.
        up    = np.zeros_like(unexplored)
        down  = np.zeros_like(unexplored)
        left  = np.zeros_like(unexplored)
        right = np.zeros_like(unexplored)
        up  [:-1, :]  = unexplored[1:,  :]   # cell above has unexplored?
        down[1:,  :]  = unexplored[:-1, :]   # cell below has unexplored?
        left[:,  :-1] = unexplored[:,  1:]   # cell to right has unexplored?
        right[:, 1:]  = unexplored[:, :-1]   # cell to left  has unexplored?

        neighbour_unexplored = up | down | left | right

        # Frontier = reachable AND at least one unexplored neighbour.
        return reachable & neighbour_unexplored

    # ---------------------------------------------------------------------- #
    def has_frontiers(self, nav, reachable):
        """Quick check: True if any frontier cells exist."""
        return bool(self.detect_cells(nav, reachable).any())

    # ---------------------------------------------------------------------- #
    @staticmethod
    def cluster(frontier_mask):
        """Group frontier cells into 8-connected clusters using BFS.

        Two frontier cells belong to the same cluster if you can walk
        between them stepping only on frontier cells (including diagonals).

        Small clusters (< FRONTIER_MIN_CELLS) are discarded as noise.

        Args:
            frontier_mask (np.ndarray bool): output of detect_cells().

        Returns:
            List of dicts, each with:
              'cells'    : list of (row, col) tuples in this cluster
              'centroid' : (row, col) of the cluster's geometric centre
                          (snapped to the nearest actual frontier cell)
              'size'     : number of cells in the cluster
        """
        visited = np.zeros_like(frontier_mask, dtype=bool)
        nrows, ncols = frontier_mask.shape
        clusters = []

        # Iterate over all frontier cells; start a new BFS whenever we
        # find one that hasn't been visited yet.
        rows, cols = np.nonzero(frontier_mask)
        for r0, c0 in zip(rows, cols):
            if visited[r0, c0]:
                continue  # already part of an earlier cluster
            cells = FrontierDetector._flood(frontier_mask, visited, r0, c0, nrows, ncols)
            if len(cells) < C.FRONTIER_MIN_CELLS:
                continue  # too small -- probably sensor noise; ignore
            cluster = FrontierDetector._summarise(cells)

            clusters.append(cluster)
        return clusters

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _flood(mask, visited, r0, c0, nrows, ncols):
        """Breadth-First Search to collect all 8-connected cells in one cluster.

        BFS uses a queue (deque).  We start with one seed cell, then
        repeatedly pop a cell, record it, and add all unvisited frontier
        neighbours to the queue.  Continues until the queue is empty,
        meaning we've found the whole connected component.

        Args:
            mask    : frontier boolean array
            visited : boolean array tracking already-processed cells
            r0, c0  : seed cell to start from
            nrows, ncols: grid dimensions for bounds checking

        Returns:
            List of (row, col) tuples belonging to this cluster.
        """
        cells = []
        queue = deque([(r0, c0)])
        visited[r0, c0] = True
        _NEIGH8 = [(-1, -1), (-1, 0), (-1, 1),
               (0,  -1),           (0,  1),
               (1,  -1),  (1, 0),  (1,  1)]

        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            # Explore all 8 neighbours.
            for dr, dc in _NEIGH8:
                rr, cc = r + dr, c + dc
                if (0 <= rr < nrows and 0 <= cc < ncols
                        and mask[rr, cc] and not visited[rr, cc]):
                    visited[rr, cc] = True
                    queue.append((rr, cc))
        return cells

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _summarise(cells):
        """Compute centroid and format a cluster summary dict.

        The centroid is the average (row, col) of all cells.
        We snap it to the NEAREST actual frontier cell so that
        the target is always a valid, reachable grid position.

        Args:
            cells: list of (row, col) tuples

        Returns:
            {'cells': ..., 'centroid': (row, col), 'size': n}
        """
        arr = np.array(cells)           # shape (N, 2)
        cr  = arr[:, 0].mean()          # average row
        cc  = arr[:, 1].mean()          # average col

        # Find the actual frontier cell closest to the computed average.
        d2 = (arr[:, 0] - cr) ** 2 + (arr[:, 1] - cc) ** 2
        sr, sc = arr[int(np.argmin(d2))]  # snapped centroid

        return {
            "cells":    cells,
            "centroid": (int(sr), int(sc)),
            "size":     len(cells),
        }
