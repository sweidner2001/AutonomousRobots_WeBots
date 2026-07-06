"""
planner.py  --  Grid path planning (A*) and frontier-target selection.
=======================================================================

WHAT DOES THIS FILE DO?
------------------------
Given the occupancy map and a set of frontier clusters, this module:
  1. INFLATES obstacles so that the planned path stays safely away from walls.
  2. Runs A* to find the shortest (cheapest) path from the robot to a target.
  3. Selects the BEST frontier cluster to drive to next.

OBSTACLE INFLATION  (also called "configuration space expansion")
-----------------------------------------------------------------
If we planned paths using the raw obstacle map, the robot might cut
corners and physically touch walls (the robot has a physical radius).

To prevent this, we expand every occupied cell outward by INFLATE_RADIUS_CELLS
in all directions BEFORE planning.  After inflation, any cell that A*
avoids is already far enough from the real wall to fit the robot.

  Real wall cells:    X
  Inflated cells:     i  (robot centre may not enter these)
  Safe free cells:    .  (robot centre can be here)

  . . . . . .
  . . i i i .
  . . i X i .     <- example 1-cell inflation around a wall cell
  . . i i i .
  . . . . . .

A* PATH PLANNING
-----------------
A* ("A-star") is a classic graph search algorithm (Hart, 1968) that
finds the shortest path between two nodes.  On a grid, each cell is a
node, and we can move to any of the 8 neighbours.

A* uses a PRIORITY QUEUE ("min-heap") ordered by:
  f(n) = g(n) + h(n)
    g(n) = exact cost to reach cell n from the start
    h(n) = heuristic estimate of cost from n to goal
           (we use the octile distance, which is exact for 8-connected grids)

By always expanding the cell with lowest f(n), A* is guaranteed to find
the optimal path.

COST FUNCTION
--------------
In a maze some cells are UNKNOWN (never observed).  The robot is allowed
to plan through unknown cells (it needs to, to actually reach a frontier),
but unknown cells are penalised with a cost multiplier UNKNOWN_TRAVERSAL_COST.
This makes A* prefer paths through KNOWN FREE space.

FRONTIER TARGET SELECTION
--------------------------
We have multiple frontier clusters.  Which one should the robot go to?
We use a simple cost function:

  cost = path_length_m - INFO_GAIN_WEIGHT * sqrt(cluster_size)

Lower cost = better target.
  - Nearer targets cost less (shorter path = lower path_length_m).
  - Larger frontier clusters cost less (more unexplored area = higher gain).
INFO_GAIN_WEIGHT controls the tradeoff between distance and information gain.
"""

import collections
import heapq   # Python's built-in min-heap priority queue
import math

import numpy as np
from scipy.ndimage import binary_dilation

import Maze4.controllers.Controller_v1.config as C


class PathPlanner:
    """Provides obstacle inflation, A* planning, and frontier-target selection."""

    # The 8 possible move directions from any grid cell.
    # Format: (delta_row, delta_col, move_cost)
    # Cardinal moves (up/down/left/right) cost 1.0.
    # Diagonal moves cost sqrt(2) ≈ 1.41421 (Pythagorean theorem).
    _NEIGH = [
        (-1,  0, 1.0    ),  # up
        ( 1,  0, 1.0    ),  # down
        ( 0, -1, 1.0    ),  # left
        ( 0,  1, 1.0    ),  # right
        (-1, -1, 1.41421),  # up-left   (diagonal)
        (-1,  1, 1.41421),  # up-right
        ( 1, -1, 1.41421),  # down-left
        ( 1,  1, 1.41421),  # down-right
    ]


    def get_block_cells_for_frontier_exploration(self, grid):
        hazard_blocked = self._inflate(grid.hazard_mask(),    C.HAZARD_INFLATE_CELLS)
        wall_blocked   = self._inflate(grid.occ_mask(),       C.INFLATE_RADIUS_CELLS)
        object_blocked = self._inflate(grid.any_object_mask(), C.OBJECT_INFLATE_CELLS)
        blocked = wall_blocked | hazard_blocked | object_blocked
        return blocked
    
    
    # def get_block_cells_for_target_navigation(self, grid):
    #     """Blocked-cell mask used when navigating TO a tracked object.

    #     Unlike get_block_cells_for_frontier_exploration(), this EXCLUDES
    #     wall cells that are really the object's own surface (see
    #     OccupancyGrid.reconciled_object_mask() for the full "why" -- in
    #     short: the lidar legitimately sees the object too, and if we
    #     didn't exclude those cells, the object's own surface would block
    #     its own destination, making it look unreachable even when the
    #     robot is standing right next to it).
    #     """
    #     hazard_blocked = self._inflate(grid.hazard_mask(), C.HAZARD_INFLATE_CELLS)

    #     object_area = grid.reconciled_object_mask("blue") | grid.reconciled_object_mask("yellow")
    #     nav_occ_mask = grid.occ_mask().copy()
    #     nav_occ_mask[object_area] = False

    #     wall_blocked = self._inflate(nav_occ_mask, C.INFLATE_RADIUS_CELLS)
    #     # Force the object's own surface open even if inflation from a
    #     # nearby, unrelated real wall would otherwise re-cover it.
    #     wall_blocked[object_area] = False

    #     blocked = wall_blocked | hazard_blocked
    #     return blocked
    

    def get_block_cells_for_target_navigation(self, grid):
        hazard_blocked = self._inflate(grid.hazard_mask(),    C.HAZARD_INFLATE_CELLS)

        blue_mask = grid.reconciled_object_mask("blue")
        yellow_mask = grid.reconciled_object_mask("yellow")
        object_area = blue_mask | yellow_mask


        object_mask_for_delete_frontiers = self._inflate(object_area, C.OBJECT_INFLATE_CELLS - 1)
        occ_mask_delete_lidar_scan_to_target = grid.occ_mask() & object_mask_for_delete_frontiers
        nav_to_target_occ_mask = grid.occ_mask().copy()
        nav_to_target_occ_mask[occ_mask_delete_lidar_scan_to_target] = False
        wall_blocked  = self._inflate(nav_to_target_occ_mask, C.INFLATE_RADIUS_CELLS)
        wall_blocked[occ_mask_delete_lidar_scan_to_target] = False

        blocked = wall_blocked | hazard_blocked
        return blocked

    # ---------------------------------------------------------------------- #
    # Navigation grid  (main entry point for planning)
    # ---------------------------------------------------------------------- #
    def build_nav_grid(self, grid, robot_xy, is_for_frontier=True):
        """Build a clean 3-value navigation grid: flood fill + obstacle inflation.

        This is the single source of truth for the planner and the frontier
        detector.  It produces three outputs that together describe the world:

          nav[r,c] = 1.0  ->  reachable FREE space (flood-filled from robot)
          nav[r,c] = 0.0  ->  BLOCKED (wall or inflation safety margin)
          nav[r,c] = 0.5  ->  UNEXPLORED (never seen by lidar)

        WHY FLOOD FILL INSTEAD OF JUST THE FREE MASK?
        -----------------------------------------------
        The raw occupancy grid can contain "orphan" free cells: areas the
        lidar saw but the robot can no longer physically reach (surrounded
        by walls).  Flood fill from the robot discards those automatically.
        Only the free-space blob that contains the robot gets label 1.0.

        GREEN FLOOR HAZARDS AND TRACKED OBJECTS COUNT AS WALLS
        -----------------------------------------------------------
        Cells flagged by the camera as green floor markings (see
        floor_hazard.py / grid.hazard_mask()) OR as a blue/yellow tracked
        object (see colored_objects.py / grid.any_object_mask()) are folded
        into `blocked` exactly like real walls, with the same safety
        inflation margin.  This means A*, the flood fill, and frontier
        detection automatically treat them as impassable -- no other module
        needs to know hazards or objects exist.

        PIPELINE
        ---------
          1. Inflate wall cells, hazard cells, AND object cells.
          2. BFS flood fill from robot position through observed, non-blocked cells.
          3. Assemble: default 0.5, set 0.0 for blocked, set 1.0 for reachable.

        Args:
            grid     : OccupancyGrid -- the current probabilistic map.
            robot_xy : (x, y) world coordinates of the robot (metres).

        Returns:
            nav       (np.ndarray float32) -- 3-value grid described above.
            reachable (np.ndarray bool)    -- True at flood-filled free cells.
            blocked   (np.ndarray bool)    -- True at inflated obstacle cells
                                              (walls, green hazards, and objects).
        """
        # Step 1: inflate obstacles -- walls, hazards, and tracked objects alike.
        if is_for_frontier:
            blocked = self.get_block_cells_for_frontier_exploration(grid)
        else:
            blocked = self.get_block_cells_for_target_navigation(grid)

        # Step 2: flood fill reachable free space from the robot's cell.
        col0, row0 = grid.world_to_grid(*robot_xy)
        reachable  = self._flood_fill_free(
            grid.observed, blocked, row0, col0, grid.log.shape
        )

        # Step 3: assemble the navigation grid.
        #   Start with everything unexplored (0.5).
        #   Mark obstacles (0.0) – includes inflated wall cells.
        #   Mark reachable cells (1.0) – these always "win" over blocked
        #   because the robot is physically present in reachable space.
        nav = np.full(grid.log.shape, 0.5, dtype=np.float32)
        nav[reachable] = 1.0
        nav[blocked]   = 0.0
        
        return nav, reachable, blocked
    

    def get_target_reachablity_mask(self, grid, robot_xy):
        blocked = self.get_block_cells_for_target_navigation(grid)
        col0, row0 = grid.world_to_grid(*robot_xy)
        reachable  = self._flood_fill_free(
            grid.observed, blocked, row0, col0, grid.log.shape
        )
        return reachable

    @staticmethod
    def _flood_fill_free(observed, blocked, start_r, start_c, shape):
        """BFS flood fill through observed, non-blocked cells from a start cell.

        Starting at (start_r, start_c) this spreads in all 8 directions,
        visiting every cell that:
          - has been observed by the lidar at least once  (observed flag is True)
          - is not inside the inflated obstacle zone      (blocked flag is False)

        This gives the set of cells the robot can PHYSICALLY REACH from its
        current position given what the map currently knows.

        If the robot's own cell is inside the inflation zone (e.g., the robot
        is very close to a wall), we search outward up to 8 cells to find the
        nearest valid start cell before flooding.

        Args:
            observed         : bool array, True where lidar has touched a cell.
            blocked          : bool array, True where the robot cannot go.
            start_r, start_c : robot's grid cell (row, col).
            shape            : (nrows, ncols) of the grid.

        Returns:
            np.ndarray bool -- True at every cell reachable from the start.
        """
        nrows, ncols = shape
        reachable    = np.zeros((nrows, ncols), dtype=bool)

        # Clamp starting cell to grid bounds.
        start_r = max(0, min(nrows - 1, start_r))
        start_c = max(0, min(ncols - 1, start_c))

        # If the robot's cell is blocked or unobserved, search outward for
        # a better seed cell (Chebyshev spiral search, same as _nearest_free).
        if blocked[start_r, start_c] or not observed[start_r, start_c]:
            found = False
            for rad in range(1, 9):
                for dr in range(-rad, rad + 1):
                    for dc in range(-rad, rad + 1):
                        if max(abs(dr), abs(dc)) != rad:
                            continue
                        r, c = start_r + dr, start_c + dc
                        if (0 <= r < nrows and 0 <= c < ncols
                                and not blocked[r, c] and observed[r, c]):
                            start_r, start_c = r, c
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                return reachable   # robot completely boxed in

        # BFS: spread to all 8-connected observed, non-blocked neighbours.
        DIRS = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]
        queue = collections.deque([(start_r, start_c)])
        reachable[start_r, start_c] = True

        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if (0 <= rr < nrows and 0 <= cc < ncols
                        and not reachable[rr, cc]
                        and not blocked[rr, cc]
                        and observed[rr, cc]):
                    reachable[rr, cc] = True
                    queue.append((rr, cc))

        return reachable

    @staticmethod
    def _inflate(occ_mask, radius_cells):
        """Expand every True cell outward by `radius_cells` in all 8 directions.

        Uses scipy binary_dilation with a 3×3 structuring element repeated
        `radius_cells` times, which is equivalent to growing each occupied
        cell by exactly one cell per iteration in all 8 directions.

        Args:
            occ_mask (np.ndarray bool): raw occupied-cell mask from the grid.
            radius_cells (int):         how many cells to grow in each direction.

        Returns:
            np.ndarray bool: inflated obstacle mask.
        """
        if radius_cells <= 0:
            return occ_mask.copy()
        # struct = np.ones((3, 3), dtype=bool)  # 8-connected structuring element
        struct = np.ones((1 + 2*radius_cells, 1 + 2*radius_cells), dtype=bool)  # 8-connected structuring element
        # return binary_dilation(occ_mask, structure=struct, iterations=radius_cells)
        return binary_dilation(occ_mask, structure=struct, iterations=1)

    # ---------------------------------------------------------------------- #
    # A* path search
    # ---------------------------------------------------------------------- #
    def astar(self, blocked, unknown, start_rc, goal_rc):
        """Find the cheapest path from start_rc to goal_rc on the grid.

        Args:
            blocked   (np.ndarray bool): cells the robot cannot enter.
            unknown   (np.ndarray bool): cells penalised by UNKNOWN_TRAVERSAL_COST.
            start_rc  (tuple): (row, col) of the starting cell.
            goal_rc   (tuple): (row, col) of the goal cell.

        Returns:
            List of (row, col) tuples from start to goal (inclusive),
            or None if no path exists.
        """
        nrows, ncols = blocked.shape
        sr, sc = start_rc
        gr, gc = goal_rc

        # --- Sanity checks -------------------------------------------------
        # Both start and goal must lie inside the grid.
        if not (0 <= sr < nrows and 0 <= sc < ncols):
            return None
        if not (0 <= gr < nrows and 0 <= gc < ncols):
            return None
        # Goal must not be a blocked (wall) cell.
        if blocked[gr, gc]:
            return None
        # Start must not be a blocked cell either.
        # This can happen when the robot is very close to a wall and the
        # inflation zone covers the robot's own grid cell.
        if blocked[sr, sc]:
            return None

        # --- Octile heuristic: exact lower bound for 8-connected grids ----
        def h(r, c):
            """Octile distance from (r,c) to goal.

            Allows diagonal moves at cost sqrt(2).
            For the Manhattan remainder, straight moves cost 1.0.
            Formula: (D + D_diag) * 1.0 + min(dr,dc) * (sqrt(2) - 2 * 1.0)
                   = dr + dc + min(dr,dc) * (sqrt(2) - 2)
            """
            dr, dc = abs(r - gr), abs(c - gc)
            return (dr + dc) + (1.41421 - 2) * min(dr, dc)

        # --- Priority queue (min-heap) ------------------------------------
        # Each entry: (f_cost, g_cost, (row, col))
        # Python's heapq is a min-heap, so the lowest f_cost pops first.
        open_heap = [(h(sr, sc), 0.0, (sr, sc))]
        came_from = {}                    # cell -> parent cell (for path reconstruction)
        g_score   = {(sr, sc): 0.0}      # best-known cost to reach each cell
        closed    = set()                 # cells already finalized

        while open_heap:
            _, g_cur, cur = heapq.heappop(open_heap)

            # Skip if this cell was already finalised with a better cost.
            if cur in closed:
                continue
            closed.add(cur)

            # Goal reached -> reconstruct and return the path.
            if cur == (gr, gc):
                return self._reconstruct(came_from, cur)

            r, c = cur
            for dr, dc, step in self._NEIGH:
                rr, cc = r + dr, c + dc
                # Skip out-of-bounds, blocked, or already-closed cells.
                if not (0 <= rr < nrows and 0 <= cc < ncols):
                    continue
                if blocked[rr, cc] or (rr, cc) in closed:
                    continue

                # Unknown cells are more expensive to cross.
                cost = step * (C.UNKNOWN_TRAVERSAL_COST if unknown[rr, cc] else 1.0)
                tentative_g = g_cur + cost

                if tentative_g < g_score.get((rr, cc), math.inf):
                    # This path to (rr, cc) is better than anything seen before.
                    g_score[(rr, cc)]  = tentative_g
                    came_from[(rr, cc)] = cur
                    heapq.heappush(
                        open_heap,
                        (tentative_g + h(rr, cc), tentative_g, (rr, cc))
                    )

        return None  # no path exists

    @staticmethod
    def _reconstruct(came_from, cur):
        """Trace back through came_from to build the path from start to goal.

        Starting at the goal, follow parent pointers until we reach the
        start (which has no entry in came_from).  Reverse the result.
        """
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    # ---------------------------------------------------------------------- #
    # Frontier target selection
    # ---------------------------------------------------------------------- #
    def choose_target(self, grid, clusters, robot_xy, blocked, unknown):
        """Pick the best reachable frontier cluster for the robot to drive to.

        Tries clusters from nearest to farthest (to minimise the number of
        A* calls before finding a good option).  Stops early once we find
        a very close cluster.

        Args:
            grid      : OccupancyGrid (for coordinate conversion).
            clusters  : list of frontier cluster dicts from FrontierDetector.
            robot_xy  : (x, y) of the robot in world coordinates (m).
            blocked   : inflated obstacle mask (from build_cost_layers).
            unknown   : unobserved cell mask.

        Returns:
            (path_rc, cluster) where:
              path_rc  : list of (row, col) tuples from robot to frontier,
                         or None if no reachable frontier was found.
              cluster  : the chosen cluster dict, or None.
        """
        rx, ry = robot_xy

        # Convert robot world position to grid cell.
        sc0, sr0 = grid.world_to_grid(rx, ry)
        start_rc = (sr0, sc0)

        # If the robot's cell is inside the inflation zone (blocked), find
        # the nearest non-blocked cell to use as the A* start point.
        # This can happen when the robot is very close to a wall and the
        # inflation margin covers the robot's own cell.
        if (0 <= sr0 < blocked.shape[0] and 0 <= sc0 < blocked.shape[1]
                and blocked[sr0, sc0]):
            alt = self._nearest_free(blocked, start_rc, max_r=5)
            if alt is None:
                return None, None  # robot completely surrounded -> give up
            start_rc = alt

        # Sort clusters by straight-line distance (nearest first).
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
            # Find a non-blocked cell near the frontier centroid for A* goal.
            goal_rc = self._nearest_free(blocked, cl["centroid"])
            if goal_rc is None:
                continue  # centroid and surroundings all blocked; skip

            path = self.astar(blocked, unknown, start_rc, goal_rc)
            tried += 1
            if path is None:
                if tried > 30:
                    break  # tried enough; no reachable frontier found
                continue

            # Compute the combined cost (distance penalty, information gain).
            length_m = (len(path) - 1) * grid.res
            cost = length_m - C.INFO_GAIN_WEIGHT * math.sqrt(cl["size"])

            if cost < best_cost:
                best_cost = cost
                best      = cl
                best_path = path

            # If the best path found so far is very short, take it immediately.
            if best is not None and length_m < 1.0:
                break

        return best_path, best

    # ---------------------------------------------------------------------- #
    # Fixed-point planning (used to drive to a known target, e.g. a tracked
    # blue/yellow object -- see explorer.py: GoToPoint)
    # ---------------------------------------------------------------------- #
    def plan_path_to(self, grid, blocked, unknown, start_xy, goal_xy):
        """Plan an A* path from start_xy to goal_xy, both given as world
        coordinates.  Unlike choose_target(), the goal is already known --
        no frontier clustering or candidate scoring is needed.

        Snaps both the start and the goal to the nearest non-blocked cell
        if they land inside the inflation margin (same fallback used by
        choose_target() when the robot is close to a wall).

        Args:
            grid     : OccupancyGrid (for coordinate conversion).
            blocked  : inflated obstacle mask (from build_nav_grid).
            unknown  : unobserved cell mask (from build_nav_grid).
            start_xy : (x, y) world coordinates of the start (usually the robot).
            goal_xy  : (x, y) world coordinates of the target.

        Returns:
            List of (row, col) grid cells from start to goal, or None if no
            path could be found (goal unreachable, or robot/goal boxed in).
        """
        # Robot start point mapping:
        sc0, sr0 = grid.world_to_grid(*start_xy)
        start_rc = (sr0, sc0)
        if (0 <= sr0 < blocked.shape[0] and 0 <= sc0 < blocked.shape[1]
                and blocked[sr0, sc0]):
            alt = self._nearest_free(blocked, start_rc, max_r=5)
            if alt is None:
                return None
            start_rc = alt


        # Goal point mapping:
        gc0, gr0 = grid.world_to_grid(*goal_xy)
        goal_rc = self._nearest_free(blocked, (gr0, gc0))
        if goal_rc is None:
            return None

        return self.astar(blocked, unknown, start_rc, goal_rc)

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _nearest_free(blocked, rc, max_r=4):
        """Find the nearest non-blocked cell to rc, searching outward.

        Searches in a square spiral around rc, checking the outermost
        "shell" of each radius in turn (Chebyshev distance).

        Args:
            blocked : blocked cell mask.
            rc      : (row, col) centre to search around.
            max_r   : maximum search radius in cells.

        Returns:
            (row, col) of the nearest non-blocked cell, or None if none
            found within max_r.
        """
        r0, c0 = rc
        nrows, ncols = blocked.shape
        for rad in range(max_r + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue  # only check the outermost shell at each radius
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < nrows and 0 <= c < ncols and not blocked[r, c]:
                        return (r, c)
        return None

    # ---------------------------------------------------------------------- #
    # Utility helpers used by the explorer
    # ---------------------------------------------------------------------- #
    @staticmethod
    def path_to_world(grid, path_rc):
        """Convert a list of (row, col) grid cells to world (x, y) points.

        Returns a list of (x, y) tuples (cell centres in metres).
        """
        return [grid.grid_to_world(c, r) for (r, c) in path_rc]

    @staticmethod
    def path_blocked(path_rc, blocked):
        """True if ANY cell in the planned path has become blocked.

        Used to detect when a previously planned path has been
        invalidated by new obstacle observations.
        """
        return any(blocked[r, c] for (r, c) in path_rc)
