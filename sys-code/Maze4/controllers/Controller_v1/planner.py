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

    def __init__(self):
        # Snapshot of the most recent A* search for visual debugging.
        self.last_astar_debug = None

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
        # Low, lidar-blind obstacles live in their own camera-only map (see
        # occupancy_grid.integrate_camera_obstacle) -- fold them in like walls.
        camera_obs_blocked = self._inflate(grid.camera_obstacle_mask(), C.INFLATE_RADIUS_CELLS)
        blocked = wall_blocked | hazard_blocked | object_blocked | camera_obs_blocked
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
    

    def get_block_cells_for_target_navigation(self, grid, 
                                              hazard_inflate_cells=C.HAZARD_INFLATE_CELLS, 
                                              object_inflate_cells=C.OBJECT_INFLATE_CELLS, 
                                              inflate_radius_cells=C.INFLATE_RADIUS_CELLS):
        hazard_blocked = self._inflate(grid.hazard_mask(),    hazard_inflate_cells)

        blue_mask = grid.reconciled_object_mask("blue")
        yellow_mask = grid.reconciled_object_mask("yellow")
        object_area = blue_mask | yellow_mask


        object_mask_for_delete_frontiers = self._inflate(object_area, object_inflate_cells - 1)
        occ_mask_delete_lidar_scan_to_target = grid.occ_mask() & object_mask_for_delete_frontiers
        nav_to_target_occ_mask = grid.occ_mask().copy()
        nav_to_target_occ_mask[occ_mask_delete_lidar_scan_to_target] = False
        wall_blocked  = self._inflate(nav_to_target_occ_mask, inflate_radius_cells)
        wall_blocked[occ_mask_delete_lidar_scan_to_target] = False

        # Low, lidar-blind obstacles (camera-only map) block the target route
        # too -- same as walls (see occupancy_grid.integrate_camera_obstacle).
        camera_obs_blocked = self._inflate(grid.camera_obstacle_mask(), inflate_radius_cells)

        blocked = wall_blocked | hazard_blocked | camera_obs_blocked
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

        def _store_debug(success, reason, came_from=None, cur=None, open_heap=None, closed=None):
            if C.VIZALIZATION_NAV_GRID is False:
                return 
            
            closed_mask = np.zeros(blocked.shape, dtype=bool)
            if closed:
                for (r, c) in closed:
                    closed_mask[r, c] = True

            open_mask = np.zeros(blocked.shape, dtype=bool)
            if open_heap:
                for _, __, (r, c) in open_heap:
                    open_mask[r, c] = True

            path_mask = np.zeros(blocked.shape, dtype=bool)
            if success and came_from is not None and cur is not None:
                path = self._reconstruct(came_from, cur)
                for (r, c) in path:
                    path_mask[r, c] = True

            self.last_astar_debug = {
                "success": bool(success),
                "reason": reason,
                "start": (sr, sc),
                "goal": (gr, gc),
                "blocked": blocked.copy(),
                "unknown": unknown.copy(),
                "closed": closed_mask,
                "open": open_mask,
                "path": path_mask,
            }

        # --- Sanity checks -------------------------------------------------
        # Both start and goal must lie inside the grid.
        if not (0 <= sr < nrows and 0 <= sc < ncols):
            _store_debug(False, "start_out_of_bounds")
            return None
        if not (0 <= gr < nrows and 0 <= gc < ncols):
            _store_debug(False, "goal_out_of_bounds")
            return None
        # Goal must not be a blocked (wall) cell.
        if blocked[gr, gc]:
            _store_debug(False, "goal_blocked")
            return None
        # Start must not be a blocked cell either.
        # This can happen when the robot is very close to a wall and the
        # inflation zone covers the robot's own grid cell.
        if blocked[sr, sc]:
            _store_debug(False, "start_blocked")
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
                _store_debug(True, "ok", came_from=came_from, cur=cur, open_heap=open_heap, closed=closed)
                return self._reconstruct(came_from, cur)

            r, c = cur
            for dr, dc, step in self._NEIGH:
                rr, cc = r + dr, c + dc
                # Skip out-of-bounds, blocked, or already-closed cells.
                if not (0 <= rr < nrows and 0 <= cc < ncols):
                    continue
                if blocked[rr, cc] or (rr, cc) in closed or unknown[rr, cc]:
                    continue

                # cutting no corners
                # if dr != 0 and dc != 0:
                #     if blocked[r + dr, c] or blocked[r, c + dc]:
                #         continue

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

        _store_debug(False, "no_path", open_heap=open_heap, closed=closed)
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

            if cost < best_cost and length_m > C.LIDAR_MIN_RANGE*2:
                best_cost = cost
                best      = cl
                best_path = path

            # If the best path found so far is very short, take it immediately.
            if best is not None and length_m < 1.0 and length_m > C.LIDAR_MIN_RANGE*2:
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
    # Fallback: get as CLOSE AS SAFELY POSSIBLE to a sealed-off target
    # ---------------------------------------------------------------------- #
    def _raw_blocked_for_target(self, grid):
        """Obstacle mask for the physical route to a target -- real walls and
        hazards, but WITHOUT the inflation safety margin, and with the
        object's own surface carved out (same carve-out as
        get_block_cells_for_target_navigation()).

        This is the "what is truly impassable" view: a maze corridor that is
        narrower than the robot's inflation margin is still PHYSICALLY open,
        so it stays free here even though the normal (inflated) planner walls
        it off.  Used only to trace where a route to the target would run --
        the robot itself always drives the inflated, safe path.
        """
        blue_mask   = grid.reconciled_object_mask("blue")
        yellow_mask = grid.reconciled_object_mask("yellow")
        object_area = blue_mask | yellow_mask

        # Carve the object's own lidar-wall surface out of the wall map, so
        # the object cannot wall off its own destination.
        obj_surface = grid.occ_mask() & self._inflate(
            object_area, C.OBJECT_INFLATE_CELLS - 1)
        walls = grid.occ_mask().copy()
        walls[obj_surface] = False

        return walls | grid.hazard_mask()

    def plan_path_near_blocked_target(self, grid, start_xy, goal_xy):
        """Plan a SAFE path that ends as close to `goal_xy` as the inflation
        margin allows, for when a direct plan to the target failed.

        WHY THIS IS NEEDED
        ------------------
        There is always a blue and a yellow object in the world, and a
        physical route to each always exists.  But the object often sits
        against a wall in a gap narrower than the robot's safety margin, so
        the normal (inflated) planner reports "no path" -- the inflation
        seals the approach even though the robot could physically squeeze in.

        HOW IT WORKS (two reachable areas)
        ----------------------------------
          1. RAW route (area 1): A* on the UN-inflated obstacle map, which
             still reaches the object because no safety margin seals the gap.
          2. SAFE area (area 2): the normal inflated flood-fill reachable set
             -- every cell the robot can reach WITHOUT clipping a wall.
          3. Walk the raw route from the object end back toward the robot and
             take the FIRST cell that is still inside the safe area.  That is
             the point where the physical route crosses out of the blocked
             (inflated) zone -- the closest the robot can safely park to the
             object.  OBJECT_REACH_TOL then covers the last short gap.

        Returns:
            List of (row, col) cells of a SAFE path to that closest point, or
            None if even the raw physical route does not exist (should not
            happen for a real object, but handled defensively).
        """
        unknown = ~grid.observed

        sc0, sr0 = grid.world_to_grid(*start_xy)
        start_rc = (sr0, sc0)

        # --- Step 1: raw physical route to the object (no inflation) --------
        
        for i in range(min(C.HAZARD_INFLATE_CELLS, C.OBJECT_INFLATE_CELLS, C.INFLATE_RADIUS_CELLS)):
            raw_blocked = self.get_block_cells_for_target_navigation(grid, C.HAZARD_INFLATE_CELLS - i, 
                                                                 C.OBJECT_INFLATE_CELLS - i, 
                                                                 C.INFLATE_RADIUS_CELLS - i)


            # raw_blocked = self._raw_blocked_for_target(grid)
            raw_start = start_rc
            if raw_blocked[sr0, sc0]:
                raw_start = self._nearest_free(raw_blocked, start_rc, max_r=5)
                if raw_start is None:
                    return None
            
            gc0, gr0 = grid.world_to_grid(*goal_xy)
            raw_goal = self._nearest_free(raw_blocked, (gr0, gc0))
            if raw_goal is None:
                return None
            raw_path = self.astar(raw_blocked, unknown, raw_start, raw_goal)
            if raw_path is not None:
                break
                # return None   # not even a physical route -- give up

        if raw_path is None:
            return None

        # --- Step 2: the safe (inflated) reachable set from the robot -------
        safe_blocked   = self.get_block_cells_for_target_navigation(grid)
        safe_reachable = self._flood_fill_free(
            grid.observed, safe_blocked, sr0, sc0, grid.log.shape)

        # --- Step 3: overlap of (inflated raw route) and safe reachable ----
        # Build a "corridor" around the physical raw route, then intersect
        # with safe-reachable cells. This avoids requiring an exact cell-wise
        # match between raw_path and safe_reachable.
        raw_mask = np.zeros(grid.log.shape, dtype=bool)
        for (r, c) in raw_path:
            raw_mask[r, c] = True

        path_radius = 3
        path_struct = np.ones((1 + 2 * path_radius, 1 + 2 * path_radius), dtype=bool)
        raw_corridor = binary_dilation(raw_mask, structure=path_struct, iterations=1)
        overlap_mask = raw_corridor & safe_reachable

        if not np.any(overlap_mask):
            return None

        # The overlap can be 2-3 cells thick. Pick one representative target
        # by maximising progress along raw_path (furthest toward the object).
        raw_index = np.full(grid.log.shape, -1, dtype=np.int32)
        for i, (r, c) in enumerate(raw_path):
            raw_index[r, c] = i

        target_rc = None
        best_idx = -1
        rows, cols = np.where(overlap_mask)
        for r, c in zip(rows, cols):
            r0 = max(0, r - path_radius)
            r1 = min(raw_index.shape[0], r + path_radius + 1)
            c0 = max(0, c - path_radius)
            c1 = min(raw_index.shape[1], c + path_radius + 1)
            local_max = int(np.max(raw_index[r0:r1, c0:c1]))
            if local_max > best_idx:
                best_idx = local_max
                target_rc = (r, c)

        if target_rc is None:
            return None

        # --- Step 4: a normal SAFE path to that closest reachable point -----
        target_xy = grid.grid_to_world(target_rc[1], target_rc[0])
        return self.plan_path_to(
            grid, safe_blocked, unknown, start_xy, target_xy)

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
