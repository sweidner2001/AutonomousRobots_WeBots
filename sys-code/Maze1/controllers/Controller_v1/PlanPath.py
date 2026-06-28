import heapq

class PlanPath:
    def __init__(self, occupancy_grid):
        self.occupancy_grid = occupancy_grid




        
    def astar(self, start, goal, blocked):
        """Find the shortest path from 'start' to 'goal' on a 2-D grid.

        This is the A* search algorithm.  It is like Dijkstra's shortest-path
        algorithm, but accelerated by a heuristic that estimates the remaining
        distance to the goal (Manhattan distance here).

        The algorithm maintains a priority queue of cells sorted by
        f = g + h  where:
            g = actual cost to reach this cell so far
            h = heuristic estimate of cost still remaining (Manhattan distance)

        It expands the cheapest cell first and stops as soon as the goal is
        popped, guaranteeing the optimal path.

        Parameters
        ----------
        start   : (col, row) — starting grid cell.
        goal    : (col, row) — target grid cell.
        blocked : 2-D list of booleans [row][col] — True = obstacle (inflated map).

        Returns
        -------
        List of (col, row) tuples from start to goal (inclusive),
        or an empty list [] if no path exists.
        """
        # Classic A* path planning on a 2D grid.
        h = len(blocked)
        w = len(blocked[0]) if h else 0

        def in_bounds(x, y):
            return 0 <= x < w and 0 <= y < h

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        neigh8 = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, 1.414),
            (-1, 1, 1.414),
            (1, -1, 1.414),
            (-1, -1, 1.414),
        ]

        visited = set()
        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cx, cy = current
            for dx, dy, cost in neigh8:
                nx, ny = cx + dx, cy + dy
                if not in_bounds(nx, ny) or blocked[ny][nx]:
                    continue
                tentative = g_score[current] + cost
                ncell = (nx, ny)
                if tentative < g_score.get(ncell, float("inf")):
                    came_from[ncell] = current
                    g_score[ncell] = tentative
                    f = tentative + heuristic(ncell, goal)
                    heapq.heappush(open_set, (f, ncell))
        return []
