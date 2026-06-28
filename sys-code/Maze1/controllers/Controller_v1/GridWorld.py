from collections import deque
from Maze1.controllers.Controller_v1.Helper import HelperMethods
import math
import cv2
import numpy as np

class GridWorld:
    """2-D grid map of the environment used for path planning and exploration.

    The world is divided into square cells of 'resolution' metres each side.
    Every cell stores one of three states:
        unknown (-1) : the robot has never observed this area.
        free    ( 0) : a lidar ray passed through here → drivable space.
        occupied( 1) : a lidar ray ended here → wall or obstacle.

    This grid is the shared data structure that connects:
        - the mapper (fills cells from lidar rays)
        - BreezySLAM (imports its output bytes here)
        - the path planner (A* reads occupied/free cells)
        - the frontier explorer (looks for unknown cells next to free ones)
    """

    def __init__(self, size_m=12.0, resolution=0.05):
        """Allocate a square grid covering 'size_m' x 'size_m' metres.

        Parameters
        ----------
        size_m     : total side length of the map in metres.
        resolution : side length of one cell in metres (e.g. 0.05 = 5 cm).

        The origin (0, 0) of the world frame is placed at the map centre.
        """
        self.resolution = resolution
        self.width = int(size_m / resolution)   # number of columns
        self.height = int(size_m / resolution)  # number of rows
        self.origin_x = -0.5 * size_m           # world-x of the left edge
        self.origin_y = -0.5 * size_m           # world-y of the bottom edge
        # 2-D list: self.grid[row][col], initialised to 'free' (0).
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.unknown = -1   # cell state: not yet seen
        self.free = 0       # cell state: laser ray passed through
        self.occ = 1        # cell state: laser ray ended here (obstacle)

    def world_to_grid(self, x, y):
        """Convert a real-world position (metres) to grid cell indices.

        Example: x=0.0, y=0.0 → the centre cell of the map.
        Returns (column_index, row_index).  No bounds check here; call
        in_bounds() afterwards when safety matters.
        """
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Convert grid cell indices back to the centre point in metres.

        The '+0.5' offsets the result to the cell centre rather than the
        cell corner, which gives smoother waypoint navigation.
        """
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return x, y

    def in_bounds(self, gx, gy):
        """Return True if (gx, gy) is a valid cell index inside the grid.

        Always call this before reading or writing a cell to avoid
        IndexError when lidar rays point outside the mapped area.
        """
        return 0 <= gx < self.width and 0 <= gy < self.height

    def set_free(self, gx, gy):
        """Mark a cell as free (drivable), but never overwrite an occupied cell.

        A cell that was already seen as occupied (a wall) should not be
        erased by a passing ray — the wall was seen first and is trusted.
        """
        if self.in_bounds(gx, gy):
            if self.grid[gy][gx] <= 0:   # only overwrite unknown or already-free
                self.grid[gy][gx] = self.free

    def set_occ(self, gx, gy):
        """Mark a cell as occupied (obstacle/wall).

        Called for the endpoint of a lidar ray that hit something before
        reaching the sensor's maximum range.
        """
        if self.in_bounds(gx, gy):
            self.grid[gy][gx] = self.occ

    def is_occ(self, gx, gy):
        """Return True if the cell contains a known obstacle.

        Out-of-bounds cells are treated as occupied so the planner
        never plans a path outside the map boundary.
        """
        if not self.in_bounds(gx, gy):
            return True
        return self.grid[gy][gx] == self.occ

    def is_free(self, gx, gy):
        """Return True if the cell is confirmed free (laser ray passed through).

        Unknown cells return False — the robot treats unseen areas as
        'not yet confirmed safe', not as guaranteed free space.
        """
        if not self.in_bounds(gx, gy):
            return False
        return self.grid[gy][gx] == self.free

    def bresenham(self, x0, y0, x1, y1):
        """Return all grid cells on the straight line from (x0,y0) to (x1,y1).

        This is Bresenham's line algorithm — an efficient way to find which
        discrete grid cells a straight lidar ray passes through.

        Used by insert_scan() to:
          - mark all intermediate cells as FREE (the ray passed through them)
          - mark the final cell as OCCUPIED (the ray hit something there)

        Returns a list of (col, row) tuples in order from start to end.
        """
        # Integer line rasterization.
        points = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points

    def insert_scan(self, pose, ranges, angle_min, angle_inc, range_max):
        """Update the occupancy grid using one full lidar scan.

        For each lidar beam:
          1. Compute the absolute direction of the beam in the world frame
             (robot heading + relative beam angle).
          2. Compute the endpoint in world coordinates from the measured range.
          3. Convert start and end to grid cells.
          4. Trace a line between them with Bresenham → mark all intermediate
             cells FREE (the laser physically passed through them).
          5. If the range is below the sensor maximum, the beam hit an object:
             mark the endpoint cell OCCUPIED.

        Parameters
        ----------
        pose      : Pose2D — current robot position and heading.
        ranges    : list of floats — measured distances for each beam (metres).
        angle_min : float — angle of the first beam relative to robot forward.
        angle_inc : float — angular step between consecutive beams.
        range_max : float — sensor maximum range; beams at this value are ignored.
        """
        # Insert one full lidar scan into the occupancy map:
        robot_gx, robot_gy = self.world_to_grid(pose.x, pose.y)
        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            r = HelperMethods.clamp(r, 0.0, range_max)
            beam = pose.theta + angle_min + i * angle_inc
            ex = pose.x + r * math.cos(beam)
            ey = pose.y + r * math.sin(beam)
            end_gx, end_gy = self.world_to_grid(ex, ey)
            ray = self.bresenham(robot_gx, robot_gy, end_gx, end_gy)
            if len(ray) < 2:
                continue
            for px, py in ray[:-1]:
                self.set_free(px, py)
            # Treat near-max returns as no-hit so we only carve free space.
            if r < 0.98 * range_max:
                self.set_occ(ray[-1][0], ray[-1][1])

    def inflated_occupancy(self, radius_cells):
        """Return a boolean 2-D grid where obstacles are expanded outward.

        Why this is necessary:
          The robot has a physical body (~26 cm wide).  If A* plans a path
          right next to a wall, the robot body will collide even though the
          centre-point path is technically clear.
          By 'inflating' every obstacle by radius_cells before planning,
          we force the planner to keep a safe margin from all walls.

        Parameters
        ----------
        radius_cells : int — number of cells to expand each obstacle outward.
                       e.g. INFLATION_RADIUS_M / MAP_RESOLUTION_M = 0.12/0.05 = 2

        Returns
        -------
        A list-of-lists of booleans: True = blocked, False = passable.
        This format is passed directly to astar().
        """
        # Inflate obstacles so planner keeps a safety distance from walls.
        inflated = [[self.is_occ(x, y) for x in range(self.width)] for y in range(self.height)]
        if radius_cells <= 0:
            return inflated
        occ_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if inflated[y][x]:
                    occ_cells.append((x, y))
        for ox, oy in occ_cells:
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy > radius_cells * radius_cells:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if self.in_bounds(nx, ny):
                        inflated[ny][nx] = True
        return inflated

    def nearest_frontier(self, start):
        """Find the nearest exploration frontier to 'start' using BFS.

        A 'frontier' is a free cell that has at least one unknown neighbour.
        It represents the boundary between explored and unexplored space.

        Exploration strategy: always drive to the nearest frontier.
        This gradually expands the explored area until the whole maze is known.

        Parameters
        ----------
        start : (col, row) — current robot cell.

        Returns
        -------
        (col, row) of the nearest frontier, or None if no frontier exists
        (meaning the entire reachable area has been explored).
        """
        # Frontier = free cell next to unknown cell.
        sx, sy = start
        if not self.in_bounds(sx, sy):
            return None

        visited = set()
        q = deque([(sx, sy)])
        visited.add((sx, sy))

        neigh4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            x, y = q.popleft()
            if self.is_frontier(x, y):
                return (x, y)
            for dx, dy in neigh4:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited or not self.in_bounds(nx, ny):
                    continue
                if not self.is_free(nx, ny):
                    continue
                visited.add((nx, ny))
                q.append((nx, ny))
        return None

    def is_frontier(self, x, y):
        """Return True if cell (x, y) qualifies as an exploration frontier.

        Conditions:
          1. The cell itself must be FREE (robot can physically be there).
          2. At least one of the 8 surrounding cells must be UNKNOWN
             (meaning there is unexplored space nearby worth visiting).
        """
        if not self.is_free(x, y):
            return False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny) and self.grid[ny][nx] == self.unknown:
                    return True
        return False

    def import_from_breezyslam_bytes(self, slam_bytes, slam_size_pixels):
        # -----------------------------------------------------------------
        # THIS IS THE CONNECTION BETWEEN BREEZYSLAM AND OUR GRID.
        #
        # BreezySLAM stores its map as a flat byte array (length = pixels^2).
        # Each byte is a value 0-255:
        #   0   = unknown / not observed
        #   < 127 = likely occupied (wall)
        #   >= 127 = likely free (drivable space)
        #
        # We read every byte, threshold it, and write the result into our
        # OccupancyGrid so the planner (A*, frontier search) can use it.
        # Without this method the two worlds (SLAM and planner) are disconnected.
        # -----------------------------------------------------------------

        # BreezySLAM pixel count may differ from our grid size.
        # We compute a scale factor so both grids map to the same real-world area.
        scale = slam_size_pixels / self.width  # e.g. 1.0 if they match

        for gy in range(self.height):
            for gx in range(self.width):
                # Find the corresponding pixel in the SLAM byte array.
                sx = int(gx * scale)
                sy = int(gy * scale)
                sx = HelperMethods.clamp(sx, 0, slam_size_pixels - 1)
                sy = HelperMethods.clamp(sy, 0, slam_size_pixels - 1)

                byte_val = slam_bytes[sy * slam_size_pixels + sx]

                # Translate BreezySLAM byte value into our three-state grid.
                if byte_val == 0:
                    # BreezySLAM uses 0 for "not yet observed".
                    self.grid[gy][gx] = self.unknown
                elif byte_val < 127:
                    # Low value = high obstacle probability → mark occupied.
                    self.grid[gy][gx] = self.occ
                else:
                    # High value = free space the robot can drive through.
                    self.grid[gy][gx] = self.free





class GridWorld2:
    UNKNOWN = 127
    OCCUPIED = 0
    FREE = 255

    def __init__(self, size_pixels, resolution):

        self.size_pixels = size_pixels
        self.resolution = resolution

        self.grid = np.ones(
            (size_pixels, size_pixels),
            dtype=np.uint8
        ) * self.UNKNOWN

        self.inflated = self.grid.copy()


    
    def update(self, mapbytes):
        self.grid = np.array(
            mapbytes,
            dtype=np.uint8
        ).reshape(
            self.size_pixels,
            self.size_pixels
        )



    def world_to_grid(self, x, y):

        gx = int(
            x / self.resolution
            + self.size_pixels / 2
        )

        gy = int(
            y / self.resolution
            + self.size_pixels / 2
        )

        return gx, gy
    
    def grid_to_world(self, gx, gy):

        x = (
            gx
            - self.size_pixels / 2
        ) * self.resolution

        y = (
            gy
            - self.size_pixels / 2
        ) * self.resolution

        return x, y
    



    def inflate(self, radius_m):

        r = int(
            radius_m /
            self.resolution
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2*r+1, 2*r+1)
        )

        occ = (
            self.grid < 100
        ).astype(np.uint8)

        occ = cv2.dilate(
            occ,
            kernel
        )

        self.inflated = np.where(
            occ == 1,
            self.OCCUPIED,
            self.FREE
        )



    def is_free(self, gx, gy):

        if gx < 0:
            return False

        if gy < 0:
            return False

        if gx >= self.size_pixels:
            return False

        if gy >= self.size_pixels:
            return False

        return self.inflated[
            gy,
            gx
        ] > 200
    



class FrontierExplorer:

    def find_frontiers(self, grid_map):

        frontiers = []

        g = grid_map.grid

        h, w = g.shape

        for y in range(1, h-1):
            for x in range(1, w-1):

                if g[y, x] < 200:
                    continue

                neighborhood = g[
                    y-1:y+2,
                    x-1:x+2
                ]

                if np.any(
                    neighborhood == 127
                ):
                    frontiers.append(
                        (x, y)
                    )

        return frontiers
    


    def choose_frontier(self, frontiers, pose, grid_map):

        if not frontiers:
            return None

        gx, gy = grid_map.world_to_grid(
            pose.x,
            pose.y
        )

        best = None
        best_d = 1e9

        for fx, fy in frontiers:

            d = (
                (fx-gx)**2 +
                (fy-gy)**2
            )

            if d < best_d:
                best_d = d
                best = (fx, fy)

        return best