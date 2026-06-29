import numpy as np

from Maze1.controllers.Controller_v5.utils import normalize_angle

class Exploration:
    EXPLORING = 'exploring'
    TURNING = 'turning'
    MOVING_TO_FRONTIER = 'moving_to_frontier'

    def __init__(self, slam_engine, map_resolution=0.05):
        self.slam = slam_engine
        self.map_resolution = map_resolution
        self.state = self.EXPLORING
        self.target_frontier = None
        self.frontier_history = []
        self.rotation_counter = 0
        
        # Movement parameters
        self.linear_speed = 1.0
        self.angular_speed = 1.5
        self.safe_distance = 0.4  # meters
        self.frontier_distance_threshold = 2.0  # meters
        
    def compute_frontiers(self, map_data, robot_pose):
        """
        Find frontier cells (boundaries between known and unknown space).
        Returns list of frontier positions in world coordinates.
        """
        map_size = map_data.shape[0]
        
        # Identify unknown cells (value 0) and known cells (value > 0.1)
        unknown = map_data < 0.01
        known = map_data > 0.1
        
        # Find frontiers: unknown cells adjacent to known cells
        frontiers = []
        for y in range(1, map_size - 1):
            for x in range(1, map_size - 1):
                if unknown[y, x]:
                    # Check neighbors
                    neighbors = [
                        known[y-1, x], known[y+1, x],
                        known[y, x-1], known[y, x+1]
                    ]
                    if any(neighbors):
                        frontiers.append((x, y))
        
        if len(frontiers) == 0:
            return []
        
        # Convert to world coordinates
        map_origin = np.array([-map_size/2, -map_size/2]) * self.map_resolution
        world_frontiers = []
        for fx, fy in frontiers:
            wx = fx * self.map_resolution + map_origin[0]
            wy = fy * self.map_resolution + map_origin[1]
            world_frontiers.append((wx, wy))
        
        return world_frontiers
    
    def select_frontier(self, frontiers, robot_pose):
        """
        Select the best frontier to explore.
        Prioritizes frontiers that are reachable and not recently visited.
        """
        if len(frontiers) == 0:
            return None
        
        robot_pos = robot_pose[:2]
        
        # Filter frontiers that are too close (already explored)
        frontiers = [f for f in frontiers 
                     if np.linalg.norm(np.array(f) - robot_pos) > 0.5]
        
        if len(frontiers) == 0:
            return None
        
        # Select the closest frontier
        distances = [np.linalg.norm(np.array(f) - robot_pos) for f in frontiers]
        best_idx = np.argmin(distances)
        
        return frontiers[best_idx]
    
    def get_control(self, lidar_ranges, lidar_angles):
        """
        Get motor commands based on exploration state.
        Returns (left_velocity, right_velocity)
        """
        map_data = self.slam.get_map()
        robot_pose = self.slam.get_pose()
        
        min_distance = float(np.min(lidar_ranges)) if len(lidar_ranges) > 0 else 10.0
        
        if min_distance < self.safe_distance:
            self.state = self.TURNING
            self.rotation_counter += 1
            if self.rotation_counter < 20:
                return -self.angular_speed * 0.5, self.angular_speed * 0.5
            self.state = self.EXPLORING
            self.rotation_counter = 0
        
        if self.state == self.EXPLORING:
            frontiers = self.compute_frontiers(map_data, robot_pose)
            
            if len(frontiers) > 0:
                target = self.select_frontier(frontiers, robot_pose)
                if target is not None:
                    self.target_frontier = target
                    self.state = self.MOVING_TO_FRONTIER
        
        if self.state == self.MOVING_TO_FRONTIER and self.target_frontier is not None:
            dx = self.target_frontier[0] - robot_pose[0]
            dy = self.target_frontier[1] - robot_pose[1]
            target_angle = np.arctan2(dy, dx)
            angle_error = normalize_angle(target_angle - robot_pose[2])
            
            distance = np.sqrt(dx**2 + dy**2)
            if distance < self.frontier_distance_threshold:
                self.state = self.EXPLORING
                self.target_frontier = None
                return self.linear_speed * 0.5, self.linear_speed * 0.5
            
            if abs(angle_error) > 0.1:
                if angle_error > 0:
                    return -self.angular_speed * 0.3, self.angular_speed * 0.3
                return self.angular_speed * 0.3, -self.angular_speed * 0.3

            return self.linear_speed * 0.7, self.linear_speed * 0.7
        
        return self.linear_speed * 0.3, self.linear_speed * 0.3

