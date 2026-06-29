import numpy as np
from scipy.spatial import KDTree
from graphslam.graph import Graph
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se2 import PoseSE2
from graphslam.vertex import Vertex
from Maze1.controllers.Controller_v5.utils import compute_relative_pose
from Maze1.controllers.Controller_v5.utils import transform_points
from Maze1.controllers.Controller_v5.utils import normalize_angle
from Maze1.controllers.Controller_v5.utils import compute_odometry

class SLAMEngine:
    def __init__(self, map_resolution=0.05, map_size_m=20):
        """
        Initialize SLAM engine.
        map_resolution: meters per pixel
        map_size_m: size of the square map in meters
        """
        self.map_resolution = map_resolution
        self.map_size = int(map_size_m / map_resolution)
        self.map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        # Pose graph state
        self.graph = None
        self.graph_vertices = []
        self.graph_edges = []
        self.vertices = []  # [{'id', 'pose', 'scan', 'timestamp'}, ...]
        self.current_vertex_id = 0
        
        # ICP parameters
        self.icp_max_iterations = 30
        self.icp_tolerance = 1e-6
        
        # Loop closure parameters
        self.loop_closure_distance_threshold = 1.5  # meters
        self.loop_closure_time_threshold = 30  # seconds between checks
        
        # Robot state
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.last_scan = None
        self.last_scan_pose = np.array([0.0, 0.0, 0.0])
        
        # Trajectory for visualization
        self.trajectory = []
        
    def add_scan(self, scan_points, timestamp, wheel_vel_left=0, wheel_vel_right=0, dt=0.032):
        """
        Add a new LiDAR scan with odometry information.
        Returns: (pose_estimate, updated_map)
        """
        if len(scan_points) < 10:
            return self.robot_pose, self.map
        
        # Compute odometry from wheel velocities.
        odom_delta = compute_odometry(wheel_vel_left, wheel_vel_right, dt)
        
        # ICP scan matching for better odometry
        if self.last_scan is not None and len(self.last_scan) > 10:
            icp_pose = self.icp_match(self.last_scan, scan_points, self.last_scan_pose)
            # Combine ICP with odometry (simple fusion)
            alpha = 0.7  # Trust ICP more
            relative_pose = alpha * icp_pose + (1 - alpha) * odom_delta
        else:
            relative_pose = odom_delta
        
        # Update robot pose.
        new_pose = self.robot_pose + relative_pose
        new_pose[2] = normalize_angle(new_pose[2])

        # Guard against numeric instability from scan matching.
        if np.any(np.isnan(new_pose)):
            print("Warning: NaN detected in new_pose. Skipping this scan update.")
            return self.robot_pose, self.map

        self.robot_pose = new_pose
        
        # Add vertex to pose graph.
        vertex_id = self.current_vertex_id
        vertex = Vertex(vertex_id, PoseSE2([new_pose[0], new_pose[1]], new_pose[2]))
        self.graph_vertices.append(vertex)
        
        # Add odometry edge from previous vertex.
        if self.current_vertex_id > 0:
            prev_pose = self.vertices[-1]['pose']
            rel_pose = compute_relative_pose(prev_pose, new_pose)
            edge = EdgeOdometry(
                [vertex_id - 1, vertex_id],
                np.eye(3) * 0.01,
                PoseSE2([rel_pose[0], rel_pose[1]], rel_pose[2]),
            )
            self.graph_edges.append(edge)
        
        # Store per-vertex metadata.
        self.vertices.append({
            'id': vertex_id,
            'pose': new_pose.copy(),
            'scan': scan_points.copy(),
            'timestamp': timestamp
        })
        self.current_vertex_id += 1
        self.trajectory.append(new_pose.copy())
        
        # Update map.
        self.update_map(scan_points, new_pose)
        
        # Check for loop closures.
        self.detect_loop_closures(timestamp)
        
        # Optimize pose graph periodically.
        if self.current_vertex_id % 10 == 0 and self.current_vertex_id > 1:
            self.optimize_pose_graph()

        # Keep scan history for ICP on the next frame.
        self.last_scan = scan_points.copy()
        self.last_scan_pose = new_pose.copy()
        
        return self.robot_pose, self.map
    
    def icp_match(self, source_points, target_points, initial_guess=None):
        """
        Iterative Closest Point algorithm for 2D scan matching.
        """
        if initial_guess is None:
            initial_guess = np.array([0.0, 0.0, 0.0])
        
        # Convert to homogeneous coordinates
        P = np.hstack([source_points, np.ones((source_points.shape[0], 1))])
        Q = target_points
        
        # Build KDTree for target
        tree = KDTree(Q)
        
        pose = initial_guess.copy()
        
        for _ in range(self.icp_max_iterations):
            # Transform source points by current pose
            c, s = np.cos(pose[2]), np.sin(pose[2])
            R = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
            P_transformed = (R @ P.T).T[:, :2]
            
            # Find nearest neighbors
            distances, indices = tree.query(P_transformed)
            
            # Filter outliers
            valid = distances < 0.5
            if np.sum(valid) < 10:
                break
            
            P_matched = P_transformed[valid]
            Q_matched = Q[indices[valid]]
            
            # Compute centroids
            centroid_P = np.mean(P_matched, axis=0)
            centroid_Q = np.mean(Q_matched, axis=0)
            
            # Cross-covariance matrix
            H = (P_matched - centroid_P).T @ (Q_matched - centroid_Q)
            
            # SVD for rotation
            try:
                U, S, Vt = np.linalg.svd(H)
            except np.linalg.LinAlgError:
                return initial_guess  # Return initial guess if SVD fails

            # --- Check if SVD returned valid numbers ---
            if np.any(np.isnan(U)) or np.any(np.isnan(S)) or np.any(np.isnan(Vt)):
                return initial_guess
    
            R_opt = Vt.T @ U.T
            
            # Ensure proper rotation (det = 1)
            if np.linalg.det(R_opt) < 0:
                Vt[-1, :] *= -1
                R_opt = Vt.T @ U.T
            
            # Extract rotation angle
            delta_theta = np.arctan2(R_opt[1, 0], R_opt[0, 0])
            
            # Translation
            delta_t = centroid_Q - R_opt @ centroid_P
            
            # Update pose
            pose[0] += delta_t[0]
            pose[1] += delta_t[1]
            pose[2] += delta_theta
            pose[2] = normalize_angle(pose[2])
            
            # Check convergence
            if np.linalg.norm(delta_t) < self.icp_tolerance and abs(delta_theta) < self.icp_tolerance:
                break
        
        return pose
    
    def detect_loop_closures(self, timestamp):
        """
        Detect potential loop closures in the trajectory.
        """
        if len(self.vertices) < 5:
            return
        
        current_pose = self.vertices[-1]['pose']
        
        # Check against previous vertices
        for i, vertex in enumerate(self.vertices[:-1]):
            # Skip recent vertices
            if len(self.vertices) - i < 5:
                continue
            
            # Check distance
            prev_pose = vertex['pose']
            distance = np.linalg.norm(current_pose[:2] - prev_pose[:2])
            
            if distance < self.loop_closure_distance_threshold:
                # Check time difference
                time_diff = timestamp - vertex['timestamp']
                if time_diff > self.loop_closure_time_threshold:
                    # Found a loop closure candidate
                    self.add_loop_constraint(i, len(self.vertices) - 1)
                    print(f"Loop closure detected between vertices {i} and {len(self.vertices)-1}")
                    break
    
    def add_loop_constraint(self, vertex_id1, vertex_id2):
        """
        Add a loop closure constraint between two vertices.
        """
        pose1 = self.vertices[vertex_id1]['pose']
        pose2 = self.vertices[vertex_id2]['pose']
        rel_pose = compute_relative_pose(pose1, pose2)
        
        # Add edge with high confidence
        edge = EdgeOdometry(
            [vertex_id1, vertex_id2],
            np.eye(3) * 0.001,
            PoseSE2([rel_pose[0], rel_pose[1]], rel_pose[2]),
        )
        self.graph_edges.append(edge)
    def optimize_pose_graph(self):
        """
        Run pose graph optimization safely.
        """
        if len(self.graph_vertices) < 3:
            return

        # Skip optimization if motion is negligible.
        if len(self.vertices) >= 2:
            last_pose = self.vertices[-1]['pose']
            prev_pose = self.vertices[-2]['pose']
            distance = np.linalg.norm(last_pose[:2] - prev_pose[:2])
            if distance < 0.01:
                return

        try:
            self.graph = Graph(self.graph_edges, self.graph_vertices)
            self.graph.optimize(verbose=False)

            # Update vertex poses, but only if they are valid numbers.
            for vertex in self.graph_vertices:
                vid = vertex.id
                if vid >= len(self.vertices):
                    continue

                est = np.array([vertex.pose[0], vertex.pose[1], vertex.pose[2]], dtype=np.float64)
                if np.any(np.isnan(est)):
                    print(f"Warning: Optimization produced NaN for vertex {vid}. Skipping update.")
                    continue

                self.vertices[vid]['pose'] = est

            # Update current robot pose.
            if len(self.vertices) > 0:
                last_est = self.vertices[-1]['pose']
                if not np.any(np.isnan(last_est)):
                    self.robot_pose = last_est.copy()
                else:
                    print("Warning: Optimized pose is NaN. Keeping previous robot pose.")

        except Exception as e:
            print(f"Pose graph optimization failed: {e}")

    def update_map(self, scan_points, robot_pose):
        """
        Update the occupancy grid map with new scan data.
        """
        # Transform scan points to world coordinates
        world_points = transform_points(scan_points, robot_pose)
        
        # Convert to map coordinates
        map_origin = np.array([-self.map_size/2, -self.map_size/2]) * self.map_resolution
        
        for point in world_points:
            # Check if point is within map bounds
            mx = int((point[0] - map_origin[0]) / self.map_resolution)
            my = int((point[1] - map_origin[1]) / self.map_resolution)
            
            if 0 <= mx < self.map_size and 0 <= my < self.map_size:
                self.map[my, mx] = min(1.0, self.map[my, mx] + 0.1)
        
        # Add robot position as free space (simple ray casting would be better)
        rx = int((robot_pose[0] - map_origin[0]) / self.map_resolution)
        ry = int((robot_pose[1] - map_origin[1]) / self.map_resolution)
        if 0 <= rx < self.map_size and 0 <= ry < self.map_size:
            self.map[ry, rx] = max(0, self.map[ry, rx] - 0.01)
    
    def get_map(self):
        """Return the current occupancy grid map."""
        return self.map
    
    def get_pose(self):
        """Return the current robot pose."""
        return self.robot_pose
    
    def get_trajectory(self):
        """Return the robot trajectory."""
        return np.array(self.trajectory)
