"""ROSbot 2 SLAM Controller for Webots"""

from controller import Robot
import numpy as np
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Maze1.controllers.Controller_v5.slam_engine import SLAMEngine
from Maze1.controllers.Controller_v5.exploration import Exploration
from Maze1.controllers.Controller_v5.utils import polar_to_cartesian

class ROSbotSLAM:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.timestep_sec = self.time_step / 1000.0

        self._init_motors()
        self._init_lidar()

        # Initialize SLAM engine
        self.slam = SLAMEngine(map_resolution=0.05, map_size_m=20)
        self.exploration = Exploration(self.slam, map_resolution=0.05)

        self.running = True
        self.step_count = 0

        print("ROSbot SLAM initialized")
        print(f"LIDAR: {self.lidar_num_points} points, FOV: {self.lidar_fov:.2f} rad")

    def _init_motors(self):
        """Configure all wheel motors for velocity control."""
        # Get all four wheel motors
        self.front_left_motor = self.robot.getDevice("fl_wheel_joint")
        self.front_right_motor = self.robot.getDevice("fr_wheel_joint")
        self.rear_left_motor = self.robot.getDevice("rl_wheel_joint")
        self.rear_right_motor = self.robot.getDevice("rr_wheel_joint")

        for motor in [self.front_left_motor, self.front_right_motor,
                      self.rear_left_motor, self.rear_right_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

    def _init_lidar(self):
        """Initialize LiDAR and precompute beam angles."""
        self.lidar = self.robot.getDevice("laser")
        self.lidar.enable(self.time_step)

        self.lidar_fov = self.lidar.getFov()  # Should be 2*pi
        self.lidar_num_points = self.lidar.getHorizontalResolution()
        self.lidar_angles = np.linspace(-self.lidar_fov / 2, self.lidar_fov / 2, self.lidar_num_points)
    
    def get_lidar_data(self):
        """Get and process LiDAR data."""
        ranges = self.lidar.getRangeImage()
        
        # Convert to numpy array and filter
        ranges = np.array(ranges)
        
        # Replace inf and nan with max range
        max_range = self.lidar.getMaxRange()
        ranges = np.where(np.isfinite(ranges), ranges, max_range)
        ranges = np.clip(ranges, 0, max_range)
        
        # Get angles
        angles = self.lidar_angles.copy()
        
        return ranges, angles
    
    def get_wheel_velocities(self):
        """Get current wheel velocities."""
        left_vel = self.front_left_motor.getVelocity()
        right_vel = self.front_right_motor.getVelocity()
        return left_vel, right_vel
    
    def run(self):
        """Main control loop."""
        while self.robot.step(self.time_step) != -1 and self.running:
            self.step_count += 1
            
            # Get sensor data
            ranges, angles = self.get_lidar_data()
            
            # Convert to Cartesian points
            scan_points = polar_to_cartesian(ranges, angles)
            
            # Get odometry
            left_vel, right_vel = self.get_wheel_velocities()
            
            # Update SLAM
            pose, map_data = self.slam.add_scan(
                scan_points, 
                self.robot.getTime(),
                left_vel, 
                right_vel,
                self.timestep_sec
            )
            
            # Get exploration control
            left_cmd, right_cmd = self.exploration.get_control(ranges, angles)
            
            # Apply motor commands
            self.front_left_motor.setVelocity(left_cmd)
            self.rear_left_motor.setVelocity(left_cmd)
            self.front_right_motor.setVelocity(right_cmd)
            self.rear_right_motor.setVelocity(right_cmd)
            
            # Print status periodically
            if self.step_count % 100 == 0:
                print(f"Step: {self.step_count}, Pose: [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}]")
                print(f"Map cells: {np.sum(map_data > 0.1)}")
                print(f"Vertices: {len(self.slam.vertices)}")
            
            # Check if exploration is complete
            if self.step_count > 1000 and len(self.slam.vertices) > 50:
                # Check if frontiers remain
                frontiers = self.exploration.compute_frontiers(map_data, pose)
                if len(frontiers) < 10:
                    print("Exploration complete!")
                    self.front_left_motor.setVelocity(0.0)
                    self.rear_left_motor.setVelocity(0.0)
                    self.front_right_motor.setVelocity(0.0)
                    self.rear_right_motor.setVelocity(0.0)
                    self.running = False
                    break

if __name__ == "__main__":
    controller = ROSbotSLAM()
    controller.run()