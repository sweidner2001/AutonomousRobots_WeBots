"""
slam -- a small, self-contained 2-D Graph-SLAM library.
=======================================================
Pure Python + NumPy (no Webots imports), so it can be unit-tested offline.

Public building blocks
----------------------
    Pose2D        SE(2) pose / rigid transform (geometry.py)
    wrap_angle    fold an angle into (-pi, pi]
    Scan          one lidar scan -> filtered point cloud (lidar_scan.py)
    ScanMatcher   ICP alignment of two scans (scan_matcher.py)
    PoseGraph     pose-graph container + Gauss-Newton optimiser (pose_graph.py)
    GraphSlam     the front-end that ties it all together (graph_slam.py)
"""

from Maze5.controllers.Controller_v1.slam.geometry import Pose2D, wrap_angle
from Maze5.controllers.Controller_v1.slam.lidar_scan import Scan
from Maze5.controllers.Controller_v1.slam.scan_matcher import ScanMatcher, MatchResult
from Maze5.controllers.Controller_v1.slam.pose_graph import PoseGraph, PoseEdge
from Maze5.controllers.Controller_v1.slam.graph_slam import GraphSlam, Keyframe

__all__ = [
    "Pose2D", "wrap_angle", "Scan",
    "ScanMatcher", "MatchResult",
    "PoseGraph", "PoseEdge",
    "GraphSlam", "Keyframe",
]
