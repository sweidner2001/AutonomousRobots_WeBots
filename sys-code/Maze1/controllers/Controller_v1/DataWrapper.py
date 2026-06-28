from dataclasses import dataclass
import numpy as np



@dataclass
class SensorData:
    scan: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray



@dataclass
class Pose2D:
    """Holds the robot's estimated position and heading in the 2-D world frame.

    Attributes
    ----------
    x     : float  — east/west position in metres (positive = east).
    y     : float  — north/south position in metres (positive = north).
    theta : float  — heading angle in radians, measured counter-clockwise
                     from the positive x-axis.
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0