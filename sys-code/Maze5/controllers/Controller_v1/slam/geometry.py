"""
slam/geometry.py
================
SE(2) pose algebra -- the mathematical foundation the whole SLAM system
speaks.  Instead of passing raw ``(x, y, theta)`` tuples around and
re-deriving rotation maths in every file, every module uses one small,
well-tested ``Pose2D`` class.

WHAT IS A "POSE" / SE(2)?
-------------------------
A 2-D robot pose is a position ``(x, y)`` plus a heading ``theta``.  It is
also a rigid-body TRANSFORM: "rotate by theta, then translate by (x, y)".
Mathematicians call the set of these transforms ``SE(2)`` (the Special
Euclidean group in 2-D).  Two facts make poses powerful:

  * They COMPOSE.  If ``A`` is the pose of the robot in the world and ``B``
    is the pose of something in the robot's frame, then ``A * B`` is the
    pose of that something in the world.
  * They INVERT.  ``A.inverse()`` is the transform that undoes ``A`` -- it
    maps world coordinates back into ``A``'s local frame.

A pose can be written as a 3x3 homogeneous matrix::

        | cos t   -sin t   x |
    T = | sin t    cos t   y |
        |   0        0      1 |

Composition is matrix multiplication; inversion is matrix inversion.  This
file gives the same operations in cheap closed form (no matrices needed for
the common cases).
"""

import math

import numpy as np


def wrap_angle(a):
    """Wrap an angle to the range (-pi, pi].

    Headings live on a circle: +pi and -pi are the same direction.  After
    adding/subtracting angles we must fold the result back into one
    canonical range, otherwise error terms blow up near the +/-pi seam.
    """
    return math.atan2(math.sin(a), math.cos(a))


class Pose2D:
    """A 2-D rigid transform / robot pose ``(x, y, theta)``.

    Treated as immutable: every operation returns a NEW ``Pose2D`` rather
    than mutating ``self``.  This makes poses safe to store inside the pose
    graph and pass around without surprise side effects.
    """

    __slots__ = ("x", "y", "theta")

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)

    # ---- construction helpers -------------------------------------------- #
    @classmethod
    def from_vector(cls, v):
        """Build a pose from a length-3 array/sequence ``[x, y, theta]``."""
        return cls(v[0], v[1], v[2])

    def as_vector(self):
        """Return this pose as a NumPy array ``[x, y, theta]``."""
        return np.array([self.x, self.y, self.theta], dtype=float)

    def as_matrix(self):
        """Return the 3x3 homogeneous transform matrix for this pose."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([[c, -s, self.x],
                         [s,  c, self.y],
                         [0,  0, 1.0]], dtype=float)

    def copy(self):
        return Pose2D(self.x, self.y, self.theta)

    # ---- group operations ------------------------------------------------ #
    def compose(self, other):
        """Return ``self * other`` (apply ``self`` to ``other``).

        If ``self`` is the pose of frame B in frame A, and ``other`` is the
        pose of frame C in frame B, the result is the pose of C in A.
        """
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Pose2D(
            self.x + c * other.x - s * other.y,
            self.y + s * other.x + c * other.y,
            wrap_angle(self.theta + other.theta),
        )

    def __mul__(self, other):
        """``A * B`` is shorthand for ``A.compose(B)``."""
        return self.compose(other)

    def inverse(self):
        """Return the transform that undoes this one (``self^-1``)."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Pose2D(
            -(c * self.x + s * self.y),
            (s * self.x - c * self.y),
            -self.theta,
        )

    def between(self, other):
        """Relative pose FROM ``self`` TO ``other`` (= ``self^-1 * other``).

        This answers: "what motion, expressed in ``self``'s frame, takes the
        robot from pose ``self`` to pose ``other``?"  It is exactly the
        measurement stored on an odometry edge of the pose graph.
        """
        return self.inverse().compose(other)

    # ---- acting on points ------------------------------------------------ #
    def transform_points(self, pts):
        """Map an ``(N, 2)`` array of points from this frame into the parent.

        Each local point ``p`` becomes ``R(theta) @ p + (x, y)``.  Used to
        place a scan's hit points (given in the robot frame) into the world
        map frame.
        """
        pts = np.asarray(pts, dtype=float)
        if pts.size == 0:
            return pts.reshape(0, 2)
        c, s = math.cos(self.theta), math.sin(self.theta)
        # pts @ R^T  +  translation.   R^T = [[c, s], [-s, c]]
        rot = np.array([[c, s], [-s, c]], dtype=float)
        return pts @ rot + np.array([self.x, self.y])

    def transform_point(self, x, y):
        """Map a single point ``(x, y)`` from this frame into the parent."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (self.x + c * x - s * y, self.y + s * x + c * y)

    # ---- misc ------------------------------------------------------------ #
    def translation_norm(self):
        """Length of the translation part (handy for thresholds)."""
        return math.hypot(self.x, self.y)

    def __repr__(self):
        return "Pose2D(x=%.3f, y=%.3f, theta=%.1f deg)" % (
            self.x, self.y, math.degrees(self.theta))
