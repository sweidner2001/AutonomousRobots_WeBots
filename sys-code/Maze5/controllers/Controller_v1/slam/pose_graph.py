r"""
slam/pose_graph.py
=================
The SLAM *back-end*: the actual "graph" in Graph-SLAM, plus the optimiser
that straightens it out.

THE PICTURE
-----------
Think of the robot's journey as a graph:

    (x0)---odom---(x1)---odom---(x2)---odom---(x3)
      \                                        /
       \---------------- loop ----------------/

  * NODES  (x0, x1, ...) are robot poses at chosen keyframes.
  * EDGES  are measured relative transforms between two poses, each with an
    INFORMATION matrix saying how trustworthy it is:
      - "odom" edges come from odometry refined by scan matching;
      - the "loop" edge appears when the robot recognises an earlier place.

Because of sensor noise the edges DISAGREE: following the odometry edges
around the loop does not bring you back to where the loop edge says you
should be.  Optimisation nudges every pose a little so that ALL edges are
satisfied as well as possible at once -- this is the moment the map snaps
straight.

THE MATHS (least squares on SE(2))
----------------------------------
Each edge ``i->j`` with measurement ``Z`` contributes an error

    e_ij(x) = t2v( Z^-1 * (X_i^-1 * X_j) )          (a 3-vector)

We want the poses that minimise ``sum_ij  e_ij^T * Omega_ij * e_ij``.
That is a nonlinear least-squares problem; we solve it with Gauss-Newton:

    1. Linearise every edge: e ~ e0 + A*dx_i + B*dx_j  (A, B are Jacobians).
    2. Stack them into the normal equations  H * dx = -b  where
         H = sum J^T Omega J     and     b = sum J^T Omega e .
    3. Anchor the first pose (otherwise the whole map could slide/rotate
       freely -- "gauge freedom") and solve for the increment dx.
    4. Apply dx to every pose and repeat until it converges.

The node count here is small (one node per keyframe), so a dense solve with
``numpy.linalg.solve`` is plenty fast and keeps the code readable.
"""

import numpy as np

from Maze5.controllers.Controller_v1.slam.geometry import Pose2D, wrap_angle


class PoseEdge:
    """A relative-pose constraint between two nodes.

    Attributes:
        i, j   : node ids the edge connects (measured motion is i -> j).
        z      : Pose2D, the measured relative transform.
        omega  : 3x3 information matrix (inverse covariance); larger = more
                 trusted.
        kind   : "odom" or "loop" -- used only for visualisation/printing.
    """

    __slots__ = ("i", "j", "z", "omega", "kind")

    def __init__(self, i, j, z, omega, kind="odom"):
        self.i = int(i)
        self.j = int(j)
        self.z = z
        self.omega = np.asarray(omega, dtype=float)
        self.kind = kind


class PoseGraph:
    """A graph of 2-D poses (nodes) and relative constraints (edges)."""

    def __init__(self):
        self.nodes = []   # list of Pose2D, indexed by node id
        self.edges = []   # list of PoseEdge

    # ---- building the graph --------------------------------------------- #
    def add_node(self, pose):
        """Add a node initialised at ``pose``; return its integer id."""
        self.nodes.append(pose.copy())
        return len(self.nodes) - 1

    def add_edge(self, i, j, z, omega, kind="odom"):
        """Add a relative constraint ``z`` (with information ``omega``)."""
        self.edges.append(PoseEdge(i, j, z, omega, kind))

    def node_pose(self, i):
        return self.nodes[i]

    def set_node_pose(self, i, pose):
        self.nodes[i] = pose.copy()

    # ---- error reporting ------------------------------------------------- #
    def chi2(self):
        """Total weighted squared error ``sum e^T Omega e`` over all edges.

        A single number summarising how badly the current poses violate the
        constraints.  Optimisation should drive it down.
        """
        total = 0.0
        for e in self.edges:
            err = self._edge_error(self.nodes[e.i], self.nodes[e.j], e.z)
            total += float(err @ e.omega @ err)
        return total

    # ---- optimisation ---------------------------------------------------- #
    def optimize(self, iterations=20, anchor=1e6, damping=1e-9):
        """Gauss-Newton optimisation of all node poses.

        Args:
            iterations: maximum Gauss-Newton iterations.
            anchor    : how hard to pin node 0 in place (removes gauge
                        freedom -- the global position/heading the whole map
                        is free to slide in).
            damping   : tiny value added to the diagonal for numerical
                        safety when the system is near-singular.

        Returns:
            The final chi2 error.
        """
        n = len(self.nodes)
        if n == 0 or not self.edges:
            return self.chi2()

        for _ in range(iterations):
            H = np.zeros((3 * n, 3 * n), dtype=float)
            b = np.zeros(3 * n, dtype=float)

            # Accumulate every edge's contribution to the normal equations.
            for e in self.edges:
                xi = self.nodes[e.i].as_vector()
                xj = self.nodes[e.j].as_vector()
                err, A, B = self._linearise(xi, xj, e.z.as_vector())
                Om = e.omega

                ii = slice(3 * e.i, 3 * e.i + 3)
                jj = slice(3 * e.j, 3 * e.j + 3)

                H[ii, ii] += A.T @ Om @ A
                H[ii, jj] += A.T @ Om @ B
                H[jj, ii] += B.T @ Om @ A
                H[jj, jj] += B.T @ Om @ B
                b[ii] += A.T @ Om @ err
                b[jj] += B.T @ Om @ err

            # Fix the first node (gauge) and add a whisker of damping.
            H[0:3, 0:3] += anchor * np.eye(3)
            H += damping * np.eye(3 * n)

            # Solve H dx = -b for the pose increments.
            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                break

            # Apply the increment to every pose (angles wrapped).
            for k in range(n):
                p = self.nodes[k]
                self.nodes[k] = Pose2D(
                    p.x + dx[3 * k],
                    p.y + dx[3 * k + 1],
                    wrap_angle(p.theta + dx[3 * k + 2]),
                )

            # Stop early once the update is negligible.
            if np.linalg.norm(dx) < 1e-6:
                break

        return self.chi2()

    # ---- error + Jacobians for one edge --------------------------------- #
    @staticmethod
    def _edge_error(pose_i, pose_j, z):
        """Error vector e = t2v( Z^-1 * (X_i^-1 * X_j) ) as ``[ex, ey, et]``."""
        predicted = pose_i.between(pose_j)      # X_i^-1 * X_j
        err_pose = z.between(predicted)         # Z^-1 * predicted
        return np.array([err_pose.x, err_pose.y, err_pose.theta], dtype=float)

    @staticmethod
    def _linearise(xi, xj, z):
        """Return (error, A, B) for one edge.

        ``xi, xj, z`` are length-3 vectors (x, y, theta).  ``A = de/dx_i`` and
        ``B = de/dx_j`` are the 3x3 Jacobians used to build the normal
        equations.  The closed forms below are the standard 2-D pose-graph
        result (Grisetti et al., "A Tutorial on Graph-Based SLAM").
        """
        ti = xi[:2]
        tj = xj[:2]
        th_i, th_j, th_z = xi[2], xj[2], z[2]

        ci, si = np.cos(th_i), np.sin(th_i)
        cz, sz = np.cos(th_z), np.sin(th_z)

        Ri_T = np.array([[ci, si], [-si, ci]])         # R(theta_i)^T
        Rz_T = np.array([[cz, sz], [-sz, cz]])         # R(theta_z)^T
        dRi_T = np.array([[-si, ci], [-ci, -si]])      # d R_i^T / d theta_i

        dt = tj - ti

        # Error.
        e_trans = Rz_T @ (Ri_T @ dt - z[:2])
        e_theta = wrap_angle(th_j - th_i - th_z)
        err = np.array([e_trans[0], e_trans[1], e_theta], dtype=float)

        # A = de/dx_i.
        A = np.zeros((3, 3))
        A[:2, :2] = -Rz_T @ Ri_T
        A[:2, 2] = Rz_T @ dRi_T @ dt
        A[2, 2] = -1.0

        # B = de/dx_j.
        B = np.zeros((3, 3))
        B[:2, :2] = Rz_T @ Ri_T
        B[2, 2] = 1.0

        return err, A, B
