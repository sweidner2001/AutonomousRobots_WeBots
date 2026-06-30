"""
slam/graph_slam.py
=================
The conductor of the SLAM system.  It owns the pose graph and decides, every
step, how to fold a new (odometry, scan) observation into it.

ONE UPDATE, STEP BY STEP
------------------------
Given the latest raw odometry pose and lidar scan:

  1. PREDICT.  Take the odometry motion since the last keyframe and apply it
     to the last keyframe's (corrected) pose -> a rough guess of where we are.
  2. REFINE.  Scan-match the current scan against the last keyframe's scan,
     seeded with that guess, to get a much better relative motion.  This is
     the live, corrected pose used for driving and drawing.
  3. KEYFRAME?  If we have moved/turned enough since the last keyframe, store
     a new node and connect it with an "odom" edge (the refined motion).
  4. LOOP CLOSURE.  Compare the new keyframe against earlier, non-recent
     keyframes that are physically close.  If a scan-match is convincing, add
     a "loop" edge and OPTIMISE the whole graph -- every pose shifts to agree.

WHAT IT EXPOSES
---------------
  * ``current_pose``           -- best estimate of the live robot pose.
  * ``keyframes``              -- list of Keyframe(id, pose, scan) for mapping.
  * ``edges_for_view()``       -- node positions + edges for visualisation.
  * ``take_map_dirty()``       -- True once after an optimisation, so the app
                                  knows to rebuild the occupancy map.
"""

import numpy as np

from Maze5.controllers.Controller_v1.slam.geometry import Pose2D
from Maze5.controllers.Controller_v1.slam.pose_graph import PoseGraph


class Keyframe:
    """A stored pose + the scan taken there.

    The pose lives in the pose graph (so it updates on optimisation); this
    object keeps the node id and the scan that the map and loop-closure
    matching need.
    """

    __slots__ = ("node_id", "scan")

    def __init__(self, node_id, scan):
        self.node_id = node_id
        self.scan = scan


class GraphSlam:
    """Front-end that builds and maintains the pose graph."""

    def __init__(self, scan_matcher,
                 keyframe_dist=0.25, keyframe_angle=0.30,
                 loop_search_radius=1.5, loop_min_gap=12,
                 loop_fitness_min=0.55, loop_residual_max=0.10,
                 info_odom=(200.0, 400.0), info_loop=(600.0, 800.0),
                 opt_iters=20):
        self.matcher = scan_matcher
        self.keyframe_dist = float(keyframe_dist)
        self.keyframe_angle = float(keyframe_angle)
        self.loop_search_radius = float(loop_search_radius)
        self.loop_min_gap = int(loop_min_gap)
        self.loop_fitness_min = float(loop_fitness_min)
        self.loop_residual_max = float(loop_residual_max)
        self.opt_iters = int(opt_iters)

        # Information matrices for the two edge types (diagonal x, y, theta).
        self._omega_odom = np.diag([info_odom[0], info_odom[0], info_odom[1]])
        self._omega_loop = np.diag([info_loop[0], info_loop[0], info_loop[1]])

        self.graph = PoseGraph()
        self.keyframes = []           # list[Keyframe]
        self.current_pose = Pose2D()  # best live estimate

        self._last_kf_odom = None     # raw odometry pose at the last keyframe
        self._map_dirty = False       # set when an optimisation changes poses

    # ---------------------------------------------------------------------- #
    @property
    def last_keyframe(self):
        return self.keyframes[-1] if self.keyframes else None

    def take_map_dirty(self):
        """Return True once after the graph was re-optimised, then reset."""
        was = self._map_dirty
        self._map_dirty = False
        return was

    # ---------------------------------------------------------------------- #
    def update(self, odom_pose, scan):
        """Fold one (odometry, scan) observation into the graph.

        Args:
            odom_pose : Pose2D, the raw wheel+IMU odometry pose.
            scan      : the current Scan.

        Returns:
            current_pose (Pose2D) -- the best live estimate.
        """
        # --- First observation: seed keyframe 0 at the origin. -------------
        if not self.keyframes:
            node0 = self.graph.add_node(Pose2D(0.0, 0.0, 0.0))
            self.keyframes.append(Keyframe(node0, scan))
            self._last_kf_odom = odom_pose.copy()
            self.current_pose = Pose2D(0.0, 0.0, 0.0)
            return self.current_pose

        kf = self.last_keyframe
        kf_pose = self.graph.node_pose(kf.node_id)

        # --- 1. PREDICT relative motion from odometry since last keyframe. -
        odom_rel = self._last_kf_odom.between(odom_pose)

        # --- 2. REFINE it by scan-matching against the last keyframe scan. -
        result = self.matcher.match(kf.scan, scan, odom_rel)
        # Use the scan-match only if it found real overlap; else trust odom.
        rel = result.pose if result.fitness > 0.0 else odom_rel

        self.current_pose = kf_pose.compose(rel)

        # --- 3. KEYFRAME? Enough motion since the last one? ----------------
        if (rel.translation_norm() >= self.keyframe_dist
                or abs(rel.theta) >= self.keyframe_angle):
            self._add_keyframe(scan, rel, odom_pose)

        return self.current_pose

    # ---------------------------------------------------------------------- #
    def _add_keyframe(self, scan, rel, odom_pose):
        """Create a new node, link it to the previous one, try loop closure."""
        prev = self.last_keyframe
        new_pose = self.current_pose

        node_id = self.graph.add_node(new_pose)
        # Odometry/scan-match edge from the previous keyframe to this one.
        self.graph.add_edge(prev.node_id, node_id, rel, self._omega_odom,
                            kind="odom")

        new_kf = Keyframe(node_id, scan)
        self.keyframes.append(new_kf)
        self._last_kf_odom = odom_pose.copy()

        # --- 4. LOOP CLOSURE ----------------------------------------------
        if self._try_loop_closures(new_kf):
            self.graph.optimize(self.opt_iters)
            self._map_dirty = True
            # Re-read the (now corrected) live pose from the graph.
            self.current_pose = self.graph.node_pose(node_id)

    def _try_loop_closures(self, new_kf):
        """Scan-match the new keyframe against older candidates.

        Returns True if at least one loop edge was added.
        """
        new_pose = self.graph.node_pose(new_kf.node_id)
        new_idx = len(self.keyframes) - 1
        found = False

        for old_idx in range(new_idx - self.loop_min_gap + 1):
            old_kf = self.keyframes[old_idx]
            old_pose = self.graph.node_pose(old_kf.node_id)

            # Only bother with keyframes that are physically near our estimate.
            if old_pose.between(new_pose).translation_norm() > self.loop_search_radius:
                continue

            guess = old_pose.between(new_pose)
            result = self.matcher.match(old_kf.scan, new_kf.scan, guess)
            if result.is_good(self.loop_fitness_min, self.loop_residual_max):
                self.graph.add_edge(old_kf.node_id, new_kf.node_id,
                                    result.pose, self._omega_loop, kind="loop")
                found = True
                print("[slam] loop closure %d <-> %d  (fitness %.2f, "
                      "residual %.3f m)" % (old_kf.node_id, new_kf.node_id,
                                            result.fitness, result.residual))

        return found

    # ---------------------------------------------------------------------- #
    # Accessors for mapping / visualisation
    # ---------------------------------------------------------------------- #
    def keyframe_poses(self):
        """List of (Pose2D, Scan) for every keyframe, at current estimates."""
        return [(self.graph.node_pose(kf.node_id), kf.scan)
                for kf in self.keyframes]

    def edges_for_view(self):
        """Return (node_xy, odom_pairs, loop_pairs) for drawing the graph.

        node_xy    : (N, 2) array of node positions.
        odom_pairs : list of (i, j) index pairs for odometry edges.
        loop_pairs : list of (i, j) index pairs for loop-closure edges.
        """
        node_xy = np.array([[p.x, p.y] for p in self.graph.nodes]) \
            if self.graph.nodes else np.zeros((0, 2))
        odom_pairs = [(e.i, e.j) for e in self.graph.edges if e.kind == "odom"]
        loop_pairs = [(e.i, e.j) for e in self.graph.edges if e.kind == "loop"]
        return node_xy, odom_pairs, loop_pairs
