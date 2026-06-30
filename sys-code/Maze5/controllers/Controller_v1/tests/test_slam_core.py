"""
tests/test_slam_core.py
=======================
Offline checks for the Graph-SLAM library -- they run WITHOUT Webots
(pure NumPy), so the maths can be validated before opening the simulator.

Run it directly:

    python tests/test_slam_core.py

or with pytest:

    pytest tests/test_slam_core.py

Three things are checked:
  1. Pose2D algebra  -- compose/inverse/between behave like real transforms.
  2. ICP             -- recovers a known rigid transform between two clouds.
  3. PoseGraph       -- optimisation straightens a drifted loop and slashes the
                        total error.
"""

import math
import os
import sys

import numpy as np

# Make the Maze5 package importable: climb from this file up to "sys-code".
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Maze5.controllers.Controller_v1.slam.geometry import Pose2D, wrap_angle
from Maze5.controllers.Controller_v1.slam.lidar_scan import Scan
from Maze5.controllers.Controller_v1.slam.scan_matcher import ScanMatcher
from Maze5.controllers.Controller_v1.slam.pose_graph import PoseGraph


def _pose_close(a, b, tol_xy=1e-6, tol_th=1e-6):
    return (abs(a.x - b.x) < tol_xy and abs(a.y - b.y) < tol_xy
            and abs(wrap_angle(a.theta - b.theta)) < tol_th)


# ===========================================================================
def test_pose_algebra():
    """compose/inverse/between obey the rigid-transform identities."""
    a = Pose2D(1.0, -2.0, 0.7)
    b = Pose2D(-0.5, 0.4, -1.2)

    # A composed with its inverse is the identity.
    assert _pose_close(a.compose(a.inverse()), Pose2D(0, 0, 0))
    assert _pose_close(a.inverse().compose(a), Pose2D(0, 0, 0))

    # between() is the relative transform: a * (a.between(b)) == b.
    assert _pose_close(a.compose(a.between(b)), b)

    # transform_points matches manual rotation+translation.
    pts = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, -1.0]])
    out = a.transform_points(pts)
    c, s = math.cos(a.theta), math.sin(a.theta)
    for p, o in zip(pts, out):
        exp = np.array([a.x + c * p[0] - s * p[1],
                        a.y + s * p[0] + c * p[1]])
        assert np.allclose(o, exp)
    print("  [ok] pose algebra: inverse, between, transform_points")


# ===========================================================================
def test_icp_recovers_transform():
    """ICP should recover a known transform between a cloud and its copy."""
    rng = np.random.default_rng(0)

    # A structured "corner" cloud (two perpendicular walls) -- enough geometry
    # for a unique alignment.
    wall_a = np.column_stack([np.linspace(0, 2, 120), np.zeros(120)])
    wall_b = np.column_stack([np.zeros(120), np.linspace(0, 2, 120)])
    ref_pts = np.vstack([wall_a, wall_b])
    ref_pts += rng.normal(scale=0.002, size=ref_pts.shape)  # a touch of noise

    # The true relative motion we want ICP to find (ref -> src).
    true = Pose2D(0.30, -0.20, math.radians(12.0))
    # src points are the ref cloud seen from the moved frame, so that
    # true.transform_points(src) == ref.
    src_pts = true.inverse().transform_points(ref_pts)

    ref = Scan.from_points(ref_pts)
    src = Scan.from_points(src_pts)

    matcher = ScanMatcher(max_iters=40, max_corr_dist=0.5, min_points=20)
    guess = Pose2D(0.2, -0.1, math.radians(6.0))   # deliberately rough
    res = matcher.match(ref, src, guess)

    print("  recovered %s  (true %s)" % (res.pose, true))
    print("  fitness=%.3f residual=%.4f m" % (res.fitness, res.residual))
    assert res.fitness > 0.9, "ICP found too few correspondences"
    assert _pose_close(res.pose, true, tol_xy=0.02, tol_th=math.radians(1.5))
    print("  [ok] ICP recovered the transform")


# ===========================================================================
def test_pose_graph_optimization():
    """A drifted square loop should snap shut and the error should collapse."""
    # Ground-truth poses at the four corners of a unit square (turning left).
    true = [
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(1.0, 0.0, math.pi / 2),
        Pose2D(1.0, 1.0, math.pi),
        Pose2D(0.0, 1.0, -math.pi / 2),
    ]
    # Perfect relative measurements for the four edges (the last closes the loop).
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    measurements = [true[i].between(true[j]) for (i, j) in pairs]

    # Build the graph with DRIFTED initial estimates: chain the odometry with a
    # small repeated bias so the loop does not close on its own.
    bias = Pose2D(0.03, -0.02, math.radians(4.0))
    graph = PoseGraph()
    est = Pose2D(0.0, 0.0, 0.0)
    graph.add_node(est)
    for k in range(1, 4):
        est = est.compose(measurements[k - 1].compose(bias))
        graph.add_node(est)

    omega = np.diag([200.0, 200.0, 400.0])
    for (i, j), z in zip(pairs, measurements):
        graph.add_edge(i, j, z, omega)

    chi2_before = graph.chi2()
    chi2_after = graph.optimize(iterations=30)
    print("  chi2: %.4f -> %.6f" % (chi2_before, chi2_after))

    assert chi2_after < chi2_before * 1e-2, "optimisation barely reduced error"

    # With perfect measurements and node 0 anchored, the optimum is ground truth.
    max_err = max(math.hypot(graph.node_pose(k).x - true[k].x,
                             graph.node_pose(k).y - true[k].y)
                  for k in range(4))
    print("  max node position error after optimisation: %.4f m" % max_err)
    assert max_err < 0.03, "optimised poses did not converge to ground truth"
    print("  [ok] pose-graph optimisation closed the loop")


# ===========================================================================
def main():
    tests = [
        ("Pose2D algebra", test_pose_algebra),
        ("ICP scan matching", test_icp_recovers_transform),
        ("Pose-graph optimisation", test_pose_graph_optimization),
    ]
    failed = 0
    for name, fn in tests:
        print("\n== %s ==" % name)
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("  [FAIL] %s" % exc)
        except Exception as exc:                    # noqa: BLE001
            failed += 1
            print("  [ERROR] %s: %s" % (type(exc).__name__, exc))

    print("\n%s" % ("-" * 50))
    if failed == 0:
        print("ALL SLAM CORE TESTS PASSED")
        return 0
    print("%d TEST(S) FAILED" % failed)
    return 1


if __name__ == "__main__":
    sys.exit(main())
