"""
slam/scan_matcher.py
===================
The SLAM *front-end*: line up two lidar scans to measure how the robot moved
between them.  This is what corrects the drift of wheel + IMU odometry and,
later, what recognises a place the robot has visited before (loop closure).

ALGORITHM: ICP (ITERATIVE CLOSEST POINT)
----------------------------------------
ICP is the classic way to align two point clouds (Besl & McKay, 1992).
Given a REFERENCE cloud (an older scan) and a SOURCE cloud (a newer scan)
plus a rough initial guess of the transform between them, it repeats:

  1. Transform the source points by the current guess.
  2. For each transformed source point, find the CLOSEST reference point.
     (These pairs are our guesses at "this beam and that beam hit the same
     bit of wall".)
  3. Throw away pairs that are implausibly far apart (outliers).
  4. Compute the single rigid rotation+translation that best snaps the
     remaining source points onto their matched reference points
     (closed-form, via the cross-covariance SVD -- the "Kabsch" method).
  5. Fold that correction into the guess and repeat until it stops changing.

The better the initial guess, the faster and more reliably ICP converges --
which is exactly why we seed it with odometry instead of starting blind.

OUTPUT
------
``match()`` returns the refined relative pose plus a ``fitness`` (fraction of
points that found a close match) and the mean residual distance.  Those two
numbers tell the caller HOW MUCH to trust the result -- essential for
rejecting bad loop closures.
"""

import math

import numpy as np

from Maze5.controllers.Controller_v1.slam.geometry import Pose2D

# Optional speed-up: SciPy's KD-tree makes nearest-neighbour search O(log n)
# instead of O(n).  We fall back to a vectorised NumPy search if SciPy is
# not installed, so the code always runs.
try:
    from scipy.spatial import cKDTree as _KDTree
except Exception:                       # pragma: no cover - SciPy optional
    _KDTree = None


class MatchResult:
    """Outcome of a scan match.

    Attributes:
        pose     : Pose2D -- refined transform mapping SOURCE into the
                   REFERENCE frame (i.e. the relative motion ref -> src).
        fitness  : float in [0, 1] -- fraction of source points that found an
                   inlier match.  Higher = more of the two scans overlap.
        residual : float -- mean distance (m) of the inlier matches.
                   Lower = the overlapping parts line up more tightly.
        converged: bool -- True if ICP settled rather than hitting the
                   iteration cap.
    """

    __slots__ = ("pose", "fitness", "residual", "converged")

    def __init__(self, pose, fitness, residual, converged):
        self.pose = pose
        self.fitness = fitness
        self.residual = residual
        self.converged = converged

    def is_good(self, fitness_min, residual_max):
        """True if this match is trustworthy enough to use as a constraint."""
        return self.fitness >= fitness_min and self.residual <= residual_max


class ScanMatcher:
    """Point-to-point ICP aligner."""

    def __init__(self, max_iters=30, max_corr_dist=0.40,
                 converge_eps=1e-4, min_points=25):
        self.max_iters = int(max_iters)
        self.max_corr_dist = float(max_corr_dist)
        self.converge_eps = float(converge_eps)
        self.min_points = int(min_points)

    # ---------------------------------------------------------------------- #
    def match(self, ref_scan, src_scan, init_guess):
        """Align ``src_scan`` onto ``ref_scan``.

        Args:
            ref_scan  : the reference Scan (older keyframe).
            src_scan  : the source Scan (to be aligned).
            init_guess: Pose2D, the initial estimate of the relative motion
                        (ref -> src), usually from odometry.

        Returns:
            MatchResult.
        """
        ref = ref_scan.points
        src = src_scan.points

        # Not enough geometry to match -> trust the prior, report no fitness.
        if ref.shape[0] < self.min_points or src.shape[0] < self.min_points:
            return MatchResult(init_guess, 0.0, math.inf, False)

        tree = _KDTree(ref) if _KDTree is not None else None

        T = init_guess.copy()
        converged = False
        fitness = 0.0
        residual = math.inf

        for _ in range(self.max_iters):
            # Step 1: place the source cloud using the current estimate.
            moved = T.transform_points(src)

            # Step 2: nearest reference point for each source point.
            dist, idx = self._nearest(tree, ref, moved)

            # Step 3: keep only close-enough correspondences (reject outliers).
            inlier = dist <= self.max_corr_dist
            n_in = int(inlier.sum())
            if n_in < self.min_points:
                break

            a = moved[inlier]          # source points (already moved by T)
            b = ref[idx[inlier]]       # their matched reference points

            # Step 4: best-fit rigid correction dT mapping ``a`` onto ``b``.
            dT = self._fit_rigid(a, b)

            # Step 5: fold the correction into the estimate.
            T = dT.compose(T)

            fitness = n_in / src.shape[0]
            residual = float(dist[inlier].mean())

            # Converged once the correction becomes negligible.
            if (dT.translation_norm() < self.converge_eps
                    and abs(dT.theta) < self.converge_eps):
                converged = True
                break

        return MatchResult(T, fitness, residual, converged)

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _nearest(tree, ref, query):
        """Nearest reference point for every query point.

        Returns (distances, indices) as arrays aligned with ``query``.
        """
        if tree is not None:
            dist, idx = tree.query(query)
            return dist, idx
        # NumPy fallback: full pairwise distance matrix (fine for a few
        # hundred points).  diff has shape (Nq, Nref, 2).
        diff = query[:, None, :] - ref[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)   # squared distances
        idx = np.argmin(d2, axis=1)
        dist = np.sqrt(d2[np.arange(query.shape[0]), idx])
        return dist, idx

    @staticmethod
    def _fit_rigid(a, b):
        """Best-fit rotation+translation mapping points ``a`` onto ``b``.

        Closed-form Kabsch / Umeyama solution:
          1. Subtract the centroids of both clouds.
          2. Form the 2x2 cross-covariance ``H = sum (a_c)(b_c)^T``.
          3. SVD: ``H = U S V^T``;  the optimal rotation is ``R = V U^T``
             (with a sign fix so it is a proper rotation, not a reflection).
          4. Translation lines the centroids back up: ``t = mu_b - R mu_a``.

        Returns the correction as a Pose2D.
        """
        mu_a = a.mean(axis=0)
        mu_b = b.mean(axis=0)
        ac = a - mu_a
        bc = b - mu_b

        H = ac.T @ bc                      # 2x2 cross-covariance
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.array([[1.0, 0.0], [0.0, d]])
        R = Vt.T @ D @ U.T                 # proper 2x2 rotation

        t = mu_b - R @ mu_a
        theta = math.atan2(R[1, 0], R[0, 0])
        return Pose2D(t[0], t[1], theta)
