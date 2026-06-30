"""
slam/lidar_scan.py
==================
A single LIDAR measurement, turned into a clean 2-D point cloud.

WHY WRAP A SCAN IN A CLASS?
---------------------------
The raw lidar gives a flat array of distances (one per ray).  Almost every
SLAM step actually wants the (x, y) HIT POINTS instead -- the dots where the
beams struck a wall.  ``Scan`` does that conversion once, caches it, and
filters out useless rays (too near, too far, or "no return" = inf).

FRAME CONVENTION
----------------
The points are expressed in the ROBOT's own frame at the instant of the
scan:  +x is straight ahead, +y is to the left.  We treat the lidar as
sitting at the robot's reference point (its ~2 cm forward offset is far
smaller than one 5 cm map cell, so we safely ignore it).

A point for ray ``i`` with range ``r`` and bearing ``b`` is simply::

    x = r * cos(b)
    y = r * sin(b)
"""

import numpy as np


class Scan:
    """One lidar scan -> filtered (N, 2) point cloud in the robot frame."""

    def __init__(self, ranges, bearings,
                 min_range=0.20, max_range=6.0, max_points=360):
        """
        Args:
            ranges    : 1-D array of measured distances (m); inf = no return.
            bearings  : 1-D array of per-ray angles (rad, robot frame).
                        Must be the same length as ``ranges``.
            min_range : drop returns nearer than this (sensor blind zone).
            max_range : drop returns farther than this (noisy / irrelevant).
            max_points: keep at most this many points (evenly strided) so
                        scan matching stays fast.
        """
        self.ranges = np.asarray(ranges, dtype=float)
        self.bearings = np.asarray(bearings, dtype=float)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.max_points = int(max_points)

        self._points = None   # lazily computed (N, 2) cloud

    # ---------------------------------------------------------------------- #
    @classmethod
    def from_points(cls, pts):
        """Build a Scan directly from an ``(N, 2)`` point cloud.

        Convenience for tests and for code that already has Cartesian points
        rather than raw ranges/bearings.
        """
        s = cls(np.zeros(0), np.zeros(0))
        s._points = np.asarray(pts, dtype=float).reshape(-1, 2)
        return s

    # ---------------------------------------------------------------------- #
    @property
    def points(self):
        """Return the cached ``(N, 2)`` hit-point cloud (robot frame)."""
        if self._points is None:
            self._points = self._build_points()
        return self._points

    def _build_points(self):
        r = self.ranges
        b = self.bearings

        # Keep only rays that returned a usable distance.
        valid = np.isfinite(r) & (r >= self.min_range) & (r <= self.max_range)
        r = r[valid]
        b = b[valid]
        if r.size == 0:
            return np.zeros((0, 2), dtype=float)

        # Down-sample to at most max_points by taking every k-th ray.
        if r.size > self.max_points:
            step = int(np.ceil(r.size / self.max_points))
            r = r[::step]
            b = b[::step]

        # Polar -> Cartesian in the robot frame.
        xy = np.empty((r.size, 2), dtype=float)
        xy[:, 0] = r * np.cos(b)
        xy[:, 1] = r * np.sin(b)
        return xy

    def __len__(self):
        return self.points.shape[0]
