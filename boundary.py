"""Boundary geometry and immersed-boundary (IB) helpers.

This module provides:

- PolygonBoundary: a lightweight polygon boundary in the (x,z) plane that
  supports point-in-polygon queries and nearest-point projection onto the wall.

- ImmersedBoundary: grid-based immersed-boundary utilities that operate on a
  boundary-like object (composition). It identifies ghost cells, border cells,
  and computes wall intercept + image points for ghost cells.

- Geometry (Path construction, cleanup, orientation) lives in
  PolygonBoundary.
- ImmersedBoundary does *not* inherit from PolygonBoundary; it takes a boundary
  object.
"""

from __future__ import annotations
from typing import Tuple, TypeAlias
import copy

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import pyplot as plt
import numpy as np

import utils
from utils import DataArray, MaskArray

#Useful type for sets of 2d (x,z usually) data.
CoordPairs: TypeAlias = Tuple[DataArray, DataArray]

class PolygonBoundary:
    """Polygon wall boundary in (x,z)."""

    def __init__(self, xbdy: DataArray, zbdy: DataArray,
            tol: utils.Tolerances = utils.DEFAULT_TOL) -> None:
        x = np.asarray(xbdy, dtype=float)
        z = np.asarray(zbdy, dtype=float)

        if x.shape != z.shape:
            raise ValueError(f"x and z must have same shape; got {x.shape} vs {z.shape}")
        if x.ndim != 1 or x.size < 3:
            raise ValueError("Need 1D arrays with at least 3 vertices for a polygon.")

        #Make a copy of default tolerances, so it can be updated locally.
        self.tol = copy.deepcopy(tol)

        #Clean up and generate boundary from points.
        self.xbdy, self.zbdy = self._clean_up_points(x, z)
        self.num_pts = int(self.xbdy.size)

        self.path = self._build_path()

        self.ccw = self._compute_orientation_ccw()
        #Set edge bias, which depends on orientation.
        self._set_edge_bias_sign()

    def _build_path(self) -> Path:
        """Build a matplotlib Path from straight line segments and explicitly mark closed."""
        verts = np.column_stack([self.xbdy, self.zbdy])
        codes = np.full(len(verts), Path.LINETO, dtype=np.uint8)
        #Set start and end vertex information per docs.
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        return Path(verts, codes)

    def _compute_orientation_ccw(self) -> bool:
        """Return True if polygon is CCW using shoelace area sign."""
        x, y = self.path.vertices[:, 0], self.path.vertices[:, 1]
        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        if np.isclose(area, 0.0, atol=self.tol.path_tol):
            raise ValueError("Path has zero (or numerically zero) area.")
        return bool(np.sign(area) > 0)

    def _set_edge_bias_sign(self) -> None:
        """Flip `path_edge_bias` to be consistent with polygon orientation."""
        bias_sign = np.sign(self.tol.path_edge_bias)
        # Convention: bias should point *inside*. If orientation disagrees, flip.
        if (self.ccw and bias_sign < 0) or ((not self.ccw) and bias_sign > 0):
            self.tol.path_edge_bias *= -1

    def _clean_up_points(self, xpts: DataArray, zpts: DataArray) -> CoordPairs:
        """Ensure closure, drop zero-length segments, drop redundant midpoints.

        Returns cleaned (x, z) arrays.
        """
        #Exact-closure check
        abs_tol = self.tol.closed_path_tol
        end_gap = np.hypot(xpts[-1] - xpts[0], zpts[-1] - zpts[0])
        if end_gap > abs_tol:
            utils.logger.warn(f"First and last wall points not equal within tol {abs_tol} (gap={end_gap}). "
                f"Closing polygon by appending the first point {(xpts[0],zpts[0])};"
                f"recommend visual verification of closed wall.")
            xpts = np.append(xpts, xpts[0])
            zpts = np.append(zpts, zpts[0])

        #Remove zero-length segments (preserve final closure point)
        x_ok = np.abs(np.diff(xpts)) > abs_tol
        z_ok = np.abs(np.diff(zpts)) > abs_tol
        keep = np.concatenate([x_ok | z_ok, np.array([True], dtype=bool)])
        xpts, zpts = xpts[keep], zpts[keep]

        #Remove unnecessary points on straight horizontal/vertical runs.
        #TODO: Clean up slanted lines too? Need angular tolerance.
        straight_tol = self.tol.path_tol
        drop_mid_x = (np.abs(np.diff(xpts[:-1])) <= straight_tol) & (np.abs(np.diff(xpts[1:])) <= straight_tol)
        drop_mid_z = (np.abs(np.diff(zpts[:-1])) <= straight_tol) & (np.abs(np.diff(zpts[1:])) <= straight_tol)
        drop_mid = np.concatenate([[False], drop_mid_x | drop_mid_z, [False]])
        keep = ~drop_mid
        xpts, zpts = xpts[keep], zpts[keep]

        return xpts, zpts

    def contains(self, xpts, zpts, combined: bool = False) -> MaskArray | bool:
        """
        Vectorized point-in-polygon check with edge bias.
        Note, if points on boundary biased inside, then solver
        needs data at the border to run those cells.
        """
        xb, zb = np.broadcast_arrays(xpts, zpts)
        pts    = np.column_stack([xb.ravel(), zb.ravel()])
        inside = self.path.contains_points(pts, radius=float(self.tol.path_edge_bias))
        inside = inside.reshape(xb.shape)
        #Return 0-D ndarray for scalars which is truth-y; return mask array for higher dims.
        #If all is False, then combine all bools to single value.
        if inside.ndim == 0:
            return inside
        else:
            if combined:
                return np.all(inside)
            else:
                return inside

    def plot(self, ax=None, *, show_vertices: bool = True, **kwargs):
        """Convenience plot for quick debugging."""
        if ax is None:
            _, ax = plt.subplots()
        patch = PathPatch(self.path, facecolor="none", edgecolor="k", lw=2, **kwargs)
        ax.add_patch(patch)
        if show_vertices:
            ax.scatter(self.path.vertices[:, 0], self.path.vertices[:, 1], s=12, c="k")
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        return ax