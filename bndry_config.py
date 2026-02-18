from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from boundary import CoordPairs, PolygonBoundary
from grid import StructuredGrid

@dataclass(frozen=True, slots=True)
class BaseBoundaryConfig(ABC):
    """
    Abstract boundary config. Subclasses implement to_polygon().
    Code expects ccw boundaries so enforce ccw by default.
    If frac is True one can provide fractional information for grid values.
    """
    ccw:  bool = True
    frac: bool = True

    #TODO: Use CoordPairs type?
    @abstractmethod
    def _from_vals(self) -> CoordPairs:
        """Return (xb, zb) in physical coordinates."""

    @abstractmethod
    def _from_frac(self, grid: StructuredGrid) -> CoordPairs:
        """Return (xb, zb) in physical coordinates from fractional config."""

    def to_boundary(self, grid: StructuredGrid) -> PolygonBoundary:
        """Method which can generate a PolygonBoundary from config info.
        Subclasses override how to use the config info to make points."""
        xb, zb = self._from_frac(grid) if self.frac is True else \
                 self._from_vals()

        #Check all boundary points good at once.
        inside = grid.boundary.contains(xb, zb, combined=True)
        if not inside:
            raise ValueError("Boundary points not contained within simulation domain...")

        return PolygonBoundary(xb, zb)

@dataclass(frozen=True, slots=True)
class RectBoundaryConfig(BaseBoundaryConfig):
    """Simple boundary configuration class which can take four fractional
    locations and generate a rect boundary."""
    x0: float = 0.25
    x1: float = 0.75
    z0: float = 0.25
    z1: float = 0.75
    mode: str = 'coords'

    def _make_vals(self, x0: float, x1: float, z0: float, z1: float) \
             -> CoordPairs:
        xb = np.asarray([x0, x0, x1, x1])
        zb = np.asarray([z0, z1, z1, z0])

        if not self.ccw:
            xb = xb[::-1]
            zb = zb[::-1]

        # Enforce closure after orientation.
        xb = np.append(xb, xb[0])
        zb = np.append(zb, zb[0])

        return xb, zb

    def _from_vals(self) -> CoordPairs:
        """Set up physical values from physical config info."""
        return self._make_vals(self.x0, self.x1, self.z0, self.z1)

    def _from_frac(self, grid: StructuredGrid) -> CoordPairs:
        """
        Creates a rectangular polygon from fractional locations on domain.
        Useful for MMS testing on gridpoints, faces, and between.
        Also works on any domain.
        """
        if (self.x0 >= self.x1 or self.z0 >= self.z1):
            raise ValueError("Invalid rectangle bounds: require left < right.")
        if ((self.x0 < 0 or self.x1 > 1) or (self.z0 < 0 or self.z1 > 1)):
            raise ValueError("Rect boundary config fractions must be in [0,1].")

        xmin, xmax = grid.x[0], grid.x[-1]
        zmin, zmax = grid.z[0], grid.z[-1]

        #Direct domain coords
        if self.mode == 'coords':
            x0 = xmin + self.x0 * (xmax - xmin)
            x1 = xmin + self.x1 * (xmax - xmin)
            z0 = zmin + self.z0 * (zmax - zmin)
            z1 = zmin + self.z1 * (zmax - zmin)
        else:
            #Get nearest index to fractional position.
            i0 = int(grid.x.size * self.x0)
            i1 = int(grid.x.size * self.x1)
            j0 = int(grid.z.size * self.z0)
            j1 = int(grid.z.size * self.z1)

            #Get shift to intended positions.
            shift1 = shift2 = 0.0, 0.0
            if self.mode == 'centers':
                shift1 = shift2 = 0.0
            elif self.mode == 'faces':
                shift1 = shift2 = 0.5
            elif self.mode == 'mid':
                #Shift differently to match bottom-left and top-right for midway.
                shift1, shift2 = 0.75, 0.25
            else:
                raise ValueError("Unknown rect boundary config mode.")

            x0 = xmin + (i0+shift1)*grid.dx
            z0 = zmin + (j0+shift1)*grid.dz
            x1 = xmin + (i1+shift2)*grid.dx
            z1 = zmin + (j1+shift2)*grid.dz

        return self._make_vals(x0, x1, z0, z1)

@dataclass(frozen=True, slots=True)
class CircleBoundaryConfig(BaseBoundaryConfig):
    """Boundary configuration for a circular polygon.

    If frac=False: x0, z0, a are physical values.
    If frac=True:  x0, z0 are fractional positions in the domain and
                   a is a fractional radius (scaled by min(domain lengths)).
    """
    x0: float = 0.50
    z0: float = 0.50
    a: float  = 1/3
    n: int    = 512

    def _make_vals(self, x0: float, z0: float, a: float) -> CoordPairs:
        """Create coordinate pairs from configuration data."""
        #Force endpoint so the boundary is closed.
        th = np.linspace(0.0, 2.0 * np.pi, self.n, endpoint=False)
        if not self.ccw:
            th = th[::-1]

        xb = x0 + a * np.cos(th)
        zb = z0 + a * np.sin(th)

        # Enforce closure after orientation.
        xb = np.append(xb, xb[0])
        zb = np.append(zb, zb[0])

        return xb, zb

    def _from_vals(self) -> CoordPairs:
        """Set up physical values from physical config info."""
        # Interpret x0, z0, a as physical values
        return self._make_vals(self.x0, self.z0, self.a)

    def _from_frac(self, grid: StructuredGrid) -> CoordPairs:
        """
        Creates a circluar polygon from fractional locations on domain.
        Useful for MMS testing and works on any domain.
        """
        if np.min([self.x0, self.z0, self.a]) < 0 \
                or np.max([self.x0, self.z0, self.a]) > 1:
            raise ValueError("Invalid circle boundary fractions, must be in [0,1] at least.")

        # Interpret x0, z0 as domain fractions; a as fractional radius
        xmin, xmax = grid.x[0], grid.x[-1]
        zmin, zmax = grid.z[0], grid.z[-1]

        Lx = xmax - xmin
        Lz = zmax - zmin

        x0 = xmin + self.x0 * Lx
        z0 = zmin + self.z0 * Lz

        #Scale radius by a characteristic domain length.
        #Using the min is probably best so a centered circle always fits.
        a  = self.a * min(Lx, Lz)

        return self._make_vals(x0, z0, a)