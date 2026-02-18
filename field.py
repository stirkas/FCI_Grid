#!/usr/bin/env python3
"""Magnetic field definitions and field-line tracing utilities."""

from __future__ import annotations
from typing import Callable, Optional, Tuple

import numpy as np
import scipy as sp

from grid import StructuredGrid
from boundary import PolygonBoundary
from utils import DataArray

FieldFunc = Callable[[DataArray, DataArray, DataArray], DataArray]

class FieldTracer:
    """Trace field lines in (x, z) using y as the independent variable.

    The tracer consumes precomputed arrays for dx/dy and dz/dy on a regular
    (x, z) grid and builds cached interpolators.
    """

    #TODO: Add errors to tolerance class in utils?
    def __init__(self, x: DataArray, z: DataArray, dx_dy: DataArray,
            dz_dy: DataArray, *, rtol: float = 1e-10, atol: float = 1e-12) -> None:
        self._rtol = rtol
        self._atol = atol

        self._dx_itp = sp.interpolate.RectBivariateSpline(x, z, dx_dy)
        self._dz_itp = sp.interpolate.RectBivariateSpline(x, z, dz_dy)

    def rhs(self, y: float, pos: DataArray) -> DataArray:
        """RHS for field-line integration in y."""
        x, z = float(pos[0]), float(pos[1])

        dx = float(np.squeeze(self._dx_itp(x,z)))
        dz = float(np.squeeze(self._dz_itp(x,z)))

        return np.array([dx, dz], dtype=float)

    def trace(self, x0: float, z0: float, y0: float, y1: float):
        """Integrate a single segment from y0 to y1."""
        return sp.integrate.solve_ivp(
            self.rhs, (y0, y1), [x0, z0], method="RK45",
            rtol=self._rtol, atol=self._atol, dense_output=False)

    def trace_until_wall(self, x0: float, z0: float, y0: float, dy: float,
            wall: PolygonBoundary, *, direction: int = 1) -> Tuple[DataArray, DataArray]:
        """Step forward in y until wall.contains(R, Z) becomes False."""
        direction = 1 if direction >= 0 else -1

        x_vals, z_vals, y = [float(x0)], [float(z0)], y0

        while True:
            y_next = y + direction * dy
            sol = self.trace(x_vals[-1], z_vals[-1], y, y_next)

            r_new = float(sol.y[0, -1])
            z_new = float(sol.y[1, -1])

            if not wall.contains(r_new, z_new):
                break

            x_vals.append(r_new)
            z_vals.append(z_new)
            y = y_next

        return x_vals, z_vals

#TODO: For 3d generate tracers at each plane. Maybe tracers then belong to the device and are made from the field.
class MagneticField:
    """Magnetic field stored on a StructuredGrid using logical components (Bx, By, Bz)."""

    def __init__(self, grid: StructuredGrid, Bx: DataArray, By: DataArray,
            Bz: DataArray, *, direction: int = 1,
            pres: Optional[DataArray] = None) -> None:
        self.grid = grid
        self.direction = 1 if direction >= 0 else -1

        self._Bx = Bx
        self._By = By
        self._Bz = Bz

        #TODO: Better way to handle equilibrium quantities?
        if pres is not None:
            self.pres = pres
        else:
            self.pres = np.ones_like(Bx)

        self._validate_shapes()

        self._tracer = self._make_tracer()

    def _validate_shapes(self) -> None:
        expected = (int(self.grid.x.size), int(self.grid.z.size))
        if (self._Bx.shape, self._By.shape, self._Bz.shape) != (expected, expected, expected):
            raise ValueError(
                "Field arrays must be shaped (nx, nz) at cell centers. "
                f"Expected {expected}, got bx={self._Bx.shape}, by={self._By.shape}, bz={self._Bz.shape}.")

    @property
    def Bx(self) -> DataArray:
        """Getter for Bx component."""
        return self._Bx

    @property
    def By(self) -> DataArray:
        """Getter for By component."""
        return self._By

    @property
    def Bz(self) -> DataArray:
        """Getter for Bz component."""
        return self._Bz

    @property
    def Bmag(self) -> DataArray:
        """Getter for field magnitude."""
        return np.sqrt(self._Bx**2 + self._By**2 + self._Bz**2)
    
    @property
    def tracer(self) -> FieldTracer:
        """Getter for access to field line tracer."""
        return self._tracer
    
    def _make_tracer(self) -> FieldTracer:
        """Create a y-parameterized tracer."""

        if np.min(np.abs(self._By)) < 1e-14:
            raise ValueError("By is too small somewhere on the grid; can't divide by By to trace...")

        par_fac = self._trace_metric_factor()

        dx_dy = par_fac * self._Bx / self._By
        dz_dy = par_fac * self._Bz / self._By

        return FieldTracer(x=self.grid.x, z=self.grid.z, dx_dy=dx_dy, dz_dy=dz_dy)
    
    def _trace_metric_factor(self) -> DataArray:
        """
        Factor multiplying (Bx/By, Bz/By) for field-line tracing.
        Default is 1 (Cartesian-style y parameterization). Subclasses
        can override as needed for now.
        """
        return np.ones_like(self._Bx, dtype=float)

    @classmethod
    def from_functions(cls, grid: StructuredGrid,
            Bx_func: FieldFunc, By_func: FieldFunc, Bz_func: FieldFunc,
            *, direction: int = 1, y0: float = 0.0) -> MagneticField:
        """Build a field by sampling f(x, y, z) at cell centers.

        Notes:
            - The callable signature is always (x, y, z) to avoid 2D/3D ambiguity.
            - For axisymmetric/2D cases, phi is a scalar (default 0.0).
        """
        Bx = Bx_func(grid.xx, y0, grid.zz)
        By = By_func(grid.xx, y0, grid.zz)
        Bz = Bz_func(grid.xx, y0, grid.zz)

        return cls(grid=grid, Bx=Bx, By=By, Bz=Bz, direction=direction)


class ToroidalField(MagneticField):
    """Geometry-aware toroidal field with physical cylindrical components."""

    def __init__(self, grid: StructuredGrid, B_R: DataArray, B_Z: DataArray, Bphi: DataArray,
            *, psi: DataArray = None, direction: int = 1,
            pres: Optional[DataArray] = None) -> None:
        """Initialize a toroidal field. Allow for the possibility of psi."""

        #TODO: Best way to handle psi in general?
        if psi is not None:
            self.psi = psi
        self._B_R  = B_R
        self._B_Z  = B_Z
        self._Bphi = Bphi

        super().__init__(grid=grid, Bx=self._B_R, By=self._Bphi, Bz=self._B_Z,
                    direction=direction, pres=pres)
    
    def _trace_metric_factor(self) -> DataArray:
        """Return x=R at cell centers for y=phi parameterization."""
        return self.grid.xx

    @property
    def Bpol(self) -> DataArray:
        """Getter for calculating poloidal field magnitude."""
        return np.sqrt(self._B_R**2 + self._B_Z**2)