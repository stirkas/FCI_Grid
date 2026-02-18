"""
Structured grid classes for fusion simulations.
Grids store 1D coordinate vectors at cell centers (including ghosts).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from boundary import PolygonBoundary
import utils

#TODO: Can integrate zoidberg logic here. Or use subclasses with find_index override.
class GridType(str, Enum):
    """Type of coordinate system to be used."""
    CARTESIAN = "cartesian" #Full FCI.
    POLOIDAL  = "poloidal"  #Quasi-FCI (zoidberg).

@dataclass(frozen=True)
class GhostLayers:
    """Ghost-layer widths for each axis. Default to BOUT++ expectations."""
    x: int = utils.DEFAULT_XGUARDS
    y: int = utils.DEFAULT_YGUARDS
    z: int = utils.DEFAULT_ZGUARDS

@dataclass(frozen=True)
class UniformAxisSpec:
    """Uniform axis specification for the *core* (non-ghost) region."""
    start: float
    stop: float
    n: int
    shift: float = 0.0
    endpoint: bool = False #TODO: Handle endpoint=True?

class StructuredGrid:
    """Basic grid class for use in BOUT++ with FCI method."""
    #The default y coord spec is useful for mms testing.
    DEFAULT_Y_SPEC = UniformAxisSpec(0.0, 1.0, 1)

    def __init__(self, x: UniformAxisSpec, z: UniformAxisSpec,
            grid_type: GridType = GridType.CARTESIAN,
            ghosts: GhostLayers = GhostLayers(),
            y: Optional[UniformAxisSpec] = DEFAULT_Y_SPEC,
            axisymmetric: Optional[bool] = False) -> None:
        """Create a Grid from uniform x/z bounds, optionally with a y axis."""

        x_core, dx = self._make_axis_array(spec=x)
        y_core, dy = self._make_axis_array(spec=y)
        z_core, dz = self._make_axis_array(spec=z)

        if axisymmetric:
            y_core = y_core[0]

        x_full = self._add_ghosts(core=x_core, d=dx, ng=ghosts.x)
        z_full = self._add_ghosts(core=z_core, d=dz, ng=ghosts.z)

        #Note, BOUT++ adds y ghosts manually so different here.
        self.x, self.dx = x_full, dx
        self.y, self.dy = y_core, dy
        self.z, self.dz = z_full, dz

        self.lx = self.x[-1] - self.x[0]
        #TODO: Allow for ly = 1 for mms testing purposes and cell volume calcs?
        self.ly = self.y[-1] - self.y[0] if not axisymmetric else self.dy
        self.lz = self.z[-1] - self.z[0]

        self.xx, self.zz = np.meshgrid(self.x, self.z, indexing='ij')

        self.grid_type = grid_type
        self.ghosts = ghosts

        #Create boundary, useful for testing points are in simulation domain.
        xb = [self.x[0+ghosts.x], self.x[0+ghosts.x], self.x[-1-ghosts.x],
                self.x[-1-ghosts.x], self.x[0+ghosts.x]]
        zb = [self.z[0+ghosts.z], self.z[-1-ghosts.z], self.z[-1-ghosts.z],
                self.z[0+ghosts.z], self.z[0+ghosts.z]]
        self.boundary = PolygonBoundary(xb,zb)

        self._validate()

    def _validate(self) -> None:
        if self.x.size < 2 or self.z.size < 2:
            raise ValueError("Need at least two points in x and z.")
        if self.y.size < 1:
            raise ValueError("Need at least one point in y.")
        if min(self.ghosts.x, self.ghosts.y, self.ghosts.z) < 0:
            raise ValueError("Ghost widths must be non-negative.")

    @property
    def is_poloidal(self) -> bool:
        """Return true if toroidal."""
        return self.grid_type == GridType.POLOIDAL

    @property
    #TODO: Remove this eventually and work only in 3d?
    def grid_shape_2d(self) -> Tuple[int, ...]:
        """Return the grid dimensions in (x,z)."""
        return (int(self.x.size), int(self.z.size))

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        """Return the length of the grid dimensions."""
        return (int(self.x.size), int(self.y.size), int(self.z.size))

    def _make_axis_array(self, spec: UniformAxisSpec) -> Tuple[np.ndarray, float]:
        """
        Generate cell centers for a uniform axis. Note the endpoint is dropped by arange,
        and this matches the default in BOUT++ since x is shifted and y/z are usually periodic.
        """
        if spec.n <= 0:
            raise ValueError("n must be positive.")
        if spec.stop <= spec.start:
            raise ValueError("stop must be greater than start.")

        d = (spec.stop - spec.start) / float(spec.n)
        arr = spec.start + d*(spec.shift + np.arange(spec.n))
        return arr, d

    def _add_ghosts(self, core: np.ndarray, d: float, ng: int) -> np.ndarray:
        """Pad a 1D cell-center vector with `ng` ghost points on each side."""
        if ng < 0:
            raise ValueError("ng must be non-negative.")
        if ng == 0:
            return np.asarray(core, dtype=float)

        lo = core[0]  - d * np.arange(ng, 0, -1, dtype=float)
        hi = core[-1] + d * np.arange(1, ng + 1, dtype=float)
        return np.concatenate((lo, core, hi)).astype(float)

    def centers(self, axis: str) -> np.ndarray:
        """Get a grid axis by name."""
        if axis == "x":
            return self.x
        if axis == "z":
            return self.z
        if axis == "y":
            if self.y is None:
                raise ValueError("No y-axis available for this grid.")
            return self.y
        raise ValueError(f"Unknown axis '{axis}'.")

    def find_index(self, x: utils.DataArray, z: utils.DataArray) \
             -> Tuple[utils.DataArray, utils.DataArray]:
        """Find the grid indices corresponding to specific locations."""
        if self.grid_type == GridType.POLOIDAL:
            raise ValueError("Need to implement zoidberg logic in this case.")

        xind = (x - self.x[0])/self.dx
        zind = (z - self.z[0])/self.dz

        #If original points were grid points need to round/snap due to floating errors.
        xi, zi = np.rint(xind), np.rint(zind) #Nearest integer
        #Note, rtol=0.0 just ignores rtol and only uses atol here.
        round_x = np.isclose(xind, xi, atol=utils.DEFAULT_TOL.path_tol, rtol=0.0)
        round_z = np.isclose(zind, zi, atol=utils.DEFAULT_TOL.path_tol, rtol=0.0)

        #Take snapped value when isclose True.
        xind = np.where(round_x, xi, xind)
        zind = np.where(round_z, zi, zind)

        return xind, zind