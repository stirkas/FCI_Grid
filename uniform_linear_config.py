from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

from boundary import PolygonBoundary
from device import Device, DeviceInfo, BOUT_IO
from field import MagneticField
from grid import StructuredGrid, UniformAxisSpec
from sim_config import SimConfig
import utils

@dataclass(slots=True)
class UniformLinearConfig(SimConfig):
    """
    Linear device with uniform By field.
    Uses default y-spec where y = [0,1], w/ ny=1.
    Generally useful for 2d MMS testing.
    """

    xmin: float = utils.DEFAULT_START
    xmax: float = utils.DEFAULT_STOP
    zmin: float = utils.DEFAULT_START
    zmax: float = utils.DEFAULT_STOP

    By0: float = 1.0

    def __post_init__(self) -> None:
        SimConfig.validate(self)
        if self.xmax <= self.xmin or self.zmax <= self.zmin:
            raise ValueError("Invalid bounds for linear config.")

    #TODO: pass doesn't technically match the return types here...double check what this returns.
    def _load_data(self) -> Any:
        pass

    def _build_info(self) -> DeviceInfo:
        return DeviceInfo()

    def _set_filename(self) -> None:
        pass

    def _build_grid(self, data: Any) -> StructuredGrid:
        #TODO: Dont really need to shift to match hermes anymore. Useful for MMS testing at the moment?
        x = UniformAxisSpec(self.xmin, self.xmax, self.nx, shift=0.50)
        z = UniformAxisSpec(self.zmin, self.zmax, self.nz, shift=0.50)

        return StructuredGrid(x=x, z=z, ghosts=self.ghosts)

    def _build_field(self, data: Any, grid: StructuredGrid) -> MagneticField:
        """Uniform magnetic field along y."""
        Bx = np.zeros_like(grid.xx)
        By = self.By0*np.ones_like(grid.xx)
        Bz = np.zeros_like(grid.xx)

        return MagneticField(grid=grid, Bx=Bx, By=By, Bz=Bz, direction=np.sign(self.By0))

    def _build_device_wall(self, data: Any) -> PolygonBoundary:
        pass

    def _plot_device_config(self, device: Device, bout_io: BOUT_IO, fig: Figure, ax: Axes) -> None:
        pass
