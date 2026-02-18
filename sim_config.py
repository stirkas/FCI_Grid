from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import uuid

from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from boututils import datafile as bdata
from hypnotoad import __version__ #TODO: Remove hypnotoad dependencies?

from boundary import CoordPairs, PolygonBoundary
from bndry_config import BaseBoundaryConfig, RectBoundaryConfig
from field import MagneticField
from grid import GhostLayers, StructuredGrid
from fci_generator import FCI_Generator
import utils
from device import Device, DeviceInfo, BOUT_IO

@dataclass(slots=True)
class SimConfig(ABC):
    """
    Base simulation config: build inputs, then delegate device information to a
    Generator. Prefer a dataclass here for flexible/easier instantiation, since
    this class encapsulates everything and its easier than specifying every
    __init__ var down through all classes/subclasses. However, because the base
    class sets defaults, the subclasses all need to default variables.
    """

    nx: int = 64
    nz: int = 64 + 2*utils.DEFAULT_XGUARDS #TODO: Default to 64x64? Not 68x68?
    ny: int = 64

    ghosts: GhostLayers = GhostLayers()

    #Flag for immersed boundary.
    use_ib: bool = True
    #TODO: If zoidberg logic integrated can set this True to use that.
    #TODO: Use quasi_fci to set grid config to poloidal regardless of device type?
    quasi_fci: bool = False

    #Shared wall inputs (generic across devices) ----
    #Can take either a boundary config or raw x/z point arrays.
    bdy_cfg:  Optional[BaseBoundaryConfig] = None  #Circle/square/etc.
    wall_pts: Optional[CoordPairs] = None          #Raw polygon arrays

    #Variable for storing output filename, set with abstract method.
    filename: str = ""

    #Show configuration plot after finished.
    show_plot: bool = True

    def validate(self) -> None:
        """Check that everything looks reasonable."""
        if self.nx <= 0 or self.nz <= 0 or self.ny <= 0:
            raise ValueError("nx,ny,nz must be positive.")

    #Device-specific hooks.
    @abstractmethod
    def _set_filename(self) -> None:
        """Abstract function for setting an output file name."""

    @abstractmethod
    def _load_data(self) -> Any:
        """Abstract function for loading device information from a file."""

    @abstractmethod
    def _build_grid(self, data: Any) -> StructuredGrid:
        """Abstract function for building the device-specific grid."""

    @abstractmethod
    def _build_field(self, data: Any, grid: StructuredGrid) -> MagneticField:
        """Abstract function for building the device-specific field."""

    @abstractmethod
    def _build_device_wall(self, data: Any) -> PolygonBoundary:
        """Device-specific fallback (tokamak from EQDSK, linear rectangle, etc.)
        for the boundary wall. Assumed to come from the device data file."""

    @abstractmethod
    def _build_info(self) -> DeviceInfo:
        """Abstract function for building device-specific information.
        Will default to simplest characteristics."""

    @abstractmethod
    def _plot_device_config(self, device: Device, bout_io: BOUT_IO, fig: Figure, ax: Axes) -> None:
        """Abstract function to allow subclasses to add anything to
        the final plotting routine."""

    #Shared wall builder logic.
    def _build_wall(self, data: Any, grid: StructuredGrid) -> PolygonBoundary:
        """
        Shared wall selection policy (allow for overriding device data if
        something passed in explicitly):
          1) boundary config (CircleBoundaryConfig / SquareBoundaryConfig / etc.)
          2) raw arrays wall_x/wall_z
          3) device default via _build_default_wall()
        """
        if self.bdy_cfg is not None:
            return self.bdy_cfg.to_boundary(grid)
        elif self.wall_pts is not None:
            #TODO: Check wall points in sim domain with grid.
            return PolygonBoundary(self.wall_pts[:,0], self.wall_pts[:,1])
        else:
            return self._build_device_wall(data)

    def _build(self) -> Device:
        """Build a device using functions specified by specific device configurations."""
        self.validate()
        self._set_filename()
        dvc_info  = self._build_info()
        data      = self._load_data()
        grid      = self._build_grid(data)
        mag_field = self._build_field(data, grid)
        wall      = self._build_wall(data, grid)

        return Device(data=data, grid=grid, field=mag_field, wall=wall, dvc_info=dvc_info)

    def generate(self) -> None:
        device = self._build()
        generator = FCI_Generator(nx=self.nx, ny=self.ny, nz=self.nz,
                                ib=self.use_ib, device=device)
        bout_io = generator.generate()
        self.write_output(device, bout_io)
        self.plot_config(device, bout_io)

    def write_output(self, device: Device, bout_io: BOUT_IO) -> None:
        #Write output to data file.
        gridfile = self.filename + ".fci.nc"
        print("Writing to " + str(gridfile) + "...")
        with bdata.DataFile(gridfile, write=True, create=True, format="NETCDF4") as f:
            f.write_file_attribute("title", "BOUT++ FCI grid file")
            #TODO: Need an official name...
            f.write_file_attribute("software_name", "fci-grid")
            f.write_file_attribute("software_version", __version__)
            grid_id = str(uuid.uuid1())
            f.write_file_attribute("id", grid_id)      #Conventional name
            f.write_file_attribute("grid_id", grid_id) #BOUT++ specific name

            shape = device.grid.grid_shape
            f.write("nx", shape[0])
            f.write("ny", shape[1])
            f.write("nz", shape[2])

            #For BSTING w/ full FCI set sep indices to nx+1. Used just for MPI comms in FCI code.
            #TODO: Z not poloidal anymore so not sure one limit works?
            #TODO: For quasi-fci or general inner boundary integrate zoidberg logic?
            f.write("ixseps1", shape[0] + 1)
            f.write("ixseps2", shape[0] + 1)

            for key, value in bout_io.metric.items():
                f.write(key, value)

            for key, value in bout_io.attributes.items():
                f.write(key, value)

            for key, value in bout_io.maps.items():
                f.write(key, value)

    def plot_config(self, device: Device, bout_io: BOUT_IO) -> None:
        """Device-agnostic configuration plot with optional overlays."""
        print("Plotting equilibrium configuration...")
        #TODO: Depend on domain size/aspect ratio?
        fig, ax = plt.subplots(figsize=(8,10))

        #TODO: What to plot for ny != 1?
        #Note, use Bmag if psi not defined/known.
        background = device.field.Bmag if not hasattr(device.field, "psi") else device.field.psi.T
        plot_func = ax.contour if utils.DEBUG_FLAG is True else ax.contourf
        cf = plot_func(device.grid.x, device.grid.z, background, levels=100, cmap='viridis')
        plt.colorbar(cf, ax=ax)
        
        #Plot various boundaries.
        #Build the ghost boundary wall. Plotted for now just as a reminder that its used in BOUT++ (and MZG = 0 always).
        ghost_bndry = RectBoundaryConfig(x0=device.grid.x[0  + self.ghosts.x],
                                         x1=device.grid.x[-1 - self.ghosts.x],
                                         z0=device.grid.z[0  + self.ghosts.z],
                                         z1=device.grid.z[-1 - self.ghosts.z],
                                         frac=False).to_boundary(device.grid)
        ghost_path = PathPatch(ghost_bndry.path, fill=False, edgecolor='k', label='ghost',
                    clip_on=False, lw=1.5, linestyle='--') #Clip false since ghost bndry can be on edge if 0 ghosts.
        ax.add_patch(ghost_path)
        if device.wall is not None:
            wall_path = PathPatch(device.wall.path, fill=False, edgecolor='k', lw=2)
            ax.add_patch(wall_path)

        if (device.dvc_info.toroidal is True):
            ax.set_xlabel('R')
            ax.set_ylabel('Z')
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('z')

        self._plot_device_config(device, bout_io, fig, ax)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper center',
            bbox_to_anchor=(0.5, 1.06), ncol=len(labels), fancybox=True, shadow=True)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        if self.show_plot is True:
            plt.show()