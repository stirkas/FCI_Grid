from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

#TODO: Use freegs? Where hypnotoad gets it from initially. Or write own pythonically (cough cough chatGPT).
from hypnotoad.utils.critical import find_critical as fc

from boundary import PolygonBoundary
from sim_config import SimConfig
from field import MagneticField, ToroidalField
from grid import StructuredGrid, UniformAxisSpec
from data import TokamakData
from device import DeviceInfo, Device, BOUT_IO
import utils

@dataclass(slots=True)
class TokamakConfig(SimConfig):
    """Tokamak simulations setup with axisymmetry and toroidal curvature."""
    gfile: str = ""
    sptx: Optional[PolygonBoundary] = None
    xpts: Optional[utils.DataArray] = None
    opts: Optional[utils.DataArray] = None

    def __post_init__(self) -> None:
        SimConfig.validate(self) #Can't call super.validate() for some reason...
        if not self.gfile:
            raise ValueError("TokamakConfig requires gfile path.")

    def _set_filename(self) -> None:
        self.filename = "tokamak"

    def _load_data(self) -> TokamakData:
        """Read device data from EQDSK file."""
        return TokamakData(self.gfile)
    
    def _build_info(self) -> DeviceInfo:
        """Set up tokamak device info."""
        return DeviceInfo(axisymmetric=True, toroidal=True)

    def _build_grid(self, data: TokamakData) -> StructuredGrid:
        """Setup the grid, specific to a tokamak configuration."""
        x = UniformAxisSpec(data.rmin, data.rmax, self.nx)
        z = UniformAxisSpec(data.zmin, data.zmax, self.nz)
        y = UniformAxisSpec(0.0, 2*np.pi, self.ny)

        return StructuredGrid(x=x, z=z, y=y, ghosts=self.ghosts, axisymmetric=True)

    def _build_field(self, data: TokamakData, grid: StructuredGrid) -> MagneticField:
        psi =  data.psi_func(grid.x, grid.z)
        B_R =  data.sign_ip*data.psi_func(grid.x, grid.z, dy=1)/grid.xx
        B_Z = -data.sign_ip*data.psi_func(grid.x, grid.z, dx=1)/grid.xx
        Bphi = data.f_spl(psi)/grid.xx

        return ToroidalField(grid=grid, B_R=B_R, B_Z=B_Z, Bphi=Bphi,
                            psi=psi, direction=data.sign_b0, pres=data.pres)

    def _build_device_wall(self, data: TokamakData) -> PolygonBoundary:
        #Build separatrix and wall from gfile data.
        #TODO: For zoidberg, allow overriding wall w/ sptx or specific flux surface?
        print("Generating separatrix boundary path...")
        self.sptx = PolygonBoundary(data.rbdy, data.zbdy)
        #For now manually remove problematic points from tokamak walls.
        print("Generating wall boundary path...")
        wall_pts_x, wall_pts_z = self._remove_target_pts(data.rlmt, data.zlmt, nx=self.nx, ny=self.ny, nz=self.nz)
        wall = PolygonBoundary(wall_pts_x, wall_pts_z)

        #TODO: Maybe have a final setup function?
        #Do this here to remove pts in wall, but if sptx or flux surface don't?
        self.xpts, self.opts = self._crit_pts(data, wall)

        return wall

    def _crit_pts(self, eq_data: TokamakData, wall: PolygonBoundary) -> Tuple[utils.DataArray, utils.DataArray]:
        #Use default settings from hypnotoad examples.
        sep_atol, sep_maxits = 1e-5, 1000
        #TODO: Does this require eq data? Or can be done on this grid? Then would need field...
        opoints, xpoints = fc(eq_data.rr, eq_data.zz, eq_data.psi, sep_atol, sep_maxits)
        #Remove points outside the wall. Not enough to remove all non-important points it turns out.
        #TODO: Can try removing points within certain flux surface outside of LCFS?
        # |--->  Probably use psi spline from R,Z to drop points a bit outside LCFS.
        for point in opoints[:]:
            if not wall.contains(point[0], point[1]):
                opoints.remove(point)
        for point in xpoints[:]:
            if not wall.contains(point[0], point[1]):
                xpoints.remove(point)

        return np.asarray(xpoints), np.asarray(opoints)

    def _remove_target_pts(self, x, z, nx: Optional[int], ny: Optional[int], nz: Optional[int], tol=1e-2):
        #Note: Manual points from default gfile...
        if "DIIID" in self.gfile:
            #DIIID sharp areas.
            utils.logger.info("Removing sharp features manually from DIIID wall. Check ok with --debug.")
            targets = np.array([
                (2.3770,  0.3890),
                (2.3770, -0.3890),
                (1.0121,  1.1648),
                (1.0009,  1.2172),
                (1.0293,  1.2170)])
        elif "TCV" in self.gfile:
            #TCV wall very close to sim domain so check for sufficient grid res.
            #TODO: Dont need nx,ny,nz args if these checks not needed.
            if (nx < 256 or nz < 256):
                raise ValueError("Resolution for x,z too small for TCV. Wall is near gfile domain, so use resolutions higher than 256.")
            if (nz < 2*nx):
                raise ValueError("Due to TCV elongation, forcing nz >= 2*nx at the moment.")
            if (ny < 128):
                raise ValueError("TCV field following doesnt always work great, increasing parallel res helps. Forcing ny >= 128 for now.")
            #TCV sharp areas.
            utils.logger.info("Removing sharp features manually from TCV wall. Check ok with --debug.")
            targets = [] 
            for i, xpt in enumerate(x):
                #Remove all points within this larger region to clean up the shape.
                if x[i] >= 0.90 and (z[i] <= -0.23 and z[i] >= -0.60):
                    #Keep a couple points to retain the general shape.
                    if (not (np.abs(x[i] - 1.14) <= tol and \
                             np.abs(z[i] - -0.55887) <= tol)) or \
                       (not (np.abs(x[i] - 1.14)  <= tol and \
                             np.abs(z[i] - -0.55188) <= tol)):
                        targets.append((x[i],z[i]))
        else:
            utils.logger.warn("Tokamak device not specified for manual boundary point removal." \
                " Sharp boundary wall regions will cause problems if using immersed boundary methods.")

        pts = np.column_stack([x, z])
        remove = np.zeros(len(pts), dtype=bool)
        for xt, zt in targets:
            d2 = (pts[:,0]-xt)**2 + (pts[:,1]-zt)**2
            j = np.argmin(d2)
            if d2[j] <= tol**2:
                remove[j] = True
        kept = pts[~remove]
        removed_pts = pts[remove]

        if utils.DEBUG_FLAG:
            fig, ax = plt.subplots()
            
            # --- Original full boundary (dashed red) ---
            ax.plot(pts[:, 0], pts[:, 1],
                    linestyle="--", color="red", linewidth=1.5,
                    label="Original boundary")

            # --- Final kept boundary (solid black) ---
            ax.plot(kept[:, 0], kept[:, 1],
                    linestyle="-", color="black", linewidth=2,
                    label="Final boundary")

            #Plot kept points as black dots
            ax.plot(kept[:, 0], kept[:, 1],'k.')

            #Plot removed points as red X's
            ax.plot(removed_pts[:, 0], removed_pts[:, 1], 'rx', markersize=8)

            ax.set_aspect("equal", "box")
            ax.set_xlabel("R")
            ax.set_ylabel("Z")
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper center',
                bbox_to_anchor=(0.5, 1.06), ncol=len(labels), fancybox=True, shadow=True)
            plt.show()

        return kept[:,0], kept[:,1]

    def _plot_device_config(self, device: Device, bout_io: BOUT_IO, fig: Figure, ax: Axes) -> None:
        field, grid, wall = device.field, device.grid, device.wall

        #Add separatrix to plot.
        sptx = PathPatch(self.sptx.path, fill=False, lw=2,
                    edgecolor='orange', label='LCFS', linestyle='-.')
        ax.add_patch(sptx)

        #Trace field lines in both directions.
        offset = 0.005 #Use minor radial offset from separatrix.
        rsep, zsep = self.sptx.path.vertices[:,0], self.sptx.path.vertices[:,1]
        #Get max major radius to use as starting point for a trace.
        sep_idx = np.argmax(self.sptx.path.vertices[:,0])
        R1, Z1     = rsep[sep_idx] + offset, zsep[sep_idx]

        print("Tracing field line to wall in forward direction"
            f" from {R1,Z1} near separatrix...")
        Rvals_pos, Zvals_pos = field.tracer.trace_until_wall(R1, Z1, grid.y, grid.dy,
                                                    wall, direction=field.direction)
        print("Tracing field line to wall in backward direction"
            f" from {R1,Z1} near separatrix...")
        Rvals_neg, Zvals_neg = field.tracer.trace_until_wall(R1, Z1, grid.y, grid.dy,
                                                    wall, direction=-field.direction)
        phi_dir, neg_phi_dir = ('+','-') if field.direction == 1 else ('-','+')
        ax.plot(Rvals_pos, Zvals_pos, '.', color='red',
            label='$+\\hat{b}_{\\phi} = ' + phi_dir     + '\\hat{\\phi}$')
        ax.plot(Rvals_neg, Zvals_neg, '.', color='cyan',
            label='$-\\hat{b}_{\\phi} = ' + neg_phi_dir + '\\hat{\\phi}$')

        #Test field line tracing on grid.
        if (utils.DEBUG_FLAG):
            print("Testing field line following on gridpoints...")
            #Get scatter point data to plot and test grid point following in general.
            gridPts = np.column_stack((grid.xx.ravel(), grid.zz.ravel()))
            fwdPts  = np.column_stack((bout_io.maps["forward_R"].ravel(),  bout_io.maps["forward_Z"].ravel()))
            bwdPts  = np.column_stack((bout_io.maps["backward_R"].ravel(), bout_io.maps["backward_Z"].ravel()))
            step    = gridPts.shape[0]//100 #Divide grid into 100 equally spaced points.
            indices = np.arange(0, gridPts.shape[0], step)
            gridPts = gridPts[indices]
            fwdPts  = fwdPts[indices]
            bwdPts  = bwdPts[indices]
            #Remove points outside the wall for all three arrays at once.
            keptPts = [(g, f, b) for (g, f, b) in zip(gridPts, fwdPts, bwdPts)
                    if wall.contains(g[0], g[1])]
            gridPts, fwdPts, bwdPts = map(list, zip(*keptPts))

            for idx, point in enumerate(gridPts):
                x0, y0 = gridPts[idx]
                xf, yf = fwdPts[idx]
                xb, yb = bwdPts[idx]
                ax.plot([x0, xf], [y0, yf], '-', color='red', linewidth=2)
                ax.plot([x0, xb], [y0, yb], '--', color='cyan', linewidth=2)
                ax.scatter(x0, y0, color='k', s=100, marker='*', zorder=2)

        #Add critical points.
        for point in self.opts:
            ax.plot(point[0], point[1], 'o', label='O',
                    markerfacecolor='none', markeredgecolor='lime')
        for point in self.xpts:
            ax.plot(point[0], point[1], 'x', color='lime', label='X')