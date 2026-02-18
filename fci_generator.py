from __future__ import annotations
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
import numpy as np

from boundary import PolygonBoundary
from imm_bndry import ImmersedBoundary
import cut_cell as cc
from field import MagneticField
from fv import DAGP_FV
from grid import StructuredGrid
import mapping as mpg
import utils
from device import Device, DeviceInfo, BOUT_IO
import weights

class FCI_Generator:
    """
    Single generator for FCI setup in Hermes-3/BOUT++.
    """
    dvc: Device = None
    maps:   dict = {}
    metric: dict = {}
    attr:   dict = {}

    def __init__(self, nx: int, ny: int, nz: int, device: Device, ib: bool = False) -> None:
        #TODO: Encapsulate these somehow?
        self.nx = nx
        self.ny = 1 if device.dvc_info.axisymmetric is True else ny
        self.nz = nz
        self.ib = ib

        self.dvc = device

    @property
    def field(self) -> MagneticField:
        """Easy getter for the field."""
        return self.dvc.field
    
    @property
    def grid(self) -> StructuredGrid:
        """Easy getter for the grid."""
        return self.dvc.grid
    
    @property
    def wall(self) -> PolygonBoundary:
        """Easy getter for the wall."""
        return self.dvc.wall

    @property
    def dvc_info(self) -> DeviceInfo:
        """Easy getter for device information."""
        return self.dvc.dvc_info

    def generate(self) -> BOUT_IO:
        """Generate all the output for BOUT++."""
        self._generate_output()

        return BOUT_IO(maps=self.maps, metric=self.metric, attributes=self.attr)

    def _generate_output(self) -> None:
        """Generate all dict values for file output."""
        print("Generating FCI mapping and metric data...")
        self._build_maps()
        self._build_metric()

        if self.ib is True:
            self._apply_ib()

        if hasattr(self.field, "psi"):
            self.attr["psi"] = self._make_3d(getattr(self.field,"psi"))

    def _build_maps(self) -> None:
        self.maps.update({
            "R": self._make_3d(self.grid.xx),
            "Z": self._make_3d(self.grid.zz),
            "MXG": self.grid.ghosts.x,
            "MYG": self.grid.ghosts.y})

        print("Generating forward and backward points on whole grid...")
        grid, field = self.grid, self.field

        #TODO: How to handle y when 3d? Need to loop over y array.
        grid_pts = np.column_stack((grid.xx.ravel(), grid.zz.ravel()))
        fwd_pts  = np.zeros_like(grid_pts)
        bwd_pts  = np.zeros_like(grid_pts)
        for idx, (x, z) in enumerate(tqdm(grid_pts, desc="Tracing", unit="pt")):
            sln = field.tracer.trace(x, z, grid.y, field.direction*grid.dy)
            fwd_pts[idx, 0], fwd_pts[idx, 1] = sln.y[0, -1], sln.y[1, -1]

            sln = field.tracer.trace(x, z, grid.y, -field.direction*grid.dy)
            bwd_pts[idx, 0], bwd_pts[idx, 1] = sln.y[0, -1], sln.y[1, -1]

        #Convert mapping points back to 2d arrays.
        Rfwd = fwd_pts[:,0].reshape(grid.grid_shape_2d)
        Zfwd = fwd_pts[:,1].reshape(grid.grid_shape_2d)
        Rbwd = bwd_pts[:,0].reshape(grid.grid_shape_2d)
        Zbwd = bwd_pts[:,1].reshape(grid.grid_shape_2d)

        fwd_xtp, fwd_ztp = self._find_inside_index(Rfwd, Zfwd)
        bwd_xtp, bwd_ztp = self._find_inside_index(Rbwd, Zbwd)

        self.maps.update({
            "forward_R":  self._make_3d(Rfwd),
            "forward_Z":  self._make_3d(Zfwd),
            "backward_R": self._make_3d(Rbwd),
            "backward_Z": self._make_3d(Zbwd),
            "forward_xt_prime":  self._make_3d(fwd_xtp),
            "forward_zt_prime":  self._make_3d(fwd_ztp),
            "backward_xt_prime": self._make_3d(bwd_xtp),
            "backward_zt_prime": self._make_3d(bwd_ztp)})

    def _build_metric(self) -> None:
        """Create metric information for all coordinate mappings."""
        #Use config sizes instead of grid sizes since new grid will re-add ghosts inside.
        #TODO: How to handle 2d vs 3d?
        ctr_map = mpg.CoordinateMapping(self.nx, self.nz,
                                        self.maps["R"][:,0,:], self.maps["Z"][:,0,:])
        fwd_map = mpg.CoordinateMapping(self.nx, self.nz,
                                        self.maps["forward_R"][:,0,:],
                                        self.maps["forward_Z"][:,0,:])
        bwd_map = mpg.CoordinateMapping(self.nx, self.nz,
                                        self.maps["backward_R"][:,0,:],
                                        self.maps["backward_Z"][:,0,:])

        Bmag3D = self._make_3d(self.field.Bmag)
        By3D   = self._make_3d(self.field.By)
        R3D    = self.maps["R"]
        pressure = self._make_3d(self.field.pres)
        self.metric.update({
            "Rxy":  R3D,
            "Bxy":  Bmag3D,
            "B":    Bmag3D,
            "pressure": pressure})

        #TODO: Can probably make a function for the ctr/fwd/bwd logic.
        #TODO: Also g22 depends on if toroidal or not.
        if self.dvc_info.toroidal is True:
            g22_fac = R3D**2
        else:
            g22_fac = np.ones_like(R3D)

        self.metric.update({
            "dx":   np.full_like(R3D, ctr_map.metric["dx"]),
            "dy":   np.full_like(R3D, self.grid.dy),
            "dz":   np.full_like(R3D, ctr_map.metric["dz"]),
            "g11":  self._make_3d(ctr_map.metric["gxx"]),
            "g_11": self._make_3d(ctr_map.metric["gxx"]),
            "g13":  self._make_3d(ctr_map.metric["gxz"]),
            "g_13": self._make_3d(ctr_map.metric["g_xz"]),
            "g22":  1/g22_fac,
            "g_22": g22_fac,
            "g33":  self._make_3d(ctr_map.metric["gzz"]),
            "g_33": self._make_3d(ctr_map.metric["g_zz"]),
            "J":    self._make_3d(ctr_map.metric["J"])})
        
        self.metric.update({
            "forward_dx":    np.full_like(R3D, fwd_map.metric["dx"]),
            "forward_dy":    np.full_like(R3D, self.grid.dy),
            "forward_dz":    np.full_like(R3D, fwd_map.metric["dz"]),
            "forward_g11":   self._make_3d(fwd_map.metric["gxx"]),
            "forward_g_11":  self._make_3d(fwd_map.metric["gxx"]),
            "forward_g13":   self._make_3d(fwd_map.metric["gxz"]),
            "forward_g_13":  self._make_3d(fwd_map.metric["g_xz"]),
            "forward_g22":   1/g22_fac,
            "forward_g_22":  g22_fac,
            "forward_g33":   self._make_3d(fwd_map.metric["gzz"]),
            "forward_g_33":  self._make_3d(fwd_map.metric["g_zz"]),
            "forward_J":     self._make_3d(fwd_map.metric["J"])})

        self.metric.update({
            "backward_dx":   np.full_like(R3D, bwd_map.metric["dx"]),
            "backward_dy":   np.full_like(R3D, self.grid.dy),
            "backward_dz":   np.full_like(R3D, bwd_map.metric["dz"]),
            "backward_g11":  self._make_3d(bwd_map.metric["gxx"]),
            "backward_g_11": self._make_3d(bwd_map.metric["gxx"]),
            "backward_g13":  self._make_3d(bwd_map.metric["gxz"]),
            "backward_g_13": self._make_3d(bwd_map.metric["g_xz"]),
            "backward_g22":  1/g22_fac,
            "backward_g_22": g22_fac,
            "backward_g33":  self._make_3d(bwd_map.metric["gzz"]),
            "backward_g_33": self._make_3d(bwd_map.metric["g_zz"]),
            "backward_J":    self._make_3d(bwd_map.metric["J"])})

        #Update gyy's with field line following factors for parallel operators.
        if self.dvc_info.toroidal is True:
            par_fac = Bmag3D/By3D
            self.metric.update({k: v/par_fac**2 for k,v in self.metric.items()
                            if k in ("g22", "forward_g22", "backward_g22")})
            self.metric.update({k: v*par_fac**2 for k,v in self.metric.items()
                            if k in ("g_22", "forward_g_22", "backward_g_22")})
            
        #Generate finite volume operators from central mapping.
        #Note, goes on maps but done here because need metric.
        dagp_fv = DAGP_FV(ctr_map.grid, ctr_map.metric)
        dagp_vars = dagp_fv.calc_dagp()
        for key, value in dagp_vars.items():
            dagp_vars[key] = self._make_3d(value)
        self.maps.update(dagp_vars)

    def _apply_ib(self) -> None:
        print("Generating immersed boundary data...")
        ib = ImmersedBoundary(self.wall)
        grid = self.grid

        ib_info = ib.compute_ghost_info(grid.xx, grid.zz, show=utils.DEBUG_FLAG)

        #Use cell numbers to map to ghost array.
        ghost_id = np.full(ib_info.ghost_mask.shape, -1.0)
        ghost_id[ib_info.ghost_mask] = np.arange(ib_info.ghost_mask.sum())

        #Get indices required for interpolation.
        xi, zi = ib_info.image_points
        indr_i, indz_i = self._find_inside_index(xi, zi, bound=False, plot=False)

        #TODO: Not really necessary, setup vandermonde solve before runtime first.
        wghts, w_in, nw = weights.calc_perp_weights(indr_i, indz_i, ib_info.in_mask)

        #TODO: Check all float64 items need to convert to int correctly in C++??? Eventually make ints. Check bound_id below too.
        #TODO: Add debug check for images missing 4 corners if need more ghosts?
        #TODO: How to deal with image points on boundary? Can happen because if gridpt == boundry then considered inside.
        #TODO: So really how to deal with grid points on boundary too.
        self.maps.update({
            "in_mask":    self._make_3d(ib_info.in_mask).astype(float),
            "ghost_id":   self._make_3d(ghost_id).astype(float),
            "ng":         ib_info.ghost_mask.sum(),
            "ghost_pts":  np.stack(ib_info.ghost_points, axis=-1),
            "image_pts":  np.stack(ib_info.image_points, axis=-1),
            "bndry_pts":  np.stack(ib_info.wall_points,  axis=-1),
            "normals":    np.stack(ib_info.normals,      axis=-1),
            "image_inds": np.stack([indr_i, indz_i],     axis=-1),
            "norm_dist":  ib_info.norm_dist,
            "nw":         nw,
            "is_plasma":  w_in.astype(float),
            "weights":    wghts})

        #TODO: This should be contained in IB class logic. If so is there any issue doing this after dagp vars are already padded.
        #Just dont want to have to pass the boundary in for this to work now.
        #TODO: Fix 32x32 circle bug with cut-cell at the top I think it was?
        #Also fix TCV/DIIID cut-cell issues? Saw problems with cut face logic. With low res sometimes have two intersection points...
        vol_frac, fx_plus_frac, fz_plus_frac, geom = \
            cc.compute_cutcell_fractions_with_bound(grid.x,
                grid.z, grid.dx, grid.dz, self.wall)

        #Store cut-cell boundary info
        #TODO: Find better way to make3d? So dont need name?
        #TODO: Also make integer, but need to allow reading ints in BOUT.
        geom["bound_id"] = self._make_3d(geom["bound_id"]).astype(float)
        self.maps.update(geom)

        vol_frac =     self._make_3d(vol_frac)
        fx_plus_frac = self._make_3d(fx_plus_frac)
        fz_plus_frac = self._make_3d(fz_plus_frac)

        #Update fv info with cut_cell factors.
        self.maps["dagp_fv_XX"] *= fx_plus_frac
        self.maps["dagp_fv_XZ"] *= fx_plus_frac
        self.maps["dagp_fv_ZX"] *= fz_plus_frac
        self.maps["dagp_fv_ZZ"] *= fz_plus_frac
        self.maps["dagp_fv_volume"] *= vol_frac

        #Useful to store for post processing.
        self.maps.update({
            "vol_frac": vol_frac,
            "face_fac_x": fx_plus_frac,
            "face_fac_z": fz_plus_frac})

    #TODO: This functionality can probably go away eventually w/ loops over y even when y=1.
    def _make_3d(self, arr: utils.DataArray) -> utils.DataArray:
        """
        Extend arrays to 3d as needed before storing as output.
        Only do this if they're still 2d and ny > 0.
        """
        if self.ny > 0 and arr.ndim == 2:
            arr = np.repeat(arr[:,np.newaxis,:], self.ny, axis=1)
        return arr

    def _find_inside_index(self, x: utils.DataArray, z: utils.DataArray,
                        bound: bool = True, plot: bool = True):
        if bound is False:
            utils.logger.warn("Assuming image points possibly outside bounds..."
                " need to deal with complex boundaries to remove this flag.")

        xind, zind = self.grid.find_index(x, z)

        #Mask out points around boundary. How parallel bounds are handled but not perp...
        inside = self.wall.contains(x,z)
        if bound:
            xind = np.where(inside, xind, self.grid.x.size)
            zind = np.where(inside, zind, self.grid.z.size)

        #Show which cells are inside/outside if debug flag on.
        if (utils.DEBUG_FLAG is True and plot is True): #TODO: remove double check on plot.
            _, ax = plt.subplots(figsize=(6,6))

            #Draw the wall Path.
            patch = PathPatch(self.wall.path, facecolor='none', edgecolor='k', lw=2)
            ax.add_patch(patch)

            #Scatter the points inside (green) and outside (red).
            marker_size, alpha = 10, 0.6
            ax.scatter(x[inside], z[inside], s=marker_size, c='g',
                    marker='o', label='inside', alpha=alpha)
            #TODO: Fix deprecated ~inside?
            ax.scatter(x[~inside], z[~inside], s=marker_size, c='r',
                    marker='x', label='outside', alpha=alpha)

            ax.set_aspect('equal', 'box')
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.legend(loc='upper right')
            plt.show()

        return xind, zind
