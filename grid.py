#TODO: Circ imp because field depends on grid but grid uses field later on... How to deal with dependency?
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from field import MagneticField #Get around circular import for now.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from tqdm import tqdm

#TODO: Use freegs? Where hypnotoad gets it from initially. Or write own pythonically (cough cough chatGPT).
from hypnotoad.utils.critical import find_critical as fc

import boundary as bdy
import cut_cell as cc
import mapping as mpg
import utils
import weights

import numpy as np

def rect_boundary_points_on_faces(R_centers, Z_centers, R0, Z0, n_half=5.5,
                                  include_corners=True, closed=False, ccw=True):
    """
    Build a rectangle (square if ΔR==ΔZ) centered near (R0,Z0) whose edges lie on faces
    at ±n_half cells (e.g., n_half=5.5). Samples along each edge at *center* locations,
    so the points align with your stagger (R centers, Z centers).

    R_centers: 1D array of R cell centers (length nr_total, incl. ghosts)
    Z_centers: 1D array of Z cell centers (length nz_total, incl. ghosts)
    R0, Z0   : target center (in same units as arrays)
    n_half   : half-side in cells (5.5 puts edges on faces)
    """
    # find nearest center indices to the requested center
    ic = int(np.argmin(np.abs(R_centers - R0)))
    jc = int(np.argmin(np.abs(Z_centers - Z0)))

    # how many *center* steps to sample along each side
    m = int(2 * n_half)          # e.g. 11 samples per side
    off = int(np.floor(n_half))  # 5 for 5.5

    # sanity: ensure we have room in the arrays (ghosts usually make this true)
    if ic - off - 1 < 0 or ic + off + 1 >= len(R_centers):
        raise IndexError("Not enough R ghost cells to place ±n_half faces.")
    if jc - off - 1 < 0 or jc + off + 1 >= len(Z_centers):
        raise IndexError("Not enough Z ghost cells to place ±n_half faces.")

    # center sequences (length m) used for sampling along the edges
    R_seq = R_centers[ic - off : ic + off + 1]   # R centers: ic-5 .. ic+5
    Z_seq = Z_centers[jc - off : jc + off + 1]   # Z centers: jc-5 .. jc+5

    # face coordinates at ±5.5 cells from center
    # right face between ic+5 and ic+6; left face between ic-6 and ic-5
    R_face_right = 0.5 * (R_centers[ic + off] + R_centers[ic + off + 1])
    R_face_left  = 0.5 * (R_centers[ic - off - 1] + R_centers[ic - off])
    # top face between jc+5 and jc+6; bottom between jc-6 and jc-5
    Z_face_top    = 0.5 * (Z_centers[jc + off] + Z_centers[jc + off + 1])
    Z_face_bottom = 0.5 * (Z_centers[jc - off - 1] + Z_centers[jc - off])

    # build edges in CCW order: Top, Right, Bottom, Left
    r_top = R_seq
    z_top = np.full(m, Z_face_top)
    nr_top = np.zeros(m);  nz_top = np.ones(m)

    r_right = np.full(m, R_face_right)
    z_right = Z_seq[::-1]               # go downwards for CCW
    nr_right = np.ones(m); nz_right = np.zeros(m)

    r_bot = R_seq[::-1]
    z_bot = np.full(m, Z_face_bottom)
    nr_bot = np.zeros(m);  nz_bot = -np.ones(m)

    r_left = np.full(m, R_face_left)
    z_left = Z_seq
    nr_left = -np.ones(m); nz_left = np.zeros(m)

    # arc-length weights (use local spacings in case ghosts differ)
    dR = R_centers[ic + 1] - R_centers[ic]
    dZ = Z_centers[jc + 1] - Z_centers[jc]
    w_top = np.full(m, dR)
    w_bot = np.full(m, dR)
    w_right = np.full(m, dZ)
    w_left  = np.full(m, dZ)

    # concatenate
    r_blocks  = [r_top,   r_right, r_bot,   r_left]
    z_blocks  = [z_top,   z_right, z_bot,   z_left]
    nr_blocks = [nr_top,  nr_right, nr_bot,  nr_left]
    nz_blocks = [nz_top,  nz_right, nz_bot,  nz_left]
    w_blocks  = [w_top,   w_right, w_bot,   w_left]

    if include_corners:
        # Explicit corners
        RT_r, RT_z = R_face_right, Z_face_top
        RB_r, RB_z = R_face_right, Z_face_bottom
        LB_r, LB_z = R_face_left,  Z_face_bottom
        LT_r, LT_z = R_face_left,  Z_face_top

        # Start at top-left corner so the first edge (Top) has a corner before it
        r_list = [np.array([LT_r])]; z_list = [np.array([LT_z])]
        # corner normal: use upcoming edge's normal (Top)
        nr_list = [nr_blocks[0][0:1]]; nz_list = [nz_blocks[0][0:1]]
        w_list  = [np.array([0.0])]

        # Top edge, then top-right corner
        r_list += [r_blocks[0], np.array([RT_r])]
        z_list += [z_blocks[0], np.array([RT_z])]
        nr_list += [nr_blocks[0], nr_blocks[1][0:1]]
        nz_list += [nz_blocks[0], nz_blocks[1][0:1]]
        w_list  += [w_blocks[0], np.array([0.0])]

        # Right edge, then bottom-right corner
        r_list += [r_blocks[1], np.array([RB_r])]
        z_list += [z_blocks[1], np.array([RB_z])]
        nr_list += [nr_blocks[1], nr_blocks[2][0:1]]
        nz_list += [nz_blocks[1], nz_blocks[2][0:1]]
        w_list  += [w_blocks[1], np.array([0.0])]

        # Bottom edge, then bottom-left corner
        r_list += [r_blocks[2], np.array([LB_r])]
        z_list += [z_blocks[2], np.array([LB_z])]
        nr_list += [nr_blocks[2], nr_blocks[3][0:1]]
        nz_list += [nz_blocks[2], nz_blocks[3][0:1]]
        w_list  += [w_blocks[2], np.array([0.0])]

        # Left edge (ends back at top-left corner if closed=True)
        r_list += [r_blocks[3]]
        z_list += [z_blocks[3]]
        nr_list += [nr_blocks[3]]
        nz_list += [nz_blocks[3]]
        w_list  += [w_blocks[3]]

        r   = np.concatenate(r_list)
        z   = np.concatenate(z_list)
        n_r = np.concatenate(nr_list)
        n_z = np.concatenate(nz_list)
        w   = np.concatenate(w_list)
    else:
        r   = np.concatenate(r_blocks)
        z   = np.concatenate(z_blocks)
        n_r = np.concatenate(nr_blocks)
        n_z = np.concatenate(nz_blocks)
        w   = np.concatenate(w_blocks)

    if not ccw:
        r, z, n_r, n_z, w = r[::-1], z[::-1], n_r[::-1], n_z[::-1], w[::-1]

    if closed:
        r = np.append(r, r[0]); z = np.append(z, z[0])
        n_r = np.append(n_r, n_r[0]); n_z = np.append(n_z, n_z[0])
        w = np.append(w, 0.0)

    return r, z, n_r, n_z, w

def circle_boundary_points(R0, Z0, a, N=512, theta0=0.0, theta1=2*np.pi, closed=False, ccw=True):
    """
    Return r,z points on a circle centered at (R0,Z0) with radius a.

    Params
    ------
    R0, Z0 : float        # center
    a      : float        # radius
    N      : int          # number of samples (uniform in angle)
    theta0 : float        # start angle (radians)
    theta1 : float        # end angle (radians), can be < 2π for an arc
    closed : bool         # if True, duplicate the first point at the end
    ccw    : bool         # orientation; False gives clockwise ordering

    Returns
    -------
    r, z           : (N[, +1],) arrays of coordinates
    n_r, n_z       : outward unit normals at each point
    w              : arc-length weights (midpoint rule), sum ≈ arc length
    """
    # parameter angles (midpoint rule -> better for integrals than including endpoints)
    th = np.linspace(theta0, theta1, N, endpoint=False)
    if not ccw:
        th = th[::-1]

    r = R0 + a*np.cos(th)
    z = Z0 + a*np.sin(th)

    # outward normals for a circle are just (cos, sin)
    n_r = np.cos(th)
    n_z = np.sin(th)

    # uniform angular spacing
    dth = (theta1 - theta0)/N
    # arc-length weights for midpoint rule
    w = a * dth * np.ones_like(th)

    if closed:
        # optionally append the first point to close the polygon
        r = np.append(r, r[0])
        z = np.append(z, z[0])
        n_r = np.append(n_r, n_r[0])
        n_z = np.append(n_z, n_z[0])
        w = np.append(w, 0.0)  # last weight zero so sum stays the same

    return r, z, n_r, n_z, w

class StructuredPoloidalGrid(object):
    """Represents a structured poloidal grid in R,phi,Z."""
    #TODO: Move 3D logic outside poloidal grid class? Into general grid class?

    def __init__(self, eq_data, nr=utils.DEFAULT_NR, nphi=utils.DEFAULT_NPHI, nz=utils.DEFAULT_NZ, make3D=False, MRG=2, MYG=1, MZG=0):
        #TODO: Is default padding in Z ok? Still periodic in BOUT...
        self.Lr = eq_data.rmax - eq_data.rmin
        self.Lz = eq_data.zmax - eq_data.zmin

        self.R0 = eq_data.R0
        self.Z0 = eq_data.Z0

        #Generate 2D grid with new resolution, and add ghost points (MRG, MZG).
        self.MRG, self.MYG, self.MZG = MRG, MYG, MZG
        self.nr = nr + 2*self.MRG
        self.nz = nz + 2*self.MZG
        self.dR = self.Lr/nr
        self.dZ = self.Lz/nz
        #TODO Create R and Z like this to match hermes/BOUT output.
        #TODO With z not periodic its preferable to keep the endpoint, no? Does anything in BOUT change?
        self.R  = eq_data.rmin + (0.5 + np.arange(0,nr))*self.dR
        self.Z  = eq_data.zmin + np.arange(0,nz)*self.dZ
        ghosts_lo_R = self.R[0]  - self.dR*np.arange(self.MRG, 0, -1)
        ghosts_lo_Z = self.Z[0]  - self.dZ*np.arange(self.MZG, 0, -1)
        ghosts_hi_R = self.R[-1] + self.dR*np.arange(1, self.MRG+1)
        ghosts_hi_Z = self.Z[-1] + self.dZ*np.arange(1, self.MZG+1)
        self.R = np.concatenate((ghosts_lo_R, self.R, ghosts_hi_R))
        self.Z = np.concatenate((ghosts_lo_Z, self.Z, ghosts_hi_Z))
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z, indexing='ij')
        self.dr, self.dz = 1/nr, 1/nz #Note, normalized.
        self.rmid = (self.R[-1] + self.R[0])/2
        self.zmid = (self.Z[-1] + self.Z[0])/2

        #Create new wall for MMS testing...
        #Rectangular boundary on faces...
        #r, z = np.array([
        #    (self.R0 - self.Lr/4, self.Z0 - self.Lr/4 - self.dZ/2),
        #    (self.R0 - self.Lr/4, self.Z0 + self.Lr/4 + self.dZ/2),
        #    (self.R0 + self.Lr/4, self.Z0 + self.Lr/4 + self.dZ/2),
        #    (self.R0 + self.Lr/4, self.Z0 - self.Lr/4 - self.dZ/2)
        #]).T
        #Rectangular boundary on grid points...
        #r, z = np.array([
        #    (self.R0 - self.Lr/4 + self.dR/2, self.Z0 - self.Lr/4),
        #    (self.R0 - self.Lr/4 + self.dR/2, self.Z0 + self.Lr/4),
        #    (self.R0 + self.Lr/4 - self.dR/2, self.Z0 + self.Lr/4),
        #    (self.R0 + self.Lr/4 - self.dR/2, self.Z0 - self.Lr/4)
        #]).T
        #Rectangular boundary between grid pts and faces...
        #r, z = np.array([
        #    (self.R0 - self.Lr/4 + self.dR/4, self.Z0 - self.Lr/4 - self.dZ/4),
        #    (self.R0 - self.Lr/4 + self.dR/4, self.Z0 + self.Lr/4 + self.dZ/4),
        #    (self.R0 + self.Lr/4 - self.dR/4, self.Z0 + self.Lr/4 + self.dZ/4),
        #    (self.R0 + self.Lr/4 - self.dR/4, self.Z0 - self.Lr/4 - self.dZ/4)
        #]).T
        r, z, n_r, n_z, w = circle_boundary_points(self.rmid, self.zmid, a=self.Lr/3, N=512)

        #Create the separatrix and wall boundaries. #TODO: Separate to device class? Which holds grid, boundaries, field, etc.
        print("Generating separatrix boundary path...") #1.75, 3.25
        self.sep_idx = np.argmax(eq_data.rbdy)          #-0.77205882, 0.77205882
        self.sptx = bdy.PolygonBoundary(eq_data.rbdy, eq_data.zbdy)
        print("Generating wall boundary path...")
        self.wall = bdy.PolygonBoundary(r,z)
        #self.wall = bdy.PolygonBoundary(eq_data.rlmt, eq_data.zlmt)
        #Also create ghost point boundary for final plotting.
        #TODO: Currently stop at zmax - dz because z periodic in BOUT++ so not shifted by 1/2 from bdry...
        self.ghst = bdy.PolygonBoundary([eq_data.rmin, eq_data.rmin, eq_data.rmax, eq_data.rmax, eq_data.rmin],
                                        [eq_data.zmin, eq_data.zmax-self.dZ, eq_data.zmax-self.dZ, eq_data.zmin, eq_data.zmin])

        #Handle toroidal direction.
        self.phi  = 0.0
        self.dphi = 2*np.pi/nphi
        #TODO: This info only needs to be output if tokamak...Maybe mention cant support turbulence sims w/o extending.
        utils.logger.info("Generating 3D configuration with one poloidal plane." if make3D is False \
            else "Extending axisymmetric configuration to multiple poloidal planes in 3D.")
        #Endpoint false because periodic.
        self.phi_arr = [self.dphi] if make3D is False else np.linspace(self.phi, self.phi + 2*np.pi, nphi, endpoint=False)
        #Reset to 1 for generating 3D grid info.
        if make3D is False:
            self.nphi = 1

        self.xpts, self.opts = self._crit_pts(eq_data)

        #For BSTING set sep indices to nx+1. Used just for MPI comms in FCI code.
        #TODO: Z not poloidal anymore so not sure one limit works?
        out_idx = self.nr + 1
        self.ixseps1 = out_idx
        self.ixseps2 = out_idx

    def __repr__(self):
        return (f"StructuredPoloidalGrid({0},{1},{2},{3},R0={4},Z0={5})".format(
            self.nr, self.nz, self.Lr, self.Lz, self.R0, self.Z0))
    
    def attach_field(self, field: "MagneticField"):
        self.field = field
    
    #For now repeat tokamak data if 3d needed.
    #TODO: Take out of grid? Uses self.nphi right now...
    def make_3d(self, arr_2d: np.ndarray) -> np.ndarray:
        """
        Repeat a 2D array (R, Z) along the middle axis to shape (R, ny, Z).
        If writable=False, returns a broadcasted (read-only) view.
        """
        return np.repeat(arr_2d[:, np.newaxis, :], self.nphi, axis=1)
    
    def _crit_pts(self, eq_data):
        #Use default settings from hypnotoad examples.
        sep_atol, sep_maxits = 1e-5, 1000
        #TODO: Does this require eq data? Or can be done on this grid? Then would need field...
        opoints, xpoints = fc(eq_data.rr, eq_data.zz, eq_data.psi, sep_atol, sep_maxits)
        #Remove points outside the wall. Not enough to remove all non-important points it turns out.
        #TODO: Can try removing points within certain flux surface outside of LCFS?
        # |--->  Probably use psi spline from R,Z to drop points a bit outside LCFS.
        for point in opoints[:]:
            if not self.wall.contains(point[0], point[1]):
                opoints.remove(point)
        for point in xpoints[:]:
            if not self.wall.contains(point[0], point[1]):
                xpoints.remove(point)

        return xpoints, opoints

    def _make_mapping(self, RR=None, ZZ=None):
        #Note maps should always use central grid dimensions as reference points.
        RR = self.RR if RR is None else RR
        ZZ = self.ZZ if ZZ is None else ZZ

        #TODO: Pass self, RR, ZZ. But thats a cyclic dep. Can make a small grid info dataclass.
        return mpg.CoordinateMapping(self.nr, self.nz, self.dr, self.dz, RR, ZZ, self.wall)

    def _find_index(self, R, Z, bound=True, show=False):
        """
        Finds the (x,z) index corresponding to the given (R,Z) coordinate.

        Parameters
        ----------
        R, Z : array_like
            Locations to find indices for.

        Returns
        -------
        x, z : (ndarray, ndarray)
            Index as a float, same shape as R,Z.
        """
        if bound == False:
            utils.logger.warn("Assuming image points possibly outside bounds..."
                " need to deal with complex boundaries to remove this flag.")

        rind = (R - self.R[0])/self.dR
        zind = (Z - self.Z[0])/self.dZ

        #If original points were grid points need to round/snap due to floating errors.
        ri, zi = np.rint(rind), np.rint(zind) #Nearest integer
        #Note, rtol=0.0 just ignores rtol and only uses atol here.
        round_r = np.isclose(rind, ri, atol=utils.DEFAULT_TOL.path_tol, rtol=0.0)
        round_z = np.isclose(zind, zi, atol=utils.DEFAULT_TOL.path_tol, rtol=0.0)

        #Take snapped value when isclose True.
        rind = np.where(round_r, ri, rind)
        zind = np.where(round_z, zi, zind)

        #Mask out points around boundary. How parallel bounds are handled but not perp...
        #TODO: Find way around needing bound flag?
        inside = self.wall.contains(R,Z) #self.wall.contains(R,Z) TIRKAS
        if (bound):
            rind[~inside] = self.nr
            zind[~inside] = self.nz

        if (show):
            fig, ax = plt.subplots(figsize=(6,6))

            #1) Draw the wall Path itself.
            patch = PathPatch(self.wall.path, facecolor='none', edgecolor='k', lw=2)
            ax.add_patch(patch)

            #2) Scatter the points inside (green) and outside (red).
            ax.scatter(R[inside], Z[inside], s=10, c='g',
                    marker='o', label='inside', alpha=0.6)
            ax.scatter(R[~inside], Z[~inside], s=10, c='r',
                    marker='x', label='outside', alpha=0.6)

            ax.set_aspect('equal', 'box')
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.legend(loc='upper right')
            plt.show()
        
        return rind, zind
    
    def _trace_grid(self):
        print("Generating forward and backward points on whole grid...")
        gridPts = np.column_stack((self.RR.ravel(), self.ZZ.ravel()))
        fwdPts  = np.zeros_like(gridPts)
        bwdPts  = np.zeros_like(gridPts)
        for idx, (r0, z0) in enumerate(tqdm(gridPts,desc="Tracing",unit="pt")):
            sln = self.field.trace_field_line(r0, z0, self.phi, self.field.dir*self.dphi)
            fwdPts[idx, 0], fwdPts[idx, 1] = sln.y[0, -1], sln.y[1, -1]

            sln = self.field.trace_field_line(r0, z0, self.phi, -self.field.dir*self.dphi)
            bwdPts[idx, 0], bwdPts[idx, 1] = sln.y[0, -1], sln.y[1, -1]

        #Convert mapping points back to 2d arrays.
        Rfwd = fwdPts[:,0].reshape(self.nr, self.nz)
        Zfwd = fwdPts[:,1].reshape(self.nr, self.nz)
        Rbwd = bwdPts[:,0].reshape(self.nr, self.nz)
        Zbwd = bwdPts[:,1].reshape(self.nr, self.nz)

        return Rfwd, Zfwd, Rbwd, Zbwd
    
    def generate_bounds(self, maps, metrics):
        print("Generating ghost cell image points and boundary data...")
        #TODO: Create a vector data class for points. Use inside get_image_pts.
        #Dont stack below? Define length/unit functions in dataclass.
        in_mask, ghost_mask, (Rg, Zg), (Rb, Zb), (Ri, Zi), (Rn, Zn) \
            = self.wall.get_image_pts(self.RR, self.ZZ, show=utils.DEBUG_FLAG)
        
        #Get distance from wall to image. (I-B) dot n.
        nhat = utils.unit_vecs(np.stack([Rn,Zn], axis=-1))
        Rhat, Zhat = nhat[..., 0], nhat[..., 1]
        norm_dist = (Ri-Rb)*Rhat + (Zi-Zb)*Zhat

        #Use cell numbers to map to ghost array.
        ghost_id  = np.full(ghost_mask.shape, -1)
        ghost_id[ghost_mask] = np.arange(ghost_mask.sum())

        #Get indices required for interpolation.
        indr_i, indz_i = self._find_index(Ri, Zi, bound=False)
        wghts, wghts_in, num_wghts = weights.calc_perp_weights(indr_i, indz_i, in_mask)

        #TODO: Should this logic be handled by boundary class? If so is there any issue doing this after dagp vars are already padded.
        #TODO: Just dont want to have to pass the boundary in for this to work now.
        #TODO: Need to get matplotlib Path as well not just boundary points, but grid handles that...
        #TODO: Clean up ChatGPTs algorithm for this method.
        #TODO: Note this only applies to the central plane currently. But dont need FV operators on fwd/bwd planes.
        vol_frac, fx_plus_frac, fz_plus_frac = cc.compute_cutcell_fractions(
            self.R, self.Z, self.dR, self.dZ, self.wall)
        vol_frac, fx_plus_frac, fz_plus_frac = self.make_3d(vol_frac), \
            self.make_3d(fx_plus_frac), self.make_3d(fz_plus_frac)

        #Update finite volume face/vol quantities.
        maps["dagp_fv_XX"] *= fx_plus_frac
        maps["dagp_fv_XZ"] *= fx_plus_frac
        maps["dagp_fv_ZX"] *= fz_plus_frac
        maps["dagp_fv_ZZ"] *= fz_plus_frac
        maps["dagp_fv_volume"] *= vol_frac

        #Useful to store for post processing.
        maps.update({
            "vol_frac": vol_frac,
            "face_fac_x": fx_plus_frac,
            "face_fac_z": fz_plus_frac
        })

        #TODO: Add debug check for images missing 4 corners if need more ghosts.
        #TODO: How to deal with image points on boundary? Can happen because if gridpt == boundry then considered inside.
        #TODO: So really how to deal with grid points on boundary too.
        maps.update({
            "in_mask":     self.make_3d(in_mask).astype(np.float64),
            "ghost_id":    self.make_3d(ghost_id).astype(np.float64),
            "ng":          Rg.size,
            "ghost_pts":   np.stack([Rg, Zg], axis=-1),
            "image_pts":   np.stack([Ri, Zi], axis=-1),
            "bndry_pts":   np.stack([Rb, Zb], axis=-1),
            "normals":     np.stack([Rhat, Zhat], axis=-1),
            "image_inds":  np.stack([indr_i, indz_i], axis=-1),
            "norm_dist":   norm_dist,
            "nw":          num_wghts,
            "is_plasma":   wghts_in.astype(np.float64),
            "weights":     wghts})

    def generate_maps(self):
        print("Generating metric and map data for output file...")
        Rfwd, Zfwd, Rbwd, Zbwd = self._trace_grid()
        fwd_xtp, fwd_ztp = self._find_index(Rfwd, Zfwd, show=utils.DEBUG_FLAG)
        bwd_xtp, bwd_ztp = self._find_index(Rbwd, Zbwd, show=utils.DEBUG_FLAG)

        #Need to do this all in 3D now, didn't need the complication before.
        R3, Z3 = self.make_3d(self.RR), self.make_3d(self.ZZ)
        maps = {
            "R": R3,
            "Z": Z3,
            "MXG": self.MRG,
            "MYG": self.MYG,
            "forward_R":  self.make_3d(Rfwd),
            "forward_Z":  self.make_3d(Zfwd),
            "backward_R": self.make_3d(Rbwd),
            "backward_Z": self.make_3d(Zbwd),
            "forward_xt_prime":  self.make_3d(fwd_xtp),
            "forward_zt_prime":  self.make_3d(fwd_ztp),
            "backward_xt_prime": self.make_3d(bwd_xtp),
            "backward_zt_prime": self.make_3d(bwd_ztp)}
        
        #Store metric info. Store everything in 3D.
        coord_map_ctr = self._make_mapping()
        coord_map_fwd = self._make_mapping(Rfwd, Zfwd)
        coord_map_bwd = self._make_mapping(Rbwd, Zbwd)

        Bmag3D = self.make_3d(self.field.Bmag)
        Bphi3D = self.make_3d(self.field.Bphi)
        parFac = Bmag3D/Bphi3D
        metric = {
            "Rxy":  R3,
            "Bxy":  Bmag3D,
            "dx":   np.full_like(R3, coord_map_ctr.metric["dx"]),
            "dy":   np.full_like(R3, self.dphi),
            "dz":   np.full_like(R3, coord_map_ctr.metric["dz"]),
            "g11":  self.make_3d(coord_map_ctr.metric["gxx"]),
            "g_11": self.make_3d(coord_map_ctr.metric["gxx"]),
            "g13":  self.make_3d(coord_map_ctr.metric["gxz"]),
            "g_13": self.make_3d(coord_map_ctr.metric["g_xz"]),
            "g22":  1/R3**2,
            "g_22": R3**2,
            "g33":  self.make_3d(coord_map_ctr.metric["gzz"]),
            "g_33": self.make_3d(coord_map_ctr.metric["g_zz"]),
            "J":    self.make_3d(coord_map_ctr.metric["J"])}
        
        metric.update({
            "forward_dx":    np.full_like(R3, coord_map_fwd.metric["dx"]),
            "forward_dy":    np.full_like(R3, self.dphi),
            "forward_dz":    np.full_like(R3, coord_map_fwd.metric["dz"]),
            "forward_g11":   self.make_3d(coord_map_fwd.metric["gxx"]),
            "forward_g_11":  self.make_3d(coord_map_fwd.metric["gxx"]),
            "forward_g13":   self.make_3d(coord_map_fwd.metric["gxz"]),
            "forward_g_13":  self.make_3d(coord_map_fwd.metric["g_xz"]),
            "forward_g22":   1/R3**2,
            "forward_g_22":  R3**2,
            "forward_g33":   self.make_3d(coord_map_fwd.metric["gzz"]),
            "forward_g_33":  self.make_3d(coord_map_fwd.metric["g_zz"]),
            "forward_J":     self.make_3d(coord_map_fwd.metric["J"])})

        metric.update({
            "backward_dx":   np.full_like(R3, coord_map_bwd.metric["dx"]),
            "backward_dy":   np.full_like(R3, self.dphi),
            "backward_dz":   np.full_like(R3, coord_map_bwd.metric["dz"]),
            "backward_g11":  self.make_3d(coord_map_bwd.metric["gxx"]),
            "backward_g_11": self.make_3d(coord_map_bwd.metric["gxx"]),
            "backward_g13":  self.make_3d(coord_map_bwd.metric["gxz"]),
            "backward_g_13": self.make_3d(coord_map_bwd.metric["g_xz"]),
            "backward_g22":  1/R3**2,
            "backward_g_22": R3**2,
            "backward_g33":  self.make_3d(coord_map_bwd.metric["gzz"]),
            "backward_g_33": self.make_3d(coord_map_bwd.metric["g_zz"]),
            "backward_J":    self.make_3d(coord_map_bwd.metric["J"])})

        #Update gyy's with field line following factors for parallel operators, since this is handled along field lines.
        parFac = Bmag3D/Bphi3D
        #metric.update({k: v/parFac**2 for k,v in metric.items() if k in ("g22", "forward_g22", "backward_g22")})
        #metric.update({k: v*parFac**2 for k,v in metric.items() if k in ("g_22", "forward_g_22", "backward_g_22")})

        #Get finite volume operators for primary grid.
        for key, value in coord_map_ctr.dagp_vars.items():
            coord_map_ctr.dagp_vars[key] = self.make_3d(value)
        maps.update(coord_map_ctr.dagp_vars)

        return maps, metric

    def plotConfig(self, psi, maps):
        print("Plotting equilibrium configuration...")
        fig, ax = plt.subplots(figsize=(8,10)) #TODO: Base figsize on device dimensions?
        #TODO: Also show contour if debug else just contourf.
        cf = ax.contour(self.R, self.Z, psi.T, levels=100, cmap='viridis')
        plt.colorbar(cf, ax=ax)

        #Plot various boundaries.
        sptx = PathPatch(self.sptx.path, fill=False, lw=2,
                    edgecolor='orange', label='LCFS', linestyle='-.')
        patch = PathPatch(self.wall.path, fill=False, edgecolor='k', lw=2)
        ghost = PathPatch(self.ghst.path, fill=False, edgecolor='k', label='ghost',
                    clip_on=False, lw=1.5, linestyle='--')
        ax.add_patch(patch)
        ax.add_patch(ghost)
        #ax.add_patch(sptx)

        #Trace field lines in both directions.
        offset = 0.005 #Use minor radial offset from separatrix.
        rsep, zsep = self.sptx.path.vertices[:,0], self.sptx.path.vertices[:,1]
        #TODO: Add back when done removing points manually and messing up indices...
        #R1, Z1     = rsep[self.sep_idx] + offset, zsep[self.sep_idx]
        #print(f"Tracing field line to wall in forward direction \
        #    from {R1,Z1} near separatrix...")
        #Rvals_pos, Zvals_pos = self.field.trace_until_wall(R1, Z1, self.phi, self.dphi,
        #                                            self.wall, direction=self.field.dir)
        #print(f"Tracing field line to wall in backward direction \
        #    from {R1,Z1} near separatrix...")
        #Rvals_neg, Zvals_neg = self.field.trace_until_wall(R1, Z1, self.phi, self.dphi,
        #                                            self.wall, direction=-self.field.dir)
        #phi_dir, neg_phi_dir = ('+','-') if self.field.dir == 1 else ('-','+')
        #ax.plot(Rvals_pos, Zvals_pos, '.', color='red',
        #    label='$+\\hat{b}_{\\phi} = ' + phi_dir     + '\\hat{\\phi}$')
        #ax.plot(Rvals_neg, Zvals_neg, '.', color='cyan',
        #    label='$-\\hat{b}_{\\phi} = ' + neg_phi_dir + '\\hat{\\phi}$')

        #Test field line tracing on grid.
        if (utils.DEBUG_FLAG):
            print("Testing field line following on gridpoints...")
            #Get scatter point data to plot and test grid point following in general.
            gridPts = np.column_stack((self.RR.ravel(), self.ZZ.ravel()))
            fwdPts  = np.column_stack((maps["forward_R"].ravel(),  maps["forward_Z"].ravel()))
            bwdPts  = np.column_stack((maps["backward_R"].ravel(), maps["backward_Z"].ravel()))
            step    = gridPts.shape[0]//100 #Divide grid into 100 equally spaced points.
            indices = np.arange(0, gridPts.shape[0], step)
            gridPts = gridPts[indices]
            fwdPts  = fwdPts[indices]
            bwdPts  = bwdPts[indices]
            #Remove points outside the wall for all three arrays at once.
            keptPts = [(g, f, b) for (g, f, b) in zip(gridPts, fwdPts, bwdPts)
                    if self.wall.contains(g[0], g[1])]
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

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper center',
            bbox_to_anchor=(0.5, 1.06), ncol=len(labels), fancybox=True, shadow=True)
        ax.set_xlabel('R')
        ax.set_ylabel('Z')
        ax.grid(True)
        plt.tight_layout()
        plt.show()