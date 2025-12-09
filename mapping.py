import numpy as np
from scipy import interpolate

import boundary as bdy
import utils

class CoordinateMapping(object):
    """Represents a coordinate mapping from cartesian R,Z to normalized x,z coordinates."""
    def __init__(self, nx, nz, dx, dz, RR, ZZ, bdry):
        #TODO: Dont need to pass all this in? But dont need to duplicate logic to create these?
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.RR = RR
        self.ZZ = ZZ

        #Generate info for metric/derivative calculations.
        self.xinds  = (0.5 + np.arange(self.nx))*self.dx
        self.zinds  = (np.arange(self.nz))*self.dz
        self.xxinds, self.zzinds = np.meshgrid(self.xinds, self.zinds, indexing='ij')

        self.spl_x = interpolate.RectBivariateSpline(self.xinds, self.zinds, self.RR)
        self.spl_z = interpolate.RectBivariateSpline(self.xinds, self.zinds, self.ZZ)

        self.bdry = self._map_bdy(bdry)

        #Generate metric.
        self.metric = self._make_metric()

        #Calculate finite volume operators. #TODO: Only really needs to happen in center...
        #So call from outside or do outside from metric?
        self.dagp_vars = self._create_dagp()

    def _get_coordinates(self, dx=0, dz=0):
        #TODO: Is it ok now that nx != nz? zoidberg had lots of asserts.
        """Get coordinates (R, Z) at given (xind, zind) index

        Parameters
        ----------
        dx : int, optional
            Order of x derivative
        dz : int, optional
            Order of z derivative

        Returns
        -------
        R, Z : (ndarray, ndarray)
            Locations of point or derivatives of R,Z with respect to
            indices if dx,dz != 0
        """
        R = self.spl_x(self.xinds, self.zinds, dx=dx, dy=dz, grid=True)
        Z = self.spl_z(self.xinds, self.zinds, dx=dx, dy=dz, grid=True)

        return np.asarray(R), np.asarray(Z)
    
    def _map_bdy(self, bdry):
        """Take the boundary in (R,Z) coordinates and map to (x,z)
        to handle cut cell methods for finite volume operators."""
        R_min, R_max = 1.0, 4.0
        Z_min, Z_max = -1.5, 1.5

        L_R = R_max - R_min
        L_Z = Z_max - Z_min

        Rb = np.asarray(bdry.path.vertices[:,0])
        Zb = np.asarray(bdry.path.vertices[:,1])
        x_boundary = (Rb - R_min) / L_R
        z_boundary = (Zb - Z_min) / L_Z

        return bdy.PolygonBoundary(x_boundary, z_boundary)

    def _make_metric(self):
        """Return the metric tensor, dx and dz

        Returns
        -------
        dict
            Dictionary containing:
            - **dx, dz**: Grid spacing
            - **gxx, gxz, gzz**: Covariant components
            - **g_xx, g_xz, g_zz**: Contravariant components

        """
        #Partial derivatives of (R,Z) when stepping in computational directions (r,z)
        parR_x, parZ_x = self._get_coordinates(dx=1, dz=0)
        parR_z, parZ_z = self._get_coordinates(dx=0, dz=1)
        #J[k,i,...] with k∈{x,z}, i∈{R,Z}  → (2,2,nR,nZ)
        J = np.stack([[parR_x, parR_z],
                      [parZ_x, parZ_z]], axis=0)

        #Metric tensor g = J^T J.
        g = np.einsum('ki...,kj...->ij...', J, J)

        assert np.all(g[0, 0] > 0), \
                f"g[0, 0] is expected to be positive, but some values are not (minimum {np.min(g[0, 0])})"
        assert np.all(g[1, 1] > 0), \
                f"g[1, 1] is expected to be positive, but some values are not (minimum {np.min(g[1, 1])})"
        g = g.transpose(2, 3, 0, 1) #Move data to front indices.
        assert np.all(np.linalg.det(g) > 0), \
                f"All determinants of g should be positive, but some are not (minimum {np.min(np.linalg.det(g))})"
        ginv = np.linalg.inv(g)

        gxx   = ginv[..., 0, 0]
        gxz   = ginv[..., 0, 1]
        gzz   = ginv[..., 1, 1]
        g_xx  = g[..., 0, 0]
        g_xz  = g[..., 0, 1]
        g_zz  = g[..., 1, 1]
        jac   = np.sqrt(g_xx*g_zz-g_xz**2)
        metric = {"dx": self.dx, "dz": self.dz,
                  "gxx": gxx, "g_xx": g_xx,
                  "gxz": gxz, "g_xz": g_xz,
                  "gzz": gzz, "g_zz": g_zz,
                  "J": jac}
        return metric
    
    
    def _calc_vol(self):
        """
        Per-radian control volume A_R at centers from radius Rc and poloidal Jacobian Jp.
        Rc, Jp can be (nx,nz) arrays; dx,dz are logical spacings.
        Multiply by 2*pi for full volume.
        """
        return self.metric["J"]*self.dx*self.dz
        #return self.RR*self.metric["J"]*self.dx*self.dz
        #TODO: Add toroidal flag for this, and field, and face factors below.
    
    def _face_metrics(self):
        """
        Inputs (cell-centered, shape = (nx, nz)):
          gxx_c, gxz_c, gzz_c : contravariant metric entries at cell centers.

        Returns dict with face-centered tensors:
          xface: contravariant (gxx,gxz,gzz) on +x faces shape (nx-1, nz),
                 covariant    (g_xx,g_xz,g_zz) on +x faces shape (nx-1, nz)
          zface: contravariant (gxx,gxz,gzz) on +z faces shape (nx, nz-1),
                 covariant    (g_xx,g_xz,g_zz) on +z faces shape (nx, nz-1)
        """

        # --- arithmetic averages of contravariant entries to faces ---
        # +x faces between i and i+1
        gxx_x = 0.5*(self.metric["gxx"][:-1, :] + self.metric["gxx"][1:, :])
        gxz_x = 0.5*(self.metric["gxz"][:-1, :] + self.metric["gxz"][1:, :])
        gzz_x = 0.5*(self.metric["gzz"][:-1, :] + self.metric["gzz"][1:, :])

        # +z faces between j and j+1
        gxx_z = 0.5*(self.metric["gxx"][:, :-1] + self.metric["gxx"][:, 1:])
        gxz_z = 0.5*(self.metric["gxz"][:, :-1] + self.metric["gxz"][:, 1:])
        gzz_z = 0.5*(self.metric["gzz"][:, :-1] + self.metric["gzz"][:, 1:])

        # --- arithmetic averages of covariant entries to faces ---
        # +x faces between i and i+1
        g_xx_x = 0.5*(self.metric["g_xx"][:-1, :] + self.metric["g_xx"][1:, :])
        g_xz_x = 0.5*(self.metric["g_xz"][:-1, :] + self.metric["g_xz"][1:, :])
        g_zz_x = 0.5*(self.metric["g_zz"][:-1, :] + self.metric["g_zz"][1:, :])

        # +z faces between j and j+1
        g_xx_z = 0.5*(self.metric["g_xx"][:, :-1] + self.metric["g_xx"][:, 1:])
        g_xz_z = 0.5*(self.metric["g_xz"][:, :-1] + self.metric["g_xz"][:, 1:])
        g_zz_z = 0.5*(self.metric["g_zz"][:, :-1] + self.metric["g_zz"][:, 1:])

        return dict(xface=dict(ctr=(gxx_x, gxz_x, gzz_x), cov=(g_xx_x, g_xz_x, g_zz_x)),
                    zface=dict(ctr=(gxx_z, gxz_z, gzz_z), cov=(g_xx_z, g_xz_z, g_zz_z)))
    
    
    def _fac_per_area(self, face_metrics):
        """
        Build the *per-area* face factors that match your raw-jump stencil:
          x-face: fac_XX = √(g^{xx})/Δx,  fac_XZ = (g^{xz}/√(g^{xx})) * 1/(2Δz)
          z-face: fac_ZZ = √(g^{zz})/Δz,  fac_ZX = (g^{xz}/√(g^{zz})) * 1/(2Δx)
        """
        (gxx_x, gxz_x, gzz_x) = face_metrics["xface"]["ctr"]
        (gxx_z, gxz_z, gzz_z) = face_metrics["zface"]["ctr"]

        fac_XX = np.sqrt(gxx_x)/self.dx
        fac_XZ = (gxz_x/np.sqrt(gxx_x))*(1.0/(2.0*self.dz))

        fac_ZZ = np.sqrt(gzz_z)/self.dz
        fac_ZX = (gxz_z/np.sqrt(gzz_z))*(1.0/(2.0*self.dx))
        return fac_XX, fac_XZ, fac_ZZ, fac_ZX

    def _face_lengths(self, face_metrics):
        """
        If you want *integrated* flux coefficients, multiply per-area factors by face length:
          ℓ_x = ||a_z|| Δz = √(g_zz) Δz   (use covariant g_zz on x-faces)
          ℓ_z = ||a_x|| Δx = √(g_xx) Δx   (use covariant g_xx on z-faces)
        """
        (_, _, g_zz_x) = face_metrics["xface"]["cov"]
        (g_xx_z, _, _) = face_metrics["zface"]["cov"]

        #Note: Lz_x here means length in z on x face and so on.
        Lz_x = np.sqrt(g_zz_x) * self.dz   # shape (nx-1, nz)
        Lx_z = np.sqrt(g_xx_z) * self.dx   # shape (nx, nz-1)
        return Lz_x, Lx_z

    def _R_faces(self):
        """
        Rc : 2D array (nx, nz) of cell-centered R (can be signed).
        Returns:
          Rx_face : (nx-1, nz) R at +x faces
          Rz_face : (nx, nz-1) R at +z faces
        """

        # +x faces: average neighbors in x
        Rx_face = 0.5*(self.RR[1:,:] + self.RR[:-1,:])          # (nx-1, nz)

        # +z faces: average neighbors in z
        Rz_face = 0.5*(self.RR[:,1:] + self.RR[:,:-1])      # (nx, nz-1)

        return Rx_face, Rz_face
    
    #TODO: Padding value should depend on boundary conditions. Currently fill with 0s.
    #TODO: Make utility function???
    def _pad_to_full(self, arr,
                     dim='x',        # Update x or z.
                     faces="plus",   # "plus" = +x/+z faces; "minus" = left/down faces
                     pad="zero"):    # "zero" or "nan"
        """
        fx: (nx-1, nz) scalar on x-faces
        fz: (nx, nz-1) scalar on z-faces
        Returns (fx_full, fz_full) each (nx, nz), with the unused edge padded.
        """
        fill = 0.0 if pad == "zero" else np.nan
        new_arr = np.full((self.nx, self.nz), fill, float)

        key = (dim, faces)
        try:
            #Note, for slices None means all, i.e. a colon. Basically take no slice here.
            r, c = {
                ("x", "plus"):  (slice(0, self.nx-1), slice(None)),
                ("x", "minus"): (slice(1, self.nx),   slice(None)),
                ("z", "plus"):  (slice(None),   slice(0, self.nz-1)),
                ("z", "minus"): (slice(None),   slice(1, self.nz)),
            }[key]
        except KeyError as e:
            raise ValueError(f"Invalid combination: dim={dim!r}, faces={faces!r}") from e

        new_arr[r, c] = arr

        return new_arr

    def _create_dagp(self):
        dagp_fv_volume = self._calc_vol()
        fc_metric = self._face_metrics()
        fac_XX, fac_XZ, fac_ZZ, fac_ZX = self._fac_per_area(fc_metric)
        Lz_x, Lx_z = self._face_lengths(fc_metric)
        Rx_face, Rz_face = self._R_faces()

        utils.logger.info("Padding finite volume operators to 0 at outer Lx/Lz edges. "
                          "Does not affect immersed boundary if further in.")

        dagp_vars = dict(
            dagp_fv_XX = self._pad_to_full(fac_XX*Lz_x), #*Rx_face)
            dagp_fv_XZ = self._pad_to_full(fac_XZ*Lz_x), #*Rx_face)
            dagp_fv_ZZ = self._pad_to_full(fac_ZZ*Lx_z, dim='z'), #*Rz_face)
            dagp_fv_ZX = self._pad_to_full(fac_ZX*Lx_z, dim='z'), #*Rz_face)
            dagp_fv_volume = dagp_fv_volume)

        return dagp_vars
