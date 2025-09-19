import numpy as np
from scipy import interpolate

class CoordinateMapping(object):
    """Represents a coordinate mapping from cartesian R,Z to normalized x,z coordinates."""
    def __init__(self, nr, nz, dr, dz, RR, ZZ):
        #TODO: Dont need to pass all this in? But dont need to duplicate logic to create these?
        self.nx = nr
        self.nz = nz
        self.dx = dr
        self.dz = dz
        self.RR = RR
        self.ZZ = ZZ

        #Generate info for metric/derivative calculations.
        self.xinds  = np.arange(self.nx)
        self.zinds  = np.arange(self.nz)
        self.xxinds, self.zzinds = np.meshgrid(self.xinds, self.zinds, indexing='ij')
        self.spl_x = interpolate.RectBivariateSpline(self.xinds, self.zinds, self.RR)
        self.spl_z = interpolate.RectBivariateSpline(self.xinds, self.zinds, self.ZZ)

        #Generate metric.
        self.J, self.metric = self._make_metric()

        #Calculate finite volume operators. #TODO: Only really needs to happen in center...
        #So call from outside or do outside from metric?
        self.dagp_vars = self._create_dagp()

    def _get_coordinate(self, dx=0, dz=0):
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
        R = self.spl_x(self.xxinds, self.zzinds, dx=dx, dy=dz, grid=False)
        Z = self.spl_z(self.xxinds, self.zzinds, dx=dx, dy=dz, grid=False)

        return np.asarray(R), np.asarray(Z)

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
        #Differentials of (R,Z) when stepping in computational directions (r,z)
        dR_dx, dZ_dx = self._get_coordinate(dx=1, dz=0) #∆X when stepping in +r
        dR_dz, dZ_dz = self._get_coordinate(dx=0, dz=1) #∆X when stepping in +z
        #Convert differentials to partial derivatives.
        parR_x = dR_dx/self.dx
        parZ_x = dZ_dx/self.dx
        parR_z = dR_dz/self.dz
        parZ_z = dZ_dz/self.dz
        #J[k,i,...] with k∈{x,z}, i∈{R,Z}  → (2,2,nR,nZ)
        J = np.stack([[parR_x, parZ_x],
                      [parR_z, parZ_z]], axis=0)

        #Metric tensor g_{ij} = Σ_k J_{k i} J_{k j}  (D’Haeseleer 2.5.27)
        #Note our J is transposed, so g = J J^T
        g = np.einsum('ji...,ki...->jk...', J, J)#Calculate the gradient along each coordinate.

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
                  "gzz": gzz, "g_zz": g_zz}
        return jac, metric
    
    
    def _calc_vol(self):
        """
        Per-radian control volume A_R at centers from radius Rc and poloidal Jacobian Jp.
        Rc, Jp can be (nx,nz) arrays; dx,dz are logical spacings.
        Multiply by 2*pi for full volume.
        """
        return self.RR*self.J*self.dx*self.dz
    
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

        dagp_fv_XX = self._pad_to_full(fac_XX * Lz_x * Rx_face)
        dagp_fv_XZ = self._pad_to_full(fac_XZ * Lz_x * Rx_face)
        dagp_fv_ZZ = self._pad_to_full(fac_ZZ * Lx_z * Rz_face, dim='z')
        dagp_fv_ZX = self._pad_to_full(fac_ZX * Lx_z * Rz_face, dim='z')

        dagp_vars = {
            "dagp_fv_XX": dagp_fv_XX,
            "dagp_fv_XZ": dagp_fv_XZ,
            "dagp_fv_ZX": dagp_fv_ZX,
            "dagp_fv_ZZ": dagp_fv_ZZ,
            "dagp_fv_volume": dagp_fv_volume
        }

        return dagp_vars

###################
####DAGP TEST CODE: TODO: Test DAGP vars...
###
###from numpy.polynomial.legendre import leggauss
###
#### ---- You provide these two callables (splines or analytic) ----
#### Each must accept arrays x,z and return arrays of same shape.
###R   = lambda x,z: Rspl.ev(x,z)            # or analytic R(x,z)
###Rx  = lambda x,z: Rspl.ev(x,z, dx=1)      # dR/dx
###Rz  = lambda x,z: Rspl.ev(x,z, dy=1)      # dR/dz
###Z   = lambda x,z: Zspl.ev(x,z)
###Zx  = lambda x,z: Zspl.ev(x,z, dx=1)
###Zz  = lambda x,z: Zspl.ev(x,z, dy=1)
###
###def cov_metrics(x,z):
###    Rxv,Rzv,Zxv,Zzv = Rx(x,z), Rz(x,z), Zx(x,z), Zz(x,z)
###    gxx = Rxv*Rxv + Zxv*Zxv
###    gxz = Rxv*Rzv + Zxv*Zzv
###    gzz = Rzv*Rzv + Zzv*Zzv
###    rtg = np.sqrt(np.maximum(gxx*gzz - gxz*gxz, 0.0))  # |J|
###    return gxx, gxz, gzz, rtg
###
###def volumes_gauss(nx,nz, p=2, dx=1.0,dz=1.0):
###    xi,w = leggauss(p)
###    ii = np.arange(nx)[:,None,None,None]
###    jj = np.arange(nz)[None,:,None,None]
###    X = ii + 0.5*xi[None,None,:,None]
###    Z = jj + 0.5*xi[None,None,None,:]
###    X = np.broadcast_to(X, (nx,nz,p,p))
###    Z = np.broadcast_to(Z, (nx,nz,p,p))
###    gxx,gxz,gzz,rtg = cov_metrics(X,Z)
###    vol = (dx*dz)/(4.0) * np.sum((w[:,None]*w[None,:])[None,None]* R(X,Z)*rtg, axis=(2,3))
###    return vol
###
###def xface_facs_gauss(nx,nz, p=2, dz=1.0):
###    xi,w = leggauss(p)
###    X = (np.arange(nx-1)+0.5)[:,None,None]
###    Z = np.arange(nz)[None,:,None] + 0.5*xi[None,None,:]
###    X = np.broadcast_to(X, (nx-1,nz,p)); Z = np.broadcast_to(Z, (nx-1,nz,p))
###    gxx,gxz,gzz,rtg = cov_metrics(X,Z)
###    facXX = -(dz/2.0)*np.sum(w*( R(X,Z)*gzz/rtg ), axis=-1)
###    facXZ =  (dz/2.0)*np.sum(w*( R(X,Z)*gxz/rtg ), axis=-1)
###    return facXX, facXZ
###
###def zface_facs_gauss(nx,nz, p=2, dx=1.0):
###    xi,w = leggauss(p)
###    X = np.arange(nx)[:,None,None] + 0.5*xi[None,None,:]
###    Z = (np.arange(nz-1)+0.5)[None,:,None]
###    X = np.broadcast_to(X, (nx,nz-1,p)); Z = np.broadcast_to(Z, (nx,nz-1,p))
###    gxx,gxz,gzz,rtg = cov_metrics(X,Z)
###    facZX = -(dx/2.0)*np.sum(w*( R(X,Z)*gxz/rtg ), axis=-1)
###    facZZ =  (dx/2.0)*np.sum(w*( R(X,Z)*gxx/rtg ), axis=-1)
###    return facZX, facZZ
###
#### --- Reference (e.g., 8x8 and 8-pt) ---
###def volumes_ref(nx,nz):        return volumes_gauss(nx,nz,p=8)
###def xfaces_ref(nx,nz):         return xface_facs_gauss(nx,nz,p=8)
###def zfaces_ref(nx,nz):         return zface_facs_gauss(nx,nz,p=8)
###
#### --- Method 2 (midpoint) just for volumes; faces analogous at face centers ---
###def volumes_midpoint(nx,nz, dx=1.0,dz=1.0):
###    ii,jj = np.meshgrid(np.arange(nx), np.arange(nz), indexing='ij')
###    gxx,gxz,gzz,rtg = cov_metrics(ii,jj)
###    return R(ii,jj)*rtg*dx*dz
###
#### Timing/accuracy example for one grid:
###def benchmark(nx,nz,R,Z,Rx,Rz,Zx,Zz):
###    t0=time.perf_counter()
###    vref=volumes_ref(nx,nz); t1=time.perf_counter()
###    v2  =volumes_midpoint(nx,nz); t2=time.perf_counter()
###    v3  =volumes_gauss(nx,nz,p=2); t3=time.perf_counter()
###    print(f"ref 8x8: {t1-t0:.3f}s  | midpoint: {t2-t1:.3f}s  | 2x2 Gauss: {t3-t2:.3f}s")
###    rel2 = lambda a,b: np.linalg.norm((a-b).ravel())/np.linalg.norm(b.ravel())
###    relinf=lambda a,b: np.max(np.abs(a-b))/np.max(np.abs(b))
###    print(f"midpoint err L2={rel2(v2,vref):.3e}, Linf={relinf(v2,vref):.3e}")
###    print(f"2x2 Gauss err L2={rel2(v3,vref):.3e}, Linf={relinf(v3,vref):.3e}")
###
###################
