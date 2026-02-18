from typing import Optional, Tuple

import numpy as np

from grid import StructuredGrid
from utils import DataArray
import utils

class DAGP_FV:
    def __init__(self, grid: StructuredGrid, metric: DataArray,
                 Rfac: Optional[DataArray] = None) -> None:
        self.nx = grid.grid_shape_2d[0]
        self.nz = grid.grid_shape_2d[1]
        self.dx = grid.dx
        self.dz = grid.dz
        self.metric = metric

        #Additional optional factor for toroidal curvature in metric.
        #TODO: How to better handle toroidal factor? See field example.
        #Need to clean up every where its used too/fix unnecessary things.
        self.Rfac = Rfac if Rfac is not None else np.ones_like(metric["J"])

    def _calc_vol(self) -> DataArray:
        """
        Per-radian control volume A_R at centers from radius Rc and poloidal Jacobian Jp.
        Rc, Jp can be (nx,nz) arrays; dx,dz are logical spacings.
        Multiply by 2*pi for full volume.
        """
        if self.Rfac is not None:
            return self.Rfac*self.metric["J"]*self.dx*self.dz
        return self.metric["J"]*self.dx*self.dz
    
    def _face_metrics(self) -> dict:
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
    
    
    def _fac_per_area(self, face_metrics: dict) -> Tuple[DataArray, DataArray, DataArray, DataArray]:
        """Build the *per-area* face factors:"""
        (gxx_x, gxz_x, gzz_x) = face_metrics["xface"]["ctr"]
        (gxx_z, gxz_z, gzz_z) = face_metrics["zface"]["ctr"]

        fac_XX = np.sqrt(gxx_x)/self.dx
        fac_XZ = (gxz_x/np.sqrt(gxx_x))*(1.0/(2.0*self.dz))

        fac_ZZ = np.sqrt(gzz_z)/self.dz
        fac_ZX = (gxz_z/np.sqrt(gzz_z))*(1.0/(2.0*self.dx))
        return fac_XX, fac_XZ, fac_ZZ, fac_ZX

    def _face_lengths(self, face_metrics):
        """
        For *integrated* flux coefficients, multiply per-area factors by face length:
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
        Rx_face = 0.5*(self.Rfac[1:,:] + self.Rfac[:-1,:])      # (nx-1, nz)

        # +z faces: average neighbors in z
        Rz_face = 0.5*(self.Rfac[:,1:] + self.Rfac[:,:-1])      # (nx, nz-1)

        return Rx_face, Rz_face
    
    #TODO: Padding value should depend on boundary conditions? Currently fill with 0s.
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

    def calc_dagp(self) -> dict:
        dagp_fv_volume = self._calc_vol()
        fc_metric = self._face_metrics()
        fac_XX, fac_XZ, fac_ZZ, fac_ZX = self._fac_per_area(fc_metric)
        Lz_x, Lx_z = self._face_lengths(fc_metric)
        Rx_face, Rz_face = self._R_faces()

        area_fac_x = Lz_x*Rx_face if self.Rfac is not None else Lz_x
        area_fac_z = Lx_z*Rz_face if self.Rfac is not None else Lx_z

        utils.logger.info("Padding finite volume operators to 0 for last Lx/Lz faces. "
                          "Does not affect immersed boundary if further in.")

        dagp_vars = dict(
            dagp_fv_XX = self._pad_to_full(fac_XX*area_fac_x),
            dagp_fv_XZ = self._pad_to_full(fac_XZ*area_fac_x),
            dagp_fv_ZZ = self._pad_to_full(fac_ZZ*area_fac_z, dim='z'),
            dagp_fv_ZX = self._pad_to_full(fac_ZX*area_fac_z, dim='z'),
            dagp_fv_volume = dagp_fv_volume)

        return dagp_vars