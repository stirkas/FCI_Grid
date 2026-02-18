import numpy as np
from scipy import interpolate

from grid import StructuredGrid, UniformAxisSpec

class CoordinateMapping(object):
    """Represents a coordinate mapping from physical to normalized x,z coordinates."""
    def __init__(self, nx, nz, x_phys, z_phys):
        #Set up normalized grid for mapping.
        #Note, nx/nz defined without ghost points. While phys grids contain them.
        #TODO: Output x and z from here to check against hermes at run time?
        x = UniformAxisSpec(0.0, 1.0, nx, shift=0.50)
        z = UniformAxisSpec(0.0, 1.0, nz, shift=0.50)

        self.grid = StructuredGrid(x,z)

        self.spl_x = interpolate.RectBivariateSpline(self.grid.x, self.grid.z, x_phys)
        self.spl_z = interpolate.RectBivariateSpline(self.grid.x, self.grid.z, z_phys)

        #Generate metric.
        self.metric = self._make_metric()

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
        R = self.spl_x(self.grid.x, self.grid.z, dx=dx, dy=dz, grid=True)
        Z = self.spl_z(self.grid.x, self.grid.z, dx=dx, dy=dz, grid=True)

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
        metric = {"dx": self.grid.dx,
                  "dz": self.grid.dz,
                  "gxx": gxx, "g_xx": g_xx,
                  "gxz": gxz, "g_xz": g_xz,
                  "gzz": gzz, "g_zz": g_zz,
                  "J": jac}

        return metric