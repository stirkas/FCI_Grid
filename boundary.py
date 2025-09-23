from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import pyplot as plt

import utils

class PolygonBoundary:
    def __init__(self, rbdy: NDArray[np.floating], zbdy: NDArray[np.floating],
            tol: utils.Tolerances = utils.DEFAULT_TOL):
        """
        Create a polygon boundary defined by (rbdy[i], zbdy[i]) vertices.

        Parameters
        ----------
        rbdy, zbdy : array_like (1D)
            R, Z coordinates of the polygon vertices.
        tol : Tolerances
            Numerical tolerances and point-in-polygon bias.
        """
        r = np.asarray(rbdy, dtype=float)
        z = np.asarray(zbdy, dtype=float)

        if r.shape != z.shape:
            raise ValueError(f"R and Z must have the same shape; got {r.shape} vs {z.shape}")
        if r.ndim != 1 or r.size < 3:
            raise ValueError("Need 1D arrays with at least 3 vertices for a polygon.")

        self.tol = tol

        self.rbdy, self.zbdy = self._clean_up_points(r, z, abs_tol=self.tol.closed_path_tol)
        self.num_pts = len(self.rbdy)
        self.path = self._build_path()

        #Store segment vector information.
        #TODO: Use length functions? Handle vectors consistently...
        ab = np.column_stack([self.rbdy, self.zbdy])
        self._a = ab[:-1]                              #Segment start points
        self._b = ab[1:]                               #Segment end points
        self._d = self._b - self._a                    #Segment vectors
        self._dd = np.sum(self._d*self._d, axis=1)     #Segment lengths
        self._n = self._d / np.sqrt(self._dd)[:, None] #Segment unit vectors

    def _clean_up_points(self, rpts: NDArray[np.floating], zpts: NDArray[np.floating], abs_tol: float
                    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Ensure closure, drop zero-length segments, and drop redundant middle points
        on vertical/horizontal runs. Returns cleaned (r,z).
        """
        # Exact-closure check
        end_gap = float(np.hypot(rpts[-1] - rpts[0], zpts[-1] - zpts[0]))
        if end_gap > abs_tol:
            utils.logger.warn(f"First and last wall points not equal within tol {abs_tol:g} (gap={end_gap:g}). "
                "Closing polygon by appending the first point; recommend visual verification of closed wall.")
            rpts = np.append(rpts, rpts[0])
            zpts = np.append(zpts, zpts[0])

        # Remove zero-length segments (keep the wraparound final vertex)
        r_zero = np.abs(np.diff(rpts)) > abs_tol
        z_zero = np.abs(np.diff(zpts)) > abs_tol
        keep = np.concatenate([r_zero | z_zero, np.array([True], dtype=bool)])
        rpts, zpts = rpts[keep], zpts[keep]

        # Drop unnecessary middle points on straight vertical/horizontal runs
        #TODO: Update for angled lines, not just horz/vert? Use PATH_ANGLE_TOL for colinear check?
        drop_mid_r = (np.abs(np.diff(rpts[:-1])) <= abs_tol) & (np.abs(np.diff(rpts[1:])) <= abs_tol)
        drop_mid_z = (np.abs(np.diff(zpts[:-1])) <= abs_tol) & (np.abs(np.diff(zpts[1:])) <= abs_tol)
        drop_mid = np.concatenate([[False], drop_mid_r | drop_mid_z, [False]])
        keep = ~drop_mid
        rpts, zpts = rpts[keep], zpts[keep]

        return rpts, zpts
    
    def _build_path(self) -> Path:
        """Build a matplotlib Path and mark it explicitly closed."""
        verts = np.column_stack([self.rbdy, self.zbdy])
        codes = np.full(len(verts), Path.LINETO, dtype=np.uint8)
        #Set start and end vertex information per docs.
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        return Path(verts, codes)

    def contains(self, rpts, zpts) -> NDArray[np.bool_]:
        """
        Vectorized point-in-polygon check with an inside bias.
        Works with any dimensional set of points.
        """
        #Broadcast to a common shape. Raises if shapes incompatible.
        rb, zb = np.broadcast_arrays(rpts, zpts)
        pts    = np.column_stack([rb.ravel(), zb.ravel()])
        inside = self.path.contains_points(pts, radius=float(self.tol.path_edge_in_bias))
        inside = inside.reshape(rb.shape)
        #Return 0-D ndarray for scalars which is truth-y; return mask array for higher dims.
        return inside.item() if inside.ndim == 0 else inside

    def _vertex_bisector(self, Rb, Zb, j):
        """Return angle bisector to image point given boundary points."""
        #Need to ignore final (closure) point when considering last vertex.
        #Note j here is the segment index.
        jm = (j-1) % (self.num_pts-1)
        jp = (j+1) % (self.num_pts-1)

        # edge tangents (into and out of the vertex)
        Tm = utils.unit([self.rbdy[j]-self.rbdy[jm], self.zbdy[j]-self.zbdy[jm]])
        Tp = utils.unit([self.rbdy[jp]-self.rbdy[j], self.zbdy[jp]-self.zbdy[j]])

        #Reverse first segment, treating vertex as origin.
        #Sum unit vectors to get bisector.
        B = -Tm + Tp
        Bu = utils.unit(B)

        #Need to deal with concave vs convex vertex.
        #Test with tiny segment for inside since full value could lie outside if passing another wall.
        if not self.contains(Rb + self.tol.path_tol*Bu[0], Zb + self.tol.path_tol*Bu[1]):
            Bu = -Bu

        return Bu

    def _nearest_point_on_path(self, Rg, Zg):
        """
        Return (seg_index, t, Rb, Zb) where (Rb,Zb) is closest point on polyline to (Rg,Zg).

        Vector formulation from wiki page "Distance from a point to a line". See image there for illustration.
        Assuming eqn of line is x = a + tn for t in [0,1]:
        (a-p)                  #Vector from point to line start.
        (a-p) dot n            #Projection of a-p onto line 
        (a-p) - ((a-p) dot n)n #Component of a-p perp to line!
        """
        g = np.array([Rg, Zg])
        ga = self._a - g                  #Ghost point to seg start vectors
        proj = np.sum(ga*self._n, axis=1) #Projection of vector along n.
        perp = ga - proj[:, None]*self._n #Perp vector to add to g to get to intersection.

        #Clamp point on line to endpoints. t = length along ab.
        t_raw = np.sum(((g + perp - self._a)*self._d), axis=1)/self._dd #Vec formula for length along line.
        before_cond = (t_raw < 0.0)[:,None]
        after_cond  = (t_raw > 1.0)[:,None]
        perp  = np.select([before_cond, after_cond], [self._a - g, self._b - g], default=perp)

        # pick nearest segment
        j = int(np.argmin(np.sum((perp)**2, axis=1)))

        return j, t_raw[j], [float(perp[j,0]), float(perp[j,1])]

    def _reflect_ghost_across_wall(self, Rg, Zg):
        #Get info about intersection point. Segment index, location along segment,
        #and normal vector from ghost to boundary.
        idx_b, loc_b, N_b = self._nearest_point_on_path(Rg, Zg)
        Rb, Zb = Rg + N_b[0], Zg + N_b[1]

        #If boundary point is an endpoint, follow bisection angle.
        if (loc_b <= 0.0) or (loc_b >= 1.0):
            utils.logger.debug(f"Calculating bisection vector for ghost point: {Rg,Zg}.")
            if loc_b >= 1.0:
                idx_b += 1
            bsct = self._vertex_bisector(Rb, Zb, idx_b)
            #Overwrite N_b with bisector.
            N_len = np.sqrt(N_b[0]**2 + N_b[1]**2)
            N_b[0], N_b[1] = N_len*bsct[0], N_len*bsct[1]

        #Store image point.
        Rimg, Zimg = Rb + N_b[0], Zb + N_b[1]

        #Lastly, if image point is outside, remove half of image length successively.
        #TODO: Find second intercept and go halfway between two boundaries?
        while not self.contains(Rimg, Zimg):
            utils.logger.debug(f"Moving outer image point back in for ghost point: {Rg,Zg}.")
            Rout,  Zout = Rimg - Rb, Zimg - Zb
            Rimg, Zimg  = Rb + Rout/2, Zb + Zout/2

        return (Rb, Zb), (Rimg, Zimg)
    
    def handle_bounds(self, rpts: NDArray[np.floating], zpts: NDArray[np.floating],
                        show=False) -> NDArray[np.bool_]:
        """
        Generate mask of ghost points from wall and 2D gridpoint arrays.
        """
        print("Generating ghost cells and boundary conditions...")
        inside = self.contains(rpts, zpts)
        neighbors_in = utils.neighbor_mask(inside)
        ghost_mask = (~inside) & neighbors_in

        #Now generate mask for inside cells which touch outside ones.
        outside = ~inside
        neighbors_out = utils.neighbor_mask(outside)
        in_mask = inside & neighbors_out

        #Get image points from ghost cells and wall points.
        Rg = rpts[ghost_mask]
        Zg = zpts[ghost_mask]
        Rb_list,   Zb_list   = [], []
        Rimg_list, Zimg_list = [], []
        for rgi, zgi in zip(Rg, Zg):
            (Rb, Zb), (Rimg, Zimg) = self._reflect_ghost_across_wall(rgi, zgi)
            Rb_list.append(Rb)
            Zb_list.append(Zb)
            Rimg_list.append(Rimg)
            Zimg_list.append(Zimg)

        if show:
            fig, ax = plt.subplots(figsize=(8,6))
            patch = PathPatch(self.path, facecolor='none', edgecolor='k', lw=2)
            ax.add_patch(patch)

            # ghost (green) and boundary-inside (blue)
            ax.scatter(rpts[ghost_mask], zpts[ghost_mask], s=16, c='g',   marker='o', label='ghost')
            ax.scatter(rpts[in_mask],    zpts[in_mask],    s=16, c='blue',marker='o', label='inner')

            # image points (red)
            ax.scatter(Rimg_list, Zimg_list, s=20, c='red', marker='o', label='image')

            # red intercept lines: ghost → boundary intercept
            for rgi, zgi, rbi, zbi, rimi, zimi in zip(Rg, Zg, Rb_list, Zb_list, Rimg_list, Zimg_list):
                ax.plot([rgi, rbi, rimi], [zgi, zbi, zimi], 'r-', lw=2, alpha=0.6)

            ax.set_aspect('equal', 'box')
            ax.set_xlabel('R')
            ax.set_ylabel('Z')
            ax.legend(loc='best')
            plt.tight_layout()
            plt.show()

        return ghost_mask
