from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np

from boundary import CoordPairs, PolygonBoundary
import utils
from utils import DataArray, MaskArray

@dataclass(frozen=True)
class ImmBdyInfo:
    """Container for immersed-boundary results."""
    in_mask:      MaskArray
    ghost_mask:   MaskArray
    border_mask:  MaskArray
    ghost_points: CoordPairs #(xg, zg)
    wall_points:  CoordPairs #(xb, zb)
    image_points: CoordPairs #(x_img, z_img)
    normals:      CoordPairs #(xn, zn)
    norm_dist:    DataArray

class GhostConnectivity(Enum):
    """How many cells to connect to around plasma when looking for ghosts."""
    TWO_D = 4
    TWO_D_WITH_CORNERS = 8

class ImmersedBoundary:
    """Immersed boundary helper operating on a boundary-like object."""

    def __init__(self, boundary: PolygonBoundary, *,
            connectivity: GhostConnectivity = GhostConnectivity.TWO_D_WITH_CORNERS) -> None:

        #Make sure # of ghost connections supported.
        if connectivity not in (GhostConnectivity.TWO_D,
                                GhostConnectivity.TWO_D_WITH_CORNERS):
            raise ValueError(f"connectivity must be {GhostConnectivity.TWO_D_WITH_CORNERS}"
                            f"or {GhostConnectivity.TWO_D_WITH_CORNERS}.")

        self.boundary = boundary
        self.connectivity = connectivity
        #Precompute segment geometry for nearest-point queries
        self._precompute_segments()

    def _precompute_segments(self) -> None:
        """Cache segment endpoints and unit tangents for distance queries."""
        #TODO: Use length functions? Handle vectors consistently...
        ab = np.column_stack([self.boundary.xbdy, self.boundary.zbdy])
        self._a  = ab[:-1]                              #Segment start points
        self._b  = ab[1:]                               #Segment end points
        self._d  = self._b - self._a                    #Segment vectors
        self._dd = np.sum(self._d * self._d, axis=1)    #Segment lengths
        self._n  = self._d / np.sqrt(self._dd)[:, None] #Segment unit vectors

    def _neighbor_mask(self, point_mask, nb_num=1):
        """
        Given a set of points which are true in point_mask, return a mask with their neighbors.
        Changing connectivity to 8 includes diagonals.
        """
        #Neighbors (booleans says: does this cell have a neighbor in that direction?)
        left  = np.zeros_like(point_mask, bool)
        right = np.zeros_like(point_mask, bool)
        down  = np.zeros_like(point_mask, bool)
        up    = np.zeros_like(point_mask, bool)

        left[nb_num:,:]   = point_mask[:-nb_num,:]
        right[:-nb_num,:] = point_mask[nb_num:,:]
        down[:,nb_num:]   = point_mask[:,:-nb_num]
        up[:,:-nb_num]    = point_mask[:,nb_num:]

        neighbor_any = left | right | up | down

        if self.connectivity == GhostConnectivity.TWO_D_WITH_CORNERS:
            ul = np.zeros_like(point_mask, bool)
            ur = np.zeros_like(point_mask, bool)
            dl = np.zeros_like(point_mask, bool)
            dr = np.zeros_like(point_mask, bool)

            ul[nb_num:, nb_num:]  = point_mask[:-nb_num, :-nb_num]
            ur[:-nb_num,nb_num:]  = point_mask[nb_num:,  :-nb_num]
            dl[nb_num:, :-nb_num] = point_mask[:-nb_num, nb_num: ]
            dr[:-nb_num,:-nb_num] = point_mask[nb_num:,  nb_num: ]

            neighbor_any |= (ul | ur | dl | dr)

        return neighbor_any

    def _reflect_point(self, xg: float, zg: float) -> Tuple[CoordPairs, CoordPairs, CoordPairs]:
        """Reflect a ghost point across the boundary.

        Returns:
            (xb, zb): nearest points on the boundary
            (x_img, z_img): image points (mirrored across boundary point)
            (nx, nz): vectors from ghost to boundary (normal-like)
        """
        #Get info about intersection point.
        #Segment index, location along segment, and normal vector from ghost to boundary.
        _, loc_b, n_b = self._nearest_point_on_path(xg, zg)
        xb, zb = xg + n_b[0], zg + n_b[1]

        #If boundary point is an endpoint, print a debug warning for convenience.
        #Note, leads to problems at sharp boundary regions.
        if (loc_b <= 0.0) or (loc_b >= 1.0):
            utils.logger.debug(f"Vertex bisection for ghost point: {xg,zg}.")

        #Store image point.
        x_img, z_img = xb + n_b[0], zb + n_b[1]

        #Lastly, if image point is outside, remove half of image length successively.
        #TODO: Right now it's best to just increase resolution to deal with this.
        #Note, this currently requires manually removing sharp boundary features.
        #A robust way to get around this is to upgrade to the Seo&Mittal (2011) IB algorithm.
        #Setting fix_edge_cases=True would make values not equidistant and lower the convergence order.
        fix_edge_cases = False
        if not self.boundary.contains(x_img, z_img):
            if fix_edge_cases is False:
                utils.logger.warn(f"Image point outside of domain for ghost point: {xg,zg}."
                                  " Try increasing resolution, generally 512x512 or 1024x1024 works best.")
            else:
                utils.logger.warn(f"Moving outer image point back in for ghost point: {xg,zg}.")
                x_out, z_out = x_img - xb, z_img - zb
                x_img, z_img  = xb + x_out/2, zb + z_out/2

        return (xb, zb), (x_img, z_img), (n_b[0], n_b[1])

    def _nearest_point_on_path(self, xg: float, zg: float) -> Tuple[int, float, CoordPairs]:
        """
        Return (seg_index, t, (xb,zb)) where (xb,zb) is closest point on polyline to (xg,zg).

        Vector formulation from wiki page "Distance from a point to a line". See image there for illustration.
        Assuming eqn of line is x = a + tn for t in [0,1]:
        (a-p)                  #Vector from point to line start.
        (a-p) dot n            #Projection of a-p onto line 
        (a-p) - ((a-p) dot n)n #Component of a-p perp to line!
        """
        g = np.array([xg, zg])
        ga = self._a - g                  #Ghost point to seg start vectors
        proj = np.sum(ga*self._n, axis=1) #Projection of vector along n.
        perp = ga - proj[:, None]*self._n #Perp vector to add to g to get to intersection.

        #Vector formula for length along line:
        t_raw = np.sum(((g + perp - self._a)*self._d), axis=1)/self._dd
        #Clamp point on line to endpoints.
        before_cond = (t_raw < 0.0)[:,None]
        after_cond  = (t_raw > 1.0)[:,None]
        perp  = np.select([before_cond, after_cond], [self._a - g, self._b - g], default=perp)

        # pick nearest segment
        j = int(np.argmin(np.sum((perp)**2, axis=1)))

        return j, t_raw[j], (perp[j, 0], perp[j, 1])

    def compute_ghost_info(self, xpts: DataArray, zpts: DataArray,
            *, show: bool = False) -> ImmBdyInfo:
        """Compute ghost cells + image/wall points from a boundary and grid arrays."""
        inside = np.asarray(self.boundary.contains(xpts, zpts), dtype=bool)

        #TODO: Use connectivity=4 for finding ghosts.
        #TODO: Use conn=8 now for some ghosts. Wont be used in code loops but will be ghost cells?
        #TODO: Have an issue when a second layer of ghost cells is needed at sharp areas.
        neighbors_in = self._neighbor_mask(inside)
        ghost_mask = (~inside) & neighbors_in

        #Now generate mask for inside cells which touch outside ones.
        outside = ~inside
        neighbors_out = self._neighbor_mask(outside)
        border_mask = inside & neighbors_out

        xg,zg = xpts[ghost_mask], zpts[ghost_mask]

        if xg.size == 0:
            raise ValueError("No ghost points were found for this boundary.")

        #Get boundary, image, and connecting normal info for each ghost point.
        reflections = [self._reflect_point(float(xgi), float(zgi)) for xgi, zgi in zip(xg, zg)]

        #Get x,z info per point.
        walls   = np.vstack([r[0] for r in reflections])
        images  = np.vstack([r[1] for r in reflections])
        normals = np.vstack([r[2] for r in reflections])
        nhat    = utils.unit_vecs(normals)

        xb, zb = walls[:,0], walls[:,1]
        xi, zi = images[:,0], images[:,1]
        xn, zn = nhat[:, 0], nhat[:, 1]

        #Get distance from wall to image. (I-B) dot n.
        nhat = utils.unit_vecs(np.stack([xn,zn], axis=-1))
        xhat, zhat = nhat[..., 0], nhat[..., 1]
        norm_dist = (xi-xb)*xhat + (zi-zb)*zhat

        imm_bdy_info = ImmBdyInfo(
            in_mask=inside,
            ghost_mask=ghost_mask,
            border_mask=border_mask,
            ghost_points=(xg, zg),
            wall_points=(xb, zb),
            image_points=(xi, zi),
            normals=(xn, zn),
            norm_dist=norm_dist)

        if show:
            self._plot_debug(xpts, zpts, imm_bdy_info)

        return imm_bdy_info

    def _plot_debug(self, xpts: DataArray, zpts: DataArray, info: ImmBdyInfo) -> None:
        """Debug plot of ghost/image construction."""
        _, ax = plt.subplots(figsize=(8, 6))

        self.boundary.plot(ax=ax)

        #Ghost (green), image pts (red), and boundary-inside (blue).
        ax.scatter(xpts[info.ghost_mask],  zpts[info.ghost_mask],  s=16, c="g",    marker="o", label="ghost")
        ax.scatter(xpts[info.border_mask], zpts[info.border_mask], s=16, c="blue", marker="o", label="inner")
        ax.scatter(info.image_points[0],   info.image_points[1],   s=20, c="red",  marker="o", label="image")

        #Red intercept lines: ghost → boundary intercept
        xg, zg = info.ghost_points
        xb, zb = info.wall_points
        x_img, z_img = info.image_points
        for xgi, zgi, xbi, zbi, ximi, zimi in zip(xg, zg, xb, zb, x_img, z_img):
            ax.plot([xgi, xbi, ximi], [zgi, zbi, zimi], "r-", lw=2, alpha=0.6)

        ax.set_aspect("equal", "box")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()