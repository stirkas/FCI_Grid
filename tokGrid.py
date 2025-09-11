#!/usr/bin/env python

import sys
import uuid

from matplotlib import path as path
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.patches import Rectangle
import numpy as np
from scipy import integrate,interpolate
from scipy.spatial import cKDTree as KDTree

from boututils import datafile as bdata
from hypnotoad import __version__
from hypnotoad.geqdsk._geqdsk import read as gq_read #TODO: Use FreeQDSK like zoidberg?
from hypnotoad.utils.critical import find_critical as fc

#TODO:Add unit tests to make sure functionality is reasonable? Convert to classes with type hints, type checks, and header comments?

#Colored output, works for linux...
YELLOW = "\033[33m"
RED    = "\033[31m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
def info(message):
    print(BLUE + "INFO: " + message + RESET)
def warn(message):
    print(YELLOW + "WARNING: " + message + RESET)
def error(message):
    print(RED + "ERROR: " + message + RESET)

def calc_vol(Rc, Jp, dx=1.0, dz=1.0):
    """
    Per-radian control volume A_R at centers from radius Rc and poloidal Jacobian Jp.
    Rc, Jp can be (nx,nz) arrays; dx,dz are logical spacings.
    Multiply by 2*pi for full volume.
    """
    return Rc*Jp*dx*dz

#TODO: Padding value should depend on boundary conditions. Currently fill with 0s.
def pad_to_full(arr, nx, nz, *,
                dim='x',        # Update x or z.
                faces="plus",   # "plus" = +x/+z faces; "minus" = left/down faces
                pad="zero"):    # "zero" or "nan"
    """
    fx: (nx-1, nz) scalar on x-faces
    fz: (nx, nz-1) scalar on z-faces
    Returns (fx_full, fz_full) each (nx, nz), with the unused edge padded.
    """
    fill = 0.0 if pad == "zero" else np.nan
    new_arr = np.full((nx, nz), fill, float)

    key = (dim, faces)
    try:
        #Note, for slices None means all, i.e. a colon. Basically take no slice here.
        r, c = {
            ("x", "plus"):  (slice(0, nx-1), slice(None)),
            ("x", "minus"): (slice(1, nx),   slice(None)),
            ("z", "plus"):  (slice(None),   slice(0, nz-1)),
            ("z", "minus"): (slice(None),   slice(1, nz)),
        }[key]
    except KeyError as e:
        raise ValueError(f"Invalid combination: dim={dim!r}, faces={faces!r}") from e
    
    new_arr[r, c] = arr

    return new_arr

def face_metrics_from_centers(gxx_c, gxz_c, gzz_c, g_xx_c, g_xz_c, g_zz_c):
    """
    Inputs (cell-centered, shape = (nx, nz)):
      gxx_c, gxz_c, gzz_c : contravariant metric entries at cell centers.

    Returns dict with face-centered tensors:
      xface: contravariant (gxx,gxz,gzz) on +x faces shape (nx-1, nz),
             covariant    (g_xx,g_xz,g_zz) on +x faces shape (nx-1, nz)
      zface: contravariant (gxx,gxz,gzz) on +z faces shape (nx, nz-1),
             covariant    (g_xx,g_xz,g_zz) on +z faces shape (nx, nz-1)
    """
    #TODO: Remove np.asarray and use numpy arrays by default everywhere.
    gxx_c = np.asarray(gxx_c); gxz_c = np.asarray(gxz_c); gzz_c = np.asarray(gzz_c)
    g_xx_c = np.asarray(gxx_c); g_xz_c = np.asarray(gxz_c); g_zz_c = np.asarray(gzz_c)
    nx, nz = gxx_c.shape

    # --- arithmetic averages of contravariant entries to faces ---
    # +x faces between i and i+1
    gxx_x = 0.5*(gxx_c[:-1, :] + gxx_c[1:, :])
    gxz_x = 0.5*(gxz_c[:-1, :] + gxz_c[1:, :])
    gzz_x = 0.5*(gzz_c[:-1, :] + gzz_c[1:, :])

    # +z faces between j and j+1
    gxx_z = 0.5*(gxx_c[:, :-1] + gxx_c[:, 1:])
    gxz_z = 0.5*(gxz_c[:, :-1] + gxz_c[:, 1:])
    gzz_z = 0.5*(gzz_c[:, :-1] + gzz_c[:, 1:])

    # --- arithmetic averages of covariant entries to faces ---
    # +x faces between i and i+1
    g_xx_x = 0.5*(g_xx_c[:-1, :] + g_xx_c[1:, :])
    g_xz_x = 0.5*(g_xz_c[:-1, :] + g_xz_c[1:, :])
    g_zz_x = 0.5*(g_zz_c[:-1, :] + g_zz_c[1:, :])

    # +z faces between j and j+1
    g_xx_z = 0.5*(g_xx_c[:, :-1] + g_xx_c[:, 1:])
    g_xz_z = 0.5*(g_xz_c[:, :-1] + g_xz_c[:, 1:])
    g_zz_z = 0.5*(g_zz_c[:, :-1] + g_zz_c[:, 1:])

    return dict(
        xface=dict(ctr=(gxx_x, gxz_x, gzz_x), cov=(g_xx_x, g_xz_x, g_zz_x)),
        zface=dict(ctr=(gxx_z, gxz_z, gzz_z), cov=(g_xx_z, g_xz_z, g_zz_z)),
    )

def R_faces_from_centers(Rc, *, periodic_z=False, use_abs=False):
    """
    Rc : 2D array (nx, nz) of cell-centered R (can be signed).
    Returns:
      Rx_face : (nx-1, nz) R at +x faces
      Rz_face : (nx, nz-1) R at +z faces
    """
    Rc = np.asarray(Rc, float)

    # +x faces: average neighbors in x
    Rx_face = 0.5 * (Rc[1:, :] + Rc[:-1, :])          # (nx-1, nz)

    # +z faces: average neighbors in z
    if periodic_z:
        Rz_face = 0.5 * (Rc + np.roll(Rc, -1, axis=1))[:, :-1]  # drop last to keep (nx, nz-1)
    else:
        Rz_face = 0.5 * (Rc[:, 1:] + Rc[:, :-1])      # (nx, nz-1)

    if use_abs:
        Rx_face = np.abs(Rx_face)
        Rz_face = np.abs(Rz_face)

    return Rx_face, Rz_face

def fac_per_area_from_faces(face_metrics, dx, dz):
    """
    Build the *per-area* face factors that match your raw-jump stencil:
      x-face: fac_XX = √(g^{xx})/Δx,  fac_XZ = (g^{xz}/√(g^{xx})) * 1/(2Δz)
      z-face: fac_ZZ = √(g^{zz})/Δz,  fac_ZX = (g^{xz}/√(g^{zz})) * 1/(2Δx)
    """
    (gxx_x, gxz_x, gzz_x) = face_metrics["xface"]["ctr"]
    (gxx_z, gxz_z, gzz_z) = face_metrics["zface"]["ctr"]

    fac_XX = np.sqrt(gxx_x) / dx
    fac_XZ = (gxz_x / np.sqrt(gxx_x)) * (1.0 / (2.0*dz))

    fac_ZZ = np.sqrt(gzz_z) / dz
    fac_ZX = (gxz_z / np.sqrt(gzz_z)) * (1.0 / (2.0*dx))
    return fac_XX, fac_XZ, fac_ZZ, fac_ZX

def face_lengths_from_faces(face_metrics, dx, dz):
    """
    If you want *integrated* flux coefficients, multiply per-area factors by face length:
      ℓ_x = ||a_z|| Δz = √(g_zz) Δz   (use covariant g_zz on x-faces)
      ℓ_z = ||a_x|| Δx = √(g_xx) Δx   (use covariant g_xx on z-faces)
    """
    (_, _, g_zz_x) = face_metrics["xface"]["cov"]  # covariant on x-faces
    (g_xx_z, _, _) = face_metrics["zface"]["cov"]

    #Note: Lz_x here means length in z on x face and so on.
    Lz_x = np.sqrt(g_zz_x) * dz   # shape (nx-1, nz)
    Lx_z = np.sqrt(g_xx_z) * dx   # shape (nx, nz-1)
    return Lz_x, Lx_z

def make_3d(arr_2d: np.ndarray, ny: int) -> np.ndarray:
    """
    Repeat a 2D array (R, Z) along the middle axis to shape (R, ny, Z).
    If writable=False, returns a broadcasted (read-only) view.
    """

    return np.repeat(arr_2d[:, np.newaxis, :], ny, axis=1)

def length(v):
    return np.sqrt(np.dot(v,v))

def unit(v):
    return v/length(v)

def vertex_bisector(R, Z, Rb, Zb, wall_path, j):
    #Wrap around for last index.
    if j == len(R)-1: j = 0

    # edge tangents (into and out of the vertex)
    Tm = unit([R[j]  - R[j-1], Z[j]  - Z[j-1]])
    Tp = unit([R[j+1] - R[j],  Z[j+1] - Z[j]])

    #Reverse first segment and treat vertex as origin.
    #Sum unit vectors to get bisector.
    B = -Tm + Tp
    Bu = unit(B)

    #Need to deal with concave vs convex vertex.
    #Use tiny segment to test correctly inside since full value could lie outside if passing another wall.
    if not wall_path.contains_point((Rb + PATH_TOL*Bu[0], Zb + PATH_TOL*Bu[1])):
        Bu = -Bu

    return Bu

def nearest_point_on_path(Rw, Zw, Rg, Zg):
    """Return (seg_index, t, Rb, Zb) where (Rb,Zb) is closest point on polyline to (Rg,Zg).
        
    Vector formulation from wiki page "Distance from a point to a line". See image there for illustration.
    Assuming eqn of line is x = a + tn for t in [0,1]:
    (a-p)                  #Vector from point to line start.
    (a-p) dot n            #Projection of a-p onto line 
    (a-p) - ((a-p) dot n)n #Component of a-p perp to line!
    """
    #Get segment vectors and vector to starts from ghost points.
    a = np.column_stack([Rw[:-1], Zw[:-1]]) #Segment start points
    b = np.column_stack([Rw[1:],  Zw[1: ]]) #Segment end points
    g = np.array([Rg, Zg])                  #Ghost points
    d = b - a                               #Segment vectors
    dd = np.sum(d*d, axis=1)
    l  = np.sqrt(dd) #TODO: Use length functions? Handle vectors consistently...
    n = d/l[:,None]                         #Segment unit vectors

    ga = a - g                  #Ghost point to seg start vectors
    proj = np.sum(ga*n, axis=1) #Projection of vector along n.
    perp = ga - proj[:, None]*n #Perp vector to add to g to get to intersection.

    #Clamp point on line to endpoints. t = length along ab.
    t_raw = np.sum(((g + perp - a)*d), axis=1)/dd
    before_cond = (t_raw < 0.0)[:,None]
    after_cond  = (t_raw > 1.0)[:,None]
    perp  = np.select([before_cond, after_cond], [a - g, b - g], default=perp)

    # pick nearest segment
    j = int(np.argmin(np.sum((perp)**2, axis=1)))

    return j, t_raw[j], [float(perp[j,0]), float(perp[j,1])]

def reflect_ghost_across_wall(wall_path, Rg, Zg):
    #Get wall points.
    Rw, Zw = wall_path.vertices[:,0], wall_path.vertices[:,1]
    #Get info about intersection point. Segment index, location along segment,
    #and normal vector from ghost to boundary.
    idx_b, loc_b, N_b = nearest_point_on_path(Rw, Zw, Rg, Zg)
    Rb, Zb = Rg + N_b[0], Zg + N_b[1]

    #If boundary point is an endpoint, follow bisection angle.
    if (loc_b <= 0.0) or (loc_b >= 1.0):
        print("Calculating bisection vector.")
        print(Rg, Zg)
        if loc_b >= 1.0:
            idx_b += 1
        bsct = vertex_bisector(Rw, Zw, Rb, Zb, wall_path, idx_b)
        #Overwrite N_b with bisector.
        N_len = np.sqrt(N_b[0]**2 + N_b[1]**2)
        N_b[0], N_b[1] = N_len*bsct[0], N_len*bsct[1]

    #Lastly, if image point is outside, divide extra length in half successively.
    #TODO: Find second intercept and go halfway between two boundaries?
    Rimg, Zimg = Rb + N_b[0], Zb + N_b[1]
    while not wall_path.contains_points([[Rimg, Zimg]], radius=PATH_EDGE_IN_BIAS):
        print("Moving outer image point back in.")
        print(Rg, Zg)
        Rout,  Zout = Rimg - Rb, Zimg - Zb
        Rimg, Zimg  = Rb + Rout/2, Zb + Zout/2

    return (Rb, Zb), (Rimg, Zimg)

def get_neighbor_mask(point_mask, connectivity=4):
    """
    Given a set of points which are true in point_mask, return a mask with their neighbors.
    Changing connectivity to 8 includes diagonals.
    """
    # neighbors (booleans says: does this cell have a neighbor in that direction?)
    left  = np.zeros_like(point_mask, bool)
    right = np.zeros_like(point_mask, bool)
    down  = np.zeros_like(point_mask, bool)
    up    = np.zeros_like(point_mask, bool)
    
    left[1:,:]   = point_mask[:-1,:]
    right[:-1,:] = point_mask[1:,:]
    down[:,1:]   = point_mask[:,:-1]
    up[:,:-1]    = point_mask[:,1:]

    neighbor_any = left | right | up | down

    if connectivity == 8:
        ul = np.zeros_like(point_mask, bool)
        ur = np.zeros_like(point_mask, bool)
        dl = np.zeros_like(point_mask, bool)
        dr = np.zeros_like(point_mask, bool)

        ul[1:, 1:]  = point_mask[:-1, :-1]
        ur[:-1,1:]  = point_mask[1:,  :-1]
        dl[1:, :-1] = point_mask[:-1, 1: ]
        dr[:-1,:-1] = point_mask[1:,   1: ]

        neighbor_any |= (ul | ur | dl | dr)

    return neighbor_any

def handle_bounds(RR, ZZ, wall_path, show=False):
    """
    Generate ghost mask and boundary conditions.
    """
    pts = np.column_stack([RR.ravel(), ZZ.ravel()])
    inside = wall_path.contains_points(pts, radius=PATH_EDGE_IN_BIAS)
    inside = inside.reshape(RR.shape)
    neigbors_in = get_neighbor_mask(inside)
    ghost_mask = (~inside) & neigbors_in

    #Now generate mask for inside cells which touch outside ones.
    outside = ~inside
    neighbors_out = get_neighbor_mask(outside)
    in_mask = inside & neighbors_out

    #Get image points from ghost cells and wall points.
    Rg = RR[ghost_mask]
    Zg = ZZ[ghost_mask]
    Rb_list,   Zb_list   = [], []
    Rimg_list, Zimg_list = [], []
    for rgi, zgi in zip(Rg, Zg):
        (Rb, Zb), (Rimg, Zimg) = reflect_ghost_across_wall(wall_path, rgi, zgi)
        Rb_list.append(Rb);     Zb_list.append(Zb)
        Rimg_list.append(Rimg); Zimg_list.append(Zimg)

    if show:
        fig, ax = plt.subplots(figsize=(8,6))
        patch = PathPatch(wall_path, facecolor='none', edgecolor='k', lw=2)
        ax.add_patch(patch)

        # ghost (green) and boundary-inside (blue)
        ax.scatter(RR[ghost_mask], ZZ[ghost_mask], s=16, c='g',   marker='o', label='ghost')
        ax.scatter(RR[in_mask],    ZZ[in_mask],    s=16, c='blue',marker='o', label='inner')

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

CLOSED_PATH_TOL   =  0.0
PATH_TOL          =  1e-12
PATH_EDGE_IN_BIAS = -PATH_TOL #Use to bias points on wall to inside.
PATH_ANGLE_TOL    =  1e-12
def make_path(rpts, zpts, abs_tol=CLOSED_PATH_TOL):
    if abs_tol == CLOSED_PATH_TOL:
        warn("Constructing closed path with tol of " + str(CLOSED_PATH_TOL) + \
              ". This seems generally ok for gfiles.")

    if rpts.shape != zpts.shape or rpts.size < 3:
        raise ValueError("Need matching size R,Z arrays with ≥4 points (a closed triangle at minimum).")

    #Exact-closure check by default.
    end_gap = float(np.hypot(rpts[-1]-rpts[0], zpts[-1]-zpts[0]))
    if (end_gap > abs_tol):
        warn("First and last wall points not exactly equal within tol: " + str(abs_tol) + ". " \
            + "Forcing wall closure at final point (" + str(rpts[0]) + ", " + str(zpts[0]) + "),\
              double check wall looks closed correctly.")
        rpts = np.append(rpts, rpts[0])
        zpts = np.append(zpts, zpts[0])

    #Remove segments with zero length, but keep final segment which wraps around to beginning.
    r_zero = np.abs(np.diff(rpts)) > abs_tol
    z_zero = np.abs(np.diff(zpts)) > abs_tol
    keep = np.concatenate([(r_zero | z_zero), np.array([True])])
    rpts, zpts = rpts[keep], zpts[keep]
    #Also remove unnecessary segments (middle segments along a vert/horz line).
    #TODO: Update for angled lines, not just horz/vert? Use PATH_ANGLE_TOL for colinear check?
    drop_mid_r = (np.abs(np.diff(rpts[:-1])) <= abs_tol) & (np.abs(np.diff(rpts[1:])) <= abs_tol)
    drop_mid_z = (np.abs(np.diff(zpts[:-1])) <= abs_tol) & (np.abs(np.diff(zpts[1:])) <= abs_tol)
    drop_mid   = drop_mid_r | drop_mid_z
    drop_mid   = np.concatenate([np.array([False]), drop_mid, np.array([False])])
    keep       = ~drop_mid
    rpts, zpts = rpts[keep], zpts[keep]

    #Build a Path with explicit CLOSEPOLY...i.e. tell Path() class it is explicitly closed.
    verts = np.column_stack([rpts, zpts])
    codes = np.full(len(verts), path.Path.LINETO, dtype=np.uint8)
    #Set start and end vertex information per docs.
    codes[0]  = path.Path.MOVETO
    codes[-1] = path.Path.CLOSEPOLY

    wall_path = path.Path(verts, codes)

    return wall_path, rpts, zpts

def setup_field_line_interpolation(R_grid, Z_grid, dRdphi, dZdphi, linear=0):
    """
    Set up interpolators for field line derivatives
    """
    # Create interpolators
    if (~linear):
        dR_itp = interpolate.RectBivariateSpline(R_grid, Z_grid, dRdphi)
        dZ_itp = interpolate.RectBivariateSpline(R_grid, Z_grid, dZdphi)
    else:
        dR_itp = interpolate.RegularGridInterpolator(
            (R_grid, Z_grid), 
            dRdphi,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        dZ_itp = interpolate.RegularGridInterpolator(
            (R_grid, Z_grid), 
            dZdphi,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

    def field_line_rhs(zeta, pos):
        """
        RHS function for field line integration
        """
        R, Z = pos
        
        # Interpolate derivatives at current position
        dRdphi = np.squeeze(dR_itp(R, Z))
        dZdphi = np.squeeze(dZ_itp(R, Z))
        
        return np.array([dRdphi, dZdphi])
    
    return field_line_rhs

def trace_until_wall(R0, Z0, phi_init, dphi, field_line_rhs, wall_path, direction=1):
    """
    Trace a field line starting at (R0, Z0) from phi_init in steps of dphi
    (sign given by direction), stopping as soon as the point exits wall_path.
    """
    Rvals = [R0]
    Zvals = [Z0]

    phi_current = phi_init

    while True:
        # advance one step in phi
        phi_next = phi_current + direction*dphi

        # trace from phi_current → phi_next
        sol = trace_field_line(
            Rvals[-1], Zvals[-1],
            phi_current, phi_next,
            field_line_rhs
        )
        Rn, Zn = sol.y[0, -1], sol.y[1, -1]

        # stop if we’ve left the wall
        if not wall_path.contains_point((Rn, Zn)):
            break

        # otherwise record and continue
        Rvals.append(Rn)
        Zvals.append(Zn)
        phi_current = phi_next

    return Rvals, Zvals

def trace_field_line(R0, Z0, zeta_init, zeta_target, field_line_rhs):
    """
    Trace a field line from starting position
    """
    sol = integrate.solve_ivp(
        field_line_rhs,
        (zeta_init, zeta_target),
        [R0, Z0],
        method="RK45", #method='DOP853' better for stiff problems
        rtol=1e-10,
        atol=1e-12,
        dense_output=True #Generates interpolatable data.
    )
    
    return sol

#TODO: Clean up this functionality? Recombine with getCoordinate. Also clean up down to findIndex().
#Note column stack for tree, so nr != nz is ok. Need to make sure this is fine in zoidberg to update and remove asserts.
#Also remove temp files and access classes in zoidberg directly (and transfer bug fixes therein?)
#Also have meshgrids multiple places now.
def getCoordSpline(R, Z):
    # Get arrays of indices
    nx, nz = R.shape
    xinds = np.arange(nx)
    zinds = np.arange(nz)

    position = np.column_stack((R.ravel(),Z.ravel()))
    tree = KDTree(position)

    _spl_r = interpolate.RectBivariateSpline(xinds, zinds, R)
    _spl_z = interpolate.RectBivariateSpline(xinds, zinds, Z)

    return _spl_r, _spl_z, tree


def getCoordinate(R, Z, spl_r, spl_z, xind, zind, dx=0, dz=0):
    nx, nz = R.shape

    if (np.amin(xind) < 0) or (np.amax(xind) > nx - 1):
        raise ValueError("x index out of range")
    if (np.amin(zind) < 0) or (np.amax(zind) > nz - 1):
        raise ValueError("z index out of range")

    R = spl_r(xind, zind, dx=dx, dy=dz, grid=False)
    Z = spl_z(xind, zind, dx=dx, dy=dz, grid=False)

    return R, Z

def getCoordSplineOrig(R, Z):
    assert R.shape == Z.shape

    # Get arrays of indices
    nx, nz = R.shape
    xinds = np.arange(nx)
    zinds = np.arange(nz)
    
    # Create a KDTree for quick lookup of nearest points
    n = R.size
    data = np.concatenate((R.reshape((n, 1)), Z.reshape((n, 1))), axis=1)
    tree = KDTree(data)

    xinds = np.arange(nx)
    zinds = np.arange(nz * 3)
    # Repeat the data in z, to approximate periodicity
    R_ext = np.concatenate((R, R, R), axis=1)
    Z_ext = np.concatenate((Z, Z, Z), axis=1)

    _spl_r = interpolate.RectBivariateSpline(xinds, zinds, R_ext)
    _spl_z = interpolate.RectBivariateSpline(xinds, zinds, Z_ext)

    return _spl_r, _spl_z, tree

def getCoordinateOrig(R, Z, spl_r, spl_z, xind, zind, dx=0, dz=0):
    nx, nz = R.shape
    if (np.amin(xind) < 0) or (np.amax(xind) > nx - 1):
        raise ValueError("x index out of range")
    
    # Periodic in y (z!)
    zind = np.remainder(zind, nz)

    R = spl_r(xind, zind + nz, dx=dx, dy=dz, grid=False)
    Z = spl_z(xind, zind + nz, dx=dx, dy=dz, grid=False)

    return R, Z

#NOTE: Taken from zoidberg StructuredPoloidalGrid.
def get_metric(R, Z, nx, nz, dx, dz):
    #Get arrays of indices.
    xind, zind = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")

    spl_r, spl_z, tree = getCoordSpline(R, Z)

    #Calculate the gradient along each coordinate.
    dolddnew = np.array(
        [getCoordinate(R, Z, spl_r, spl_z, xind, zind, dx=a, dz=b) for a, b in ((1, 0), (0, 1))]
    )
    #Dims: 0 : dx or dz?
    #      1 : R or z?
    #      2 : spatial: r
    #      3 : spatial: \theta
    ddist = np.sqrt(np.sum(dolddnew**2, axis=1)) #Sum R + Z
    nx, nz = ddist.shape[1:]
    ddist[0] = dx
    ddist[1] = dz
    dolddnew /= ddist[:,None,...]

    #g_ij = J_ki J_kj
    #(2.5.27) from D'Haeseleer 1991
    #Note: our J is transposed
    J = dolddnew
    g = np.sum(
        np.array(
            [
                [[J[j, i] * J[k, i] for i in range(2)] for j in range(2)]
                for k in range(2)
            ]
        ),
        axis=2,
    )

    assert np.all(
        g[0, 0] > 0
    ), f"g[0, 0] is expected to be positive, but some values are not (minimum {np.min(g[0, 0])})"
    assert np.all(
        g[1, 1] > 0
    ), f"g[1, 1] is expected to be positive, but some values are not (minimum {np.min(g[1, 1])})"
    g = g.transpose(2, 3, 0, 1)
    assert np.all(
        np.linalg.det(g) > 0
    ), f"All determinants of g should be positive, but some are not (minimum {np.min(np.linalg.det(g))})"
    ginv = np.linalg.inv(g)
    #Jacobian from BOUT++
    JB = R * (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
    return {
        "dx": ddist[0],
        "dz": ddist[1], #Grid spacing
        "gxx": ginv[..., 0, 0],
        "g_xx": g[..., 0, 0],
        "gxz": ginv[..., 0, 1],
        "g_xz": g[..., 0, 1],
        "gzz": ginv[..., 1, 1],
        "g_zz": g[..., 1, 1],
        #"J": JB,
    }

def findIndex(R, Z, Rmin, Zmin, dR, dZ, bdry, show=False):
    """Finds the (x,z) index corresponding to the given (R,Z) coordinate

    Parameters
    ----------
    R, Z : array_like
        Locations to find indices for

    Returns
    -------
    x, z : (ndarray, ndarray)
        Index as a float, same shape as R,Z
    """
    
    # Make sure inputs are NumPy arrays
    R = np.asarray(R)
    Z = np.asarray(Z)

    # Check that they have the same shape
    assert R.shape == Z.shape

    #Note mins and dR,dZ come from central grid here. R,Z from mapped grids.
    xind = (R - Rmin) / dR
    zind = (Z - Zmin) / dZ

    # Note: These indices may be outside the domain,
    # but this is handled in BOUT++, and useful for periodic
    # domains.

    #Mask out points around boundary.
    pts = np.column_stack([R.ravel(), Z.ravel()])
    inside = bdry.contains_points(pts, radius=PATH_EDGE_IN_BIAS)
    inside = inside.reshape((R.shape[0], R.shape[1]))

    nrp, nzp = len(R), len(Z)
    xind[~inside] = nrp
    zind[~inside] = nzp #TODO: Double check this in BOUT. Maybe helps with boundary issues. Should be +1 for x and z like seps though???

    if (show):
        fig, ax = plt.subplots(figsize=(6,6))

        # 1) draw the wall Path itself
        patch = PathPatch(bdry, facecolor='none', edgecolor='k', lw=2)
        ax.add_patch(patch)

        # 2) scatter the points inside (green) and outside (red)
        ax.scatter(
            R[inside], Z[inside],
            s=10, c='g', marker='o', label='inside', alpha=0.6
        )
        ax.scatter(
            R[~inside], Z[~inside],
            s=10, c='r', marker='x', label='outside', alpha=0.6
        )

        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.legend(loc='upper right')
        plt.show()

    return xind, zind

def plot(R, Z, psi, ghost, rbdy, zbdy, rlmt, zlmt, sign_b0,
        Rvals_pos, Zvals_pos, Rvals_neg, Zvals_neg, opoints, xpoints,
        checkPts, gridPts, fwdPts, bwdPts):
    #Plot contours, LCFS, wall and field line.
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    fig, ax = plt.subplots(figsize=(8,10))
    cf = ax.contourf(R, Z, psi.T, levels=100, cmap='viridis')
    #cf = ax.contourf(RR, ZZ, psi, levels=100, cmap='viridis') #How to plot without tranposing on a meshgrid.
    plt.colorbar(cf, ax=ax)
    ax.plot(rbdy, zbdy, '.-', color='orange', label='LCFS')
    ax.plot(rlmt, zlmt, '-',  color='black',  label='Wall')
    ax.add_patch(ghost)
    phi_dir, neg_phi_dir = ('+','-') if sign_b0 == 1 else ('-','+')
    ax.plot(Rvals_pos, Zvals_pos, '.', color='red',  label='$+B_{\\phi} = ' + phi_dir     + '\\phi$')
    ax.plot(Rvals_neg, Zvals_neg, '.', color='cyan', label='$-B_{\\phi} = ' + neg_phi_dir + '\\phi$')
    #Plotting assuming one x-point at the moment.
    for point in opoints:
        ax.plot(point[0], point[1], 'o', label='O',
                markerfacecolor='none', markeredgecolor='lime')
    for point in xpoints:
        ax.plot(point[0], point[1], 'x', color='lime', label='X')

    if checkPts:
        for idx, point in enumerate(gridPts):
            x0, y0 = gridPts[idx]
            xf, yf = fwdPts[idx]
            xb, yb = bwdPts[idx]
            ax.plot([x0, xf], [y0, yf], '-', color='red', linewidth=2)
            ax.plot([x0, xb], [y0, yb], '--', color='cyan', linewidth=2)
            ax.scatter(x0, y0, color='k', s=100, marker='*', zorder=2)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.06),
            ncol=len(labels), fancybox=True, shadow=True)
    ax.set_xlabel('R')
    ax.set_ylabel('Z')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def main(args):
    #Read eqdsk file.
    gfile_dir = "/home/tirkas1/Workspace/TokData/"
    device = "DIIID" + "/"
    #device = "TCV" + "/"
    gfile1 = "g162940.02944_670" #Old ql one.
    gfile2 = "g163241.03500" #Old DIIID one.
    #Ben's test cases for varying Ip and B0 directions.
    gfile3 = "g172208.03000"
    gfile4 = "g174791.03000"
    gfile5 = "g176413.03000"
    gfile6 = "g176312.03000"
    #TCV Case for simpler geometry.
    gfile7 = "65402_t1.eqdsk" #TODO: Need to make sure nr != nz is ok. TCV quite elongated.
    gfilename = gfile1
    gfilepath = gfile_dir + device + gfilename
    print("Reading EQDSK file...")
    with open(gfilepath, "r", encoding="utf-8") as file:
        gfile = gq_read(file)
    print("Finished reading EQDSK file...")

    nrg, nzg = gfile["nx"], gfile["ny"]
    rmin, rmax = gfile["rleft"], gfile["rleft"] + gfile["rdim"]
    zmin, zmax = gfile["zmid"] - 0.5*gfile["zdim"], gfile["zmid"] + 0.5*gfile["zdim"]
    Rg = np.linspace(rmin, rmax, nrg)
    Zg = np.linspace(zmin, zmax, nzg)

    R0    = gfile["rmagx"]
    Z0    = gfile["zmagx"]
    psi   = gfile["psi"]
    fpol  = gfile["fpol"] # f = R*B_t
    qpsi  = gfile["qpsi"] # q(psi)
    prsr  = gfile["pres"]
    pcur  = gfile["cpasma"]
    bcntr = gfile["bcentr"]
    paxis = gfile["simagx"]
    pbdry = gfile["sibdry"]
    rbdy = np.array(gfile["rbdry"])
    zbdy = np.array(gfile["zbdry"])
    rlmt = np.array(gfile["rlim"])
    zlmt = np.array(gfile["zlim"])

    #Get direction of field and current. Assuming (R,phi,Z) is RHS with phi CCW from top of tokamak per COCOS.
    sign_ip = np.sign(pcur)
    sign_b0 = np.sign(bcntr)

    #Calculate field data.
    print("Calculating field data...")
    psin = (psi - paxis) / (pbdry - paxis)
    psi_func = interpolate.RectBivariateSpline(Rg, Zg, psi)

    #Toroidal field component and q(psi).
    psi1D = np.linspace(paxis, pbdry, nrg)
    #psi1D = np.linspace(pbdry, paxis, nrg) #TODO: For TCV it seems r is backwards??? Different cocos convention?
    #TODO: Why doesnt ext=0 work ok when tracing field?
    f_spl = interpolate.InterpolatedUnivariateSpline(psi1D, fpol, ext=3) #ext=3 uses boundary values outside range.
    q_spl = interpolate.InterpolatedUnivariateSpline(psi1D, qpsi, ext=3) #ext=0 uses extrapolation as with RectBivSpline on 2D but doesnt work in the integrator.
    p_spl = interpolate.InterpolatedUnivariateSpline(psi1D, prsr, ext=3)

    #Generate simulation grid (lower res + guard cells)
    #TODO: Can pad in z as well in full FCI case, but wont matter if gfile large outside mask. But Z periodic in BOUT...
    rpad, phipad, zpad = 2, 1, 2 #Padding/ghost cells on each end.
    r_res, phi_res, z_res = 64, 64, 64
    nr, nphi, nz = r_res, 1, z_res
    phi_val = 0 #Take phi=0 to be starting point.
    dphi = 2*np.pi/phi_res
    phi_arr = [dphi]
    #Note, 2d really means 3d with len(phi) == 1. Use full 3d for turbulence sims.
    METRIC_2D = True
    metric_info = "Generating one poloidal plane in 3D." if METRIC_2D == True \
             else "Extending info to multiple poloidal planes in 3D."
    info(metric_info)
    if not METRIC_2D:
        phi_arr = np.linspace(phi_val, 2*np.pi, nphi, endpoint=False) #Periodic so dont go all the way.
        dphi = phi_arr[1]-phi_arr[0]
        nphi = phi_res

    #Add ghost cell padding to R and Z arrays.
    nrp, nzp = nr + 2*rpad, nz + 2*zpad
    R, Z = np.linspace(rmin, rmax, nr), np.linspace(zmin, zmax, nz)
    dR = R[1]-R[0]
    ghosts_lo_R = R[0]  - dR*np.arange(rpad, 0, -1)
    ghosts_hi_R = R[-1] + dR*np.arange(1, rpad+1)
    dZ = Z[1]-Z[0]
    ghosts_lo_Z = Z[0]  - dZ * np.arange(zpad, 0, -1)
    ghosts_hi_Z = Z[-1] + dZ * np.arange(1, zpad+1)
    R = np.concatenate((ghosts_lo_R, R, ghosts_hi_R))
    Z = np.concatenate((ghosts_lo_Z, Z, ghosts_hi_Z))
    RR, ZZ = np.meshgrid(R,Z,indexing='ij')
    dr, dz = 1/(nr-1), 1/(nz-1)

    #Calculate field components following COCOS convention.
    Bp_R =  sign_ip*psi_func(R, Z, dy=1)/RR
    Bp_Z = -sign_ip*psi_func(R, Z, dx=1)/RR
    Bp   = np.sqrt(Bp_R**2 + Bp_Z**2)
    Bphi = f_spl(psi_func(R, Z))/RR
    Bmag = np.sqrt(Bp_R**2 + Bp_Z**2 + Bphi**2)
    pres = p_spl(psi_func(R,Z))

    #Compute derivatives along phi NOT B_phi. So need to pass field line direction when following below.
    #R factor comes from cylindrical geometry.
    dRdphi = RR*Bp_R/Bphi
    dZdphi = RR*Bp_Z/Bphi

    #Set up interpolation
    print("Setting up interpolation...")
    #Cubic splines seem to work well but can try linear if regions of high gradients exist and splines get messy.
    field_line_rhs = setup_field_line_interpolation(R, Z, dRdphi, dZdphi)
    #field_line_rhs_lin = setup_field_line_interpolation(R, Z, dRphi, dZphi, linear=1)

    #Choose a starting point for tracing single field line.
    offset = 0.005
    sep_idx = np.argmax(rbdy)
    R1, Z1 = rbdy[sep_idx] + offset, zbdy[sep_idx] #Minor offset from separatrix.
    #R1, Z1 = R0, Z0               #Magnetic axis
    #R1, Z1 = R[3*nr//4], Z[nz//2] #Core point.
    #R1, Z1 = R[7*nr//8], Z[nz//2] #Outer point.
    #R1, Z1 = rbdy[sep_idx], zbdy[sep_idx] #Separatrix

    #Grab wall points to test points in domain.
    print("Generating wall boundary path...")
    wall_path, rlmt2, zlmt2 = make_path(rlmt,zlmt)

    R2D, Z2D = np.meshgrid(Rg, Zg, indexing="ij")
    sep_atol, sep_maxits = 1e-5, 1000 #Store default settings from hypnotoad examples.
    opoints, xpoints = fc(R2D, Z2D, psi, sep_atol, sep_maxits)
    #Remove points outside the wall. Not enough to remove all non-important points it turns out.
    #TODO: Can try removing points within certain flux surface outside of LCFS?
    # |--->  Probably use psi spline from R,Z to drop points a bit outside LCFS.
    for points in opoints[:]:
        if not wall_path.contains_point((points[0], points[1])):
            opoints.remove(points)
    for points in xpoints[:]:
        if not wall_path.contains_point((points[0], points[1])):
            xpoints.remove(points)

    #For BSTING just set to nx + 1. Used for mpi communication for FCI, so don't separate anything.
    ixseps1 = ixseps2 = nrp + 1

    #Trace field lines in both directions.
    print("Tracing field line in forward direction...")
    Rvals_pos, Zvals_pos = trace_until_wall(R1, Z1, phi_val, dphi, field_line_rhs,
                                         wall_path, direction=sign_b0)
    #Rvals_pos_lin, Zvals_pos_lin = trace_until_wall(R1, Z1, zeta_arr, field_line_rhs_lin, wall_pts, direction=sign_b0)
    print("Tracing field line in backward direction...")
    Rvals_neg, Zvals_neg = trace_until_wall(R1, Z1, phi_val, dphi, field_line_rhs,
                                         wall_path, direction=-sign_b0)

    #Trace all grid points once back and forth.
    gridPts = np.column_stack((RR.ravel(), ZZ.ravel()))
    fwdPts = np.zeros_like(gridPts)
    bwdPts = np.zeros_like(gridPts)
    print("Generating forward and backward points on whole grid...")
    for idx, (r0, z0) in enumerate(gridPts):
        sln = trace_field_line(r0, z0, phi_val,
                            sign_b0*dphi, field_line_rhs)
        fwdPts[idx, 0], fwdPts[idx, 1] = sln.y[0, -1], sln.y[1, -1]
        sln = trace_field_line(r0, z0, phi_val,
                            -sign_b0*dphi, field_line_rhs)
        bwdPts[idx, 0], bwdPts[idx, 1] = sln.y[0, -1], sln.y[1, -1]

    #Generate ghost point mask and BC information.
    #TODO: Add parallel BCs as well based on traced points from above.
    print("Generating ghost cells and boundary conditions...")
    ghosts = handle_bounds(RR, ZZ, wall_path, show=True)

    #Convert mapping points back to 2d arrays.
    Rfwd = fwdPts[:,0].reshape(nrp, nzp)
    Zfwd = fwdPts[:,1].reshape(nrp, nzp)
    Rbwd = bwdPts[:,0].reshape(nrp, nzp)
    Zbwd = bwdPts[:,1].reshape(nrp, nzp)

    #Get scatter point data to plot and test grid point following in general.
    step = 100
    indices = np.arange(0, gridPts.shape[0], step)
    gridPtsFinal = gridPts[indices]
    fwdPtsFinal = fwdPts[indices]
    bwdPtsFinal = bwdPts[indices]
    #Remove points outside the wall for all three arrays at once.
    keptPts = [(g, f, b) for (g, f, b) in zip(gridPtsFinal, fwdPtsFinal, bwdPtsFinal)
            if wall_path.contains_point((g[0], g[1]))]
    gridPtsFinal, fwdPtsFinal, bwdPtsFinal = map(list, zip(*keptPts))

    #Generate metric and maps and so on to write out for BSTING.
    print("Generating metric and map data for output file...")
    psi = psi_func(R,Z)
    attributes = {
        "psi": make_3d(psi, nphi)
    }
    #Need to do this all in 3D now, didn't need the complication before.
    R3, phi3, Z3 = np.meshgrid(R, phi_arr, Z, indexing='ij')
    fwd_xtp, fwd_ztp = findIndex(Rfwd, Zfwd, R[0], Z[0], dR, dZ, wall_path, show=True)
    bwd_xtp, bwd_ztp = findIndex(Rbwd, Zbwd, R[0], Z[0], dR, dZ, wall_path, show=True)

    maps = {
        "R": R3,
        "Z": Z3,
        "MXG": rpad,
        "MYG": phipad,
        "forward_R": make_3d(Rfwd, nphi),
        "forward_Z": make_3d(Zfwd, nphi),
        "backward_R": make_3d(Rbwd, nphi),
        "backward_Z": make_3d(Zbwd, nphi),
        "forward_xt_prime":  make_3d(fwd_xtp, nphi),
        "forward_zt_prime":  make_3d(fwd_ztp, nphi),
        "backward_xt_prime": make_3d(bwd_xtp, nphi),
        "backward_zt_prime": make_3d(bwd_ztp, nphi)
    }

    #Store metric info.
    #And work in 3D for stellarator/mirror cases.
    ctr_metric = get_metric(R3[:,0,:], Z3[:,0,:], nrp, nzp, dr, dz)
    fwd_metric = get_metric(Rfwd, Zfwd, nrp, nzp, dr, dz)
    bwd_metric = get_metric(Rbwd, Zbwd, nrp, nzp, dr, dz)
    Bmag3D = make_3d(Bmag, nphi)
    Bphi3D = make_3d(Bphi, nphi)
    parFac = Bmag3D/Bphi3D
    metric = {
        "Rxy":  R3,
        "Bxy":  Bmag3D,
        "dx":   make_3d(ctr_metric["dx"], nphi),
        "dy":   np.full_like(R3, dphi),
        "dz":   make_3d(ctr_metric["dz"], nphi),
        "g11":  make_3d(ctr_metric["gxx"], nphi),
        "g_11": make_3d(ctr_metric["gxx"], nphi),
        "g13":  make_3d(ctr_metric["gxz"], nphi),
        "g_13": make_3d(ctr_metric["g_xz"], nphi),
        "g22":  1/R3**2,
        "g_22": R3**2,
        "g33":  make_3d(ctr_metric["gzz"], nphi),
        "g_33": make_3d(ctr_metric["g_zz"], nphi),
        "forward_dx":    make_3d(fwd_metric["dx"], nphi),
        "forward_dy":    np.full_like(R3, dphi),
        "forward_dz":    make_3d(fwd_metric["dz"], nphi),
        "forward_g11":   make_3d(fwd_metric["gxx"], nphi),
        "forward_g_11":  make_3d(fwd_metric["gxx"], nphi),
        "forward_g13":   make_3d(fwd_metric["gxz"], nphi),
        "forward_g_13":  make_3d(fwd_metric["g_xz"], nphi),
        "forward_g22":   1/R3**2,
        "forward_g_22":  R3**2,
        "forward_g33":   make_3d(fwd_metric["gzz"], nphi),
        "forward_g_33":  make_3d(fwd_metric["g_zz"], nphi),
        "backward_dx":   make_3d(bwd_metric["dx"], nphi),
        "backward_dy":   np.full_like(R3, dphi),
        "backward_dz":   make_3d(bwd_metric["dz"], nphi),
        "backward_g11":  make_3d(bwd_metric["gxx"], nphi),
        "backward_g_11": make_3d(bwd_metric["gxx"], nphi),
        "backward_g13":  make_3d(bwd_metric["gxz"], nphi),
        "backward_g_13": make_3d(bwd_metric["g_xz"], nphi),
        "backward_g22":  1/R3**2,
        "backward_g_22": R3**2,
        "backward_g33":  make_3d(bwd_metric["gzz"], nphi),
        "backward_g_33": make_3d(bwd_metric["g_zz"], nphi)
    }
    #Update gyy's with field line following factors for parallel operators, since this is handled along field lines.
    metric.update({k: v/parFac**2 for k,v in metric.items() if k in ("g22", "forward_g22", "backward_g22")})
    metric.update({k: v*parFac**2 for k,v in metric.items() if k in ("g_22", "forward_g_22", "backward_g_22")})

    #Generate finite volume operators following zoidberg stencil code.
    #Jacobian assuming gxz = gzx.
    jac = np.sqrt(ctr_metric["g_xx"]*ctr_metric["g_zz"]-ctr_metric["g_xz"]**2)
    dagp_fv_volume = calc_vol(RR, jac, dr, dz)
    faces = face_metrics_from_centers(ctr_metric["gxx"], ctr_metric["gxz"], ctr_metric["gzz"],
                                      ctr_metric["g_xx"], ctr_metric["g_xz"], ctr_metric["g_zz"],)
    fac_XX, fac_XZ, fac_ZZ, fac_ZX = fac_per_area_from_faces(faces, dr, dz)
    # If you actually want the *integrated* coefficients (face-length included):
    Lz_x, Lx_z = face_lengths_from_faces(faces, dr, dz)
    Rx_face, Rz_face = R_faces_from_centers(RR, periodic_z=False, use_abs=False)
    dagp_fv_XX = pad_to_full(fac_XX * Lz_x * Rx_face, nrp, nzp)
    dagp_fv_XZ = pad_to_full(fac_XZ * Lz_x * Rx_face, nrp, nzp)
    dagp_fv_ZZ = pad_to_full(fac_ZZ * Lx_z * Rz_face, nrp, nzp, dim='z')
    dagp_fv_ZX = pad_to_full(fac_ZX * Lx_z * Rz_face, nrp, nzp, dim='z')

    dagp_vars = {
        "dagp_fv_XX": make_3d(dagp_fv_XX, nphi),
        "dagp_fv_XZ": make_3d(dagp_fv_XZ, nphi),
        "dagp_fv_ZX": make_3d(dagp_fv_ZX, nphi),
        "dagp_fv_ZZ": make_3d(dagp_fv_ZZ, nphi),
        "dagp_fv_volume": make_3d(dagp_fv_volume, nphi)
    }
    maps.update(dagp_vars)

    #Calculate interpolation weights directly in python rather than BSTING.
    #TODO: BSTING would need to read in weights still...BSTING can do bilinear or cubic hermite spline itself at the moment.
    #weights = calc_weights(maps)

    #Write output to data file.
    gridfile = gfilename + ".fci.nc"
    print("Writing to " + str(gridfile) + "...")
    with bdata.DataFile(gridfile, write=True, create=True, format="NETCDF4") as f:
        f.write_file_attribute("title", "BOUT++ grid file")
        f.write_file_attribute("software_name", "zoidberg")
        f.write_file_attribute("software_version", __version__)
        grid_id = str(uuid.uuid1())
        f.write_file_attribute("id", grid_id)      #Conventional name
        f.write_file_attribute("grid_id", grid_id) #BOUT++ specific name

        f.write("nx", nrp)
        f.write("ny", nphi)
        f.write("nz", nzp)

        f.write("dx", metric["dx"])
        f.write("dy", metric["dy"])
        f.write("dz", metric["dz"])

        f.write("ixseps1", ixseps1)
        f.write("ixseps2", ixseps2)

        for key, value in metric.items():
            f.write(key, value)

        f.write("B", Bmag3D)

        f.write("pressure", make_3d(pres, nphi))

        for key, value in attributes.items():
            f.write(key, value)

        for key, value in maps.items():
            f.write(key, value)

    #Build a Rectangle at (rmin, zmin) with width/height to show ghost point border.
    ghost = Rectangle((rmin, zmin),
                     rmax - rmin,
                     zmax - zmin,
                     fill=False,      #No fill, just border
                     linestyle='--',
                     edgecolor='k',
                     label = 'ghost',
                     clip_on = False, #Draw at top of bounds.
                     linewidth=1.5)

    plotting = True
    checkPts = True
    if (plotting):
        plot(R, Z, psi, ghost, rbdy, zbdy, rlmt2, zlmt2,
            sign_b0, Rvals_pos, Zvals_pos, Rvals_neg, Zvals_neg,
            opoints, xpoints, checkPts, gridPtsFinal, fwdPtsFinal, bwdPtsFinal)

if __name__ == "__main__":
    main(sys.argv[1:])

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