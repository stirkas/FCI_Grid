#!/usr/bin/env python

import sys
import uuid

from matplotlib import path as path
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.patches import Rectangle
from scipy import integrate,interpolate
from scipy.spatial import cKDTree as KDTree
import numpy as np

from boututils import datafile as bdata
from hypnotoad import __version__
from hypnotoad.geqdsk._geqdsk import read as gq_read
from hypnotoad.utils.critical import find_critical as fc

import stencil_dagp_fv
import poloidal_grid2

def make_3d(arr_2d: np.ndarray, ny: int) -> np.ndarray:
    """
    Repeat a 2D array (R, Z) along the middle axis to shape (R, ny, Z).
    If writable=False, returns a broadcasted (read-only) view.
    """

    return np.repeat(arr_2d[:, np.newaxis, :], ny, axis=1)

def edges_from_centers(c):
    """Face edges from center coords via linear extrapolation at both ends."""
    c = np.asarray(c, dtype=float)
    if c.size < 2:
        raise ValueError("Need at least 2 centers to form edges.")
    e = np.empty(c.size + 1, dtype=float)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0]    = c[0] - 0.5 * (c[1] - c[0])
    e[-1]   = c[-1] + 0.5 * (c[-1] - c[-2])
    return e

def Ar_from_centers(R_centers, Z_centers):
    """
    Compute A_R = ∬ R dA for each rectangular cell, returned at cell centers.
    R_centers, Z_centers: 1D arrays of cell-center coordinates (in meters).
    Returns Ar with shape (len(R_centers), len(Z_centers)).
    """
    Re = edges_from_centers(R_centers)  # shape NR+1
    Ze = edges_from_centers(Z_centers)  # shape NZ+1

    Rm, Rp = Re[:-1], Re[1:]            # left/right faces per cell
    Zm, Zp = Ze[:-1], Ze[1:]            # bottom/top faces per cell

    # Ar[i,j] = 0.5*(R_{i+1/2,+}^2 - R_{i+1/2,-}^2) * (Z_{j+1/2,+} - Z_{j+1/2,-})
    Ar = 0.5 * (Rp**2 - Rm**2)[:, None] * (Zp - Zm)[None, :]
    return Ar

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

#TODO: Clean up this functionality? Recombine with getCoordinate.
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
def get_metric(R, Z, nx, nz, nxpad, nzpad):
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
    ddist[0] = 1 / (nx - 2*nxpad - 1)
    ddist[1] = 1 / (nz - 2*nzpad - 1)
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

def findIndex(R, Z, Rmin, Zmin, dR, dZ, rbdy, zbdy, show=False):
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

    xind = (R - Rmin) / dR
    zind = (Z - Zmin) / dZ

    # Note: These indices may be outside the domain,
    # but this is handled in BOUT++, and useful for periodic
    # domains.

    #Mask out points around separatrix.
    sep_pts = np.column_stack([rbdy,zbdy])
    sep_path = path.Path(sep_pts, closed=True)
    pts = np.column_stack([R.ravel(), Z.ravel()])
    inside = sep_path.contains_points(pts)
    inside = inside.reshape((R.shape[0], R.shape[1]))

    nrp = len(R)
    xind[~inside] = nrp
    #zind[~inside] = -1 #Only need to do x?

    if (show):
        fig, ax = plt.subplots(figsize=(6,6))

        # 1) draw the wall Path itself
        patch = PathPatch(sep_path, facecolor='none', edgecolor='k', lw=2)
        ax.add_patch(patch)

        # 2) scatter the points inside (green) and outside (red)
        ax.scatter(
            R[ inside], Z[ inside],
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
    gfile_dir = "/home/tirkas1/Workspace/TokData/DIIID/"
    gfile1 = "g162940.02944_670" #Old ql one.
    gfile2 = "g163241.03500" #Old DIIID one.
    #Ben's test cases for varying Ip and B0 directions.
    gfile3 = "g172208.03000"
    gfile4 = "g174791.03000"
    gfile5 = "g176413.03000"
    gfile6 = "g176312.03000"
    gfilename = gfile1
    gfilepath = gfile_dir + gfilename
    print("Reading EQDSK file...")
    with open(gfilepath, "r", encoding="utf-8") as file:
        gfile = gq_read(file)
    print("Finished reading EQDSK file...")

    nrg, nzg = gfile["nx"], gfile["ny"]
    rmin, rmax = gfile["rleft"], gfile["rleft"] + gfile["rdim"]
    zmin, zmax = gfile["zmid"] - 0.5 * gfile["zdim"], gfile["zmid"] + 0.5 * gfile["zdim"]
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
    #TODO: Why doesnt ext=0 work ok when tracing field?
    f_spl = interpolate.InterpolatedUnivariateSpline(psi1D, fpol, ext=3) #ext=3 uses boundary values outside range.
    q_spl = interpolate.InterpolatedUnivariateSpline(psi1D, qpsi, ext=3) #ext=0 uses extrapolation as with RectBivSpline on 2D but doesnt work in the integrator.
    p_spl = interpolate.InterpolatedUnivariateSpline(psi1D, prsr, ext=3)

    #Generate simulation grid (lower res + guard cells)
    #TODO: Can pad in z as well in full FCI case, but wont matter if gfile large outside mask.
    rpad, phipad, zpad = 2,1,0 #0 #Padding/ghost cells in R. No padding in Z, and 1 in Y according to zoidberg, why?
    grid_res, phi_res = 64, 16
    nr, nphi, nz = grid_res, 1, grid_res
    phi_val = 0 #Take phi=0 to be starting point.
    dphi = 2*np.pi/phi_res
    phi_arr = [dphi]
    copy3d = False
    if copy3d == True:
        phi_arr = np.linspace(phi_val, 2*np.pi, nphi, endpoint=False) #Periodic so dont go all the way.
        dphi = phi_arr[1]-phi_arr[0]
        nphi = phi_res
    #nrp, nzp = nr + 2*rpad, nz + 2*zpad
    ##Method 1: Concatenate ghost points onto initial grid so 64x64 goes to 68x64.
    ##dR, dZ = R[1]-R[0], Z[1]-Z[0]
    ##rp_min, rp_max = rmin - rpad*dR, rmax + rpad*dR
    ##zp_min, zp_max = zmin - zpad*dZ, zmax + zpad*dZ
    ##ghosts_lo_R, ghosts_hi_R = R[0] - dR*np.arange(rpad, 0, -1), R[-1] + dR*np.arange(1, rpad + 1)
    ##ghosts_lo_Z, ghosts_hi_Z = Z[0] - dZ*np.arange(zpad, 0, -1), Z[-1] + dZ*np.arange(1, zpad + 1)
    ###Replace R and Z with padded arrays.
    ##R = np.concatenate((ghosts_lo_R, R, ghosts_hi_R))
    ##Z = np.concatenate((ghosts_lo_Z, Z, ghosts_hi_Z))
    # --- 1) Strip ghosts -> 60 x 64 ---
    #R_int = R[rpad:-rpad]                         # length 60
    #Z_int = Z if zpad == 0 else Z[zpad:-zpad]     # still 64 here
    #psi_int = psi[rpad:-rpad, :] if zpad == 0 else psi[rpad:-rpad, zpad:-zpad]
    # psi_int.shape == (60, 64)

    #Method 2: Replace original points with ghost points so grid 64x64 for Structured class which requires nx == nz for now.
    nrp, nzp = nr, nz
    R, Z = np.linspace(rmin, rmax, nr - 2*rpad), np.linspace(zmin, zmax, nz - 2*zpad)
    dR = R[1] - R[0]
    ghosts_lo_R = R[0]  - dR * np.arange(rpad, 0, -1)
    ghosts_hi_R = R[-1] + dR * np.arange(1, rpad+1)
    dZ = Z[1] - Z[0]
    ghosts_lo_Z = Z[0]  - dZ * np.arange(zpad, 0, -1)
    ghosts_hi_Z = Z[-1] + dZ * np.arange(1, zpad+1)
    R = np.concatenate((ghosts_lo_R, R, ghosts_hi_R))
    Z = np.concatenate((ghosts_lo_Z, Z, ghosts_hi_Z))
    RR, ZZ = np.meshgrid(R,Z,indexing='ij')

    #Calculate field components following COCOS convention.
    Bp_R =  sign_ip*psi_func(R, Z, dy=1)/RR
    Bp_Z = -sign_ip*psi_func(R, Z, dx=1)/RR
    Bp   = np.sqrt(Bp_R**2 + Bp_Z**2)
    Bphi = f_spl(psi_func(R, Z))/RR
    Bmag = np.sqrt(Bp_R**2 + Bp_Z**2 + Bphi**2)
    pres = p_spl(psi_func(R,Z))

    #Copmute derivatives along phi NOT B_phi. So need to pass field line direction when following below.
    #R factor comes from cylindrical geometry.
    dRdphi = RR * Bp_R / Bphi
    dZdphi = RR * Bp_Z / Bphi

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
    wall_pts = np.column_stack([rlmt,zlmt])
    wall_path = path.Path(wall_pts, closed=True)

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
    #if len(xpoints) >= 1:
    #    ixseps1 = np.argmin(np.abs(R - xpoints[0][0]))
    #if len(xpoints) >= 2:
    #    ixseps2 = np.argmin(np.abs(R - xpoints[1][0]))

    #Trace field lines in both directions.
    print("Tracing field line in forward direction...")
    Rvals_pos, Zvals_pos = trace_until_wall(R1, Z1, phi_val, dphi, field_line_rhs,
                                         wall_path, direction=sign_b0)
    #Rvals_pos_lin, Zvals_pos_lin = trace_until_wall(R1, Z1, zeta_arr, field_line_rhs_lin, wall_pts, direction=sign_b0)
    print("Tracing field line in backward direction...")
    Rvals_neg, Zvals_neg = trace_until_wall(R1, Z1, phi_val, dphi, field_line_rhs,
                                         wall_path, direction=-sign_b0)

    #Trace all grid points once back and forth.
    #Use low-res toroidal distance.
    #TODO: Need to apply a mask for BSTING? Ben says to use the wall surface rather than the separatrix.
    gridPts = np.column_stack((RR.ravel(), ZZ.ravel()))
    fwdPts = np.zeros_like(gridPts)
    bwdPts = np.zeros_like(gridPts)
    print("Generating forward and backward points on whole grid...")
    for idx, (r0, z0) in enumerate(gridPts):
        sol = trace_field_line(r0, z0, phi_val,
                            sign_b0*dphi, field_line_rhs)
        fwdPts[idx, 0], fwdPts[idx, 1] = sol.y[0, -1], sol.y[1, -1]
        sol = trace_field_line(r0, z0, phi_val,
                            -sign_b0*dphi, field_line_rhs)
        bwdPts[idx, 0], bwdPts[idx, 1] = sol.y[0, -1], sol.y[1, -1]

    #Convert mapping points back to 2d arrays.
    Rfwd = fwdPts[:,0].reshape(nrp, nzp)
    Zfwd = fwdPts[:,1].reshape(nrp, nzp)
    Rbwd = bwdPts[:,0].reshape(nrp, nzp)
    Zbwd = bwdPts[:,1].reshape(nrp, nzp)

    #Plot points scattered to test grid point following in general.
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
    #TODO: Generate interpolation weights per zoidberg? Can run BSTING without it first and add later.
    psi = psi_func(R,Z)
    attributes = {
        "psi": make_3d(psi, nphi)
    }
    #Need to do this all in 3D now, didn't need the complication before.
    R3, phi3, Z3 = np.meshgrid(R, phi_arr, Z, indexing='ij')
    fwd_xtp, fwd_ztp = findIndex(Rfwd, Zfwd, R[0], Z[0], R[1]-R[0], Z[1]-Z[0], rbdy, zbdy)
    bwd_xtp, bwd_ztp = findIndex(Rbwd, Zbwd, R[0], Z[0], R[1]-R[0], Z[1]-Z[0], rbdy, zbdy)

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

    #Store metric info. #TODO: Tokamak metric 3D for now, can update zoidberg to 2D, removing phi?
    #And work in 3D for stellarator/mirror cases.
    ctr_metric = get_metric(R3[:,0,:], Z3[:,0,:], nrp, nzp, rpad, zpad)
    fwd_metric = get_metric(Rfwd, Zfwd, nrp, nzp, rpad, zpad)
    bwd_metric = get_metric(Rbwd, Zbwd, nrp, nzp, rpad, zpad)
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
    metric.update({k: v/parFac**2 for k, v in metric.items() if k in ("g22", "forward_g22", "backward_g22")})
    metric.update({k: v*parFac**2 for k, v in metric.items() if k in ("g_22", "forward_g_22", "backward_g_22")})

    #Generate divergence operators as well.
    #dagp_vars = stencil_dagp_fv.doit([poloidal_grid2.StructuredPoloidalGrid(RR,ZZ)],plot=True)
    ##TODO: Using non-periodic z and nx != nz seems to cause nans in dagp vars.
    ##Replace with best fit based on data for now.
    #for val in dagp_vars["dagp_fv_XX"]:
    #    val[0][0] = val[0][1]
    #    val[0][-1] = val[0][-2]
    #np.nan_to_num(dagp_vars["dagp_fv_XZ"], copy=False, nan=0.0)
    ##Extend from length 1 to nphi.
    #for key,val in dagp_vars.items():
    #    dagp_vars[key] = np.repeat(val, nphi, axis=1)
    #TODO: Make sure these values are correct/Get stencil code working for all cases?
    dagp_fv_XX = np.full_like(R3, np.sqrt(metric["g11"])/(1/nrp))
    dagp_fv_ZZ = np.full_like(R3, np.sqrt(metric["g33"])/(1/nzp))
    dagp_fv_XZ = np.zeros_like(R3)
    dagp_fv_ZX = np.zeros_like(R3)
    dagp_fv_volume = make_3d(Ar_from_centers(R,Z), nphi)
    dagp_vars = {
        "dagp_fv_XX": dagp_fv_XX,
        "dagp_fv_XZ": dagp_fv_XZ,
        "dagp_fv_ZX": dagp_fv_ZX,
        "dagp_fv_ZZ": dagp_fv_ZZ,
        "dagp_fv_volume": dagp_fv_volume
    }
    maps.update(dagp_vars)

    #Write output to data file.
    gridfile = gfilename + ".fci.nc"
    print("Writing to " + str(gridfile) + "...")
    with bdata.DataFile(gridfile, write=True, create=True, format="NETCDF4") as f:
        f.write_file_attribute("title", "BOUT++ grid file")
        f.write_file_attribute("software_name", "zoidberg")
        f.write_file_attribute("software_version", __version__) #TODO: Use zoidberg version eventually...?
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
        plot(R, Z, psi, ghost, rbdy, zbdy, rlmt, zlmt,
            sign_b0, Rvals_pos, Zvals_pos, Rvals_neg, Zvals_neg,
            opoints, xpoints, checkPts, gridPtsFinal, fwdPtsFinal, bwdPtsFinal)

if __name__ == "__main__":
    main(sys.argv[1:])
