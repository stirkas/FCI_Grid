#!/usr/bin/env python

import sys
import uuid

from matplotlib import path as path
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import integrate,interpolate
import numpy as np

from boututils import datafile as bdata
from hypnotoad import __version__
from hypnotoad.geqdsk._geqdsk import read as gq_read
from hypnotoad.utils.critical import find_critical as fc

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

def trace_until_wall(R0, Z0, zeta_arr, field_line_rhs, wall_path, direction=1):
    """
    Trace a field line starting at (R0,Z0) along the toroidal angles in zeta_arr,
    in the given direction (+1 or -1), stopping as soon as the point exits wall_path.
    """

    Rvals = [R0]
    Zvals = [Z0]

    #Loop over each segment in zeta_arr.
    for i in range(len(zeta_arr) - 1):
        z0 = direction * zeta_arr[i]
        z1 = direction * zeta_arr[i + 1]

        sol = trace_field_line(
            Rvals[-1], Zvals[-1],
            z0, z1,
            field_line_rhs
        )
        Rn, Zn = sol.y[0, -1], sol.y[1, -1]

        #Stop if that new point lies outside the wall.
        if not wall_path.contains_point((Rn, Zn)):
            break

        Rvals.append(Rn)
        Zvals.append(Zn)

    return Rvals, Zvals

def trace_field_line(R0, Z0, zeta_init, zeta_target, field_line_rhs):
    """
    Trace a field line from starting position
    """
    sol = integrate.solve_ivp(
        field_line_rhs,
        (zeta_init, zeta_target),
        [R0, Z0],
        method="RK45",
        #method='DOP853',
        rtol=1e-10,
        atol=1e-12,
        dense_output=True #What does this do?
    )

    #Match what zoidberg does as close as possible.
    #sol = integrate.solve_ivp(
    #    field_line_rhs,
    #    (0, zeta_target),
    #    [R0, Z0],
    #    method='LSODA',   # Match odeint default
    #    rtol=1.49012e-8,  # Match odeint default
    #    atol=1.49012e-11, # Match odeint default
    #    dense_output=False
    #)
    
    return sol

def getCoordinate(R, Z, xind, zind, dx=0, dz=0):
    # Get arrays of indices
    nx = np.shape(R)[0]
    nz = np.shape(Z)[0]
    xinds = np.arange(nx * 3)
    zinds = np.arange(nz * 3)
        
    # Repeat the data in z, to approximate periodicity
    R_ext = np.concatenate((R, R, R), axis=0)
    Z_ext = np.concatenate((Z, Z, Z), axis=0)

    _spl_r = interpolate.RectBivariateSpline(xinds, zinds, R_ext)
    _spl_z = interpolate.RectBivariateSpline(xinds, zinds, Z_ext)

    nx, nz = R.shape
    if (np.amin(xind) < 0) or (np.amax(xind) > nx - 1):
        raise ValueError("x index out of range")
    
    # Periodic in y
    zind = np.remainder(zind, nz)
    R = _spl_r(xind, zind + nz, dx=dx, dy=dz, grid=False)
    Z = _spl_z(xind, zind + nz, dx=dx, dy=dz, grid=False)

    return R, Z

def metric(R, Z, nx, nz):
        #Get arrays of indices.
        xind, zind = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")

        # Calculate the gradient along each coordinate.
        dolddnew = np.array(
            [getCoordinate(R, Z, xind, zind, dx=a, dz=b) for a, b in ((1, 0), (0, 1))]
        )
        # dims: 0 : dx or dz?
        #       1 : R or z?
        #       2 : spatial: r
        #       3 : spatial: \theta
        ddist = np.sqrt(np.sum(dolddnew**2, axis=1))  # sum R + z
        nx, nz = ddist.shape[1:]
        ddist[0] = 1 / nx
        ddist[1] = 1 / nz
        dolddnew /= ddist[:, None, ...]

        # g_ij = J_ki J_kj
        # (2.5.27) from D'Haeseleer 1991
        # Note: our J is transposed
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
        # Jacobian from BOUT++
        JB = R * (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
        return {
            "dx": ddist[0],
            "dz": ddist[1],  # Grid spacing
            "gxx": ginv[..., 0, 0],
            "g_xx": g[..., 0, 0],
            "gxz": ginv[..., 0, 1],
            "g_xz": g[..., 0, 1],
            "gzz": ginv[..., 1, 1],
            "g_zz": g[..., 1, 1],
            # "J": JB,
        }

def plot(R, Z, psi, ghost, rbdy, zbdy, rlmt, zlmt, sign_b0, Rvals_pos, Zvals_pos, Rvals_neg, Zvals_neg, opoints, xpoints):
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
    ax.plot(opoints[0][0], opoints[0][1], 'o', label='O',
            markerfacecolor='none', markeredgecolor='lime')
    ax.plot(xpoints[0][0], xpoints[0][1], 'x', color='lime', label='X')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
            ncol=len(labels), fancybox=True, shadow=True)
    ax.set_xlabel('R')
    ax.set_ylabel('Z')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    #fig.savefig('flux_surf.png', dpi=600)
    #fig.savefig('flux_surf.pdf')

    ##Can look for high threshold gradient regions (>95% of array) to see where splines may be problematic
    ##and linear interp might be better.
    #dpsi_dR, dpsi_dZ = np.gradient(psi, R, Z, edge_order=2)
    #grad_mag = Bp #np.sqrt(dpsi_dR**2 + dpsi_dZ**2) #Bp
    #thresh = 95
    #sharp_mask = grad_mag >= np.percentile(grad_mag, thresh)
    #RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    #fig, ax = plt.subplots(figsize=(6,8))
    #cf = ax.contourf(RR, ZZ, grad_mag, levels=50, cmap='viridis')
    #ax.contour(RR, ZZ, sharp_mask, levels=[0.5], colors='white', linewidths=2)
    #plt.colorbar(cf, ax=ax, label='|∇ψ|')
    #ax.set_xlabel('R'); ax.set_ylabel('Z')
    #plt.title(f"Regions with |∇ψ| ≥ {thresh:.3g}%")
    #plt.show()

def main(args):
    #Read eqdsk file.
    gfile_dir = "/home/tirkas1/Workspace/TokData/DIIID/"
    gfile1 = "g162940.02944_670" #Old ql one.
    gfile2 = "g163241.03500" #Old DIIID one.
    gfile3 = "g172208.03000" #Ben's test cases.
    gfile4 = "g174791.03000" #Ben's test cases.
    gfile5 = "g176413.03000" #Ben's test cases.
    gfile6 = "g176312.03000" #Ben's test cases.
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
    rpad, phipad, zpad = 2,1,0 #0 #Padding/ghost cells in R. No padding in Z, and 1 in Y according to zoidberg, why?
    grid_res = 64
    nr, nphi, nz = grid_res, 1, grid_res
    nrp, nzp = nr + 2*rpad, nz + 2*zpad
    phi_val = 0 #Take phi=0 to be reference point.
    R, Z = np.linspace(rmin, rmax, nr), np.linspace(zmin, zmax, nz)
    dR, dZ = R[1]-R[0], Z[1]-Z[0]
    rp_min, rp_max = rmin - rpad*dR, rmax + rpad*dR
    zp_min, zp_max = zmin - zpad*dZ, zmax + zpad*dZ
    ghosts_lo_R, ghosts_hi_R = R[0] - dR*np.arange(rpad, 0, -1), R[-1] + dR*np.arange(1, rpad + 1)
    ghosts_lo_Z, ghosts_hi_Z = Z[0] - dZ*np.arange(zpad, 0, -1), Z[-1] + dZ*np.arange(1, zpad + 1)
    #Replace R and Z with padded arrays.
    R = np.concatenate((ghosts_lo_R, R, ghosts_hi_R)) 
    Z = np.concatenate((ghosts_lo_Z, Z, ghosts_hi_Z)) 

    #Calculate field components following COCOS.
    #TODO: Make R and Z 2D so don't have to put [:, None] everywhere?
    Bp_R  =  sign_ip*psi_func(R, Z, dy=1)/R[:, None]
    Bp_Z  = -sign_ip*psi_func(R, Z, dx=1)/R[:, None]
    Bp    = np.sqrt(Bp_R**2 + Bp_Z**2)
    Bphi  = f_spl(psi_func(R, Z))/R[:, None]
    Bmag = np.sqrt(Bp_R**2 + Bp_Z**2 + Bphi**2)
    pres = p_spl(psi_func(R,Z))

    #Copmute derivatives along phi NOT B_phi. So need to pass field line direction when following below.
    #R factor comes from cylindrical geometry.
    dRdphi = R[:,None] * Bp_R / Bphi
    dZdphi = R[:,None] * Bp_Z / Bphi

    #Set up interpolation
    print("Setting up interpolation...")
    #Cubic splines seem to work well but can try linear if regions of high gradients exist and splines get messy.
    field_line_rhs = setup_field_line_interpolation(R, Z, dRdphi, dZdphi)
    #field_line_rhs_lin = setup_field_line_interpolation(R, Z, dRphi, dZphi, linear=1)

    #Choose a starting point.
    offset = 0.005
    sep_idx = np.argmax(rbdy)
    #R1, Z1 = R0, Z0               #Magnetic axis
    #R1, Z1 = R[3*nr//4], Z[nz//2] #Core point.
    #R1, Z1 = R[7*nr//8], Z[nz//2] #Outer point.
    #R1, Z1 = rbdy[sep_idx], zbdy[sep_idx]          #Separatrix
    R1, Z1 = rbdy[sep_idx] + offset, zbdy[sep_idx] #Minor offset from separatrix.
    #R1, Z1 = np.min(R), Z[nz//2]
    q1 = q_spl(psi_func(R1,Z1)[0,0]) #Getting single point so access output as [0,0].
    #Create toroidal angle array in radians.
    num_zeta = 180
    zeta_arr = np.linspace(0, np.pi, num_zeta + 1)*q1 #q extends to all poloidal angles.
    if (R1 == rbdy[sep_idx]): #Need more points at separatrix. Maybe pass angular resolution and do a while loop for points instead.
        zeta_arr *= 1.5
    #Grab wall points to test points in domain.
    wall_pts = np.column_stack([rlmt,zlmt])
    wall_path = path.Path(wall_pts, closed=True)

    R2D, Z2D = np.meshgrid(Rg, Zg, indexing="ij")
    sep_atol, sep_maxits = 1e-5, 1000 #Store default settings from hypnotoad examples.
    opoints, xpoints = fc(R2D, Z2D, psi, sep_atol, sep_maxits)
    #Remove points outside the wall. Not enough to remove all non-important points it turns out.
    #TODO: Can try removing points within certain flux surface outside of LCFS?
    for points in opoints[:]:
        if not wall_path.contains_point((points[0], points[1])):
            opoints.remove(points)
    for points in xpoints[:]:
        if not wall_path.contains_point((points[0], points[1])):
            xpoints.remove(points)
    #TODO: Figure out separatrix scenario.
    ixseps = nr + 1
    #Use ixseps1 = xpoints[0](R), ixseps2 = nx + 1?

    #Trace field lines in both directions.
    #TODO: Trace whole grid forward and backward once.
    print("Tracing field line in forward direction...")
    Rvals_pos, Zvals_pos = trace_until_wall(R1, Z1, zeta_arr, field_line_rhs,
                                         wall_path, direction=sign_b0)
    #Rvals_pos_lin, Zvals_pos_lin = trace_until_wall(R1, Z1, zeta_arr, field_line_rhs_lin, wall_pts, direction=sign_b0)
    print("Tracing field line in backward direction...")
    Rvals_neg, Zvals_neg = trace_until_wall(R1, Z1, zeta_arr, field_line_rhs,
                                         wall_path, direction=-sign_b0)

    #Generate metric and maps and so on to write out for BSTING.
    #TODO: Generate basic weights and eventually anti-symmetric weights.
    attributes = {
        "psi": psi[:, np.newaxis, :]
    }
    #Need to do this all in 3D now, didn't need the complication before.
    #TODO Map back and forward R,Z. And fix xt/zt primes? Shouldn't all be -1, test old gfile.
    R3, phi3, Z3 = np.meshgrid(R, phi_val, Z, indexing='ij')
    maps = {
        "R": R3,
        "Z": Z3,
        "MXG": rpad,
        "MYG": phipad,
        "forward_R": R3,
        "forward_Z": Z3,
        "backward_R": R3,
        "backward_Z": Z3,
        "forward_xt_prime": -1*np.ones_like(R3),
        "forward_zt_prime": -1*np.ones_like(R3),
        "backward_xt_prime": -1*np.ones_like(R3),
        "backward_zt_prime": -1*np.ones_like(R3)

    }
    #Store metric info. Note metric is expected 2D (R,phi) not 3D.
    #Note: No g12 or g23?
    #TODO: Add forward and backward metric.
    R2 = R3[:,:,0]
    metric = {
        "dx": np.full_like(R2, R[1]-R[0]),
        "dy": np.full_like(R2, zeta_arr[1]-zeta_arr[0]),
        "dz": Z[1]-Z[0],
        "g11": np.ones_like(R2),
        "g_11": np.ones_like(R2),
        "g13": np.zeros_like(R2),
        "g_13": np.zeros_like(R2),
        "g22": 1/(R2**2),
        "g_22": R2**2,
        "g33": np.ones_like(R2),
        "g_33": np.ones_like(R2),
        "Rxy": R2,
        "Bxy": Bmag[:,np.newaxis,0]
    }

    #Write output to data file.
    gridfile = gfilename + ".fci.nc"
    with bdata.DataFile(gridfile, write=True, create=True, format="NETCDF4") as f:
        f.write_file_attribute("title", "BOUT++ grid file")
        f.write_file_attribute("software_name", "zoidberg")
        f.write_file_attribute("software_version", __version__)
        grid_id = str(uuid.uuid1())
        f.write_file_attribute("id", grid_id)  # conventional name
        f.write_file_attribute("grid_id", grid_id)  # BOUT++ specific name

        f.write("nx", nr)
        f.write("ny", nphi)
        f.write("nz", nz)

        f.write("dx", metric["dx"])
        f.write("dy", metric["dy"])
        f.write("dz", metric["dz"])

        f.write("ixseps1", ixseps)
        f.write("ixseps2", ixseps)

        # Metric tensor
        for key, value in metric.items():
            f.write(key, value)

        # Magnetic field
        f.write("B", Bmag[:,np.newaxis,:])

        # Pressure
        f.write("pressure", pres[:,np.newaxis,:])

        # Attributes
        for key, value in attributes.items():
            f.write(key, value)

        # Maps - write everything to file
        for key, value in maps.items():
            f.write(key, value)

    #Generate data for plotting on full grid with ghost points.
    psi = psi_func(R,Z)
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
    if (plotting):
        plot(R, Z, psi, ghost, rbdy, zbdy, rlmt, zlmt,
            sign_b0, Rvals_pos, Zvals_pos, Rvals_neg, Zvals_neg,
            opoints, xpoints)

if __name__ == "__main__":
    main(sys.argv[1:])