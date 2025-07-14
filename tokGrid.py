#!/usr/bin/env python

import sys

import matplotlib.pyplot as plt
import matplotlib.path as path
import numpy as np
from scipy import integrate,interpolate

from hypnotoad.geqdsk._geqdsk import read as gq_read

def setup_field_line_interpolation(R_grid, Z_grid, dR_dzeta, dZ_dzeta):
    """
    Set up interpolators for field line derivatives
    """
    # Create interpolators
    dR_itp = interpolate.RegularGridInterpolator(
        (R_grid, Z_grid), 
        dR_dzeta,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )
    
    dZ_itp = interpolate.RegularGridInterpolator(
        (R_grid, Z_grid), 
        dZ_dzeta,
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
        dR_dzt = np.squeeze(dR_itp([R, Z]))
        dZ_dzt = np.squeeze(dZ_itp([R, Z]))
        
        return np.array([dR_dzt, dZ_dzt])
    
    return field_line_rhs

def trace_until_wall(R0, Z0, zeta_arr, field_line_rhs, wall_pts, direction=1):
    """
    Trace a field line starting at (R0,Z0) along the toroidal angles in zeta_arr,
    in the given direction (+1 or -1), stopping as soon as the point exits wall_path.
    """

    Rvals = [R0]
    Zvals = [Z0]
    wall_path = path.Path(wall_pts, closed=True)

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

def main(args):

    #Read eqdsk file.
    print("Reading EQDSK file...")
    with open("/home/tirkas1/Workspace/TokData/DIIID/g162940.02944_670", "r", encoding="utf-8") as file:
        gfile = gq_read(file)
    print("Finished reading EQDSK file...")

    nr = gfile["nx"]
    nz = gfile["ny"]
    R = np.linspace(gfile["rleft"], gfile["rleft"] + gfile["rdim"], nr)
    Z = np.linspace(gfile["zmid"] - 0.5 * gfile["zdim"],
                    gfile["zmid"] + 0.5 * gfile["zdim"],
                    nz)

    #TODO: Figure out how to handle sign conventions. Ben says all 4 options should be possible to find in the gfile data.

    R0    = gfile["rmagx"]
    Z0    = gfile["zmagx"]
    psi   = gfile["psi"]
    fpol  = gfile["fpol"] # f = R*B_t
    qpsi  = gfile["qpsi"] # q(psi)
    paxis = gfile["simagx"]
    pbdry = gfile["sibdry"]
    rbdy = np.array(gfile["rbdry"])
    zbdy = np.array(gfile["zbdry"])
    rlmt = np.array(gfile["rlim"])
    zlmt = np.array(gfile["zlim"])

    #Calculate field data.
    print("Calculating field data...")
    psin = (psi - paxis) / (pbdry - paxis)
    psi_func = interpolate.RectBivariateSpline(R, Z, psi)

    #Poloidal field components.
    Bp_R =  psi_func(R, Z, dy=1)/R
    Bp_Z = -psi_func(R, Z, dx=1)/R

    #Toroidal field component and q(psi).
    psi1D = np.linspace(paxis, pbdry, nr)
    f_spl = interpolate.InterpolatedUnivariateSpline(psi1D, fpol, ext=3) #ext=3 uses boundary values outside range.
    q_spl = interpolate.InterpolatedUnivariateSpline(psi1D, qpsi, ext=3)
    Bzeta = f_spl(psi_func(R, Z))/R
    Bp = np.sqrt(Bp_R**2 + Bp_Z**2)

    #Copmute derivatives dR/dzeta and dZ/dzeta.
    #R factor from cylindrical geometry.
    dRdzt = R[:,None] * Bp_R / Bzeta
    dZdzt = R[:,None] * Bp_Z / Bzeta

    #Set up interpolation
    print("Setting up interpolation...")
    field_line_rhs = setup_field_line_interpolation(R, Z, dRdzt, dZdzt)

    #Choose a starting point.
    offset = 0.005
    sep_idx = np.argmax(rbdy)
    #R1, Z1 = R0, Z0               #Magnetic axis
    #R1, Z1 = R[3*nr//4], Z[nz//2] #Core point.
    R1, Z1 = rbdy[sep_idx], zbdy[sep_idx]          #Separatrix
    #R1, Z1 = rbdy[sep_idx] + offset, zbdy[sep_idx] #Minor offset from separatrix.
    q1 = float(q_spl(psi_func(R1,Z1)[0,0])) #Getting single point so access output as [0,0].
    #Create toroidal angle array in radians.
    num_zeta = 60
    zeta_arr = np.linspace(0, np.pi, num_zeta + 1)*q1 #q extends to all poloidal angles.
    if (R1 == rbdy[sep_idx]): #Need more points at separatrix. Maybe pass angular resolution and do a while loop for points instead.
        zeta_arr *= 1.5
    #Grab wall points to test points in domain.
    wall_pts   = np.column_stack([rlmt,zlmt])

    #Trace field lines in both directions.
    print("Tracing field line in positive direction...")
    Rvals_pos, Zvals_pos = trace_until_wall(R1, Z1, zeta_arr, field_line_rhs, wall_pts, direction=1)
    print("Tracing field line in negative direction...")
    Rvals_neg, Zvals_neg = trace_until_wall(R1, Z1, zeta_arr, field_line_rhs, wall_pts, direction=-1)

    #Plot contours, LCFS, wall and field line.
    fig, ax = plt.subplots(figsize=(6, 8))
    cf = ax.contourf(R, Z, psi.T, levels=100, cmap='viridis')
    plt.colorbar(cf, ax=ax)
    ax.plot(Rvals_pos, Zvals_pos, '.', color='red',   label='Fld+')
    ax.plot(Rvals_neg, Zvals_neg, '.', color='white', label='Fld-')
    ax.plot(rbdy, zbdy, '.-', color='orange', label='LCFS')
    ax.plot(rlmt, zlmt, '-', color='black', label='Wall')
    ax.set_xlabel('R')
    ax.set_ylabel('Z')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])