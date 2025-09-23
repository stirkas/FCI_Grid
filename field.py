import numpy as np
import scipy as sp

from data import TokamakData
from grid import StructuredPoloidalGrid

#Generate magnetic field on the given grid from the equilibrium data.
class MagneticField(object): #TODO: How best to derive from general base class?
    def __init__(self, eqb: TokamakData, grd: StructuredPoloidalGrid):
        self.psi  = eqb.psi_func(grd.R, grd.Z)
        self.Bp_R = eqb.sign_ip*eqb.psi_func(grd.R, grd.Z, dy=1)/grd.RR
        self.Bp_Z = -eqb.sign_ip*eqb.psi_func(grd.R, grd.Z, dx=1)/grd.RR
        self.Bphi = eqb.f_spl(eqb.psi_func(grd.R, grd.Z))/grd.RR
        self.Bp   = np.sqrt(self.Bp_R**2 + self.Bp_Z**2)
        self.Bmag = np.sqrt(self.Bp_R**2 + self.Bp_Z**2 + self.Bphi**2)
        self.pres = eqb.p_spl(eqb.psi_func(grd.R,grd.Z)) #Not really field related...

        self.dir  = eqb.sign_b0

        #Compute derivatives along phi NOT B_phi. R factor comes from cylindrical geometry.
        dRdphi = grd.RR*self.Bp_R/self.Bphi
        dZdphi = grd.RR*self.Bp_Z/self.Bphi
        self._field_line_rhs = self._setup_field_line_interpolation(grd.R, grd.Z, dRdphi, dZdphi)
        
    def _setup_field_line_interpolation(self, R, Z, dRdphi, dZdphi, linear=False):
        """
        Set up interpolators for field line derivatives
        """
        # Create interpolators
        if (not linear):
            dR_itp = sp.interpolate.RectBivariateSpline(R, Z, dRdphi)
            dZ_itp = sp.interpolate.RectBivariateSpline(R, Z, dZdphi)
        else:
            dR_itp = sp.interpolate.RegularGridInterpolator(
                (R, Z), dRdphi, method='linear',
                bounds_error=False, fill_value=0.0)
            dZ_itp = sp.interpolate.RegularGridInterpolator(
                (R, Z), dZdphi, method='linear',
                bounds_error=False, fill_value=0.0)

        def field_line_rhs(phi, pos):
            """
            RHS function for field line integration
            """
            R, Z = pos

            # Interpolate derivatives at current position
            dRdphi = np.squeeze(dR_itp(R, Z))
            dZdphi = np.squeeze(dZ_itp(R, Z))

            return np.array([dRdphi, dZdphi])

        return field_line_rhs

    def trace_field_line(self, R0, Z0, phi_init, phi_target):
        """
        Trace a field line from a starting position.
        """
        sol = sp.integrate.solve_ivp(
            self._field_line_rhs,
            (phi_init, phi_target), [R0, Z0],
            method="RK45", #method='DOP853' better for stiff problems
            rtol=1e-10, atol=1e-12, dense_output=True) #Generates interpolatable data.

        return sol

    def trace_until_wall(self, R0, Z0, phi_init, dphi, wall, direction=1):
        """
        Trace a field line starting at (R0, Z0) from phi_init in steps of dphi
        (sign given by direction), stopping as soon as the point exits wall_path.
        """
        Rvals = [R0]
        Zvals = [Z0]

        phi_current = phi_init

        while True:
            #Advance one step in phi
            phi_next = phi_current + direction*dphi

            #Trace from phi_current → phi_next
            sol = self.trace_field_line(
                Rvals[-1], Zvals[-1],
                phi_current, phi_next)

            Rn, Zn = sol.y[0, -1], sol.y[1, -1]

            # stop if we’ve left the wall
            if not wall.contains(Rn, Zn):
                break

            # otherwise record and continue
            Rvals.append(Rn)
            Zvals.append(Zn)
            phi_current = phi_next

        return Rvals, Zvals