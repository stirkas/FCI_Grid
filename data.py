import numpy as np
from scipy import interpolate

from freeqdsk import geqdsk

class TokamakData():
    """Read a EFIT G-Eqdsk file for a toroidal equilibrium

    This generates a grid in cylindrical geometry

    Parameters
    ----------
    gfile : str
        Name of the file to open
    """
    def __init__(self, gfile):
        #Read gfile data.
        print("Reading EQDSK file...")
        with open(gfile, "r", encoding="utf-8") as f:
            data = geqdsk.read(f)

        #Store equilibrium data grid info.
        self.nr = data["nx"]
        self.nz = data["ny"]

        # Get the range of major radius
        self.rmin = data["rleft"]
        self.rmax = data["rdim"] + self.rmin

        # Range of height
        self.zmin = data["zmid"] - 0.5*data["zdim"]
        self.zmax = data["zmid"] + 0.5*data["zdim"]

        self.R0 = data["rmagx"]
        self.Z0 = data["zmagx"]

        self.r = np.linspace(self.rmin, self.rmax, self.nr)
        self.z = np.linspace(self.zmin, self.zmax, self.nz)
        self.rr, self.zz = np.meshgrid(self.r, self.z, indexing="ij")

        #Field/plasma data.
        self.psi   = data["psi"]
        self.fpol  = data["fpol"] # f = R*B_t
        self.qpsi  = data["qpsi"] # q(psi)
        self.prsr  = data["pres"]
        self.pcur  = data["cpasma"]
        self.bcntr = data["bcentr"]
        self.paxis = data["simagx"]
        self.pbdry = data["sibdry"]

        #Get direction of field and current. Assuming (R,phi,Z) is RHS with phi CCW from top of tokamak per COCOS.
        self.sign_ip = np.sign(self.pcur)
        self.sign_b0 = np.sign(self.bcntr)

        #Separatrix/wall boundaries.
        self.rbdy = np.asarray(data["rbdry"])
        self.zbdy = np.asarray(data["zbdry"])
        self.rlmt = np.asarray(data["rlim"])
        self.zlmt = np.asarray(data["zlim"])

        #Create a 2D spline interpolation for psi
        self.psin      = (self.psi-self.paxis)/(self.pbdry-self.paxis) #psinorm
        self.psi_func  = interpolate.RectBivariateSpline(self.r, self.z, self.psi)
        self.psin_func = interpolate.RectBivariateSpline(self.r, self.z, self.psin)

        #Toroidal field component and q(psi).
        self.psi1D = np.linspace(self.paxis, self.pbdry, self.nr)
        #self.psi1D = np.flip(self.psi1D) #TODO: For TCV it seems r is backwards??? Different cocos convention?
        #self.psi1D = np.linspace(self.pbdry, self.paxis, nrg)
        #TODO: Why doesnt ext=0 work ok when tracing field?
        self.f_spl = interpolate.InterpolatedUnivariateSpline(self.psi1D, self.fpol, ext=3) #ext=3 uses boundary values outside range.
        self.q_spl = interpolate.InterpolatedUnivariateSpline(self.psi1D, self.qpsi, ext=3) #ext=0 uses extrapolation as with RectBivSpline on 2D but doesnt work in the integrator.
        self.p_spl = interpolate.InterpolatedUnivariateSpline(self.psi1D, self.prsr, ext=3)