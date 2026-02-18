"""Equilibrium reader for EFIT GEQDSK files.

This module wraps `freeqdsk.geqdsk` and exposes convenient splines and
metadata used by the grid and field modules.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy import interpolate

from freeqdsk import geqdsk

class TokamakData:
    """Read an EFIT G-EQDSK file and expose interpolants.

    The class produces 1-D coordinate arrays `.r`, `.z` and 2-D meshgrids
    `.rr`, `.zz`. It also exposes commonly used splines for psi, normalized
    psi, f(psi) and q(psi).
    """

    def __init__(self, gfile: str) -> None:
        """Load equilibrium from `gfile`."""
        path = Path(gfile)
        if not path.exists():
            raise FileNotFoundError(f"EQDSK file not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Path exists but is not a file: {path}")

        print("Reading EQDSK file...")
        with open(gfile, "r", encoding="utf-8") as f:
            data = geqdsk.read(f)

        # Basic grid metadata.
        self.nr   = data["nx"]
        self.nz   = data["ny"]
        self.rmin = data["rleft"]
        self.rmax = data["rdim"] + self.rmin
        self.zmin = data["zmid"] - 0.5*data["zdim"]
        self.zmax = data["zmid"] + 0.5*data["zdim"]

        #Center coordinates
        self.R0 = data["rmagx"]
        self.Z0 = data["zmagx"]

        # 1-D coordinate arrays and 2-D mesh.
        self.r = np.linspace(self.rmin, self.rmax, self.nr)
        self.z = np.linspace(self.zmin, self.zmax, self.nz)
        self.rr, self.zz = np.meshgrid(self.r, self.z, indexing="ij")

        #Field/plasma data.
        self.psi   = np.asarray(data["psi"])
        self.fpol  = np.asarray(data["fpol"]) # f = R*B_t
        self.qpsi  = np.asarray(data["qpsi"]) # q(psi)
        self.pres  = np.asarray(data["pres"])
        self.pcur  = np.asarray(data["cpasma"])
        self.bcntr = np.asarray(data["bcentr"])
        self.paxis = np.asarray(data["simagx"])
        self.pbdry = np.asarray(data["sibdry"])

        #Get direction of field and current.
        #Assuming (R,phi,Z) is RHS with phi CCW from top of tokamak per COCOS.
        self.sign_ip = np.sign(self.pcur)
        self.sign_b0 = np.sign(self.bcntr)

        #Separatrix and wall boundaries.
        self.rbdy = np.asarray(data["rbdry"])
        self.zbdy = np.asarray(data["zbdry"])
        self.rlmt = np.asarray(data["rlim"])
        self.zlmt = np.asarray(data["zlim"])

        #Create a 2D spline interpolation for psi
        self.psin      = (self.psi-self.paxis)/(self.pbdry-self.paxis) #psinorm
        self.psi_func  = interpolate.RectBivariateSpline(self.r, self.z, self.psi)
        self.psin_func = interpolate.RectBivariateSpline(self.r, self.z, self.psin)

        # 1-D psi array for f(psi), q(psi), p(psi) splines.
        self.psi1D = np.linspace(self.paxis, self.pbdry, max(self.nr, 2))
        if self.pbdry < self.paxis:
            self.psi1D = np.flip(self.psi1D)

        #Toroidal field component and q(psi).
        self.psi1D = np.linspace(self.paxis, self.pbdry, self.nr)
        #TODO: If minor radius decreasing means diff COCOS convention, does anything else need to change?
        if (self.pbdry < self.paxis):
            self.psi1D = np.flip(self.psi1D)
        #Note, ext=3 uses boundary values outside range, but doing interpolation (ext=0) crashes when tracing field.
        self.f_spl = interpolate.InterpolatedUnivariateSpline(self.psi1D, self.fpol, ext=3)
        self.q_spl = interpolate.InterpolatedUnivariateSpline(self.psi1D, self.qpsi, ext=3)
        self.p_spl = interpolate.InterpolatedUnivariateSpline(self.psi1D, self.pres, ext=3)
