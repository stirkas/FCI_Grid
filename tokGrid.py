#!/usr/bin/env python

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from hypnotoad.geqdsk._geqdsk import read as gq_read

def main(args):

    with open("/home/tirkas1/Workspace/TokData/DIIID/g162940.02944_670", "r", encoding="utf-8") as file:
        gfile = gq_read(file)

    R = np.linspace(gfile["rleft"], gfile["rleft"] + gfile["rdim"], gfile["nx"])
    Z = np.linspace(gfile["zmid"] - 0.5 * gfile["zdim"],
                    gfile["zmid"] + 0.5 * gfile["zdim"],
                    gfile["ny"])
    
    for item in gfile.items():
        print(item[0])

    Rmin = R[0]
    Zmin = Z[0]
    Rmax = R[-1]
    Zmax = Z[-1]
    RR, ZZ = np.meshgrid(R,Z)

    #TODO: Figure out how to handle sign conventions. Ben says all 4 options should be possible to find in the gfile data.

    rbdy = np.array(gfile["rbdry"])
    zbdy = np.array(gfile["zbdry"])
    rlmt = np.array(gfile["rlim"])
    zlmt = np.array(gfile["zlim"])

    psizr = gfile["psi"]
    psirz = np.transpose(psizr)
    paxis = gfile["simagx"]
    pbdry = gfile["sibdry"]
    fpol = gfile["fpol"] # f = R*B_t
    psin = (psirz - paxis) / (pbdry - paxis)
    psi_func = interpolate.RectBivariateSpline(R, Z, psirz)
    Bp_R =  psi_func(R, Z, dx=1)/R #Note dx=Z here which is confusing. But dx=R if using psizr as in hypnotoad...but then have to transpose.
    Bp_Z = -psi_func(R, Z, dy=1)/R

    psi1D = np.linspace(paxis, pbdry, gfile["nx"])
    f_spl = interpolate.InterpolatedUnivariateSpline(psi1D, fpol, ext=3) #ext=3 uses boundary values outside range.
    Bzeta = f_spl(psi_func(R, Z))/R
    Bp = np.sqrt(Bp_R**2 + Bp_Z**2)

    fig = plt.figure(figsize=(6,8))
    contour_plot = plt.contourf(R, Z, Bzeta, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour_plot)
    plt.scatter(rbdy, zbdy, color='orange', marker='.')
    plt.plot(rlmt, zlmt, color='black', linestyle='-')
    plt.xlabel('R', fontsize=20)
    plt.ylabel('Z', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])