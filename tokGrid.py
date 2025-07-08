#!/usr/bin/env python

import sys

import matplotlib.pyplot as plt
import numpy as np

from hypnotoad.geqdsk._geqdsk import read as gq_read

def main(args):

    with open("/home/tirkas1/Workspace/TokData/DIIID/g162940.02944_670", "r", encoding="utf-8") as file:
        gfile = gq_read(file)
    
    for item in gfile.items():
        print(item[0])

    R = np.linspace(gfile["rleft"], gfile["rleft"] + gfile["rdim"], gfile["nx"])
    Z = np.linspace(gfile["zmid"] - 0.5 * gfile["zdim"],
                    gfile["zmid"] + 0.5 * gfile["zdim"],
                    gfile["ny"])
    Rmin = R[0]
    Zmin = Z[0]
    Rmax = R[-1]
    Zmax = Z[-1]
    RR, ZZ = np.meshgrid(R,Z)

    rbdy = np.array(gfile["rbdry"])
    zbdy = np.array(gfile["zbdry"])
    rlmt = np.array(gfile["rlim"])
    zlmt = np.array(gfile["zlim"])

    psirz = np.transpose(gfile["psi"])

    fig = plt.figure(figsize=(6,8))
    plt.contourf(RR, ZZ, psirz, levels=50)
    plt.scatter(rbdy, zbdy, color='orange', marker='.')
    plt.plot(rlmt, zlmt, color='black', linestyle='-')
    plt.xlabel('R', fontsize=20)
    plt.ylabel('Z', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])