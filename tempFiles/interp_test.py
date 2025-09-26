# Demo 2: Show equivalence between
#   (i) Coupled ghost equations (unknowns = g00, g01, g11, phi_ip) solved by GS and by direct solve
#  (ii) 4x4 "row-replacement" bilinear system (unknowns = A,B,C,D) with 3 BC rows + 1 fluid row
#
# Geometry: one IP inside the unit cell with corners:
#   (0,0)-> ghost g00, (1,0)-> fluid f, (0,1)-> ghost g01, (1,1)-> ghost g11
# Dirichlet wall at the 3 ghost corners with given *wall values* B00, B01, B11
# Ghost formula (symmetric): g = 2*B - phi_ip
#
# Bilinear basis: phi(x,z) = A*x*z + B*x + C*z + D
# Let e = [xI*zI, xI, zI, 1], q00=[0,0,0,1], q10=[0,1,0,1], q01=[0,0,1,1], q11=[1,1,1,1]
#
# Row-replacement system:
#   (q00 + e)·c = 2*B00
#   (q01 + e)·c = 2*B01
#   (q11 + e)·c = 2*B11
#    q10      ·c =    f
# Solve for c=[A,B,C,D]; then phi_ip = e·c, and g00= q00·c, etc.
#
# Coupled system in (g00, g01, g11, phi_ip):
#   g00 + phi_ip = 2*B00
#   g01 + phi_ip = 2*B01
#   g11 + phi_ip = 2*B11
#  -w00*g00 - w01*g01 - w11*g11 + phi_ip = w10*f
#
# We'll compare: (a) direct solve of this 4x4 vs. (b) GS on it vs. (c) row-replacement bilinear result.

import numpy as np

# pick an IP location (s,t) and compute bilinear weights
s, t = 0.37, 0.58
w00 = (1-s)*(1-t)
w10 = s*(1-t)
w01 = (1-s)*t
w11 = s*t

# fluid value at (1,0)
f = 1.23
# wall Dirichlet values at the three ghost corners (these are phi_B at those ghost corners' BIs)
B00, B01, B11 = 0.9, 1.1, 1.05

# --- (i) Solve the coupled 4x4 system in [g00, g01, g11, phi_ip] ---

A = np.array([
    [ 1.0,  0.0,  0.0,  1.0],         # g00 + phi_ip = 2*B00
    [ 0.0,  1.0,  0.0,  1.0],         # g01 + phi_ip = 2*B01
    [ 0.0,  0.0,  1.0,  1.0],         # g11 + phi_ip = 2*B11
    [-w00, -w01, -w11,  1.0],         # -w00 g00 - w01 g01 - w11 g11 + phi_ip = w10 f
], dtype=float)

b = np.array([2*B00, 2*B01, 2*B11, w10*f], dtype=float)

sol_direct = np.linalg.solve(A, b)
g00_d, g01_d, g11_d, phi_ip_d = sol_direct

# Gauss–Seidel on the same 4x4
def gauss_seidel(A, b, x0=None, sweeps=100, tol=1e-14):
    A = np.asarray(A, float); b = np.asarray(b, float)
    n = A.shape[0]; x = np.zeros(n) if x0 is None else np.array(x0, float, copy=True)
    for k in range(sweeps):
        x_prev = x.copy()
        for i in range(n):
            s1 = A[i,:i] @ x[:i]
            s2 = A[i,i+1:] @ x_prev[i+1:]
            x[i] = (b[i] - s1 - s2) / A[i,i]
        if np.linalg.norm(x - x_prev, np.inf) < tol:
            return x, k+1
    return x, sweeps

sol_gs, iters = gauss_seidel(A, b, x0=np.zeros(4), sweeps=200, tol=1e-14)
g00_gs, g01_gs, g11_gs, phi_ip_gs = sol_gs

# --- (ii) Row-replacement 4x4 in bilinear coefficients [A,B,C,D] ---

xI, zI = s, t
e   = np.array([xI*zI, xI, zI, 1.0])
q00 = np.array([0.0,   0.0, 0.0, 1.0])
q10 = np.array([0.0,   1.0, 0.0, 1.0])
q01 = np.array([0.0,   0.0, 1.0, 1.0])
q11 = np.array([1.0,   1.0, 1.0, 1.0])

V = np.vstack([
    q00 + e,   # (q00 + e)·c = 2*B00
    q01 + e,   # (q01 + e)·c = 2*B01
    q11 + e,   # (q11 + e)·c = 2*B11
    q10,       # q10·c = f
])
rhs = np.array([2*B00, 2*B01, 2*B11, f], dtype=float)

coeff = np.linalg.solve(V, rhs)  # [A,B,C,D]
phi_ip_rr = e @ coeff
g00_rr    = q00 @ coeff
g01_rr    = q01 @ coeff
g11_rr    = q11 @ coeff

# Cross-check: recompute phi_ip from weights using row-replacement ghost values
phi_ip_from_weights = w00*g00_rr + w10*f + w01*g01_rr + w11*g11_rr

# Report
print("Weights: w00=%.6f w10=%.6f w01=%.6f w11=%.6f" % (w00,w10,w01,w11))
print("Gauss–Seidel iterations:", iters)
print("\n--- Coupled 4x4 (direct) ---")
print("g00=%.12f  g01=%.12f  g11=%.12f  phi_ip=%.12f" % (g00_d, g01_d, g11_d, phi_ip_d))
print("\n--- Coupled 4x4 (GS)     ---")
print("g00=%.12f  g01=%.12f  g11=%.12f  phi_ip=%.12f" % (g00_gs, g01_gs, g11_gs, phi_ip_gs))
print("\n--- Row-replacement 4x4 ---")
print("g00=%.12f  g01=%.12f  g11=%.12f  phi_ip=%.12f" % (g00_rr, g01_rr, g11_rr, phi_ip_rr))
print("\nphi_ip (weights from RR ghosts) = %.12f" % phi_ip_from_weights)

# Differences
print("\nMax |coupled_direct - RR| on [g00,g01,g11,phi_ip] =",
      np.max(np.abs(np.array([g00_d,g01_d,g11_d,phi_ip_d]) -
                    np.array([g00_rr,g01_rr,g11_rr,phi_ip_rr]))))
