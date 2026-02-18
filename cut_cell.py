import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np

import boundary as bdy
import utils

OUTSIDE = 0
INSIDE  = 1
CUT     = 2

def segment_intersection(p, r, q, s, eps=1e-14):
    """
    Intersection of segments p->p+r and q->q+s.
    Returns (hit, pt). pt is intersection point.
    """
    p = np.asarray(p, float); r = np.asarray(r, float)
    q = np.asarray(q, float); s = np.asarray(s, float)

    rxs = r[0]*s[1] - r[1]*s[0]
    if abs(rxs) < eps:
        return False, None  # parallel/colinear; ignore

    qmp = q - p
    t = (qmp[0]*s[1] - qmp[1]*s[0]) / rxs
    u = (qmp[0]*r[1] - qmp[1]*r[0]) / rxs

    if -1e-12 <= t <= 1.0 + 1e-12 and -1e-12 <= u <= 1.0 + 1e-12:
        t = min(max(t, 0.0), 1.0)
        pt = p + t*r
        return True, pt

    return False, None


def cell_boundary_intersections(bdry_verts, x_min, x_max, z_min, z_max, eps=1e-14):
    """
    Intersect boundary polygon edges with the cell rectangle boundary.
    Returns unique intersection points as (M,2).
    """
    bdry_verts = np.asarray(bdry_verts, float)
    hits = []

    rect = np.array([
        [x_min, z_min],
        [x_max, z_min],
        [x_max, z_max],
        [x_min, z_max],
    ], float)
    rect_edges = [(rect[k], rect[(k+1) % 4]) for k in range(4)]

    n = len(bdry_verts)
    for k in range(n):
        p0 = bdry_verts[k]
        p1 = bdry_verts[(k+1) % n]
        p = p0
        r = p1 - p0

        for (q0, q1) in rect_edges:
            q = q0
            s = q1 - q0
            hit, pt = segment_intersection(p, r, q, s, eps=eps)
            if hit:
                hits.append(pt)

    if not hits:
        return np.zeros((0, 2), float)

    hits = np.asarray(hits, float)

    # Deduplicate (corners can create duplicates)
    uniq = [hits[0]]
    for h in hits[1:]:
        if min(np.linalg.norm(h - u) for u in uniq) > 1e-10:
            uniq.append(h)
    return np.asarray(uniq, float)


def furthest_pair(points):
    """
    Choose the two points with maximum separation.
    points: (M,2)
    Returns p0, p1 or (None,None) if insufficient.
    """
    points = np.asarray(points, float)
    m = len(points)
    if m < 2:
        return None, None
    if m == 2:
        return points[0], points[1]

    best = (0, 1)
    bestd = -1.0
    for a in range(m):
        for b in range(a+1, m):
            d = np.linalg.norm(points[a] - points[b])
            if d > bestd:
                bestd = d
                best = (a, b)
    return points[best[0]], points[best[1]]

def bilinear_base_and_weights(point, xc, zc, dx, dz):
    """
    Bilinear interpolation weights for a point (x,z) using *cell-centered* values.

    Returns:
      i0, j0, w00, w01, w10, w11

    Stencil layout (cell centers):
        (i0,   j0)     (i0,   j0+1)
        (i0+1, j0)     (i0+1, j0+1)

    Weight ordering matches C++:
        w00 * f[i0,   j0]
      + w01 * f[i0,   j0+1]
      + w10 * f[i0+1, j0]
      + w11 * f[i0+1, j0+1]
    """

    x, z = float(point[0]), float(point[1])

    # Find i0,j0 such that point lies between centers (i0,i0+1), (j0,j0+1)
    i0 = int(np.floor((x - xc[0]) / dx))
    j0 = int(np.floor((z - zc[0]) / dz))

    # Clamp so i0+1, j0+1 are valid - unnecessary?
    #i0 = max(0, min(i0, len(xc) - 2))
    #j0 = max(0, min(j0, len(zc) - 2))

    # Local coordinates between cell centers
    x0 = xc[i0]
    z0 = zc[j0]

    xi  = (x - x0) / dx
    eta = (z - z0) / dz

    xi  = np.clip(xi, 0.0, 1.0)
    eta = np.clip(eta, 0.0, 1.0)

    # Weights (NOTE ORDER!)
    w00 = (1.0 - xi) * (1.0 - eta)   # (i0,   j0)
    w01 = (1.0 - xi) * eta           # (i0,   j0+1)
    w10 = xi * (1.0 - eta)           # (i0+1, j0)
    w11 = xi * eta                   # (i0+1, j0+1)

    return i0, j0, w00, w01, w10, w11

def classify_cell(bdry: bdy.PolygonBoundary,
                  x_min, x_max, z_min, z_max,
                  bbox, eps=1e-12):
    """
    Classify a cell [x_min,x_max]x[z_min,z_max] as:
        OUTSIDE (0) : no plasma
        INSIDE  (1) : fully inside plasma, no boundary crossing
        CUT     (2) : boundary passes through cell

    Uses:
      - fast bounding box reject
      - Path.intersects_path(rect_path, filled=True)
      - corner contains checks
    """
    plasma_path: Path = bdry.path

    px_min, px_max, pz_min, pz_max = bbox

    # --- 1. Fast global bbox reject ---
    if x_max < px_min - eps or x_min > px_max + eps:
        return OUTSIDE
    if z_max < pz_min - eps or z_min > pz_max + eps:
        return OUTSIDE

    # --- 2. Build rectangle path for this cell ---
    rect_verts = [
        (x_min, z_min),
        (x_max, z_min),
        (x_max, z_max),
        (x_min, z_max),
        (x_min, z_min),
    ]
    rect_codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    rect_path = Path(rect_verts, rect_codes)

    # --- 3. Check for intersection of the boundaries ---
    # If the boundaries intersect, the cell is definitely CUT.
    if plasma_path.intersects_path(rect_path, filled=False):
        return CUT

    # --- 4. No boundary crossing: either fully inside or fully outside ---
    # Check corners against filled region.
    corners_inside = [plasma_path.contains_point(v) for v in rect_verts[:-1]]

    if any(corners_inside):
        # At least one corner is inside and we know there is no intersection,
        # so the entire cell is inside the plasma region.
        return INSIDE

    # No intersections and no corners inside -> cell is fully outside
    return OUTSIDE

def clip_polygon_halfplane(poly, a, b, c):
    """
    Clip polygon 'poly' against the half-plane a*x + b*z + c >= 0.
    poly: array-like shape (N, 2)
    Returns new polygon (M,2). Can be empty (M = 0).
    """
    poly = np.asarray(poly, dtype=float)
    if poly.size == 0:
        return poly

    x = poly[:, 0]
    z = poly[:, 1]
    vals = a * x + b * z + c
    inside = vals >= 0.0

    out_pts = []
    n = len(poly)

    for i in range(n):
        j = (i + 1) % n
        Pi = poly[i]
        Pj = poly[j]
        fi = inside[i]
        fj = inside[j]

        if fi and fj:
            # inside -> inside: keep Pj
            out_pts.append(Pj)
        elif fi and not fj:
            # inside -> outside: keep intersection
            denom = vals[i] - vals[j]
            if abs(denom) > 1e-14:
                t = vals[i] / denom
                I = Pi + t * (Pj - Pi)
                out_pts.append(I)
        elif (not fi) and fj:
            # outside -> inside: keep intersection + Pj
            denom = vals[i] - vals[j]
            if abs(denom) > 1e-14:
                t = vals[i] / denom
                I = Pi + t * (Pj - Pi)
                out_pts.append(I)
            out_pts.append(Pj)
        # else: outside -> outside: keep nothing

    if not out_pts:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(out_pts, dtype=float)

def clean_polygon(poly, eps=1e-12): #TODO: Use tolerance class? And for all eps in this file...
    """
    Clean up a polygon (N,2) from clipping:

    - Remove consecutive duplicate points (within eps).
    - Optionally remove final point if it duplicates the first.
    - If < 3 unique points remain or area ~0, return empty array.
    """
    poly = np.asarray(poly, dtype=float)
    if poly.size == 0:
        return poly

    # remove consecutive duplicates
    cleaned = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - cleaned[-1]) > eps:
            cleaned.append(p)
    cleaned = np.array(cleaned)

    # if closed polygon repeats first vertex at end, drop the last
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) < eps:
        cleaned = cleaned[:-1]

    # if too few vertices, treat as empty
    if len(cleaned) < 3:
        return np.empty((0, 2), dtype=float)

    # optional: check area; if essentially zero, treat as empty
    A = polygon_area_line_integral(cleaned)  # your existing area function
    if abs(A) < eps:
        return np.empty((0, 2), dtype=float)

    return cleaned

def clip_polygon_to_rect(poly, x_min, x_max, z_min, z_max):
    """
    Clip polygon 'poly' to axis-aligned rectangle [x_min,x_max] x [z_min,z_max].
    poly: (N,2) array [ [x0,z0], [x1,z1], ... ]
    Returns intersection polygon (M,2). Can be empty (M=0).
    """
    poly = np.asarray(poly, dtype=float)
    if poly.size == 0:
        return poly

    # x >= x_min  ->  a= 1, b=0, c=-x_min
    poly = clip_polygon_halfplane(poly,  1.0, 0.0, -x_min)
    if len(poly) == 0: return poly

    # x <= x_max  ->  -x >= -x_max -> a=-1, b=0, c=x_max
    poly = clip_polygon_halfplane(poly, -1.0, 0.0,  x_max)
    if len(poly) == 0: return poly

    # z >= z_min  ->  a=0, b=1, c=-z_min
    poly = clip_polygon_halfplane(poly,  0.0, 1.0, -z_min)
    if len(poly) == 0: return poly

    # z <= z_max  ->  -z >= -z_max -> a=0, b=-1, c=z_max
    poly = clip_polygon_halfplane(poly,  0.0, -1.0,  z_max)

    poly = clean_polygon(poly)

    return poly

def polygon_area_line_integral(poly):
    """
    Area of a simple polygon via line integral / Green's theorem:
        A = 1/2 ∮ (x dz - z dx)
          = 1/2 Σ (x_i z_{i+1} - z_i x_{i+1})

    poly: (N,2) vertices, NOT necessarily closed. Works either way.
    Returns non-negative area.
    """
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return 0.0

    # ensure closed loop for the sum
    if not np.allclose(poly[0], poly[-1], atol=1e-14):
        V = np.vstack([poly, poly[0]])
    else:
        V = poly

    x = V[:, 0]
    z = V[:, 1]
    A = 0.5 * np.sum(x[:-1] * z[1:] - z[:-1] * x[1:])
    return float(abs(A))


def polygon_area(poly):
    """
    Signed area of polygon (N,2) vertices.
    Returns positive for CCW orientation; use abs() for area.
    """
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    z = poly[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))

def face_fraction_plus_x(cell_poly, x_max, z_min, z_max, dz):
    """
    Fraction of +x face (x = x_max, z in [z_min,z_max]) that lies inside plasma.
    cell_poly is plasma ∩ cell rectangle (M,2).
    """
    cell_poly = np.asarray(cell_poly, dtype=float)
    if len(cell_poly) == 0:
        return 0.0

    z_hits = []
    n = len(cell_poly)
    for k in range(n):
        P = cell_poly[k]
        Q = cell_poly[(k + 1) % n]
        xP, zP = P
        xQ, zQ = Q

        denom = (xQ - xP)
        if abs(denom) < 1e-14:
            continue  # segment parallel to face

        t = (x_max - xP) / denom

        #TODO: Combine logic with z-face.
        #TODO: Little annoying that path_edge_bias is +/- depending on cw/ccw, maybe base it on an always positive to use here.
        #Allow a little slack:
        bdy_eps = np.abs(utils.DEFAULT_TOL.path_edge_bias)
        if t < -bdy_eps or t > 1.0 + bdy_eps:
            # definitely outside the segment
            continue
        #Clamp into [0,1] to kill tiny overshoots like -1e-15 or 1+1e-15
        t = min(max(t, 0.0), 1.0)

        if 0.0 <= t <= 1.0:
            zI = zP + t * (zQ - zP)
            if z_min - bdy_eps <= zI <= z_max + bdy_eps:
                z_hits.append(zI)

    if len(z_hits) < 2:
        return 0.0

    z_hits.sort()
    length_inside = 0.0
    # even-odd rule: sum intervals [z0,z1], [z2,z3], ...
    for k in range(0, len(z_hits) - 1, 2):
        length_inside += (z_hits[k + 1] - z_hits[k])

    return max(0.0, min(1.0, length_inside / dz))


def face_fraction_plus_z(cell_poly, x_min, x_max, z_max, dx):
    """
    Fraction of +z face (z = z_max, x in [x_min,x_max]) that lies inside plasma.
    """
    cell_poly = np.asarray(cell_poly, dtype=float)
    if len(cell_poly) == 0:
        return 0.0

    x_hits = []
    n = len(cell_poly)
    for k in range(n):
        P = cell_poly[k]
        Q = cell_poly[(k + 1) % n]
        xP, zP = P
        xQ, zQ = Q

        denom = (zQ - zP)
        if abs(denom) < 1e-14: #TODO: What tol to use here from class?
            continue  # parallel

        t = (z_max - zP) / denom

        #TODO: Little annoying that path_edge_bias is +/- depending on cw/ccw, maybe base it on an always positive to use here.
        #Allow a little slack:
        if t < -np.abs(utils.DEFAULT_TOL.path_edge_bias) or t > 1.0 + np.abs(utils.DEFAULT_TOL.path_edge_bias):
            # definitely outside the segment
            continue
        #Clamp into [0,1] to kill tiny overshoots like -1e-15 or 1+1e-15
        t = min(max(t, 0.0), 1.0)

        if 0.0 <= t <= 1.0:
            xI = xP + t * (xQ - xP)
            if x_min <= xI <= x_max:
                x_hits.append(xI)

    if len(x_hits) < 2:
        return 0.0

    x_hits.sort()
    length_inside = 0.0
    for k in range(0, len(x_hits) - 1, 2):
        length_inside += (x_hits[k + 1] - x_hits[k])

    return max(0.0, min(1.0, length_inside / dx))

def cell_in_bounds(bdy_path: Path,
            x_min, x_max, z_min, z_max,
            bbox, eps=1e-12):
    """
    Return True if the plasma polygon (bdry.path) might intersect
    or cover the cell [x_min,x_max] x [z_min,z_max].

    Uses:
      - fast bounding-box reject
      - path.intersects_path(rect_path)
      - corner contains checks
    """
    # --- 1. fast bounding box reject ---
    (px_min, px_max, pz_min, pz_max) = bbox
    if x_max < px_min - eps or x_min > px_max + eps:
        return False
    if z_max < pz_min - eps or z_min > pz_max + eps:
        return False

    # --- 2. build rectangle as a Path ---
    rect_verts = [
        (x_min, z_min),
        (x_max, z_min),
        (x_max, z_max),
        (x_min, z_max),
        (x_min, z_min),
    ]
    rect_codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    rect_path = Path(rect_verts, rect_codes)

    # If the two shapes don't intersect at all, and no corner is inside,
    # then the cell is definitely empty.
    if not bdy_path.intersects_path(rect_path, filled=True):
        # check if the cell is completely inside or outside
        corners = rect_verts[:-1]
        if not any(bdy_path.contains_point(c) for c in corners):
            return False

    return True

def compute_cutcell_fractions(xc, zc, dx, dz, bdry: bdy.PolygonBoundary):
    """
    Compute volume and +face fractions for a Cartesian grid
    clipped by an arbitrary boundary polygon.

    boundary_poly : (Nb,2) array of (x,z) vertices (enclosing the plasma).
    xc, zc        : 1D arrays of cell-center coordinates.
    dx, dz        : scalar cell sizes.
    path_for_culling : optional matplotlib.path.Path to cheaply
                       detect fully outside cells.

    Returns:
        vol_frac      : (nx, nz)
        fx_plus_frac  : (nx-1, nz)
        fz_plus_frac  : (nx, nz-1)
    """
    #TODO: This is all in R,Z not x,z...
    nx = len(xc)
    nz = len(zc)

    #TODO: Temporary working with ones since outer cells are zeroed out anyway in BOUT. But cells outside domain should be set zero for faces.
    #TODO: Switch to np.ones and break in the loop for full face for now.
    vol_frac     = np.zeros((nx, nz), dtype=float) #np.zeros/np.ones
    fx_plus_frac = np.zeros((nx, nz), dtype=float) #np.zeros/np.ones
    fz_plus_frac = np.zeros((nx, nz), dtype=float) #np.zeros/np.ones

    #Get info for testing bounds through cell.
    px_min, pz_min = bdry.path.vertices.min(axis=0)
    px_max, pz_max = bdry.path.vertices.max(axis=0)
    bbox = (px_min, px_max, pz_min, pz_max)

    for i in range(nx):
        x_min = xc[i] - 0.5 * dx
        x_max = xc[i] + 0.5 * dx
        for j in range(nz):
            z_min = zc[j] - 0.5 * dz
            z_max = zc[j] + 0.5 * dz

            cls = classify_cell(bdry, x_min, x_max, z_min, z_max, bbox)

            #If cell fully outside boundary can ignore it and use default factors.
            if cls == OUTSIDE:
                continue

            if cls == INSIDE:
                vol_frac[i, j] = 1.0
                fx_plus_frac[i,j] = 1.0
                fz_plus_frac[i,j] = 1.0
                continue

            # Clip boundary poly to this cell
            cell_poly = clip_polygon_to_rect(bdry.path.vertices,
                            x_min, x_max, z_min, z_max)
            
            A = polygon_area_line_integral(cell_poly) #TODO: Use built in function from bdy? But this is the cut piece in the plasma.
            vol_frac[i, j] = A / (dx * dz)

            debug = False
            if debug:
                print("i, j: ", i, j)
                print("x_min, x_max: ", x_min, x_max)
                print("z_min, z_max: ", z_min, z_max)
                print("cell_poly:\n", cell_poly)
                print("cell area: ", A)
                inside_center = bdry.contains(xc[i], zc[j])
                print("center inside?", inside_center)

                fig, ax = plt.subplots()
                # draw cell
                ax.plot([x_min, x_max, x_max, x_min, x_min],
                        [z_min, z_min, z_max, z_max, z_min], 'b-')
                # draw boundary
                bp = np.asarray(bdry.path.vertices)  # or whatever you use
                ax.plot(bp[:,0], bp[:,1], 'k-')
                ax.set_aspect('equal')
                plt.show()
            
            # +x face (i,j) corresponds to face index (i, j) if i < nx-1
            if i < nx - 1:
                fx_plus_frac[i, j] = face_fraction_plus_x(
                    cell_poly, x_max, z_min, z_max, dz)
            else:
                fx_plus_frac[i,j] = 0.0 #TODO: Match padding code? And remove when initialized to zeros! Same for z below.
            
            # +z face (i,j) corresponds to (i, j) if j < nz-1
            if j < nz - 1:
                fz_plus_frac[i, j] = face_fraction_plus_z(
                    cell_poly, x_min, x_max, z_max, dx)
            else:
                fz_plus_frac[i,j] = 0.0

    #Clip to possibly avoid numerical issues.
    vol_frac = clip_cuts(vol_frac)
    fx_plus_frac = clip_cuts(fx_plus_frac)
    fz_plus_frac = clip_cuts(fz_plus_frac)

    if (utils.DEBUG_FLAG):
        print("Plotting cut cell volume fractions and faces...")
        plot_volume_fractions(xc, zc, dx, dx, vol_frac, bdry)
        plot_cut_faces(
            xc, zc, dx, dz, fx_plus_frac, fz_plus_frac, bdry)

    #TODO: Temporary fix since BOUT runs cells outside the wall too right now. Dont divide by zero volume.
    vol_eps = 1e-12
    vol_frac = np.asarray(vol_frac)
    out_mask = vol_frac < vol_eps
    vol_frac[out_mask] = 1.0

    return vol_frac, fx_plus_frac, fz_plus_frac

def clip_segment_to_rect(P, Q, x_min, x_max, z_min, z_max, eps=1e-14):
    """
    Clip segment P->Q to axis-aligned rectangle.
    Returns (hit, A, B, t0, t1) where A=P+t0*(Q-P), B=P+t1*(Q-P), and 0<=t0<=t1<=1.
    """
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    d = Q - P

    t0, t1 = 0.0, 1.0
    # inequalities: x_min <= x <= x_max, z_min <= z <= z_max
    # written as:  p*t <= q  (Liang-Barsky form)
    for p, q in [
        (-d[0], P[0] - x_min),  # x >= x_min
        ( d[0], x_max - P[0]),  # x <= x_max
        (-d[1], P[1] - z_min),  # z >= z_min
        ( d[1], z_max - P[1]),  # z <= z_max
    ]:
        if abs(p) < eps:
            if q < 0:  # parallel and outside
                return False, None, None, None, None
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 - t1 > 1e-12:
                return False, None, None, None, None

    A = P + t0 * d
    B = P + t1 * d
    # reject essentially-zero length
    if np.linalg.norm(B - A) < 1e-12:
        return False, None, None, None, None
    return True, A, B, t0, t1

def eb_endpoints_longest_run(bdry_verts, x_min, x_max, z_min, z_max, eps=1e-14): #TODO: Fix eps to tols?
    """
    Walk boundary edges, clip each edge to the cell rectangle, merge consecutive inside pieces,
    and return endpoints of the longest merged run.
    """
    V = np.asarray(bdry_verts, float)
    n = len(V)
    if n < 2:
        return None, None

    # collect inside pieces in boundary order: (k, t0, t1, A, B, len)
    pieces = []
    for k in range(n):
        P = V[k]
        Q = V[(k + 1) % n]
        hit, A, B, t0, t1 = clip_segment_to_rect(P, Q, x_min, x_max, z_min, z_max, eps=eps)
        if hit:
            pieces.append((k, t0, t1, A, B, float(np.linalg.norm(B - A))))

    if not pieces:
        return None, None

    # merge consecutive pieces along boundary index, allowing wrap by doubling list once
    pieces.sort(key=lambda x: x[0])
    pieces2 = pieces + [(k + n, t0, t1, A, B, L) for (k, t0, t1, A, B, L) in pieces]

    best_L = -1.0
    best_A = None
    best_B = None

    run_A = pieces2[0][3]
    run_B = pieces2[0][4]
    run_L = pieces2[0][5]
    prev_k = pieces2[0][0]

    def close_enough(p, q):
        return np.linalg.norm(np.asarray(p) - np.asarray(q)) < 1e-10

    for (k, t0, t1, A, B, L) in pieces2[1:]:
        # stop after spanning more than one full loop
        if k - pieces2[0][0] >= n:
            break

        consecutive = (k == prev_k + 1)
        connected = close_enough(run_B, A)  # end touches next start

        if consecutive and connected:
            run_B = B
            run_L += L
        else:
            if run_L > best_L:
                best_L, best_A, best_B = run_L, run_A, run_B
            run_A, run_B, run_L = A, B, L

        prev_k = k

    if run_L > best_L:
        best_L, best_A, best_B = run_L, run_A, run_B

    return best_A, best_B

def rect_contains(p, x_min, x_max, z_min, z_max, eps=1e-14):
    return (x_min - eps <= p[0] <= x_max + eps) and (z_min - eps <= p[1] <= z_max + eps)

def seg_seg_intersection_param(p, r, q, s, eps=1e-14):
    """Return (hit, t) where p+t r intersects q+u s, with t,u in [0,1]."""
    rxs = r[0]*s[1] - r[1]*s[0]
    if abs(rxs) < eps:
        return False, None
    qmp = q - p
    t = (qmp[0]*s[1] - qmp[1]*s[0]) / rxs
    u = (qmp[0]*r[1] - qmp[1]*r[0]) / rxs
    if -1e-12 <= t <= 1+1e-12 and -1e-12 <= u <= 1+1e-12:
        return True, min(max(t, 0.0), 1.0)
    return False, None

def point_on_poly(V, s):
    """V: (n,2), s in [0,n)."""
    n = len(V)
    k = int(np.floor(s)) % n
    t = s - np.floor(s)
    return V[k] + t*(V[(k+1)%n] - V[k])

def longest_inside_intervals_on_boundary(bdry_verts, x_min, x_max, z_min, z_max, eps=1e-14):
    """
    Return a list of endpoint pairs [(p0,p1), ...] for every contiguous boundary
    interval that lies inside the rectangle [x_min,x_max] x [z_min,z_max].

    Each p0,p1 is a numpy array (shape (2,)). If no interval is found, returns [].

    This re-uses the same seg-seg intersection event method as your original,
    but returns ALL intervals (not only the longest one).
    """
    V = np.asarray(bdry_verts, float)
    n = len(V)
    if n < 2:
        return []

    # rectangle edges
    R = np.array([[x_min, z_min],
                  [x_max, z_min],
                  [x_max, z_max],
                  [x_min, z_max]], float)
    rect_edges = [(R[i], R[(i+1) % 4]) for i in range(4)]

    # collect intersection "events" as (s_param, point)
    events = []
    for k in range(n):
        p0 = V[k]; p1 = V[(k+1) % n]
        p = p0; r = p1 - p0
        for (q0, q1) in rect_edges:
            q = q0; s = q1 - q0
            hit, t = seg_seg_intersection_param(p, r, q, s, eps=eps)
            if hit:
                sp = k + t
                events.append((sp, p0 + t*r))

    # need at least two events to form an interval
    if len(events) < 2:
        return []

    # sort and deduplicate by parameter s (corner hits can produce near-duplicate events)
    events.sort(key=lambda x: x[0])
    s_list = [events[0][0]]
    p_list = [events[0][1]]
    for sp, pt in events[1:]:
        if abs(sp - s_list[-1]) > 1e-10:
            s_list.append(sp); p_list.append(pt)
        else:
            # average near-duplicate intersection points (corner hits)
            p_list[-1] = 0.5*(p_list[-1] + pt)

    if len(s_list) < 2:
        return []

    # append wrap event to allow intervals that wrap across polygon index 0
    s_ext = s_list + [s_list[0] + n]
    p_ext = p_list + [p_list[0].copy()]

    intervals = []

    def rect_contains_local(p):
        return rect_contains(p, x_min, x_max, z_min, z_max, eps=eps)

    # For every adjacent event pair, test midpoint — if midpoint lies inside rect,
    # the interval between those two events is an "inside" run.
    for i in range(len(s_ext) - 1):
        s0, s1 = s_ext[i], s_ext[i+1]
        smid = 0.5 * (s0 + s1)
        pmid = point_on_poly(V, smid)
        if rect_contains_local(pmid):
            intervals.append((p_ext[i].copy(), p_ext[i+1].copy()))

    return intervals

def longest_inside_interval_on_boundary(bdry_verts, x_min, x_max, z_min, z_max, eps=1e-14):
    """
    Return endpoints (p0,p1) of the *longest contiguous boundary interval*
    that lies inside the rectangle, based on consecutive boundary/rect intersection events.
    """
    V = np.asarray(bdry_verts, float)
    n = len(V)
    if n < 2:
        return None, None

    # rectangle edges
    R = np.array([[x_min,z_min],[x_max,z_min],[x_max,z_max],[x_min,z_max]], float)
    rect_edges = [(R[i], R[(i+1)%4]) for i in range(4)]

    # precompute edge lengths and cumulative arclength along polygon
    elen = np.linalg.norm(np.roll(V,-1,axis=0) - V, axis=1)  # length of edge k

    def arc_len(s0, s1):
        """forward arclength from s0 to s1 where s1>=s0, allowing wrap via +n."""
        k0 = int(np.floor(s0))
        t0 = s0 - k0
        k1 = int(np.floor(s1))
        t1 = s1 - k1
        # same edge
        if k0 == k1:
            return abs(t1 - t0) * elen[k0 % n]
        L = (1.0 - t0) * elen[k0 % n]
        # full edges between
        for k in range(k0+1, k1):
            L += elen[k % n]
        # last partial
        L += t1 * elen[k1 % n]
        return L

    # collect intersection "events" as (s_param, point)
    events = []
    for k in range(n):
        p0 = V[k]; p1 = V[(k+1)%n]
        p = p0; r = p1 - p0
        for (q0,q1) in rect_edges:
            q = q0; s = q1 - q0
            hit, t = seg_seg_intersection_param(p, r, q, s, eps=eps)
            if hit:
                sp = k + t
                events.append((sp, p0 + t*r))

    if len(events) < 2:
        return None, None

    # sort and dedupe by s (corner hits create duplicates)
    events.sort(key=lambda x: x[0])
    s_list = [events[0][0]]
    p_list = [events[0][1]]
    for sp, pt in events[1:]:
        if abs(sp - s_list[-1]) > 1e-10:
            s_list.append(sp); p_list.append(pt)
        else:
            p_list[-1] = 0.5*(p_list[-1] + pt)

    if len(s_list) < 2:
        return None, None

    # append wrap event
    s_ext = s_list + [s_list[0] + n]
    p_ext = p_list + [p_list[0].copy()]

    bestL = -1.0
    best = None

    for i in range(len(s_ext)-1):
        s0, s1 = s_ext[i], s_ext[i+1]
        smid = 0.5*(s0 + s1)
        pmid = point_on_poly(V, smid)
        if rect_contains(pmid, x_min, x_max, z_min, z_max, eps=eps):
            L = arc_len(s0, s1)
            if L > bestL:
                bestL = L
                best = (p_ext[i].copy(), p_ext[i+1].copy())

    return best if best is not None else (None, None)

#TODO: Go through new code compared to old function and figure out what should be cleaned up from ChatGPT...
#TODO: Update volume calculations to use Divergence Theorem and surface fluxes. Can test 3d volume this way.
#TODO: For prev line, add cell intersections to polygon and use linear bdy approximation for volume and face factors.
#TODO: Do interpolation at just far point from boundary normal, then midpt rule to get middle point?
def compute_cutcell_fractions_with_bound(
    xc, zc, dx, dz, bdry: bdy.PolygonBoundary):
    """
    Pure (R,Z) version.

    Returns:
      (vol_frac, fx_plus_frac, fz_plus_frac) if return_geom==False
      (vol_frac, fx_plus_frac, fz_plus_frac, geom) if return_geom==True

    geom includes EB midpoint/normal and bilinear interpolation weights for two
    points along inward normal for each CUT cell.
    """
    import numpy as np

    nx = len(xc)
    nz = len(zc)

    vol_frac     = np.zeros((nx, nz), dtype=float)
    fx_plus_frac = np.zeros((nx, nz), dtype=float)  # padded to (nx,nz)
    fz_plus_frac = np.zeros((nx, nz), dtype=float)

    # ---- geometry outputs ----
    bound_id = -np.ones((nx, nz), dtype=np.int32)
    cc_mask = np.zeros((nx, nz), float)

    # Debug arrays for plotting
    eb_p0x = np.zeros((nx, nz), float); eb_p0z = np.zeros((nx, nz), float)
    eb_p1x = np.zeros((nx, nz), float); eb_p1z = np.zeros((nx, nz), float)
    eb_mx  = np.zeros((nx, nz), float); eb_mz  = np.zeros((nx, nz), float)
    eb_nx  = np.zeros((nx, nz), float); eb_nz  = np.zeros((nx, nz), float)
    eb_len = np.zeros((nx, nz), float)

    pAx = np.zeros((nx, nz), float); pAz = np.zeros((nx, nz), float)
    pBx = np.zeros((nx, nz), float); pBz = np.zeros((nx, nz), float)

    # packed lists (nb-long)
    mid_pts = []
    norms = []
    base_inds = []
    weights = []
    ds_list = []
    cell_inds = []
    segAs = []

    b = 0  # running boundary-cut index
    h = np.hypot(dx, dz) #Normal sampling distance for midpoints.
    s1, s2 = 0.5*h, h

    # bbox for quick cell classification
    px_min, pz_min = bdry.path.vertices.min(axis=0)
    px_max, pz_max = bdry.path.vertices.max(axis=0)
    bbox = (px_min, px_max, pz_min, pz_max)

    for i in range(nx):
        x_min = xc[i] - 0.5 * dx
        x_max = xc[i] + 0.5 * dx
        for j in range(nz):
            z_min = zc[j] - 0.5 * dz
            z_max = zc[j] + 0.5 * dz

            cls = classify_cell(bdry, x_min, x_max, z_min, z_max, bbox)

            if cls == OUTSIDE:
                continue

            if cls == INSIDE:
                vol_frac[i, j] = 1.0
                fx_plus_frac[i, j] = 1.0
                fz_plus_frac[i, j] = 1.0
                continue

            # ---- CUT cell volume + face fractions ----
            cell_poly = clip_polygon_to_rect(
                bdry.path.vertices, x_min, x_max, z_min, z_max
            )
            A = polygon_area(cell_poly)
            vol_frac[i, j] = A / (dx * dz)

            if i < nx - 1:
                fx_plus_frac[i, j] = face_fraction_plus_x(
                    cell_poly, x_max, z_min, z_max, dz
                )
            else:
                fx_plus_frac[i, j] = 0.0

            if j < nz - 1:
                fz_plus_frac[i, j] = face_fraction_plus_z(
                    cell_poly, x_min, x_max, z_max, dx
                )
            else:
                fz_plus_frac[i, j] = 0.0

            # ---- EB endpoints inside this cell ----
            #hits = cell_boundary_intersections(
            #    bdry.path.vertices, x_min, x_max, z_min, z_max
            #)
            #p0, p1 = furthest_pair(hits)
            #p0, p1 = eb_endpoints_longest_run(bdry.path.vertices, x_min, x_max, z_min, z_max)
            segments = longest_inside_intervals_on_boundary(
            bdry.path.vertices, x_min, x_max, z_min, z_max)

            if not segments:
                # no (robust) EB segment inside this cell
                continue
            
            # segments is a list of (p0, p1); process each separately
            for (p0, p1) in segments:
                seg = p1 - p0
                segL = np.linalg.norm(seg)
                if segL < 1e-14:
                    continue

                mid = 0.5 * (p0 + p1)

                # normal from segment tangent
                t = seg / segL
                n = np.array([-t[1], t[0]], float)  # candidate normal (not yet oriented)

                # orient to point inward (into plasma) using a finite step (more robust than tiny epsilon)
                pA_test = mid + s1 * n
                pB_test = mid + s2 * n
                #TODO: Note there is a small gap b/w real boundary and approximate boundary.
                #To be robust, test with small normal perturbation for inside approximate boundary after updating cut cell logic.
                if (not bdry.contains(pA_test[0], pA_test[1])):
                    n = -n

                # sample points along inward normal in RZ
                pA = mid + s1 * n
                pB = mid + s2 * n

                if ((not bdry.contains(pA[0], pA[1])) or \
                    (not bdry.contains(pB[0], pB[1]))): 
                    utils.logger.warn("Points along boundary normal not outside. Indicates resolution too low."
                                      f" For boundary midpoint located at {mid[0], mid[1]}")

                # bilinear interpolation weights in RZ (assumes xc/zc are uniform grids in RZ)
                ai0, aj0, aw00, aw01, aw10, aw11 = bilinear_base_and_weights(
                    pA, xc, zc, dx, dz)
                bi0, bj0, bw00, bw01, bw10, bw11 = bilinear_base_and_weights(
                    pB, xc, zc, dx, dz)

                # store debug arrays
                #TODO: Store more like mid_pts below. Dont need so many structures.
                eb_p0x[i, j], eb_p0z[i, j] = p0
                eb_p1x[i, j], eb_p1z[i, j] = p1
                eb_mx[i, j],  eb_mz[i, j]  = mid
                eb_nx[i, j],  eb_nz[i, j]  = n
                eb_len[i, j] = segL

                pAx[i, j], pAz[i, j] = pA
                pBx[i, j], pBz[i, j] = pB

                # record boundary cut index
                bound_id[i, j] = b
                cc_mask[i, j] = 1.0

                mid_pts.append([mid[0], mid[1]])
                norms.append([n[0], n[1]])
                base_inds.append([ai0, aj0, bi0, bj0])

                # Pack weights: A then B. (Keep your preferred ordering consistent with C++)
                # Here: w00,w01,w10,w11 for each point
                weights.append([aw00, aw01, aw10, aw11,
                                bw00, bw01, bw10, bw11])

                ds_list.append(s2 - s1)
                #Convert length to area by multiplying by R factor at midpoint (x location)
                segAs.append(segL*mid[0])
                cell_inds.append([i, j])

                b += 1

    # Clip fractions to [0,1]
    vol_frac = clip_cuts(vol_frac)
    fx_plus_frac = clip_cuts(fx_plus_frac)
    fz_plus_frac = clip_cuts(fz_plus_frac)

    # Safety hack you had: avoid divide-by-zero later
    vol_eps = 1e-12
    out_mask = vol_frac < vol_eps
    vol_frac[out_mask] = 1.0

    #TODO: Throw res. warning if either two points outside boundary...
    if utils.DEBUG_FLAG:
        print("Plotting cut cell volume fractions and faces...")
        plot_volume_fractions(xc, zc, dx, dz, vol_frac, bdry)
        plot_cut_faces(
            xc, zc, dx, dz, fx_plus_frac, fz_plus_frac, bdry,
            cc_mask=cc_mask,
            eb_p0x=eb_p0x, eb_p0z=eb_p0z, eb_p1x=eb_p1x, eb_p1z=eb_p1z,
            eb_mx=eb_mx, eb_mz=eb_mz, eb_nx=eb_nx, eb_nz=eb_nz,
            pAx=pAx, pAz=pAz, pBx=pBx, pBz=pBz)

    geom = {
        "bound_id": bound_id,
        "nb": b,
        "mid_pts": np.asarray(mid_pts, float),     # (nb,2) in RZ
        "bnorms":  np.asarray(norms, float),       # (nb,2) in RZ (unit-ish; depends on cell scaling)
        "bd_area":  np.asarray(segAs, float),       # (nb,) in RZ
        "s1": float(s1),
        "s2": float(s2),
        "ds": np.asarray(ds_list, float),
        "bweights": np.asarray(weights, float),    # (nb,8) A(4) + B(4)
        "base_inds": np.asarray(base_inds, float), # (nb,4)
        "cell_inds": np.asarray(cell_inds, int),   # (nb,2)
        "cc_mask": cc_mask,
    }

    return vol_frac, fx_plus_frac, fz_plus_frac, geom

def clip_cuts(arr, clip_eps=1e-12):
    """
    Clip face and volume fractions to 0.0 or 1.0 if w/in tolerance to avoid numerical issues.
    """
    arr = np.asarray(arr)

    #TODO: Do clipping based on tolerance class?
    arr [arr < clip_eps] = 0.0
    arr[arr > 1.0 - clip_eps] = 1.0
    arr = np.clip(arr, 0.0, 1.0)

    return arr

def plot_volume_fractions(xc, zc, dx, dz, vol_frac, bdry: bdy.PolygonBoundary = None):
    """
    Visualize cell volume fractions as a 2D image over the physical grid.

    xc, zc   : 1D arrays of cell centers
    dx, dz   : scalar cell sizes
    vol_frac : (nx, nz) array in [0,1]
    bdry : optional wall boundary for plotting.
    """
    xc = np.asarray(xc)
    zc = np.asarray(zc)
    vol_frac = np.asarray(vol_frac)

    nx, nz = vol_frac.shape
    assert nx == len(xc) and nz == len(zc)

    # Cell edges (for pcolormesh)
    x_edges = np.concatenate(([xc[0] - 0.5*dx],
                              0.5*(xc[:-1] + xc[1:]),
                              [xc[-1] + 0.5*dx]))
    z_edges = np.concatenate(([zc[0] - 0.5*dz],
                              0.5*(zc[:-1] + zc[1:]),
                              [zc[-1] + 0.5*dz]))

    X, Z = np.meshgrid(x_edges, z_edges, indexing='ij')

    fig, ax = plt.subplots()
    # pcolormesh expects (nx+1, nz+1) edges and (nx, nz) values
    cmap = plt.get_cmap('viridis')
    im = ax.pcolormesh(X, Z, vol_frac, cmap=cmap, shading='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Volume fraction")

    wall = PathPatch(bdry.path, fill=False, edgecolor='k', label='ghost',
                clip_on=False, lw=1.5, linestyle='--')
    ax.add_patch(wall)

    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Cell Volume Fractions")
    plt.tight_layout()
    plt.show()

def plot_cut_faces(xc, zc, dx, dz, fx_plus_frac, fz_plus_frac,
                   bdry: bdy.PolygonBoundary, eps=1e-12,
                   cc_mask=None,
                   eb_p0x=None, eb_p0z=None, eb_p1x=None, eb_p1z=None,
                   eb_mx=None, eb_mz=None, eb_nx=None, eb_nz=None,
                   pAx=None, pAz=None, pBx=None, pBz=None):
    """
    Visualize cut-cell faces:

        Blue (thin)   = fully inside (frac ~ 1)
        Gray (thin)   = fully outside (frac ~ 0)
        Green (thick) = inside portion of a cut face
        Red (thick)   = outside portion of a cut face

    For cut faces (0 < frac < 1), we:
      - check both endpoints of the face against bdry.contains(x,z)
      - if exactly one endpoint is inside, we attach the 'frac' portion
        of the face to that endpoint as the green (inside) side
      - if both or neither are inside, we fall back to assuming inside
        is attached to the lower/left endpoint.
    """
    xc = np.asarray(xc)
    zc = np.asarray(zc)
    fx_plus_frac = np.asarray(fx_plus_frac, float)
    fz_plus_frac = np.asarray(fz_plus_frac, float)

    nx = len(xc)
    nz = len(zc)
    #TODO: Does it matter that these are padded for the loops below?
    #assert fx_plus_frac.shape == (nx-1, nz)
    #assert fz_plus_frac.shape == (nx, nz-1)

    # --- snap near-0 / near-1 for cleaner logic ---
    def snap(f):
        f = np.array(f, float)
        f[np.abs(f) < eps] = 0.0
        f[np.abs(f - 1.0) < eps] = 1.0
        return np.clip(f, 0.0, 1.0)

    fx_plus_frac = snap(fx_plus_frac)
    fz_plus_frac = snap(fz_plus_frac)

    # --- cell edges ---
    x_edges = np.concatenate((
        [xc[0] - 0.5*dx],
        0.5*(xc[:-1] + xc[1:]),
        [xc[-1] + 0.5*dx]
    ))
    z_edges = np.concatenate((
        [zc[0] - 0.5*dz],
        0.5*(zc[:-1] + zc[1:]),
        [zc[-1] + 0.5*dz]
    ))

    fig, ax = plt.subplots()

    # light grid
    for x in x_edges:
        ax.plot([x, x], [z_edges[0], z_edges[-1]], color='0.90', lw=0.5, zorder=0)
    for z in z_edges:
        ax.plot([x_edges[0], x_edges[-1]], [z, z], color='0.90', lw=0.5, zorder=0)

    # boundary
    wall = PathPatch(
        bdry.path, fill=False, edgecolor='k',
        label='boundary', clip_on=False, lw=1.5, linestyle='--'
    )
    ax.add_patch(wall)

    blue_segments  = []  # full inside
    gray_segments  = []  # full outside
    green_segments = []  # inside part of cut face
    red_segments   = []  # outside part of cut face

    # --- X-FACES (vertical): +x faces of cell (i,j) ---
    for i in range(nx-1):
        x_face = x_edges[i+1]
        for j in range(nz):
            z0 = z_edges[j]
            z1 = z_edges[j+1]
            frac = fx_plus_frac[i, j]

            if frac == 0.0:
                # fully outside
                gray_segments.append([(x_face, z0), (x_face, z1)])
            elif frac == 1.0:
                # fully inside
                blue_segments.append([(x_face, z0), (x_face, z1)])
            else:
                # cut face
                full_len = z1 - z0
                inside_len = frac * full_len

                # check endpoints
                low_inside  = bdry.contains(x_face, z0)
                high_inside = bdry.contains(x_face, z1)

                if low_inside and not high_inside:
                    # inside attached to lower endpoint
                    z_split = z0 + inside_len
                    green_segments.append([(x_face, z0), (x_face, z_split)])
                    red_segments.append([(x_face, z_split), (x_face, z1)])
                elif high_inside and not low_inside:
                    # inside attached to upper endpoint
                    z_split = z1 - inside_len
                    green_segments.append([(x_face, z_split), (x_face, z1)])
                    red_segments.append([(x_face, z0), (x_face, z_split)])
                else:
                    # ambiguous: fall back to lower-end convention
                    z_split = z0 + inside_len
                    green_segments.append([(x_face, z0), (x_face, z_split)])
                    red_segments.append([(x_face, z_split), (x_face, z1)])

    # --- Z-FACES (horizontal): +z faces of cell (i,j) ---
    for i in range(nx):
        for j in range(nz-1):
            z_face = z_edges[j+1]
            x0 = x_edges[i]
            x1 = x_edges[i+1]
            frac = fz_plus_frac[i, j]

            if frac == 0.0:
                gray_segments.append([(x0, z_face), (x1, z_face)])
            elif frac == 1.0:
                blue_segments.append([(x0, z_face), (x1, z_face)])
            else:
                full_len = x1 - x0
                inside_len = frac * full_len

                low_inside  = bdry.contains(x0, z_face)
                high_inside = bdry.contains(x1, z_face)

                if low_inside and not high_inside:
                    # inside attached to left endpoint
                    x_split = x0 + inside_len
                    green_segments.append([(x0, z_face), (x_split, z_face)])
                    red_segments.append([(x_split, z_face), (x1, z_face)])
                elif high_inside and not low_inside:
                    # inside attached to right endpoint
                    x_split = x1 - inside_len
                    green_segments.append([(x_split, z_face), (x1, z_face)])
                    red_segments.append([(x0, z_face), (x_split, z_face)])
                else:
                    # ambiguous: fall back to left-end convention
                    x_split = x0 + inside_len
                    green_segments.append([(x0, z_face), (x_split, z_face)])
                    red_segments.append([(x_split, z_face), (x1, z_face)])

    # --- draw segments ---
    if gray_segments:
        ax.add_collection(LineCollection(gray_segments, colors='gray', lw=0.7, zorder=2))
    if blue_segments:
        ax.add_collection(LineCollection(blue_segments, colors='blue', lw=0.7, zorder=3))
    if green_segments:
        ax.add_collection(LineCollection(green_segments, colors='green', lw=2.0, zorder=4))
    if red_segments:
        ax.add_collection(LineCollection(red_segments, colors='red', lw=2.0, zorder=4))

    ax.set_aspect('equal')
    ax.set_xlabel("x (logical)")
    ax.set_ylabel("z (logical)")
    ax.set_title("Cut Faces")

    # --- NEW: draw EB segment in each cut cell + normal + sample points ---
    if cc_mask is not None:
        cc_mask = np.asarray(cc_mask)

        eb_segments = []
        mids = []
        norms = []
        ptsA = []
        ptsB = []

        nx = len(xc)
        nz = len(zc)
        for i in range(nx):
            for j in range(nz):
                if cc_mask[i, j] != 1:
                    continue

                # EB segment
                if eb_p0x is not None:
                    p0 = (eb_p0x[i, j], eb_p0z[i, j])
                    p1 = (eb_p1x[i, j], eb_p1z[i, j])
                    eb_segments.append([p0, p1])

                # midpoint + normal
                if eb_mx is not None:
                    mids.append((eb_mx[i, j], eb_mz[i, j]))
                    norms.append((eb_nx[i, j], eb_nz[i, j]))

                # sample points
                if pAx is not None:
                    ptsA.append((pAx[i, j], pAz[i, j]))
                if pBx is not None:
                    ptsB.append((pBx[i, j], pBz[i, j]))

        if eb_segments:
            ax.add_collection(LineCollection(eb_segments, colors='magenta', lw=2.0, zorder=6))

        # normal arrows
        if mids and norms:
            mids = np.asarray(mids, float)
            norms = np.asarray(norms, float)

            # pick an arrow scale that looks reasonable
            arrow_len = 0.35 * min(dx, dz)
            ax.quiver(
                mids[:, 0], mids[:, 1],
                norms[:, 0], norms[:, 1],
                angles='xy', scale_units='xy', scale=1.0/arrow_len,
                width=0.003, zorder=7
            )

        # sample points
        if ptsA:
            ptsA = np.asarray(ptsA, float)
            ax.scatter(ptsA[:, 0], ptsA[:, 1], s=18, marker='o', color='cyan', zorder=8, label='interp A')
        if ptsB:
            ptsB = np.asarray(ptsB, float)
            ax.scatter(ptsB[:, 0], ptsB[:, 1], s=18, marker='x', color='yellow', zorder=8, label='interp B')

    # Legend along the top
    legend_lines = [
        Line2D([0], [0], color='blue',  lw=0.7, label='inside (full)'),
        Line2D([0], [0], color='gray',  lw=0.7, label='outside (full)'),
        Line2D([0], [0], color='green', lw=2.0, label='inside (cut)'),
        Line2D([0], [0], color='red',   lw=2.0, label='outside (cut)'),
    ]
    ax.legend(handles=legend_lines,
              loc='upper center',
              bbox_to_anchor=(0.5, 1.15),
              ncol=4,
              frameon=False)

    plt.tight_layout()
    plt.show()
