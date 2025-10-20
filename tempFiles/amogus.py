#TODO: Unnecessary test case.

def amogus(size=68, scale=1.0):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    xs = np.linspace(-1,1,size)
    zs = np.linspace(-1,1,size)
    X,Z = np.meshgrid(xs,zs)

    height = np.zeros((size,size), dtype=float)

    body = (((X-0.05)/0.45)**2 + ((Z + 0.0)/0.6)**2) <= 1.0
    height[body] = 1.0

    backpack = (((X + 0.5)/ 0.22)**2 + ((Z + 0.0)/0.42)**2) <= 1.0
    height[backpack] = 1.0

    visor = (((X-0.25)/0.32)**2 + ((Z-0.02)/0.16)**2) <= 1.0
    height[visor] = 1.0

    leg1 = (((X+0.18)/0.18)**2 + ((Z-0.65)/0.15)**2) <= 1.0
    leg2 = (((X-0.38)/0.18)**2 + ((Z-0.65)/0.15)**2) <= 1.0
    height[leg1] = 1.0
    height[leg2] = 1.0

    height *= scale
    height = np.flipud(height)
    
    level = 0.0  # pick your isovalue
    fig, ax = plt.subplots()
    cs = ax.contour(height, levels=[level]) # creates a QuadContourSet
    plt.close(fig)  # we only needed it to compute contours

    segs = cs.allsegs[0]  # list of (N,2) arrays for this level
    paths = []
    for seg in segs:
        # ensure closure
        if not np.allclose(seg[0], seg[-1]):
            seg = np.vstack([seg, seg[0]])
        codes = np.full(len(seg), Path.LINETO, dtype=np.uint8)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        paths.append(Path(seg, codes))

    chunks = []
    for p in paths:
        if p.codes is None:
            chunks.append(p.vertices)
        else:
            # drop CLOSEPOLY rows (their vertex is a placeholder)
            chunks.append(p.vertices[p.codes != Path.CLOSEPOLY])

    return np.vstack(chunks) if chunks else np.empty((0, 2))

