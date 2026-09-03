"""
make_dogbone.py
===============
Generate a dimensionally-correct ASTM D412 Die C tension dogbone as an STL.

This is the UNIAXIAL TENSION specimen for material characterization (Test 1 of
the three hyperelastic modes) — a SOLID specimen of the base TPU, NOT a lattice.
Print one set per TPU type you're differentiating, pull on the UTM at 500 mm/min.

ASTM D412 Die C controlled dimensions (mm):
    overall length      115
    width at grip ends   25
    narrow gauge width    6.0   <- the dimension that sets your test force
    narrow gauge length  33
    transition radii     14 (inner, concave) + 25 (outer, convex), tangent S-curve
Thickness is yours to choose (default 3 mm; use what you can print flat and flat-
faced consistently).

Usage:
    python scripts/make_dogbone.py                      # 3 mm thick -> stl_output/
    python scripts/make_dogbone.py --thickness 2.0
    python scripts/make_dogbone.py --out path/to/file.stl
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import trimesh

# Die C controlled dimensions (mm)
L_TOTAL = 115.0
W_END = 25.0
W_GAUGE = 6.0
L_GAUGE = 33.0
R_INNER = 14.0          # concave fillet leaving the gauge section
R_OUTER = 25.0          # convex fillet meeting the grip tab


def half_profile(n_arc: int = 80) -> np.ndarray:
    """Top boundary points (x, y) for x in [0, L_TOTAL/2], y = half-width.

    Built as: straight gauge -> tangent S-curve (R_INNER then R_OUTER) -> straight
    grip tab. The two-arc S-curve is the standard way Die C flares the 6 mm neck
    out to the 25 mm ends with matched (horizontal) tangents at both ends.
    """
    yg, ye = W_GAUGE / 2.0, W_END / 2.0          # 3.0, 12.5
    dy = ye - yg                                  # 9.5
    # Common turn angle of both arcs: rise = (r1+r2)(1-cos phi).
    phi = np.arccos(1.0 - dy / (R_INNER + R_OUTER))
    x_g = L_GAUGE / 2.0                           # 16.5: gauge half-length

    # Gauge: straight at y = yg.
    gauge = np.array([[0.0, yg], [x_g, yg]])

    # Arc 1 (concave): centre directly above the gauge end, sweep 0..phi.
    c1 = np.array([x_g, yg + R_INNER])
    t1 = np.linspace(0.0, phi, n_arc)
    arc1 = np.column_stack([c1[0] + R_INNER * np.sin(t1),
                            c1[1] - R_INNER * np.cos(t1)])

    # Arc 2 (convex): tangent to arc 1, sweeps phi..0 to land horizontal at y = ye.
    p1 = arc1[-1]
    c2 = np.array([p1[0] + R_OUTER * np.sin(phi), p1[1] - R_OUTER * np.cos(phi)])
    t2 = np.linspace(phi, 0.0, n_arc)
    arc2 = np.column_stack([c2[0] - R_OUTER * np.sin(t2),
                            c2[1] + R_OUTER * np.cos(t2)])

    # Straight grip tab out to the end.
    x_tab = c2[0]
    tab = np.array([[x_tab, ye], [L_TOTAL / 2.0, ye]])

    pts = np.vstack([gauge, arc1, arc2, tab])
    # Drop duplicate seam points where segments meet.
    keep = np.concatenate([[True], np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)])
    return pts[keep]


def stations() -> np.ndarray:
    """Ordered half-width y at x-stations from -L/2 to +L/2 (symmetric)."""
    half = half_profile()
    left = half[::-1].copy()
    left[:, 0] *= -1.0                            # mirror to negative x
    full = np.vstack([left[:-1], half])           # drop the shared x=0 point once
    # ensure strictly increasing x for the loft
    order = np.argsort(full[:, 0])
    return full[order]


def build_dogbone(thickness: float) -> trimesh.Trimesh:
    """Extrude the Die C cross-section to a watertight solid (loft of rectangles)."""
    st = stations()
    n = len(st)
    t = float(thickness)

    # 4 vertices per station: (-y,0)(+y,0)(+y,t)(-y,t) in y-z, x = station x.
    V = []
    for x, y in st:
        V.extend([[x, -y, 0.0], [x, y, 0.0], [x, y, t], [x, -y, t]])
    V = np.asarray(V, dtype=float)

    F = []
    for i in range(n - 1):
        a, b, c, d = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3            # station i
        a2, b2, c2, d2 = 4 * (i + 1), 4 * (i + 1) + 1, 4 * (i + 1) + 2, 4 * (i + 1) + 3
        quads = [(a, a2, b2, b),   # z = 0 face
                 (b, b2, c2, c),   # +y wall
                 (c, c2, d2, d),   # z = t face
                 (d, d2, a2, a)]   # -y wall
        for p, q, r, s in quads:
            F.append([p, q, r]); F.append([p, r, s])

    # End caps (rectangles at first and last station).
    a, b, c, d = 0, 1, 2, 3
    F.append([a, c, b]); F.append([a, d, c])                          # x = -L/2
    a, b, c, d = 4 * (n - 1), 4 * (n - 1) + 1, 4 * (n - 1) + 2, 4 * (n - 1) + 3
    F.append([a, b, c]); F.append([a, c, d])                          # x = +L/2

    mesh = trimesh.Trimesh(V, np.asarray(F), process=True)
    mesh.fix_normals()
    return mesh


def main():
    ap = argparse.ArgumentParser(description="ASTM D412 Die C dogbone STL generator.")
    ap.add_argument("--thickness", type=float, default=3.0, help="specimen thickness mm (default 3)")
    ap.add_argument("--out", default=None, help="output STL path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else \
        root / "stl_output" / f"ASTM_D412_DieC_dogbone_t{args.thickness:g}mm.stl"
    out.parent.mkdir(parents=True, exist_ok=True)

    mesh = build_dogbone(args.thickness)
    mesh.export(out)

    ext = mesh.bounds[1] - mesh.bounds[0]
    gauge_area = W_GAUGE * args.thickness
    print(f"Saved: {out}")
    print(f"  bounding box (mm): {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.2f}  (L x W x thickness)")
    print(f"  watertight: {mesh.is_watertight}   volume: {mesh.volume/1000:.2f} cm^3")
    print(f"  gauge cross-section: {W_GAUGE:g} x {args.thickness:g} = {gauge_area:.1f} mm^2")
    print(f"  -> at e.g. 10 MPa engineering stress that's ~{gauge_area*10:.0f} N of force.")


if __name__ == "__main__":
    main()
