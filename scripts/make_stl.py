"""
make_stl.py
===========
Single-tile STL generator with a clean CLI — the bridge the Fusion add-in calls.

The Fusion 360 add-in (fusion_addin/HelmetLattice) can't run numpy/scikit-image/
trimesh inside Fusion's bundled Python, so it shells out to *this* script using the
system Python (which has the packages) and then imports the STL it writes.

Examples
--------
    python scripts/make_stl.py --topology gyroid --cell 8 --density 0.27 --out tile.stl
    python scripts/make_stl.py --topology graded --density 0.27 --gradient 0.17 --out g.stl
    python scripts/make_stl.py --topology auxetic --cell 8 --density 0.29 --out a.stl

Sizes default to the 2 x 2 x 1.5 in coupon; override with --size-x/y/z (mm).
"""

from __future__ import annotations
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_tiles as gen   # noqa: E402


def _parse_blend(spec):
    """'gyroid:0.6,schwartz:0.4' -> {'gyroid':0.6,'schwartz':0.4}."""
    if not spec:
        return {"gyroid": 0.5, "schwartz": 0.5}
    out = {}
    for part in spec.split(","):
        f = [x.strip() for x in part.split(":")]
        out[f[0]] = float(f[1]) if len(f) > 1 and f[1] else 1.0
    return out


def _parse_layers(spec, default_rho, default_cell, total_z):
    """'gyroid:0.25:8,schwartz:0.30:6' -> [(topo, rho, cell, thickness), ...].

    density/cell are optional per layer; thickness is split equally across total_z.
    Empty spec -> a 2-layer gyroid+schwartz stack at the global density/cell.
    """
    if spec:
        items = []
        for part in spec.split(","):
            f = [x.strip() for x in part.split(":")]
            topo = f[0]
            rho = float(f[1]) if len(f) > 1 and f[1] else default_rho
            cell = float(f[2]) if len(f) > 2 and f[2] else default_cell
            items.append((topo, rho, cell))
    else:
        items = [("gyroid", default_rho, default_cell),
                 ("schwartz", default_rho, default_cell)]
    n = len(items)
    return [(t, r, c, total_z / n) for (t, r, c) in items]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate one watertight lattice tile STL.")
    ap.add_argument("--topology", required=True,
                    choices=["gyroid", "schwartz", "diamond", "graded", "auxetic",
                             "alternating", "blend"])
    ap.add_argument("--layers", default=None,
                    help="alternating layers as 'topo:density:cell,topo:density:cell' "
                         "(equal thickness across --size-z); density/cell optional. "
                         "Defaults to a 2-layer gyroid+schwartz stack.")
    ap.add_argument("--blend", default=None,
                    help="GENERATIVE blend weights over the TPMS basis, e.g. "
                         "'gyroid:0.6,schwartz:0.4' or 'gyroid:0.5,diamond:0.5'. "
                         "Produces a new in-between geometry (use --topology blend).")
    ap.add_argument("--cell", type=float, default=8.0, help="cell size / pitch in X (mm)")
    ap.add_argument("--cell-aspect-y", type=float, default=1.0, dest="aspect_y",
                    help="cell size ratio y/x (1 = cubic; non-cubic -> anisotropy)")
    ap.add_argument("--cell-aspect-z", type=float, default=1.0, dest="aspect_z",
                    help="cell size ratio z/x (1 = cubic; affects directional compression)")
    ap.add_argument("--density", type=float, default=0.25, help="target relative density")
    ap.add_argument("--gradient", type=float, default=0.0,
                    help="graded: stiff-minus-soft spread in relative density")
    ap.add_argument("--pitch", type=float, default=gen.VOXEL_PITCH_MM,
                    help="marching-cubes voxel pitch (mm); smaller = finer + slower")
    ap.add_argument("--size-x", type=float, default=gen.TILE_X, help="tile X (mm)")
    ap.add_argument("--size-y", type=float, default=gen.TILE_Y, help="tile Y (mm)")
    ap.add_argument("--size-z", type=float, default=gen.TILE_Z, help="tile Z (mm)")
    ap.add_argument("--out", required=True, help="output STL path")
    a = ap.parse_args()

    # generate_tiles reads these as module globals at call time, so overriding them
    # here re-targets the tile size. RESOLUTION/tz are baked into default args, so we
    # pass res and tz explicitly below.
    gen.TILE_X, gen.TILE_Y, gen.TILE_Z = a.size_x, a.size_y, a.size_z
    gen.TILE_AREA_MM2 = a.size_x * a.size_y
    res = max(16, round(a.size_x / a.pitch))

    # Per-axis (anisotropic / non-cubic) cell: x, y=x*aspect_y, z=x*aspect_z.
    cellv = (a.cell, a.cell * a.aspect_y, a.cell * a.aspect_z)

    topo = a.topology
    if topo == "blend":
        weights = _parse_blend(a.blend)
        mesh, mass = gen.generate_blended_tile(weights, cellv, a.density,
                                               tz=a.size_z, res=res)

    elif topo in ("gyroid", "schwartz", "diamond"):
        fn = gen.TOPOLOGY_FN[topo]
        mesh, mass = gen.generate_tile(fn, cellv, a.density, tz=a.size_z, res=res)

    elif topo == "graded":
        soft = max(0.10, a.density - a.gradient / 2.0)
        stiff = min(0.55, a.density + a.gradient / 2.0)
        mesh, mass = gen.generate_graded_tile(cell_size=cellv, rho_soft=soft,
                                              rho_stiff=stiff, res=res)

    elif topo == "alternating":
        layers = _parse_layers(a.layers, a.density, a.cell, a.size_z)
        mesh, mass = gen.generate_stacked_tile(layers, res=res)

    else:  # auxetic — perforated sheet: solid sheet with square holes
        mesh, mass, rho = gen.generate_rotating_squares(
            tile_x=a.size_x, tile_y=a.size_y, tile_z=a.size_z,
            cell=a.cell, rel_density=a.density, res=res)

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mesh.export(out)
    cells = mesh.metadata.get("cell_sizes_mm")
    counts = mesh.metadata.get("cell_counts")
    # Single machine-readable line the add-in parses; keep the "OK " prefix stable.
    print(f"OK {out} mass_g={mass:.1f} watertight={mesh.is_watertight} "
          f"verts={len(mesh.vertices)} res={res} "
          f"cells={counts} cell_mm={cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
