"""
make_print_plates.py
====================
Build ready-to-slice PRINT PLATES: 5 copies of each TPU characterization
specimen, laid out non-overlapping, one STL per (type, thickness). Drop a file
into the slicer and print a full set of 5 in one job.

Covers the two thicknesses you settled on (2 mm and 3 mm) x the three
hyperelastic modes (Test 1 dogbone, Test 2 planar, Test 3 equibiaxial disk).
The single-specimen generators (make_dogbone.py, make_material_specimens.py)
stay the source of geometry; this just tiles them.

Layout note: copies are gridded with an 8 mm gap. Specimen positions don't have
to be perfect — every slicer auto-arranges on import (use "Arrange"); on a small
bed the disc plates (120 mm each) will spill onto a second plate, which the
slicer handles. The footprint printed per file tells you if it fits your bed.

Usage:
    python scripts/make_print_plates.py                 # all 6 plates -> stl_output/
    python scripts/make_print_plates.py --count 6       # 6 per plate instead of 5
    python scripts/make_print_plates.py --thickness 2   # just the 2 mm plates
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_dogbone import build_dogbone                       # noqa: E402
from make_material_specimens import build_planar, build_biaxial  # noqa: E402

# type -> (build fn, grid columns, file stem). Columns chosen to keep each
# plate's footprint as bed-friendly as the part allows.
TYPES = {
    "dogbone": (build_dogbone, 1, "PLATE_dogbone"),   # long+thin -> single column
    "planar":  (build_planar,  2, "PLATE_planar"),
    "biaxial": (build_biaxial, 3, "PLATE_biaxial"),
}
GAP_MM = 8.0


def make_plate(build_fn, thickness: float, n: int, cols: int) -> trimesh.Trimesh:
    """Tile n copies of one specimen into a grid; return one combined mesh."""
    proto = build_fn(thickness)
    ext = proto.bounds[1] - proto.bounds[0]
    dx, dy = ext[0] + GAP_MM, ext[1] + GAP_MM

    parts = []
    for i in range(n):
        row, col = divmod(i, cols)
        m = proto.copy()
        m.apply_translation([col * dx, row * dy, 0.0])
        parts.append(m)
    return trimesh.util.concatenate(parts)


def main():
    ap = argparse.ArgumentParser(description="Tile characterization specimens into print plates.")
    ap.add_argument("--count", type=int, default=5, help="copies per plate (default 5)")
    ap.add_argument("--thickness", type=float, default=None,
                    help="only this thickness (default: both 2 and 3)")
    args = ap.parse_args()

    thicknesses = [args.thickness] if args.thickness is not None else [2.0, 3.0]
    out_dir = Path(__file__).resolve().parents[1] / "stl_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in thicknesses:
        for name, (fn, cols, stem) in TYPES.items():
            plate = make_plate(fn, t, args.count, cols)
            out = out_dir / f"{stem}_x{args.count}_t{t:g}mm.stl"
            plate.export(out)
            ext = plate.bounds[1] - plate.bounds[0]
            print(f"{out.name:36s}  footprint {ext[0]:6.1f} x {ext[1]:6.1f} mm  "
                  f"({args.count} x {name}, watertight={plate.is_watertight})")


if __name__ == "__main__":
    main()
