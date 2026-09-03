"""
make_combined_bed.py
===================
Pack chosen TPU characterization specimens onto ONE build plate so they print in
a single job. Default target is the Creality K1 Max (300 x 300 mm bed). Each
specimen type gets its own tidy column; columns sit side by side, centered on the
bed, to keep travel moves short.

Geometry comes from the single-specimen generators (make_dogbone /
make_material_specimens); this only arranges copies.

Specimen keys: dogbone, planar, compression.
  --specimens "planar:5,compression:5"   -> 5 planar strips + 5 compression pucks

Usage:
    python scripts/make_combined_bed.py --specimens "planar:5,compression:5"
    python scripts/make_combined_bed.py --specimens "planar:5,compression:5" --thickness 2
    python scripts/make_combined_bed.py --specimens "dogbone:5,planar:5" --bed 256 256
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_dogbone import build_dogbone                                  # noqa: E402
from make_material_specimens import (                                    # noqa: E402
    build_planar, build_compression, build_cruciform,
)

GAP_MM = 8.0        # spacing between parts within a column
COL_GAP_MM = 12.0   # spacing between columns
MARGIN_MM = 5.0     # keep-out from each bed edge

# thickness-aware builders; compression ignores thickness (fixed puck height).
BUILDERS = {
    "dogbone": build_dogbone,
    "planar": build_planar,
    "compression": build_compression,
    "cruciform": build_cruciform,
}


def column(build_fn, thickness: float, n: int):
    """n copies of one specimen stacked in Y, centered on the origin."""
    proto = build_fn(thickness)
    w, h = (proto.bounds[1] - proto.bounds[0])[:2]
    total_h = n * h + (n - 1) * GAP_MM
    meshes = []
    for i in range(n):
        m = proto.copy()
        m.apply_translation([0.0, -total_h / 2 + h / 2 + i * (h + GAP_MM), 0.0])
        meshes.append(m)
    return meshes, w


def parse_specimens(spec_str: str):
    out = []
    for tok in spec_str.split(","):
        key, _, count = tok.strip().partition(":")
        key = key.strip()
        if key not in BUILDERS:
            raise SystemExit(f"unknown specimen '{key}'; choose from {list(BUILDERS)}")
        out.append((key, int(count)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Pack characterization specimens onto one bed.")
    ap.add_argument("--specimens", default="planar:5,compression:5",
                    help='e.g. "planar:5,compression:5"')
    ap.add_argument("--thickness", type=float, default=2.0,
                    help="sheet-specimen thickness mm (default 2; compression ignores it)")
    ap.add_argument("--bed", type=float, nargs=2, default=[300.0, 300.0],
                    metavar=("X", "Y"), help="bed size mm (default 300 300, K1 Max)")
    args = ap.parse_args()

    groups = parse_specimens(args.specimens)

    # Build each column, then place columns left-to-right.
    cols = [(key, n, *column(BUILDERS[key], args.thickness, n)) for key, n in groups]
    widths = [c[3] for c in cols]
    total_w = sum(widths) + COL_GAP_MM * (len(cols) - 1)

    parts = []
    x_cursor = -total_w / 2
    for (_, _, meshes, w) in cols:
        xc = x_cursor + w / 2
        for m in meshes:
            mm = m.copy(); mm.apply_translation([xc, 0.0, 0.0]); parts.append(mm)
        x_cursor += w + COL_GAP_MM

    bed = trimesh.util.concatenate(parts)
    bx, by = args.bed
    c = (bed.bounds[0] + bed.bounds[1]) / 2.0          # center the cluster on the bed
    bed.apply_translation([bx / 2 - c[0], by / 2 - c[1], 0.0])

    tag = "_".join(f"{k}{n}" for k, n in groups)
    out = (Path(__file__).resolve().parents[1] / "stl_output" /
           f"BED_K1Max_{tag}_t{args.thickness:g}mm.stl")
    bed.export(out)

    ext = bed.bounds[1] - bed.bounds[0]
    fits = ext[0] <= bx - 2 * MARGIN_MM and ext[1] <= by - 2 * MARGIN_MM
    print(f"Saved: {out.name}")
    print(f"  groups: {', '.join(f'{n}x {k}' for k, n in groups)}  ({len(parts)} parts)")
    print(f"  tallest part: {ext[2]:.1f} mm -> {ext[2]/0.2:.0f} layers at 0.2 mm")
    print(f"  footprint {ext[0]:.1f} x {ext[1]:.1f} mm on {bx:.0f} x {by:.0f} bed "
          f"-> fits with {MARGIN_MM:g} mm margin: {fits}")
    print(f"  watertight: {bed.is_watertight}")


if __name__ == "__main__":
    main()
