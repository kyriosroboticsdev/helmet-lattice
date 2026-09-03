"""
make_material_specimens.py
==========================
Generate the two remaining TPU hyperelastic characterization specimens for the
MTS-880 / material-model fit — Test 2 (planar / pure shear) and Test 3
(equibiaxial via disk-bulge). Test 1 (uniaxial tension dogbone) has its own
generator, scripts/make_dogbone.py.

Per the Testing/Hardware spec §4.1, SimScale's Ogden fit needs THREE independent
deformation modes. All three specimens are SOLID base TPU, printed thin/flat in
the actual print profile (NOT lattices):

  Test 1  Uniaxial tension   -> make_dogbone.py        (ASTM D412 Die C)
  Test 2  Planar/pure shear  -> here, --mode planar    (wide-short strip, ~10:1)
  Test 3  Equibiaxial        -> here, --mode biaxial   (flat disk, bubble-inflation)

Test 2 (planar / "pure shear"): a wide, short gauge so the clamped full-width
edges suppress lateral contraction -> the gauge sees pure shear. Spec example:
~100 mm wide x 10 mm tall gauge (10:1). Clamp the full top and bottom edges; the
extra material above/below the gauge is the clamp allowance.

Test 3 (equibiaxial / disk-bulge): a flat circular membrane clamped over a nozzle
and inflated; record pressure + dome apex height -> biaxial stress/strain. The
printed part is just a uniform flat disk larger than the rig aperture (the outer
ring is clamped, the inner circle inflates).

NOTE on thickness: the spec stresses an identical print profile across all three
modes. The existing dogbone is 3 mm; planar/biaxial want a thin uniform membrane,
so the default here is 2 mm. Set --thickness to match your profile / dogbone if
you'd rather hold thickness constant too.

Usage:
    python scripts/make_material_specimens.py                 # both, 2 mm -> stl_output/
    python scripts/make_material_specimens.py --mode planar
    python scripts/make_material_specimens.py --mode biaxial --thickness 2.5
    python scripts/make_material_specimens.py --out path/to/file.stl --mode planar
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import trimesh

# ── Test 2: planar / pure-shear strip (mm) ───────────────────────────────────
PLANAR_WIDTH = 100.0        # full specimen width (clamped full-width)
PLANAR_GAUGE_H = 10.0       # free gauge height between clamps -> 10:1 aspect
PLANAR_CLAMP_H = 15.0       # clamp allowance added top AND bottom

# ── Test 3: equibiaxial disk-bulge membrane (mm) ─────────────────────────────
BIAXIAL_DIAMETER = 120.0    # outer disk diameter (must exceed the rig aperture)
DISK_SECTIONS = 256         # circle facets (smooth clamp seal)


def build_planar(thickness: float) -> trimesh.Trimesh:
    """Flat rectangular strip: WIDTH x (gauge + 2*clamp) x thickness, z in [0,t]."""
    total_h = PLANAR_GAUGE_H + 2.0 * PLANAR_CLAMP_H
    mesh = trimesh.creation.box(extents=[PLANAR_WIDTH, total_h, thickness])
    # box() centers on origin; lift so it sits flat on the bed (z in [0, t]).
    mesh.apply_translation([0.0, 0.0, thickness / 2.0])
    mesh.fix_normals()
    return mesh


def build_biaxial(thickness: float) -> trimesh.Trimesh:
    """Flat uniform circular disk: diameter BIAXIAL_DIAMETER, z in [0,t]."""
    mesh = trimesh.creation.cylinder(
        radius=BIAXIAL_DIAMETER / 2.0, height=thickness, sections=DISK_SECTIONS
    )
    mesh.apply_translation([0.0, 0.0, thickness / 2.0])
    mesh.fix_normals()
    return mesh


# ── Test 3 (uniaxial-frame route): compression puck (mm) ─────────────────────
# On a UNIAXIAL frame (e.g. MTS 880) you get the equibiaxial data stream from
# lubricated uniaxial compression of a disc — equibiaxial tension == uniaxial
# compression for an incompressible elastomer. ASTM D575-style: Ø28.6 x 12.5 mm.
COMP_DIAMETER = 28.6
COMP_HEIGHT = 12.5


def build_compression(thickness: float | None = None) -> trimesh.Trimesh:
    """Solid cylindrical compression puck; `thickness` is ignored (fixed height)."""
    mesh = trimesh.creation.cylinder(
        radius=COMP_DIAMETER / 2.0, height=COMP_HEIGHT, sections=DISK_SECTIONS
    )
    mesh.apply_translation([0.0, 0.0, COMP_HEIGHT / 2.0])
    mesh.fix_normals()
    return mesh


# ── Test 3 (TRUE equibiaxial, needs a BIAXIAL / 4-arm machine): cruciform ─────
# Per the team's spec image: 120 mm tip-to-tip each axis, 30x30 gauge, 30 mm arms
# (45 mm each), ~5 mm inner fillets, 4-6 slits per arm ("fingers" so the arms
# spread and the centre sees even biaxial strain), 2 mm uniform.
CRUCI_SPAN = 120.0          # tip-to-tip, each axis
CRUCI_ARM_W = 30.0          # arm width (= central gauge size)
CRUCI_FILLET = 5.0          # inner re-entrant corner radius
CRUCI_N_SLITS = 5           # slits per arm -> N+1 fingers
CRUCI_SLIT_W = 1.2          # slit width (cut)
CRUCI_SLIT_BRIDGE = 10.0    # solid length near gauge before slits begin


def build_cruciform(thickness: float) -> trimesh.Trimesh:
    """Slitted, inner-filleted biaxial cruciform, extruded to `thickness`."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    hs, hw = CRUCI_SPAN / 2.0, CRUCI_ARM_W / 2.0          # 60, 15
    cross = unary_union([box(-hs, -hw, hs, hw), box(-hw, -hs, hw, hs)])

    # Fillet only the 4 re-entrant corners (morphological closing keeps arm tips square).
    R = CRUCI_FILLET
    poly = cross.buffer(R, join_style=1, quad_segs=24).buffer(-R, join_style=1, quad_segs=24)

    # Slit centrelines across the 30 mm width: N slits + (N+1) fingers.
    n, ws = CRUCI_N_SLITS, CRUCI_SLIT_W
    finger = (CRUCI_ARM_W - n * ws) / (n + 1)
    centers, pos = [], -hw + finger + ws / 2.0
    for _ in range(n):
        centers.append(pos); pos += ws + finger

    inner = hw + CRUCI_SLIT_BRIDGE                         # slits start here (toward gauge)
    outer = hs + 1.0                                       # overrun the tip for a clean cut
    slits = []
    for c in centers:
        slits += [box(inner, c - ws / 2, outer, c + ws / 2),     # +X arm
                  box(-outer, c - ws / 2, -inner, c + ws / 2),   # -X arm
                  box(c - ws / 2, inner, c + ws / 2, outer),     # +Y arm
                  box(c - ws / 2, -outer, c + ws / 2, -inner)]   # -Y arm
    poly = poly.difference(unary_union(slits))

    mesh = trimesh.creation.extrude_polygon(poly, height=thickness)
    mesh.fix_normals()
    return mesh


SPECS = {
    "planar": {
        "build": build_planar,
        "name": lambda t: f"Planar_pureshear_100x10gauge_t{t:g}mm.stl",
        "label": "Test 2 - planar / pure shear",
    },
    "biaxial": {
        "build": build_biaxial,
        "name": lambda t: f"Equibiaxial_bulge_disk_D{BIAXIAL_DIAMETER:g}_t{t:g}mm.stl",
        "label": "Test 3 - equibiaxial (disk-bulge)",
    },
    "compression": {
        "build": build_compression,
        "name": lambda t: f"Compression_puck_D{COMP_DIAMETER:g}_h{COMP_HEIGHT:g}mm.stl",
        "label": "Test 3 - uniaxial compression (equibiaxial-equivalent)",
    },
    "cruciform": {
        "build": build_cruciform,
        "name": lambda t: f"Cruciform_biaxial_120mm_t{t:g}mm.stl",
        "label": "Test 3 - true equibiaxial (cruciform, BIAXIAL machine only)",
    },
}


def emit(mode: str, thickness: float, out_dir: Path, out_override: Path | None):
    spec = SPECS[mode]
    out = out_override if out_override else out_dir / spec["name"](thickness)
    out.parent.mkdir(parents=True, exist_ok=True)

    mesh = spec["build"](thickness)
    mesh.export(out)

    ext = mesh.bounds[1] - mesh.bounds[0]
    print(f"[{spec['label']}]  Saved: {out}")
    print(f"  bounding box (mm): {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.2f}")
    print(f"  watertight: {mesh.is_watertight}   volume: {mesh.volume/1000:.2f} cm^3")
    if mode == "planar":
        print(f"  gauge: {PLANAR_WIDTH:g} wide x {PLANAR_GAUGE_H:g} tall "
              f"(aspect {PLANAR_WIDTH/PLANAR_GAUGE_H:.0f}:1); "
              f"clamp {PLANAR_CLAMP_H:g} mm top & bottom.")
    elif mode == "compression":
        print(f"  solid puck Ø{COMP_DIAMETER:g} x {COMP_HEIGHT:g} mm; "
              f"LUBRICATE both faces to keep compression uniaxial.")
    elif mode == "cruciform":
        print(f"  {CRUCI_SPAN:g} mm tip-to-tip, {CRUCI_ARM_W:g}x{CRUCI_ARM_W:g} gauge, "
              f"{CRUCI_N_SLITS} slits/arm, {CRUCI_FILLET:g} mm fillets; "
              f"REQUIRES a biaxial/4-arm machine.")
    else:
        print(f"  uniform membrane; clamp the outer ring, inflate the inner circle.")


def main():
    ap = argparse.ArgumentParser(description="TPU planar + equibiaxial specimen STL generator.")
    ap.add_argument("--mode",
                    choices=["planar", "biaxial", "compression", "cruciform", "all"],
                    default="all", help="which specimen(s) to generate (default: all)")
    ap.add_argument("--thickness", type=float, default=2.0,
                    help="specimen thickness mm (default 2; dogbone uses 3)")
    ap.add_argument("--out", default=None,
                    help="output STL path (only valid with a single --mode)")
    ap.add_argument("--diameter", type=float, default=None,
                    help="biaxial disc outer diameter mm (default 120); "
                         "set it larger than your bulge-rig clamp aperture")
    args = ap.parse_args()

    if args.out and args.mode == "all":
        ap.error("--out requires a single --mode (planar or biaxial), not 'all'.")

    if args.diameter is not None:
        global BIAXIAL_DIAMETER          # build_biaxial + the file name read this
        BIAXIAL_DIAMETER = args.diameter

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "stl_output"
    out_override = Path(args.out) if args.out else None

    modes = ["planar", "biaxial"] if args.mode == "all" else [args.mode]
    for m in modes:
        emit(m, args.thickness, out_dir, out_override)


if __name__ == "__main__":
    main()
