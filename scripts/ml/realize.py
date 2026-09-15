"""
realize.py
==========
Turn an optimizer-proposed design (a parameter dict) into a printable STL —
the `params -> STL` half of the Section 6 interface, restoring the auto-STL
behavior the old surrogate.py had.

Topology coverage matches scripts/generate_tiles.py:
  * gyroid, schwartz  -> generate_tile (TPMS)
  * graded            -> generate_graded_tile (soft outer / stiff inner)
  * auxetic           -> generate_rotating_squares
  * re_entrant, honeycomb, elytra, voronoi, spinodoid -> build_tile (support-free
    field library: the OTHER auxetic family + the bio-inspired set)
  * bouligand         -> generate_bouligand_tile (helicoidal ply stack)
  * bcc/kelvin/octet  -> NOT scriptable here (need FreeCAD/nTop); skipped with a
    clear message instead of crashing.

STL + a clean solid open directly in Fusion / Creality Print / SimScale.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# generate_tiles.py lives in scripts/ (the parent of this ml/ package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import generate_tiles as gen
    _HAS_GEN = True
    _IMPORT_ERR = None
except Exception as e:  # trimesh/skimage missing, etc.
    _HAS_GEN = False
    _IMPORT_ERR = e

# Topologies generate_tiles.py can build directly.
SCRIPTABLE = {"gyroid", "schwartz", "diamond", "graded", "auxetic", "alternating",
              "hybrid", "re_entrant", "honeycomb", "elytra", "voronoi", "spinodoid",
              "bouligand"}

# Stochastic topologies whose STL depends on a random seed; carry the proposal's seed
# (default 0) into the filename so the exact tile that was tested can be regenerated.
_SEEDED = {"voronoi", "spinodoid"}

# gen.suggest_cell_size bisects ~11 full tile builds per call (~65 s per wall family),
# which was 99% of realize time for a blend. It depends only on (family, density), so
# cache it on disk, with density rounded to the nearest 0.01. Note the underlying answer
# is NOISY, not monotonic: it bisects on a wall thickness measured at res=56 (~0.9 mm
# voxels vs a 1.2 mm limit), e.g. honeycomb gave 11.3 mm at rho 0.30 but ~14.6 mm at
# 0.309. Caching makes it at least reproducible; the printability gate still checks
# every built mesh.
_CELL_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "cache" / "suggest_cell_size.json"


def _suggest_cell(fam: str, rho: float) -> float:
    import json
    key = f"{fam}@{round(rho, 2):.2f}"
    try:
        cache = json.loads(_CELL_CACHE_PATH.read_text())
    except Exception:
        cache = {}
    if key not in cache:
        cache[key] = float(gen.suggest_cell_size(fam, float(key.split("@")[1]))[0])
        _CELL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:                                   # re-read so concurrent runs don't clobber
            cache = {**json.loads(_CELL_CACHE_PATH.read_text()), key: cache[key]}
        except Exception:
            pass
        _CELL_CACHE_PATH.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return cache[key]

# Printability gate for generated tiles: the fraction of surface area that faces
# downward steeply enough to need support. TPU 95A support interfaces tear the lattice
# on removal, so anything past this is rejected rather than printed. Blends drawn only
# from z-invariant families measure ~0 here by construction and always pass.
MAX_UNSUPPORTED_FRAC = 0.05

# Fill-evenness gate. The tile is split into octants and each one's solid volume is
# compared against the mean; this is the emptiest octant's share. A uniform lattice
# sits near 1.0. Well below that means the design hits its average density while
# leaving part of the cube bare -- the "doesn't take up a full section" failure, which
# average density alone cannot detect.
MIN_OCTANT_FILL = 0.55


def _fill_uniformity(mesh):
    """Emptiest octant's solid volume as a fraction of the mean octant volume.

    Uses the mesh's own bounding box split 2x2x2 and slices the mesh against each
    octant. Returns 1.0 for a perfectly uniform lattice, ->0 as a region empties out.
    Falls back to 1.0 (no opinion) if the slicing stack can't handle the mesh, so a
    library quirk can't silently reject every design.
    """
    try:
        import numpy as _np
        lo, hi = mesh.bounds
        mid = (lo + hi) / 2.0
        vols = []
        for ix in (0, 1):
            for iy in (0, 1):
                for iz in (0, 1):
                    box_lo = _np.array([lo[0] if ix == 0 else mid[0],
                                        lo[1] if iy == 0 else mid[1],
                                        lo[2] if iz == 0 else mid[2]])
                    box_hi = _np.array([mid[0] if ix == 0 else hi[0],
                                        mid[1] if iy == 0 else hi[1],
                                        mid[2] if iz == 0 else hi[2]])
                    # Count vertices in the octant as a cheap proxy for how much
                    # structure lives there -- robust and far faster than a boolean.
                    v = mesh.vertices
                    inside = ((v >= box_lo).all(axis=1) & (v <= box_hi).all(axis=1))
                    vols.append(float(inside.sum()))
        vols = _np.array(vols, dtype=float)
        mean = vols.mean()
        return 1.0 if mean <= 0 else float(vols.min() / mean)
    except Exception:
        return 1.0


def _layers_from_encoding(encoding, rho: float, cell: float, total_z: float):
    """Build a generate_stacked_tile layer list from a region_encoding string.

    Uses the JSON {"layers":[{"topology":..,"frac":..}, ...]} form when present and
    every layer topology is scriptable; otherwise falls back to a 2-layer
    gyroid+schwartz stack. Each layer's thickness = frac * total_z.
    """
    import json
    fallback = [("gyroid", rho, cell, total_z / 2.0),
                ("schwartz", rho, cell, total_z / 2.0)]
    if not encoding:
        return fallback
    try:
        obj = json.loads(str(encoding))
        layers = obj.get("layers")
        if not layers:
            return fallback
        out = []
        for l in layers:
            t = l.get("topology")
            if t not in gen.TOPOLOGY_FN:        # un-scriptable layer -> bail to fallback
                return fallback
            frac = float(l.get("frac", 1.0 / len(layers)))
            out.append((t, rho, cell, frac * total_z))
        return out
    except Exception:
        return fallback


def realize_proposal(p: dict, out_dir: Path, res: int | None = None,
                     fea_res: int | None = None, piece_cells: int | None = None) -> dict:
    """Generate an STL for one proposal dict. Returns an info dict (path or reason).

    fea_res: build a COARSE simulation mesh instead of the print mesh. It overrides
    every resolution choice (including the fine sampling wall families normally
    force) and writes into out_dir rather than stl_output/. Marching cubes at ~60
    samples gives a self-intersection-free ~160k-face surface gmsh tet-meshes in
    ~20 s; decimating the ~1M-face print mesh instead creates self-intersections
    gmsh rejects. The printability gate below still runs, as a cheap screen --
    re-check at print resolution before actually printing.

    piece_cells (blends only, with fea_res): build just `piece_cells` whole cells per
    axis of the tile's own snapped lattice, at the same sampling pitch fea_res gives
    the full tile. A 2x2x2 piece of seed 11 solves to 20% crush in ~3 min and reads
    2-4% stiffer than the full tile over 1-12% (runs/fea_debug_2026-09-14).
    """
    if not _HAS_GEN:
        return {"design": p.get("topology"), "stl_path": None,
                "status": f"STL toolchain unavailable: {_IMPORT_ERR}"}

    out_dir.mkdir(parents=True, exist_ok=True)
    topo = p["topology"]
    rho = float(p["rel_density_measured"])
    cell = float(p["cell_size_mm"])
    ay = float(p.get("cell_aspect_y", 1.0) or 1.0)
    az = float(p.get("cell_aspect_z", 1.0) or 1.0)
    cellv = (cell, cell * ay, cell * az)            # per-axis (anisotropic) cell
    grad = float(p.get("gradient_param", 0.0) or 0.0)
    res = fea_res or res or gen.RESOLUTION

    name = f"PROP_{topo}_rho{rho:.2f}_cell{cell:.1f}mm"
    if piece_cells and not (fea_res and (topo == "blend" or p.get("blend"))):
        return {"design": name, "stl_path": None,
                "status": "piece_cells needs fea_res and a blend proposal"}
    # The generator reads the tile extent from module globals at call time, so a piece
    # is built by shrinking them for this call only (restored in `finally`).
    tile_saved = (gen.TILE_X, gen.TILE_Y, gen.TILE_Z)
    try:
        if topo == "blend" or p.get("blend"):
            # GENERATIVE: a new in-between geometry blended across families, with the
            # per-family cap re-applied here so a hand-edited proposal can't sneak a
            # near-pure design past the ceiling the optimizer sampled under.
            weights = gen.normalize_blend(p.get("blend") or {"gyroid": 1.0},
                                          cap=gen.MAX_FAMILY_WEIGHT)
            # Wall families need a minimum printable wall; take the largest cell any
            # member demands so no member ends up with sub-nozzle walls.
            build_cell = cell
            for fam in weights:
                if gen.FIELD_TOPOLOGIES[fam]["distance"]:
                    need = _suggest_cell(fam, rho)
                    build_cell = max(build_cell, need)
            scale = build_cell / cell if cell > 0 else 1.0
            build_cellv = tuple(c * scale for c in cellv)
            seed = int(p.get("blend_seed", p.get("seed", 0)) or 0)
            # Put every family into its repeating configuration (voronoi
            # irregularity=0, spinodoid periodic=True), then merge the seeds on top so
            # stochastic members stay reproducible without losing periodicity.
            field_kw = gen.periodic_field_kw(
                weights, extra={f: {"seed": seed} for f in weights if f in _SEEDED})
            # Any wall family in the mix means ~1.2-2 mm features -> fine sampling.
            res_b = res if fea_res else \
                gen.RESOLUTION_FINE if any(f in gen._FINE for f in weights) else res
            tz = gen.TILE_Z
            if piece_cells:
                snapped, _ = gen.snap_cell_sizes(build_cellv, gen.TILE_X, gen.TILE_Y, gen.TILE_Z)
                pitch = gen.TILE_X / fea_res
                gen.TILE_X, gen.TILE_Y, gen.TILE_Z = (piece_cells * c for c in snapped)
                build_cellv, tz = snapped, gen.TILE_Z
                res_b = max(16, round(gen.TILE_X / pitch))
            mesh, mass = gen.generate_blended_tile(weights, build_cellv, rho, tz=tz,
                                                   res=res_b, field_kw=field_kw)
            tag = "-".join(f"{k[:3]}{int(round(v*100))}" for k, v in
                           sorted(weights.items(), key=lambda kv: -kv[1]))
            name = f"PROP_blend_{tag}_rho{rho:.2f}_cell{build_cell:.1f}mm"
            if piece_cells:
                name += f"_piece{piece_cells}"

        elif topo in ("gyroid", "schwartz", "diamond"):
            fn = gen.TOPOLOGY_FN[topo]
            mesh, mass = gen.generate_tile(fn, cellv, rho, res=res)

        elif topo == "graded":
            # Split the mean density into a soft outer + stiff inner layer.
            soft = max(0.10, rho - grad / 2.0)
            stiff = min(0.55, rho + grad / 2.0)
            mesh, mass = gen.generate_graded_tile(cell_size=cellv, rho_soft=soft,
                                                  rho_stiff=stiff, res=res)
            name = f"PROP_graded_{soft:.2f}to{stiff:.2f}_cell{cell:.1f}mm"

        elif topo == "auxetic":
            # Perforated sheet: solid sheet with square holes; printed rho = 1-(s/a)^2.
            mesh, mass, rho_actual = gen.generate_rotating_squares(
                cell=cell, rel_density=rho, res=res)
            name = f"PROP_auxetic_rho{rho_actual:.2f}_cell{cell:.1f}mm"

        elif topo in ("alternating", "hybrid"):
            # Stack scriptable TPMS layers along Z. The optimizer proposes a topology
            # but not the layer recipe, so default to a 2-layer gyroid+schwartz stack
            # (override via the design's region_encoding when available).
            layers = _layers_from_encoding(p.get("region_encoding"), rho, cellv, gen.TILE_Z)
            mesh, mass = gen.generate_stacked_tile(layers, res=res)
            descr = "-".join(l[0][:3] for l in layers)
            name = f"PROP_{topo}_{descr}_rho{rho:.2f}_cell{cell:.1f}mm"

        elif topo == "bouligand":
            mesh, mass = gen.generate_bouligand_tile(
                base="honeycomb", cell_size=cell, relative_density=rho, res=res)

        elif topo in gen.FIELD_TOPOLOGIES:
            # Support-free field library: re-entrant / honeycomb / elytra / voronoi /
            # spinodoid. Wall-based ones need a minimum printable wall, so bump the
            # proposed cell up to the smallest printable one rather than emit a tile
            # with sub-nozzle walls the optimizer had no way to know about.
            kw = {}
            if topo in _SEEDED:
                kw["seed"] = int(p.get("seed", 0) or 0)
            build_cell = cell
            if gen.FIELD_TOPOLOGIES[topo]["distance"]:
                need, _ = gen.suggest_cell_size(topo, rho, **kw)
                build_cell = max(cell, need)
            mesh, mass = gen.build_tile(topo, build_cell, rho, res=res, **kw)
            seed_tag = f"_s{kw['seed']}" if "seed" in kw else ""
            name = f"PROP_{topo}_rho{rho:.2f}_cell{build_cell:.1f}mm{seed_tag}"

        else:
            return {"design": name, "stl_path": None,
                    "status": f"'{topo}' needs FreeCAD/nTop — not scriptable here; "
                              f"build it manually (see scripts/auxetic_instructions.txt)"}

        if fea_res:
            name = f"{name}_fea{fea_res}"
            path = str(out_dir / f"{name}.stl")
            mesh.export(path)
        else:
            path = gen.save_tile(mesh, name)
        info = {"design": name, "topology": topo, "stl_path": path,
                "est_mass_g": round(float(mass), 1),
                "watertight": bool(mesh.is_watertight),
                "piece_cells": piece_cells,
                "cell_size_built_mm": round(float(locals().get("build_cell", cell)), 3),
                "extent_mm": [round(gen.TILE_X, 2), round(gen.TILE_Y, 2), round(gen.TILE_Z, 2)],
                "status": "ok"}
        # As-built density and support cost are the two facts the testing team needs
        # off the mesh (the handoff wants MEASURED density, not the requested one).
        rep = gen.printability_report(mesh, tile_volume_mm3=gen.TILE_X * gen.TILE_Y
                                      * gen.TILE_Z, verbose=False)
        info["rel_density_as_built"] = round(rep["rel_density_measured"], 3)
        info["unsupported_frac"] = round(rep["unsupported_frac"], 3)
        info["n_bodies"] = rep["n_bodies"]
        info["support_free_by_construction"] = bool(
            mesh.metadata.get("support_free_by_construction", False))
        # Printability gate: TPU 95A support interfaces tear the lattice on removal, so
        # a design past the threshold is not worth printing even if the surrogate likes
        # it. Blends of z-invariant families measure ~0 here by construction.
        info["printable"] = info["unsupported_frac"] <= MAX_UNSUPPORTED_FRAC
        reasons = []
        if not info["printable"]:
            reasons.append(f"unsupported_frac={info['unsupported_frac']:.3f} > "
                           f"{MAX_UNSUPPORTED_FRAC}")

        # A coupon has to be ONE piece. _drop_islands only removes fragments under 1%
        # of the volume, so a tile that genuinely splits in two survives it -- and a
        # two-piece coupon is not testable at all, whatever its predicted metrics.
        if info["n_bodies"] > 1:
            reasons.append(f"{info['n_bodies']} disconnected bodies")

        # And it has to fill the cube reasonably evenly. A design can hit its target
        # average density while leaving a whole region empty, which reads as "doesn't
        # take up the full section" and is not a lattice worth testing.
        fill = _fill_uniformity(mesh)
        info["min_octant_fill"] = round(fill, 3)
        if fill < MIN_OCTANT_FILL:
            reasons.append(f"uneven fill (emptiest octant at {fill:.0%} of mean)")

        info["printable"] = not reasons
        if reasons:
            info["status"] = "rejected: " + "; ".join(reasons)
        return info

    except Exception as e:
        return {"design": name, "stl_path": None, "status": f"generation failed: {e}"}
    finally:
        gen.TILE_X, gen.TILE_Y, gen.TILE_Z = tile_saved


def realize_proposals(proposals: list[dict], out_dir: Path, res: int | None = None):
    """Generate STLs for a list of proposals; print status; return info dicts."""
    print(f"[realize] generating STL(s) into {out_dir} ...")
    infos = []
    for p in proposals:
        info = realize_proposal(p, out_dir, res=res)
        infos.append(info)
        if info["stl_path"] and info.get("printable", True):
            sf = " support-free" if info.get("support_free_by_construction") else \
                 f" unsupported={info.get('unsupported_frac', 0):.1%}"
            print(f"  OK  {info['stl_path']}  (~{info['est_mass_g']}g, "
                  f"watertight={info['watertight']},{sf})")
        elif info["stl_path"]:
            print(f"  XX  {info['design']}: {info['status']}")
        else:
            print(f"  --  {info['design']}: {info['status']}")
    return infos


def realize_gated(candidates: list[dict], out_dir: Path, q: int = 1,
                  res: int | None = None):
    """Walk a ranked shortlist, meshing until `q` proposals pass the printability gate.

    The optimizer ranks on predicted injury metrics; it has no way to know whether a
    blend it likes will print, because unsupported area is a property of the MESH, not
    of the design vector. So the gate lives here, after meshing, and the shortlist
    exists to absorb rejections instead of leaving the run with nothing to print.

    Returns (accepted, all_infos) so the report can show what was rejected and why.
    """
    print(f"[realize] generating STL(s) into {out_dir} "
          f"(need {q} passing, {len(candidates)} ranked candidates) ...")
    accepted, infos = [], []
    for p in candidates:
        if len(accepted) >= q:
            break
        info = realize_proposal(p, out_dir, res=res)
        infos.append(info)
        if info["stl_path"] and info.get("printable", True):
            sf = "support-free by construction" if \
                info.get("support_free_by_construction") else \
                f"unsupported={info.get('unsupported_frac', 0):.1%}"
            print(f"  OK  {info['stl_path']}  (~{info['est_mass_g']}g, "
                  f"watertight={info['watertight']}, {sf})")
            accepted.append((p, info))
        elif info["stl_path"]:
            print(f"  XX  {info['design']}: {info['status']}")
        else:
            print(f"  --  {info['design']}: {info['status']}")
    if len(accepted) < q:
        print(f"  !!  only {len(accepted)}/{q} candidates passed the gate; "
              f"raise --shortlist or relax MAX_UNSUPPORTED_FRAC")
    return accepted, infos


if __name__ == "__main__":
    # Demo: realize one example proposal.
    root = Path(__file__).resolve().parents[2]
    example = {"topology": "gyroid", "rel_density_measured": 0.27,
               "cell_size_mm": 8.0, "gradient_param": 0.0}
    realize_proposals([example], root / "stl_output")
