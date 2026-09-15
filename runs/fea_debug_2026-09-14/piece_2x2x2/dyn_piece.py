"""Seed-11 blend as a 2x2x2-cell piece: static control vs implicit dynamic, both to 20% crush."""
import sys, os, time, json
sys.path.insert(0, r"<repo>\scripts")
sys.path.insert(0, r"<repo>\scripts\ml")
sys.path.insert(0, r"<scratchpad>")
import numpy as np
import generate_tiles as gen, seed_lhs
import dyn_lib as L, fea_evaluator as E

p = seed_lhs.proposals_for_realize(20)[11]
rho = p["rel_density_measured"]; cell = p["cell_size_mm"]
w = gen.normalize_blend(p["blend"], cap=gen.MAX_FAMILY_WEIGHT)
build = cell
for fam in w:
    if gen.FIELD_TOPOLOGIES[fam]["distance"]:
        build = max(build, gen.suggest_cell_size(fam, rho)[0])
sc = build / cell
cellv = (build, cell * p["cell_aspect_y"] * sc, cell * p["cell_aspect_z"] * sc)
# same cell counts the full tile snaps to, then keep 2 cells per axis
_, counts = gen.snap_cell_sizes(cellv, gen.TILE_X, gen.TILE_Y, gen.TILE_Z)
snapped = (gen.TILE_X / counts[0], gen.TILE_Y / counts[1], gen.TILE_Z / counts[2])
ext = [2 * s for s in snapped]
pitch = gen.TILE_X / 60                        # same sampling pitch as the fea_res=60 full tile
gen.TILE_X, gen.TILE_Y = ext[0], ext[1]
fkw = gen.periodic_field_kw(w, extra={f: {"seed": p["blend_seed"]} for f in w if f in {"voronoi", "spinodoid"}})
res = round(ext[0] / pitch)
mesh, _ = gen.generate_blended_tile(w, snapped, rho, tz=ext[2], res=res, field_kw=fkw)
out = r"<scratchpad>/seed11_piece.stl"
mesh.export(out)
print(f"full-tile cell counts {counts}, snapped cell {np.round(snapped, 2)} mm; piece {np.round(ext, 1)} mm, "
      f"res {res}, watertight {mesh.is_watertight}, rho {mesh.volume / np.prod(ext):.3f}", flush=True)
c, n, t, rep = E.mesh_for_fea(out)
print("mesh:", json.dumps(rep), flush=True)

s = L.run("PIECE static 20%", c, n, t, 0.20, "static", 3000)
np.savetxt(out.replace(".stl", "_static.csv"), np.c_[s["strain"], s["stress_kpa"]], delimiter=",", header="strain,stress_kPa")
d = L.run("PIECE dynamic 20% (T=0.05 s, HHT alpha=-0.3)", c, n, t, 0.20, "dynamic", 3600, T=0.05)
np.savetxt(out.replace(".stl", "_dynamic.csv"), np.c_[d["strain"], d["stress_kpa"]], delimiter=",", header="strain,stress_kPa")
for e in (0.02, 0.05, 0.10, 0.12, 0.15, 0.20):
    sv = np.interp(e, s["strain"], s["stress_kpa"]) if s["reached"] >= e else float("nan")
    dv = np.interp(e, d["strain"], d["stress_kpa"]) if d["reached"] >= e else float("nan")
    print(f"  {e*100:4.0f}%: static {sv:8.1f} kPa | dynamic {dv:8.1f} kPa", flush=True)
print(f"  dynamic peak KE / external work = {L.ke_ratio(d, c):.2e}")
print("DONE", flush=True)
