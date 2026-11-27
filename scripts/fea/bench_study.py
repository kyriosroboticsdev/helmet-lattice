"""
bench_study.py  --  BC audit, element / mesh-convergence and compressibility study.
==================================================================================
Two benchmarks, built the way the loop builds screening pieces (2x2x2 cells, coarse
marching-cubes surface at a fixed pitch):
  gyroid   pure gyroid, 12.7 mm cell (snaps to the 38.1 mm tile), rho 0.20
  blend    it005_0 from runs/loop_2026-09-14 (schwartz .45 / elytra .44 / diamond .11,
           rho 0.195, 14.6 mm cell) -- on both the screening front and the eff. front

Each case changes ONE thing from the loop's deck (C3D4, Neo-Hooke C10=5.005 D1=0.0435,
*STATIC, symmetry-plane platens, slab 2%, pitch 50.8/60 mm) and crushes to --crush.
Results go to runs/bench/<study>.jsonl, one line per case, appended as they finish,
so a long study can be read (and resumed) while it runs.

  python scripts/fea/bench_study.py bc        # sym vs free sides; slab thickness
  python scripts/fea/bench_study.py mesh      # C3D4 vs C3D10 x pitch refinement
  python scripts/fea/bench_study.py compress  # D1 (bulk modulus) alone, C3D4 vs C3D10
  python scripts/fea/bench_study.py shear     # shear load case on both benchmarks
  python scripts/fea/bench_study.py dynamic   # implicit HHT vs static; dt/T/alpha checks
"""
from __future__ import annotations
import argparse, json, sys, tempfile, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "scripts" / "ml"), str(ROOT / "scripts")]
import fea_evaluator as FE
import fea_variants as V

OUT = ROOT / "runs" / "bench"
CACHE = OUT / "geom"
BASE_MAT = dict(FE.NEO_HOOKE)


def build_stl(bench: str, fea_res: int) -> Path:
    if bench.startswith("loop:"):
        # loop:<design_id> -> that design's screening STL from the efficiency run,
        # exactly as the loop meshed it (fea_res is ignored)
        df = pd.read_csv(ROOT / "runs/loop_eff_2026-10-03/master_table.csv")
        return ROOT / df.set_index("design_id").loc[bench[5:], "raw_file_path"]
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{bench}_res{fea_res}.stl"
    if p.exists():
        return p
    import generate_tiles as gen
    if bench == "gyroid":
        saved = (gen.TILE_X, gen.TILE_Y, gen.TILE_Z)
        try:
            cell = 12.7; pitch = 50.8 / fea_res
            gen.TILE_X = gen.TILE_Y = gen.TILE_Z = 2 * cell
            mesh, _ = gen.generate_tile(gen.TOPOLOGY_FN["gyroid"], (cell,) * 3, 0.20,
                                        tz=2 * cell, res=max(16, round(2 * cell / pitch)))
        finally:
            gen.TILE_X, gen.TILE_Y, gen.TILE_Z = saved
        mesh.export(p)
    elif bench == "blend":
        import realize
        df = pd.read_csv(ROOT / "runs/loop_2026-09-14/master_table.csv")
        params = json.loads(df.set_index("design_id").loc["it005_0", "proposal_json"])
        tmp = Path(tempfile.mkdtemp())
        info = realize.realize_proposal(params, tmp, fea_res=fea_res, piece_cells=2)
        shutil.copy(info["stl_path"], p); shutil.rmtree(tmp, ignore_errors=True)
    else:
        raise ValueError(bench)
    return p


def wall_thickness(stl, n=3000, seed=0):
    """Local wall thickness by inward ray casting from random surface points.

    Returns (5th percentile, median) in mm. Points on the closing faces at the piece
    boundary are skipped (their inward ray runs along a cut, not across a wall).
    """
    import trimesh
    m = trimesh.load(stl)
    pts, fi = trimesh.sample.sample_surface(m, n, seed=seed)
    nrm = m.face_normals[fi]
    lo, hi = m.bounds; tol = 1e-3 * np.ptp(m.bounds, axis=0).max()
    inner = ~((np.abs(pts - lo) < tol) | (np.abs(pts - hi) < tol)).any(axis=1)
    pts, nrm = pts[inner], nrm[inner]
    loc, ray, _ = m.ray.intersects_location(pts - nrm * 1e-4, -nrm, multiple_hits=False)
    d = np.linalg.norm(loc - pts[ray], axis=1)
    d = d[d > 1e-3]
    return round(float(np.percentile(d, 5)), 3), round(float(np.median(d)), 3)


_mesh_cache = {}
def get_mesh(bench, fea_res, elem):
    key = (bench, fea_res)
    if key not in _mesh_cache:
        stl = build_stl(bench, fea_res)
        coords, nid, tets, rep = FE.mesh_for_fea(str(stl))
        idx = np.zeros(int(nid.max()) + 1, dtype=int); idx[nid] = np.arange(len(nid))
        P = coords[idx[tets]]
        edge = float(np.mean([np.linalg.norm(P[:, a] - P[:, b], axis=1).mean()
                              for a in range(4) for b in range(a + 1, 4)]))
        w5, w50 = wall_thickness(stl)
        rep.update(wall_p5_mm=w5, wall_median_mm=w50, mean_edge_mm=round(edge, 3),
                   elems_across_thin_wall=round(w5 / edge, 2))
        _mesh_cache[key] = (coords, nid, tets, rep)
    coords, nid, tets, rep = _mesh_cache[key]
    if elem == "C3D10":
        c, n, t = V.quadratic_tets(coords, nid, tets)
        rep = dict(rep, min_detJ=float(V.c3d10_jacobians(c, n, t).min()))
        return c, n, t, rep
    return coords, nid, tets, rep


def run_case(study, name, bench, *, fea_res=60, elem="C3D4", bc="sym", material=None,
             crush=0.10, step="static", dyn=None, slab_tol=0.02, threads=4, timeout=5400):
    out = OUT / f"{study}.jsonl"
    if out.exists() and any(json.loads(l).get("name") == name for l in out.read_text().splitlines() if l.strip()):
        print(f"[skip] {name} (already in {out.name})"); return
    coords, nid, elems, rep = get_mesh(bench, fea_res, elem)
    ext = np.ptp(coords[:, :3], axis=0); H = float(ext[2]); A = float(ext[0] * ext[1])
    work = Path(tempfile.mkdtemp(prefix=f"bench_{name}_"))
    disp = (crush * H) if bc == "shear" else (-crush * H)
    info = V.write_deck(work / "job.inp", coords, nid, elems, disp=disp, bc=bc, elem=elem,
                        material=material or BASE_MAT, step=step, dyn=dyn, slab_tol=slab_tol)
    r = V.run_deck(work, timeout=timeout, threads=threads)
    T = dyn["duration"] if step == "dynamic" else 1.0
    frac = r["time"] / T                                   # fraction of the applied load
    if step == "dynamic":
        _, amp = V.smooth_ramp(); tt = np.linspace(0, 1, len(amp))
        frac = np.interp(frac, tt, amp)
    k = info["rf_dof"] - 1
    strain = np.r_[0, crush * frac]
    # reaction at the driven nodes has the sign of the imposed displacement; report
    # magnitude-positive stress for compression (disp<0) and shear (disp>0) alike
    stress_kPa = np.r_[0, np.sign(disp) * r["RF"][:, k] / A * 1000] if len(r["RF"]) else np.r_[0]
    reached = float(strain[-1]) if len(strain) > 1 else 0.0
    def at(e):
        return float(np.interp(e, strain, stress_kPa)) if reached >= e - 1e-9 else None
    W = float(np.trapezoid(stress_kPa, strain))            # kPa = mJ/mm^3 * 1e3 ... energy density
    rec = dict(name=name, bench=bench, fea_res=fea_res, elem=elem, bc=bc, step=step,
               material=material or BASE_MAT, slab_tol=slab_tol, crush=crush, dyn=dyn,
               reached=round(reached, 4), sec=r["sec"], timed_out=r["timed_out"],
               errors=r["errors"], threads=threads, H_mm=round(H, 2),
               n_elems=int(len(elems)), n_nodes=int(len(nid)),
               mesh={k2: v for k2, v in rep.items() if k2 != "tets_meshed"},
               s_2_5=at(0.025), s_5=at(0.05), s_7_5=at(0.075), s_10=at(0.10),
               s_15=at(0.15), s_20=at(0.20), W_kPa=round(W, 4),
               plateau_kPa=(FE.band_mean(strain, stress_kPa, 0.5 * crush, crush)
                            if reached >= crush - 1e-9 else None),
               lateral_RF_N=(np.abs(r["RF"][-1, :2]).tolist() if len(r["RF"]) else None),
               nsets={k2: v for k2, v in info.items() if k2.startswith("n_")},
               curve=[(round(float(a), 5), round(float(b), 3)) for a, b in zip(strain, stress_kPa)])
    if step == "dynamic" and len(r.get("ELSE", [])):
        rec["KE_over_IE_max"] = float(np.max(r["ELKE"][1:] / np.maximum(r["ELSE"][1:], 1e-30)))
        rec["KE_over_IE_end"] = float(r["ELKE"][-1] / max(r["ELSE"][-1], 1e-30))
    with open(out, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    print(f"[{study}] {name}: reached {reached:.3f} in {r['sec']:.0f}s  s5={rec['s_5']} "
          f"s10={rec['s_10']} kPa  elems={len(elems)} {r['errors'] or ''}", flush=True)


def study_bc(a):
    for b in ("gyroid", "blend"):
        run_case("bc", f"{b}_sym", b, bc="sym", crush=a.crush, threads=a.threads)
        run_case("bc", f"{b}_free", b, bc="free", crush=a.crush, threads=a.threads)
        run_case("bc", f"{b}_sym_slab0.002", b, bc="sym", slab_tol=0.002, crush=a.crush, threads=a.threads)
        run_case("bc", f"{b}_free_slab0.002", b, bc="free", slab_tol=0.002, crush=a.crush, threads=a.threads)


def study_mesh(a):
    for b in ("gyroid", "blend"):
        for res in (60, 90, 120):
            run_case("mesh", f"{b}_C3D4_res{res}", b, fea_res=res, crush=a.crush, threads=a.threads)
        for res in (60, 90):
            run_case("mesh", f"{b}_C3D10_res{res}", b, fea_res=res, elem="C3D10",
                     crush=a.crush, threads=a.threads)


def study_compress(a):
    # C10 fixed; D1 alone sets K = 2/D1 -> nu 0.40 / 0.478 / 0.489. Never 0.5: C3D10 in
    # CalculiX is not a hybrid element and would lock / fail rather than converge.
    # Blend C3D10 is skipped: the static solve cannot pass the blend's ~4-8% peak with
    # C3D10 inside 90 min (mesh study: 4.2% at timeout), so it would only time out.
    for b in ("gyroid", "blend"):
        for D1 in (0.0435, 0.0087, 0.00435):
            for elem in (("C3D4", "C3D10") if b == "gyroid" else ("C3D4",)):
                run_case("compress", f"{b}_{elem}_D1{D1}", b, elem=elem,
                         material=dict(C10=BASE_MAT["C10"], D1=D1), crush=a.crush, threads=a.threads)


def study_shear(a):
    for b in ("gyroid", "blend"):
        run_case("shear", f"{b}_shear_C3D4", b, bc="shear", crush=a.crush, threads=a.threads)


def study_dynamic(a):
    # Quasi-static implicit dynamics: T >> the piece's fundamental period (~0.2-1 ms).
    b, bc = a.bench, a.bc
    tag = b.replace(":", "_")
    base = dict(duration=0.05, alpha=-0.05)
    run_case("dynamic", f"{tag}_{bc}_static", b, bc=bc, crush=a.crush, threads=a.threads)
    run_case("dynamic", f"{tag}_{bc}_dyn_base", b, bc=bc, step="dynamic", dyn=base, crush=a.crush, threads=a.threads)
    run_case("dynamic", f"{tag}_{bc}_dyn_halfdt", b, bc=bc, step="dynamic",
             dyn=dict(base, inc0=base["duration"] / 400, inc_max=base["duration"] / 100), crush=a.crush, threads=a.threads)
    run_case("dynamic", f"{tag}_{bc}_dyn_2T", b, bc=bc, step="dynamic",
             dyn=dict(base, duration=0.10), crush=a.crush, threads=a.threads)
    run_case("dynamic", f"{tag}_{bc}_dyn_alpha0.02", b, bc=bc, step="dynamic",
             dyn=dict(base, alpha=-0.02), crush=a.crush, threads=a.threads)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["bc", "mesh", "compress", "shear", "dynamic"])
    ap.add_argument("--crush", type=float, default=0.10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bench", default="blend")
    ap.add_argument("--bc", default="free", help="BC for the dynamic study (sym/free/bonded)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"study_{a.study}"](a)
