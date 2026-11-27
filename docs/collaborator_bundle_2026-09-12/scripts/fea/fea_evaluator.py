"""
fea_evaluator.py
================
The FEA evaluator node that closes the ML loop.

Takes a watertight lattice STL (from realize.py or the generator), runs a
validated CalculiX quasi-static compression using the calibrated TPU-95A Ogden
model, extracts the quasi-static targets the surrogate expects, and returns /
appends a row in the EXACT schema.py Data Handoff Contract format with
data_source="fea".

Pipeline position:
    optimize.py  ->  realize.py (params->STL)  ->  [THIS]  ->  data.py (ingest)

Requires: gmsh, numpy, and the `ccx` (CalculiX) executable on PATH.
          pymeshlab optional (used to decimate heavy STLs before meshing).

Validated material model (see button validation, mean abs error ~21%, 3% at 30%):
    Ogden N=3, mu>0 (SimScale/CalculiX-compatible), tension+compression consistent.
    Boundary conditions: FRICTIONLESS platens (the critical fix) via symmetry planes.
"""
from __future__ import annotations
import os, sys, re, subprocess, tempfile, shutil
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------------
# VALIDATED MATERIAL MODEL  (calibrated Option-B both-modes Ogden, mu>0)
# Units: N, mm  ->  stress in MPa ; mu in MPa ; D1 in 1/MPa
# ----------------------------------------------------------------------------
OGDEN_MU    = [0.25246, 0.01695, 11.30314]     # MPa
OGDEN_ALPHA = [1.9058,  5.9637,  1.9056]
OGDEN_D1    = 0.0435                            # 1/MPa (near-incompressible)
TPU_DENSITY_G_PER_MM3 = 1.22e-3                 # 1.22 g/cm^3
COMPRESS_FRACTION = 0.20                        # crush to 30% nominal strain

# ----------------------------------------------------------------------------
def _load_mesh(stl_path: str, target_faces: int = 60000):
    """Decimate (if pymeshlab present) then tet-mesh the STL surface -> (coords, nid, tets, bbox)."""
    import gmsh
    work_stl = stl_path
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet(); ms.load_new_mesh(stl_path)
        if ms.current_mesh().face_number() > target_faces:
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=target_faces, preservetopology=True, preservenormal=True)
        for fn in ("meshing_remove_duplicate_faces",
                   "meshing_remove_duplicate_vertices",
                   "meshing_repair_non_manifold_edges"):
            try: getattr(ms, fn)()
            except Exception: pass
        tmp = tempfile.NamedTemporaryFile(suffix="_dec.stl", delete=False)
        tmp.close(); ms.save_current_mesh(tmp.name); work_stl = tmp.name
    except ImportError:
        pass  # pymeshlab optional; feed STL straight to gmsh

    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.merge(work_stl)
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.model.mesh.createTopology()
        surfs = gmsh.model.getEntities(2)
        sl = gmsh.model.geo.addSurfaceLoop([e[1] for e in surfs])
        gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        nt, nc, _ = gmsh.model.mesh.getNodes(); coords = nc.reshape(-1, 3); nid = nt.astype(int)
        tets = None
        for typ, _, cn in zip(*gmsh.model.mesh.getElements(3)):
            if typ == 4: tets = cn.reshape(-1, 4).astype(int)
    finally:
        gmsh.finalize()
    if tets is None or len(tets) < 10:
        raise RuntimeError("volume meshing produced no tets (surface not watertight?)")
    return coords, nid, tets

def _mesh_volume(coords, nid, tets):
    idx = {int(t): i for i, t in enumerate(nid)}
    v = 0.0
    for e in tets:
        p = coords[[idx[x] for x in e]]
        v += abs(np.dot(p[1]-p[0], np.cross(p[2]-p[0], p[3]-p[0]))) / 6.0
    return v

def _write_inp(path, coords, nid, tets, disp):
    zc = coords[:, 2]; zmin, zmax = zc.min(), zc.max(); tol = (zmax-zmin)*0.02
    bot = [int(nid[i]) for i in range(len(nid)) if zc[i] <= zmin+tol]
    top = [int(nid[i]) for i in range(len(nid)) if zc[i] >= zmax-tol]
    # Frictionless + stable: symmetry planes at the mesh's own min-x / min-y faces
    # (origin-independent). A symmetry plane restrains only the NORMAL DOF, so the
    # faces still spread freely in-plane -> no barreling, and rigid body is removed.
    xmin = coords[:,0].min(); ymin = coords[:,1].min()
    stol = max((coords[:,0].max()-xmin), (coords[:,1].max()-ymin))*0.02
    xsym = [int(nid[i]) for i in range(len(nid)) if coords[i,0] <= xmin+stol]
    ysym = [int(nid[i]) for i in range(len(nid)) if coords[i,1] <= ymin+stol]
    with open(path, "w") as f:
        f.write("*NODE, NSET=NALL\n")
        for i, t in enumerate(nid):
            f.write(f"{int(t)}, {coords[i,0]:.5f}, {coords[i,1]:.5f}, {coords[i,2]:.5f}\n")
        f.write("*ELEMENT, TYPE=C3D4, ELSET=EALL\n")
        for i, e in enumerate(tets, 1):
            f.write(f"{i}, {e[0]}, {e[1]}, {e[2]}, {e[3]}\n")
        f.write("*NSET, NSET=BOT\n"  + ",\n".join(map(str, bot)) + "\n")
        f.write("*NSET, NSET=TOP\n"  + ",\n".join(map(str, top)) + "\n")
        f.write("*NSET, NSET=XSYM\n" + ",\n".join(map(str, sorted(set(xsym)))) + "\n")
        f.write("*NSET, NSET=YSYM\n" + ",\n".join(map(str, sorted(set(ysym)))) + "\n")
        f.write("*MATERIAL, NAME=TPU\n*HYPERELASTIC, OGDEN, N=3\n")
        f.write(f"{OGDEN_MU[0]:.5f}, {OGDEN_ALPHA[0]:.5f}, {OGDEN_MU[1]:.5f}, {OGDEN_ALPHA[1]:.5f}, "
                f"{OGDEN_MU[2]:.5f}, {OGDEN_ALPHA[2]:.5f}, {OGDEN_D1}, 0.0\n0.0\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=TPU\n")
        f.write("*STEP, NLGEOM, INC=2000\n*STATIC, STABILIZE\n0.01, 1.0, 1e-8, 0.05\n")
        # FRICTIONLESS platens (validated BC): bottom z=0, symmetry planes kill rigid body,
        # faces free to spread. DO NOT constrain top/bottom laterally.
        f.write("*BOUNDARY\nBOT, 3, 3, 0\nXSYM, 1, 1, 0\nYSYM, 2, 2, 0\n")
        f.write(f"*BOUNDARY\nTOP, 3, 3, {disp:.5f}\n")
        f.write("*NODE PRINT, NSET=TOP, TOTALS=ONLY\nRF\n*END STEP\n")
    return zmax - zmin

def _run_ccx(inp_stem, workdir, timeout=1800):
    subprocess.run(["ccx", inp_stem], cwd=workdir, capture_output=True, text=True, timeout=timeout)
    dat = Path(workdir) / f"{inp_stem}.dat"
    if not dat.exists(): raise RuntimeError("CalculiX produced no .dat (solve failed)")
    txt = dat.read_text()
    times = [float(t) for t in re.findall(r"for set TOP and time\s+([\d.E+-]+)", txt)]
    fz = [float(re.findall(r"[-\d.E+]+", b)[2])
          for b in re.findall(r"for set TOP and time[^\n]*\n\n\s*([^\n]+)", txt)]
    if len(times) < 2: raise RuntimeError("solve did not converge past first increment")
    return np.r_[0, times], np.r_[0, [-x for x in fz]]

# ----------------------------------------------------------------------------
def evaluate_stl(stl_path: str, *, footprint_area_mm2: float | None = None,
                 target_faces: int = 60000, keep_files: bool = False) -> dict:
    """Run validated quasi-static compression FEA on one lattice STL.

    Returns a dict of the schema quasi-static targets plus as-built geometry:
        rel_density_measured  (from mesh volume / bbox volume -- the REAL feature,
                               immune to the gyroid/schwartz density labeling bug)
        SEA_J_per_kg, plateau_stress_kPa, densification_strain
        + fea_reached_strain, fea_converged for QC.
    """
    coords, nid, tets = _load_mesh(stl_path, target_faces=target_faces)
    Lx = float(np.ptp(coords[:, 0])); Ly = float(np.ptp(coords[:, 1])); H = float(np.ptp(coords[:, 2]))
    bbox_vol = Lx * Ly * H
    solid_vol = _mesh_volume(coords, nid, tets)
    rel_density = solid_vol / bbox_vol
    area = footprint_area_mm2 if footprint_area_mm2 else Lx * Ly

    workdir = tempfile.mkdtemp(prefix="fea_")
    try:
        inp = os.path.join(workdir, "job.inp")
        Hm = _write_inp(inp, coords, nid, tets, disp=-COMPRESS_FRACTION*H)
        t, F = _run_ccx("job", workdir)
        strain = COMPRESS_FRACTION * t
        stress = F / area                        # MPa (nominal)
        reached = float(strain[-1]); converged = bool(abs(t[-1]-1.0) < 1e-3)

        # --- quasi-static targets ---
        W = float(np.trapezoid(stress, strain))          # MPa == mJ/mm^3 absorbed
        mass_g = rel_density * bbox_vol * TPU_DENSITY_G_PER_MM3
        SEA_J_per_kg = (W * bbox_vol * 1e-3) / (mass_g * 1e-3) if mass_g > 0 else float("nan")
        # plateau stress: mean nominal stress over the 20-40% band (fallback to available range)
        band = (strain >= 0.20) & (strain <= 0.40)
        plateau_MPa = float(np.mean(stress[band])) if band.sum() >= 2 else float(np.interp(min(0.3, reached), strain, stress))
        plateau_stress_kPa = plateau_MPa * 1000.0
        # densification strain: efficiency-curve peak (eta = integral/current stress)
        eps = np.maximum(strain, 1e-9)
        cum = np.concatenate([[0], np.cumsum((stress[1:]+stress[:-1])/2 * np.diff(strain))])
        eta = cum / np.maximum(stress, 1e-9)
        dens_strain = float(strain[int(np.argmax(eta))]) if len(eta) > 3 else float("nan")

        return dict(
            rel_density_measured=round(rel_density, 4),
            SEA_J_per_kg=round(SEA_J_per_kg, 2),
            plateau_stress_kPa=round(plateau_stress_kPa, 2),
            densification_strain=round(dens_strain, 4),
            fea_reached_strain=round(reached, 3),
            fea_converged=converged,
            fea_tets=len(tets),
            bbox_mm=(round(Lx, 2), round(Ly, 2), round(H, 2)),
        )
    finally:
        if not keep_files: shutil.rmtree(workdir, ignore_errors=True)

# ----------------------------------------------------------------------------
def schema_row(stl_path: str, design_id: str, *, replicate_id: int = 1,
               topology: str = "blend", design_meta: dict | None = None,
               **eval_kwargs) -> dict:
    """Evaluate an STL and return a row matching schema.ALL_COLUMNS exactly.

    design_meta may carry generator-known fields (rel_density_designed, cell_size_mm,
    blend region_encoding, gradient_param, etc.). FEA fills the measured/quasi-static
    fields; impact/VT targets are left NaN (they come from physical tests, not sim).
    """
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[1] / "ml"))
    import schema

    res = evaluate_stl(stl_path, **eval_kwargs)
    row = {c: np.nan for c in schema.ALL_COLUMNS}
    row.update({
        "design_id": design_id,
        "replicate_id": replicate_id,
        "data_source": "fea",
        "topology": topology,
        "rel_density_measured": res["rel_density_measured"],
        "SEA_J_per_kg": res["SEA_J_per_kg"],
        "plateau_stress_kPa": res["plateau_stress_kPa"],
        "densification_strain": res["densification_strain"],
        "temperature_C": 23.0,
        "raw_file_path": str(stl_path),
        "notes": (f"CalculiX Ogden(N3) frictionless quasi-static; "
                  f"reached {res['fea_reached_strain']*100:.0f}% strain; "
                  f"{'converged' if res['fea_converged'] else 'PARTIAL'}; "
                  f"{res['fea_tets']} tets"),
    })
    if design_meta:
        for k, v in design_meta.items():
            if k in row and (isinstance(row[k], float) and np.isnan(row[k])):
                row[k] = v
    return row

def append_to_master(row: dict, master_csv: str):
    """Append one schema row to the master table (creates it with header if absent)."""
    import pandas as pd
    p = Path(master_csv)
    df_new = pd.DataFrame([row])
    if p.exists():
        df = pd.read_csv(p); df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(p, index=False); return len(df)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Validated FEA evaluator -> schema row")
    ap.add_argument("stl"); ap.add_argument("--design-id", required=True)
    ap.add_argument("--topology", default="blend")
    ap.add_argument("--append-to", help="master CSV to append the row to")
    ap.add_argument("--target-faces", type=int, default=60000)
    a = ap.parse_args()
    row = schema_row(a.stl, a.design_id, topology=a.topology, target_faces=a.target_faces)
    print(json.dumps({k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                      for k, v in row.items()}, indent=2, default=str))
    if a.append_to:
        n = append_to_master(row, a.append_to)
        print(f"\nappended -> {a.append_to} (now {n} rows)")


def safe_schema_row(stl_path: str, design_id: str, *, replicate_id: int = 1,
                    topology: str = "blend", design_meta: dict | None = None,
                    **eval_kwargs) -> dict:
    """Graceful wrapper around schema_row(): NEVER raises. On any mesh/solve
    failure it returns a schema-valid row with data_source='fea', the quasi-static
    targets left NaN, fea_status='failed', and the error in `notes`. This lets an
    unattended loop LOG-AND-SKIP a bad design instead of crashing."""
    import numpy as np, traceback
    from pathlib import Path as _P
    import sys as _sys
    here = _P(__file__).resolve(); _sys.path.insert(0, str(here.parents[1] / "ml"))
    import schema
    try:
        row = schema_row(stl_path, design_id, replicate_id=replicate_id,
                         topology=topology, design_meta=design_meta, **eval_kwargs)
        row["fea_status"] = "ok"
        return row
    except Exception as e:
        row = {c: np.nan for c in schema.ALL_COLUMNS}
        row.update(dict(design_id=design_id, replicate_id=replicate_id,
                        data_source="fea", topology=topology, temperature_C=23.0,
                        raw_file_path=str(stl_path),
                        notes=f"FEA FAILED: {type(e).__name__}: {str(e)[:160]}"))
        if design_meta:
            for k, v in design_meta.items():
                if k in row and (isinstance(row[k], float) and np.isnan(row[k])):
                    row[k] = v
        row["fea_status"] = "failed"
        return row
