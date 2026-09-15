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

# Resolve CalculiX to an absolute path NOW, before gmsh is imported: on Windows,
# gmsh.initialize() leaves bare-name lookups failing (WinError 2) even though
# os.environ["PATH"] still lists the ccx folder. Override with the CCX env var.
CCX = os.environ.get("CCX") or shutil.which("ccx") or "ccx"

# ----------------------------------------------------------------------------
# VALIDATED MATERIAL MODEL  (calibrated Option-B both-modes Ogden, mu>0)
# Units: N, mm  ->  stress in MPa ; mu in MPa ; D1 in 1/MPa
# ----------------------------------------------------------------------------
OGDEN_MU    = [0.25246, 0.01695, 11.30314]     # MPa
OGDEN_ALPHA = [1.9058,  5.9637,  1.9056]
OGDEN_D1    = 0.0435                            # 1/MPa (K=46 MPa, nu~0.38 -- NOT near-incompressible)
# KNOWN ISSUE (2026-09-13, CalculiX 2.22): this Ogden card does not converge -- a
# solid cube diverges at increment 2 for any D1 >= ~0.011 (N=1..3, even 0.1% fixed
# steps), and the seed-11 lattice stalls at a 3-5% force residual even at D1/10.
# Kept for reference only.

# Re-fit and re-validated against the real compression data (mean abs error):
MOONEY_RIVLIN = dict(C10=7.895, C01=-1.519, D1=0.0435)   # 20.3%
NEO_HOOKE     = dict(C10=5.005, D1=0.0435)                # 33% (guaranteed-working fallback)
# C01 < 0: incompressible MR loses stability in equibiaxial stretch above ~1.51
# (uniaxial is stable over stretch 0.3-3.0).
# Seed-11 lattice test (2026-09-13): MR converges on a solid cube and to ~5.3% crush
# on the lattice, then the increment residual blows up (8.7e4 -> 7e6) and cutbacks
# shrink to <0.6% of the step without passing ~5.4% -- does not converge. Neo-Hooke
# is the working default until a stable MR/other fit exists.

MATERIAL = "neo_hooke"            # "neo_hooke" | "mooney_rivlin" (stalls ~5% crush) | "ogden" (no)


def material_card(name: str | None = None) -> str:
    """CalculiX *HYPERELASTIC card for the selected TPU model."""
    name = name or MATERIAL
    if name == "mooney_rivlin":
        m = MOONEY_RIVLIN
        return f"*HYPERELASTIC, MOONEY-RIVLIN\n{m['C10']}, {m['C01']}, {m['D1']}\n"
    if name == "neo_hooke":
        m = NEO_HOOKE
        return f"*HYPERELASTIC, NEO HOOKE\n{m['C10']}, {m['D1']}\n"
    if name == "ogden":
        return ("*HYPERELASTIC, OGDEN, N=3\n"
                f"{OGDEN_MU[0]:.5f}, {OGDEN_ALPHA[0]:.5f}, {OGDEN_MU[1]:.5f}, {OGDEN_ALPHA[1]:.5f}, "
                f"{OGDEN_MU[2]:.5f}, {OGDEN_ALPHA[2]:.5f}, {OGDEN_D1}, 0.0\n0.0\n")
    raise ValueError(f"unknown material {name!r}")


TPU_DENSITY_G_PER_MM3 = 1.22e-3                 # 1.22 g/cm^3
# Nominal crush applied. 10%, not 20%: a static implicit solve follows the elastic
# pre-buckling branch only -- seed 11 (Neo-Hooke) converged to 12.1% and then stalled
# on ever-smaller cutbacks as its walls began to buckle, and Mooney-Rivlin stalled at
# 5.3%. At 10% the plateau band and densification are NOT reached, so those two
# targets are left NaN rather than faked; SEA is the energy to fea_max_strain.
COMPRESS_FRACTION = 0.10
PLATEAU_BAND = (0.20, 0.40)                     # strain band averaged for plateau stress
FEA_TIMEOUT_S = 5400                            # ~45 min per 10% solve at 330k tets, 12 threads

# ----------------------------------------------------------------------------
SLIVER_QUALITY = 1e-4   # tet volume / mean_edge^3 below this is a sliver (regular tet = 0.118)


def tet_quality(coords, nid, tets):
    """Signed volume and shape quality (|vol| / mean_edge^3) for every tet, vectorised."""
    idx = np.zeros(int(nid.max()) + 1, dtype=int); idx[nid] = np.arange(len(nid))
    P = coords[idx[tets]]
    vol = np.einsum("ij,ij->i", P[:, 1] - P[:, 0],
                    np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0])) / 6.0
    edge = np.mean([np.linalg.norm(P[:, a] - P[:, b], axis=1)
                    for a in range(4) for b in range(a + 1, 4)], axis=0)
    return vol, np.abs(vol) / (edge ** 3 + 1e-30)


def mesh_for_fea(stl_path: str, target_faces: int | None = None, optimize: bool = True,
                 sliver_quality: float = SLIVER_QUALITY):
    """Tet-mesh an STL ONCE for the gate and the solve -> (coords, nid, tets, report).

    Feed it a coarse marching-cubes surface (realize_proposal(fea_res=...)), not a
    decimated print mesh: quadric decimation leaves self-intersecting facets and
    gmsh refuses them. target_faces decimates only if explicitly asked.

    gmsh on a marching-cubes surface still leaves ~0.1-0.3% sliver tets, a few of
    them inverted, and CalculiX aborts on the first increment with "nonpositive
    jacobian" on those. Repair: netgen optimisation, then drop tets below
    `sliver_quality` (near-zero volume, so mechanically negligible -- the dropped
    volume share is reported) and any node left unused.
    """
    import gmsh
    work_stl = stl_path
    if target_faces:
        import pymeshlab
        ms = pymeshlab.MeshSet(); ms.load_new_mesh(stl_path)
        if ms.current_mesh().face_number() > target_faces:
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=target_faces, preservetopology=True, preservenormal=True)
        tmp = tempfile.NamedTemporaryFile(suffix="_dec.stl", delete=False)
        tmp.close(); ms.save_current_mesh(tmp.name); work_stl = tmp.name

    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        if optimize:
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
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

    vol, q = tet_quality(coords, nid, tets)
    bad = q < sliver_quality
    total = float(np.abs(vol).sum())
    report = dict(tets_meshed=int(len(tets)), inverted=int((vol < 0).sum()),
                  slivers_dropped=int(bad.sum()),
                  sliver_frac=float(bad.mean()),
                  dropped_volume_frac=float(np.abs(vol[bad]).sum() / total) if total else 0.0)
    tets = tets[~bad]
    vol = vol[~bad]
    # Dropping slivers can leave a few tets attached only by an edge or a vertex. Those
    # are free to spin (zero-energy mechanisms): the solve's force residual stalls at a
    # few % while displacements have converged. Keep the largest FACE-connected set.
    keep_t = _largest_face_component(tets)
    report["detached_tets_dropped"] = int((~keep_t).sum())
    tets, vol = tets[keep_t], vol[keep_t]
    flip = vol < 0                       # any remaining inverted (non-sliver) tet: reorder
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    report["reoriented"] = int(flip.sum())
    used = np.unique(tets)
    keep = np.isin(nid, used)
    coords, nid = coords[keep], nid[keep]
    report["tets"] = int(len(tets))
    report["worst_quality_after"] = float(tet_quality(coords, nid, tets)[1].min())
    return coords, nid, tets, report

def _largest_face_component(tets):
    """Boolean mask of the tets in the largest set connected through shared FACES."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(tets)
    faces = np.concatenate([np.sort(tets[:, f], axis=1)
                            for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))])
    owner = np.tile(np.arange(n), 4)
    order = np.lexsort(faces.T[::-1]); f, o = faces[order], owner[order]
    same = np.all(f[1:] == f[:-1], axis=1)
    g = coo_matrix((np.ones(int(same.sum())), (o[:-1][same], o[1:][same])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab == np.argmax(np.bincount(lab))


def _mesh_volume(coords, nid, tets):
    return float(np.abs(tet_quality(coords, nid, tets)[0]).sum())

def _write_inp(path, coords, nid, tets, disp, material=None):
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
        f.write("*MATERIAL, NAME=TPU\n" + material_card(material))
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=TPU\n")
        # No STABILIZE: that is Abaqus syntax; CalculiX 2.22 prints "parameter not
        # recognized" and ignores it. Initial 1%, min 1e-5, max 5% of the step.
        f.write("*STEP, NLGEOM, INC=2000\n*STATIC\n0.01, 1.0, 1e-5, 0.05\n")
        # FRICTIONLESS platens (validated BC): bottom z=0, symmetry planes kill rigid body,
        # faces free to spread. DO NOT constrain top/bottom laterally.
        f.write("*BOUNDARY\nBOT, 3, 3, 0\nXSYM, 1, 1, 0\nYSYM, 2, 2, 0\n")
        f.write(f"*BOUNDARY\nTOP, 3, 3, {disp:.5f}\n")
        f.write("*NODE PRINT, NSET=TOP, TOTALS=ONLY\nRF\n*END STEP\n")
    return zmax - zmin

def _run_ccx(inp_stem, workdir, timeout=1800):
    """Run CalculiX; stdout is saved to ccx_stdout.txt so a failure has a reason.

    On a timeout the solver is killed but the curve up to its last converged increment
    is still returned (the .dat is written increment by increment). A stalled solve
    crawls on ever-smaller cutbacks, so the cap usually lands long after the useful
    part of the curve -- throwing that away cost a real 1-hour result.
    """
    try:
        r = subprocess.run([CCX, inp_stem], cwd=workdir, capture_output=True, text=True,
                           timeout=timeout)
        stdout = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stdout = out + f"\n[fea_evaluator] killed after {timeout}s timeout\n"
    Path(workdir, "ccx_stdout.txt").write_text(stdout)
    dat = Path(workdir) / f"{inp_stem}.dat"
    if not dat.exists(): raise RuntimeError("CalculiX produced no .dat (solve failed)")
    txt = dat.read_text()
    times = [float(t) for t in re.findall(r"for set TOP and time\s+([\d.E+-]+)", txt)]
    fz = [float(re.findall(r"[-\d.E+]+", b)[2])
          for b in re.findall(r"for set TOP and time[^\n]*\n\n\s*([^\n]+)", txt)]
    n = min(len(times), len(fz))          # a kill can leave the last block half-written
    if n < 2:
        errs = sorted(set(re.findall(r"\*ERROR[^\n]*", stdout)))[:3]
        raise RuntimeError("solve did not converge past first increment"
                           + (f"; ccx: {errs}" if errs else ""))
    return np.r_[0, times[:n]], np.r_[0, [-x for x in fz[:n]]]

# ----------------------------------------------------------------------------
def evaluate_stl(stl_path: str, *, footprint_area_mm2: float | None = None,
                 target_faces: int | None = None, keep_files: bool = False,
                 mesh=None, timeout: int = FEA_TIMEOUT_S, material: str | None = None,
                 crush: float | None = None) -> dict:
    """Run validated quasi-static compression FEA on one lattice STL.

    Returns a dict of the schema quasi-static targets plus as-built geometry:
        rel_density_measured  (from mesh volume / bbox volume -- the REAL feature,
                               immune to the gyroid/schwartz density labeling bug)
        SEA_J_per_kg, plateau_stress_kPa, densification_strain
        + fea_reached_strain, fea_converged for QC.
    """
    # mesh: (coords, nid, tets, report) from mesh_for_fea, so the gate's mesh is reused
    coords, nid, tets, mesh_report = mesh if mesh is not None else \
        mesh_for_fea(stl_path, target_faces=target_faces)
    Lx = float(np.ptp(coords[:, 0])); Ly = float(np.ptp(coords[:, 1])); H = float(np.ptp(coords[:, 2]))
    bbox_vol = Lx * Ly * H
    solid_vol = _mesh_volume(coords, nid, tets)
    rel_density = solid_vol / bbox_vol
    area = footprint_area_mm2 if footprint_area_mm2 else Lx * Ly

    workdir = tempfile.mkdtemp(prefix="fea_")
    try:
        inp = os.path.join(workdir, "job.inp")
        crush = crush or COMPRESS_FRACTION
        Hm = _write_inp(inp, coords, nid, tets, disp=-crush*H, material=material)
        t, F = _run_ccx("job", workdir, timeout=timeout)
        strain = crush * t
        stress = F / area                        # MPa (nominal)
        reached = float(strain[-1]); converged = bool(abs(t[-1]-1.0) < 1e-3)

        # --- quasi-static targets ---
        W = float(np.trapezoid(stress, strain))          # MPa == mJ/mm^3 absorbed
        mass_g = rel_density * bbox_vol * TPU_DENSITY_G_PER_MM3
        SEA_J_per_kg = (W * bbox_vol * 1e-3) / (mass_g * 1e-3) if mass_g > 0 else float("nan")
        # Same quantities at a FIXED 10% crush, so runs with different crush depths
        # (2x2x2 piece to 20%, full tile to 10%) can be compared directly.
        if reached >= 0.10 - 1e-9 and mass_g > 0:
            m10 = strain <= 0.10 + 1e-9
            s10 = float(np.interp(0.10, strain, stress))
            W10 = float(np.trapezoid(np.r_[stress[m10], s10], np.r_[strain[m10], 0.10]))
            stress_10_kPa = s10 * 1000.0
            SEA_10 = (W10 * bbox_vol * 1e-3) / (mass_g * 1e-3)
        else:
            stress_10_kPa = SEA_10 = float("nan")
        # plateau stress: mean nominal stress over PLATEAU_BAND -- only if the solve
        # actually covered the band. (The old fallback reported the stress at the last
        # strain under the plateau label, which is a different quantity.)
        lo, hi = PLATEAU_BAND
        band = (strain >= lo) & (strain <= hi)
        plateau_stress_kPa = float(np.mean(stress[band])) * 1000.0 \
            if (reached >= hi and band.sum() >= 2) else float("nan")
        # densification strain: efficiency-curve peak (eta = integral/current stress).
        # A peak at the last point means it lies beyond the crush applied -> unknown.
        cum = np.concatenate([[0], np.cumsum((stress[1:]+stress[:-1])/2 * np.diff(strain))])
        eta = cum / np.maximum(stress, 1e-9)
        k = int(np.argmax(eta))
        dens_strain = float(strain[k]) if (len(eta) > 3 and k < len(eta) - 1) else float("nan")

        return dict(
            rel_density_measured=round(rel_density, 4),
            SEA_J_per_kg=round(SEA_J_per_kg, 2),
            plateau_stress_kPa=round(plateau_stress_kPa, 2),
            densification_strain=round(dens_strain, 4),
            fea_reached_strain=round(reached, 3),
            stress_at_max_strain_kPa=round(float(stress[-1]) * 1000.0, 2),
            stress_at_10pct_kPa=round(stress_10_kPa, 2),
            SEA_to_10pct_J_per_kg=round(SEA_10, 2),
            fea_crush=crush,
            stress_strain=[(round(float(e), 5), round(float(s) * 1000.0, 3))
                           for e, s in zip(strain, stress)],
            fea_converged=converged,
            fea_tets=len(tets),
            material=material or MATERIAL,
            fea_mesh=mesh_report,
            bbox_mm=(round(Lx, 2), round(Ly, 2), round(H, 2)),
            workdir=workdir if keep_files else None,
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
        "notes": (f"CalculiX {res['material']} frictionless quasi-static; "
                  f"reached {res['fea_reached_strain']*100:.0f}% strain; "
                  f"{'converged' if res['fea_converged'] else 'PARTIAL'}; "
                  f"{res['fea_tets']} tets"),
    })
    # Not contract columns: the crush a row's SEA refers to, and the stress there
    # (a stiffness proxy that exists even when the plateau band was not reached).
    row["fea_max_strain"] = res["fea_reached_strain"]
    row["stress_at_max_strain_kPa"] = res["stress_at_max_strain_kPa"]
    row["stress_at_10pct_kPa"] = res["stress_at_10pct_kPa"]
    row["SEA_to_10pct_J_per_kg"] = res["SEA_to_10pct_J_per_kg"]
    row["fea_crush"] = res["fea_crush"]
    row["_stress_strain"] = res["stress_strain"]   # popped by the loop into its own file
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
    ap.add_argument("--target-faces", type=int, default=None,
                    help="decimate first (not recommended: creates self-intersections)")
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
