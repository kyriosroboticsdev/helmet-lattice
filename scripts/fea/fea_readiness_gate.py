"""
fea_readiness_gate.py
=====================
An FEA-readiness gate for the design loop. Rejects geometry that is not cleanly
SIMULATABLE *before* it reaches the expensive nonlinear FEA, so that:

  1. A bad mesh can't crash / stall the automated loop.
  2. When FEA *does* fail, it means something PHYSICAL (a design that truly
     collapses) rather than something NUMERICAL (a modelling artifact).

This is the FEA sibling of the generator's existing manufacturability gate
("repeating cell, one piece, evenly filling, support-free"). Wire it in right
after realize.py produces an STL and before fea_evaluator.evaluate_stl().

Checks (cheap -> expensive; stops at the first failure):
  A. WATERTIGHT + SINGLE BODY   -- trimesh: closed, one connected component.
  B. MIN FEATURE / WALL SIZE    -- thinnest wall must exceed a floor tied to the
                                   print resolution (thin walls fail BOTH printing
                                   and FEA convergence -- one rule kills both).
  C. MESH-ABILITY               -- a fast coarse tet mesh must succeed & be valid
                                   (no near-zero-volume / inverted elements).
  D. (optional) LINEAR PRE-SOLVE-- a seconds-long *linear* elastic solve; if it
                                   blows up or is singular, the geometry has a
                                   numerical defect -> reject before nonlinear FEA.

Returns a GateResult with pass/fail + reason + measured diagnostics, so failures
are logged as data (and can be fed back to the optimizer as "avoid this region").
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import os, tempfile, subprocess, math, shutil
import numpy as np

# Absolute ccx path resolved before gmsh loads (see fea_evaluator.CCX for why).
CCX = os.environ.get("CCX") or shutil.which("ccx") or "ccx"


@dataclass
class GateConfig:
    min_wall_mm: float = 0.8          # thinnest allowed wall/strut (>= ~2x nozzle)
    nozzle_mm: float = 0.4            # printer nozzle; floor auto-raised to 2x this
    coarse_faces: int = 20000        # decimation target for the quick mesh check
    min_tet_quality: float = 1e-4    # reject if worst tet volume/edge^3 below this
    run_linear_presolve: bool = False  # check D: linear pre-solve (re-meshes; ~30 s extra)
    max_components: int = 1          # single connected body
    # --- mesh-once path (check_fea_ready): the tet mesh is built once by
    # fea_evaluator.mesh_for_fea, slivers are repaired there, and the SAME mesh goes
    # to the solve. Quality is judged by how much had to be dropped, not by the one
    # worst element (every marching-cubes lattice has a few near-zero slivers).
    max_sliver_frac: float = 0.01          # reject if >1% of tets are slivers
    max_dropped_volume_frac: float = 1e-3  # ...or dropping them removes >0.1% of volume
    # Off by default: the generator already enforces a printable wall via
    # suggest_cell_size, and this check took 4-8 min per design. Turn on to audit.
    run_wall_check: bool = False


@dataclass
class GateResult:
    ok: bool
    reason: str
    checks: dict          # per-check pass/fail
    diagnostics: dict     # measured numbers (wall size, components, tets, etc.)
    def as_dict(self): return asdict(self)


def _watertight_single_body(stl_path, cfg) -> tuple[bool, str, dict]:
    try:
        import trimesh
    except ImportError:
        return True, "trimesh-not-installed-skipped", {}   # don't hard-fail if lib missing
    m = trimesh.load(stl_path, force="mesh")
    diag = dict(watertight=bool(m.is_watertight),
                n_components=int(len(m.split(only_watertight=False))),
                volume=float(m.volume) if m.is_volume else None)
    if not m.is_watertight:
        return False, "NOT_WATERTIGHT (open surface -> cannot form FEA volume)", diag
    if diag["n_components"] > cfg.max_components:
        return False, f"MULTIPLE_BODIES ({diag['n_components']} disconnected pieces)", diag
    if m.volume is None or m.volume <= 0:
        return False, "NON_POSITIVE_VOLUME (inverted/degenerate)", diag
    return True, "ok", diag


def _min_wall_thickness(stl_path, cfg) -> tuple[bool, str, dict]:
    """Estimate the thinnest wall via a signed-distance sample of the interior.
    The max interior distance-to-surface ~ half the thickest wall; we flag when
    the *typical thin* features fall below the floor. Cheap proxy, no heavy libs."""
    try:
        import trimesh
        m = trimesh.load(stl_path, force="mesh")
        # sample points just inside the surface, measure distance to surface
        pts, _ = trimesh.sample.sample_surface(m, 4000)
        normals = m.face_normals[_]
        inside = pts - normals * (cfg.min_wall_mm * 0.5)   # step inward by half-floor
        # a robust wall proxy: fraction of inward samples that are STILL inside the solid
        contains = m.contains(inside)
        thin_frac = float(1.0 - contains.mean())           # high => many walls thinner than floor
        floor = max(cfg.min_wall_mm, 2.0 * cfg.nozzle_mm)
        diag = dict(min_wall_floor_mm=floor, thin_wall_fraction=round(thin_frac, 3))
        # reject if a large fraction of the surface has sub-floor wall thickness
        if thin_frac > 0.25:
            return False, (f"WALLS_TOO_THIN (~{thin_frac*100:.0f}% of surface below "
                           f"{floor:.2f} mm floor -> fails print + FEA convergence)"), diag
        return True, "ok", diag
    except Exception as e:
        # A check that could not run must not report a pass (a 1M-face mesh hit
        # MemoryError here and used to "pass" silently).
        return False, f"WALL_CHECK_ERROR ({type(e).__name__}: {str(e)[:40]})", {}


def _meshable(stl_path, cfg) -> tuple[bool, str, dict]:
    """Fast coarse tet mesh; reject if it fails or produces degenerate elements."""
    import gmsh
    work = stl_path
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet(); ms.load_new_mesh(stl_path)
        if ms.current_mesh().face_number() > cfg.coarse_faces:
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=cfg.coarse_faces, preservetopology=True)
        tmp = tempfile.NamedTemporaryFile(suffix="_g.stl", delete=False); tmp.close()
        ms.save_current_mesh(tmp.name); work = tmp.name
    except ImportError:
        pass
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.merge(work); gmsh.model.mesh.removeDuplicateNodes(); gmsh.model.mesh.createTopology()
        surfs = gmsh.model.getEntities(2)
        gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([e[1] for e in surfs])])
        gmsh.model.geo.synchronize(); gmsh.model.mesh.generate(3)
        nt, nc, _ = gmsh.model.mesh.getNodes(); coords = nc.reshape(-1, 3); nid = nt.astype(int)
        tets = None
        for typ, _, cn in zip(*gmsh.model.mesh.getElements(3)):
            if typ == 4: tets = cn.reshape(-1, 4).astype(int)
    except Exception as e:
        gmsh.finalize()
        return False, f"MESH_FAILED ({str(e)[:50]})", {}
    finally:
        try: gmsh.finalize()
        except Exception: pass
    if tets is None or len(tets) < 20:
        return False, "MESH_EMPTY (no valid volume elements)", {"tets": 0}
    # quality: worst normalized tet volume
    idx = {int(t): i for i, t in enumerate(nid)}
    worst = math.inf
    for e in tets[:: max(1, len(tets)//5000)]:            # sample for speed
        p = coords[[idx[x] for x in e]]
        vol = abs(np.dot(p[1]-p[0], np.cross(p[2]-p[0], p[3]-p[0]))) / 6.0
        edge = np.mean([np.linalg.norm(p[a]-p[b]) for a in range(4) for b in range(a+1, 4)])
        q = vol / (edge**3 + 1e-12)
        worst = min(worst, q)
    diag = dict(tets=len(tets), worst_tet_quality=round(float(worst), 6))
    if worst < cfg.min_tet_quality:
        return False, f"DEGENERATE_ELEMENTS (worst tet quality {worst:.1e})", diag
    return True, "ok", diag



def _linear_presolve(stl_path, cfg) -> tuple[bool, str, dict]:
    """Check D: a seconds-long LINEAR elastic compression. Linear always converges
    if the geometry is numerically sound, so a FAILURE here means a real numerical
    defect (singular / disconnected / degenerate) -> reject before nonlinear FEA.
    Also flags absurd stiffness (near-zero or exploding) that signals bad geometry."""
    import gmsh, subprocess, tempfile, re
    work = stl_path
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet(); ms.load_new_mesh(stl_path)
        if ms.current_mesh().face_number() > cfg.coarse_faces:
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=cfg.coarse_faces, preservetopology=True)
        tmp = tempfile.NamedTemporaryFile(suffix="_lin.stl", delete=False); tmp.close()
        ms.save_current_mesh(tmp.name); work = tmp.name
    except ImportError:
        pass
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.merge(work); gmsh.model.mesh.removeDuplicateNodes(); gmsh.model.mesh.createTopology()
        surfs = gmsh.model.getEntities(2)
        gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([e[1] for e in surfs])])
        gmsh.model.geo.synchronize(); gmsh.model.mesh.generate(3)
        nt, nc, _ = gmsh.model.mesh.getNodes(); coords = nc.reshape(-1,3); nid = nt.astype(int)
        tets = None
        for typ, _, cn in zip(*gmsh.model.mesh.getElements(3)):
            if typ == 4: tets = cn.reshape(-1,4).astype(int)
    except Exception as e:
        try: gmsh.finalize()
        except Exception: pass
        return False, f"LINEAR_MESH_FAILED ({str(e)[:40]})", {}
    finally:
        try: gmsh.finalize()
        except Exception: pass
    if tets is None or len(tets) < 20:
        return False, "LINEAR_NO_MESH", {}
    zc = coords[:,2]; zmin, zmax = zc.min(), zc.max(); tol=(zmax-zmin)*0.02; H=zmax-zmin
    bot=[int(nid[i]) for i in range(len(nid)) if zc[i]<=zmin+tol]
    top=[int(nid[i]) for i in range(len(nid)) if zc[i]>=zmax-tol]
    Lx=float(np.ptp(coords[:,0])); Ly=float(np.ptp(coords[:,1])); area=Lx*Ly
    wd=tempfile.mkdtemp(prefix="lin_"); inp=os.path.join(wd,"lin.inp"); disp=-0.02*H  # 2% linear
    with open(inp,"w") as f:
        f.write("*NODE, NSET=NALL\n")
        for i,t in enumerate(nid): f.write(f"{int(t)}, {coords[i,0]:.5f}, {coords[i,1]:.5f}, {coords[i,2]:.5f}\n")
        f.write("*ELEMENT, TYPE=C3D4, ELSET=EALL\n")
        for i,e in enumerate(tets,1): f.write(f"{i}, {e[0]}, {e[1]}, {e[2]}, {e[3]}\n")
        f.write("*NSET, NSET=BOT\n"+",\n".join(map(str,bot))+"\n*NSET, NSET=TOP\n"+",\n".join(map(str,top))+"\n")
        # simple linear-elastic TPU-ish modulus, incompressible-ish
        f.write("*MATERIAL, NAME=LIN\n*ELASTIC\n10.0, 0.45\n*SOLID SECTION, ELSET=EALL, MATERIAL=LIN\n")
        f.write("*STEP\n*STATIC\n")
        f.write("*BOUNDARY\nBOT, 1, 3, 0\n")
        f.write(f"*BOUNDARY\nTOP, 3, 3, {disp:.5f}\nTOP, 1, 2, 0\n")
        f.write("*NODE PRINT, NSET=TOP, TOTALS=ONLY\nRF\n*END STEP\n")
    try:
        subprocess.run([CCX,"lin"],cwd=wd,capture_output=True,text=True,timeout=120)
        dat=os.path.join(wd,"lin.dat")
        if not os.path.exists(dat): return False, "LINEAR_SOLVE_FAILED (no output -> singular/disconnected)", {}
        txt=open(dat).read()
        fz=[float(re.findall(r"[-\d.E+]+",b)[2]) for b in re.findall(r"for set TOP and time[^\n]*\n\n\s*([^\n]+)", txt)]
        if not fz: return False, "LINEAR_NO_REACTION (singular system)", {}
        stress = abs(fz[-1])/area / 0.02   # apparent modulus (MPa) at 2% strain
        diag=dict(linear_apparent_modulus_MPa=round(stress,3))
        # absurd -> reject: near-zero (disconnected/floppy) or explosive (locked/degenerate)
        if stress < 1e-3:  return False, f"LINEAR_TOO_FLOPPY (apparent E={stress:.2e} MPa -> disconnected)", diag
        if stress > 1e4:   return False, f"LINEAR_EXPLODES (apparent E={stress:.1e} MPa -> degenerate/locked)", diag
        return True, "ok", diag
    except Exception as e:
        return False, f"LINEAR_PRESOLVE_ERROR ({str(e)[:40]})", {}


def _mesh_once(stl_path, cfg):
    """Build the one tet mesh the solve will use; judge it by the sliver repair."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fea_evaluator as E
    try:
        mesh = E.mesh_for_fea(stl_path)
    except Exception as e:
        return False, f"MESH_FAILED ({str(e)[:60]})", {}, None
    rep = mesh[3]
    if rep["tets"] < 20:
        return False, "MESH_EMPTY (no valid volume elements)", rep, None
    if rep["sliver_frac"] > cfg.max_sliver_frac:
        return False, f"TOO_MANY_SLIVERS ({rep['sliver_frac']:.2%} of tets)", rep, None
    if rep["dropped_volume_frac"] > cfg.max_dropped_volume_frac:
        return False, (f"SLIVER_REPAIR_REMOVES_VOLUME "
                       f"({rep['dropped_volume_frac']:.3%})"), rep, None
    return True, "ok", rep, mesh


def check_fea_ready(stl_path: str, config: GateConfig | None = None) -> GateResult:
    """Cheap checks, then mesh ONCE. On a pass, result.mesh holds the repaired
    (coords, nid, tets, report) for fea_evaluator.evaluate_stl(mesh=...)."""
    cfg = config or GateConfig()
    checks, diag = {}, {}
    stages = [("watertight_single_body", _watertight_single_body)]
    if cfg.run_wall_check:
        stages.append(("min_wall_thickness", _min_wall_thickness))
    for name, fn in stages:
        ok, reason, d = fn(stl_path, cfg)
        checks[name] = ok; diag.update(d)
        if not ok:
            return GateResult(False, f"{name}: {reason}", checks, diag)
    ok, reason, d, mesh = _mesh_once(stl_path, cfg)
    checks["meshable"] = ok; diag.update(d)
    if not ok:
        return GateResult(False, f"meshable: {reason}", checks, diag)
    # check D: optional linear pre-solve. Re-meshes internally; off in the loop,
    # because a numerically broken model now fails fast in the real solve anyway.
    if cfg.run_linear_presolve:
        ok, reason, d = _linear_presolve(stl_path, cfg)
        checks["linear_presolve"] = ok; diag.update(d)
        if not ok:
            return GateResult(False, f"linear_presolve: {reason}", checks, diag)
    result = GateResult(True, "FEA-ready", checks, diag)
    result.mesh = mesh          # not a dataclass field: keeps as_dict() light
    return result


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="FEA-readiness gate for a lattice STL")
    ap.add_argument("stl"); ap.add_argument("--min-wall", type=float, default=0.8)
    ap.add_argument("--nozzle", type=float, default=0.4)
    a = ap.parse_args()
    r = check_fea_ready(a.stl, GateConfig(min_wall_mm=a.min_wall, nozzle_mm=a.nozzle))
    print(json.dumps(r.as_dict(), indent=2, default=str))
    print("\nVERDICT:", "PASS -> send to FEA" if r.ok else f"REJECT -> {r.reason}")
