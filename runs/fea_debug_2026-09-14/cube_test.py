"""Same material card, BCs and step as fea_evaluator, on a clean solid cube."""
import sys, os, re, time, tempfile
sys.path.insert(0, r"<repo>\scripts\fea")
import numpy as np, gmsh
import fea_evaluator as E

def cube_mesh(L=10.0, h=1.2):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.occ.addBox(0, 0, 0, L, L, L); gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", h); gmsh.model.mesh.generate(3)
        nt, nc, _ = gmsh.model.mesh.getNodes(); coords = nc.reshape(-1, 3); nid = nt.astype(int)
        for typ, _, cn in zip(*gmsh.model.mesh.getElements(3)):
            if typ == 4: tets = cn.reshape(-1, 4).astype(int)
    finally:
        gmsh.finalize()
    used = np.unique(tets); keep = np.isin(nid, used)
    return coords[keep], nid[keep], tets

def run(label, coords, nid, tets, material_patch=None):
    wd = tempfile.mkdtemp(prefix="cube_"); inp = os.path.join(wd, "job.inp")
    H = E._write_inp(inp, coords, nid, tets, disp=-E.COMPRESS_FRACTION * np.ptp(coords[:, 2]))
    if material_patch:
        txt = open(inp).read(); txt = material_patch(txt); open(inp, "w").write(txt)
    t = time.time()
    try:
        tt, F = E._run_ccx("job", wd, timeout=600)
        area = np.ptp(coords[:, 0]) * np.ptp(coords[:, 1])
        s = F / area
        print(f"{label}: CONVERGED to t={tt[-1]:.3f} in {time.time()-t:.0f}s, {len(tt)-1} incs; "
              f"stress at 5/10/20% strain = {np.interp([.25,.5,1.0], tt, s).round(3)} MPa", flush=True)
    except Exception as e:
        sta = open(os.path.join(wd, "job.sta")).read().strip().splitlines()[-1] if os.path.exists(os.path.join(wd, "job.sta")) else ""
        print(f"{label}: FAILED in {time.time()-t:.0f}s: {str(e)[:150]} | last sta: {sta}", flush=True)

c, n, t = cube_mesh()
_, q = E.tet_quality(c, n, t)
print(f"cube: {len(t)} tets, worst quality {q.min():.3f}", flush=True)
run("A cube, Ogden card as delivered", c, n, t)
run("B cube, Ogden with D2=D3=D1", c, n, t,
    lambda s: s.replace(f"{E.OGDEN_D1}, 0.0\n0.0\n", f"{E.OGDEN_D1}, {E.OGDEN_D1}\n{E.OGDEN_D1}\n"))
run("C cube, linear elastic E=10 nu=0.45 + NLGEOM", c, n, t,
    lambda s: re.sub(r"\*HYPERELASTIC, OGDEN, N=3\n[^\n]*\n[^\n]*\n", "*ELASTIC\n10.0, 0.45\n", s))
