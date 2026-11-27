"""
fea_variants.py  --  CalculiX deck variants for auditing and benchmarking the piece FEA.
=======================================================================================
The loop's evaluator (fea_evaluator._write_inp) writes exactly one deck: C3D4,
Neo-Hooke, *STATIC, frictionless platens with symmetry planes. This module writes
that deck AND controlled departures from it, one knob at a time, so each modelling
choice can be measured instead of assumed:

  bc        "sym"    the loop's deck: BOT uz=0, TOP uz=d, XSYM ux=0 / YSYM uy=0 on the
                     mesh's min-x / min-y faces (all nodes within slab_tol of them)
            "free"   frictionless platens with FREE sides: BOT uz=0, TOP uz=d, and only
                     the 3 DOF that remove in-plane rigid motion (two bottom nodes).
                     Nothing restrains the side faces, so asymmetric buckling, sway and
                     auxetic lateral motion are all admissible.
            "shear"  bonded platens: BOT fully fixed, TOP ux=d (tangential), uy=0,
                     uz=0 (constant thickness). Sides free. Reaction RFx -> shear stress.
  elem      "C3D4" | "C3D10" (straight-sided: midside nodes at edge midpoints, so every
            element keeps its parent tet's Jacobian -- see quadratic_tets)
  material  dict(C10=, D1=) Neo-Hooke; vary D1 alone to vary bulk modulus / Poisson
  step      "static" | "dynamic" (implicit HHT; see dynamic_step)
  slab_tol  thickness of the BOT/TOP/XSYM/YSYM node slabs, as a fraction of extent
            (the loop uses 0.02)

Units throughout: N, mm, MPa, tonne/mm^3, s.
"""
from __future__ import annotations
import os, re, subprocess, time
from pathlib import Path
import numpy as np

import fea_evaluator as FE

RHO_TONNE_PER_MM3 = FE.TPU_DENSITY_G_PER_MM3 * 1e-6      # 1.22e-3 g/mm^3 -> 1.22e-9 t/mm^3


# ---------------------------------------------------------------------------
def quadratic_tets(coords, nid, tets):
    """C3D4 -> C3D10 with straight edges. Returns (coords, nid, tet10) in CalculiX order.

    CalculiX/Abaqus C3D10 node order (0-based): corners 0-3, then midsides of edges
    (0,1) (1,2) (2,0) (0,3) (1,3) (2,3). Midside nodes are placed at the edge midpoint,
    shared between neighbours. With straight edges the isoparametric map is affine,
    so the Jacobian is constant over the element and equals the parent tet's -- no
    curved-element inversion is possible. (A curved-boundary C3D10 would have to be
    re-checked; the STL surface is faceted, so there is no curvature to recover.)
    """
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    pairs = np.concatenate([np.sort(tets[:, e], axis=1) for e in edges])
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    idx = np.zeros(int(nid.max()) + 1, dtype=int); idx[nid] = np.arange(len(nid))
    mid = 0.5 * (coords[idx[uniq[:, 0]]] + coords[idx[uniq[:, 1]]])
    new_ids = np.arange(len(uniq)) + int(nid.max()) + 1
    mids = new_ids[inv.ravel()].reshape(len(edges), len(tets)).T
    return (np.vstack([coords, mid]), np.r_[nid, new_ids],
            np.hstack([tets, mids]))


def c3d10_jacobians(coords, nid, tet10):
    """det(J) at the 4 corners of every C3D10 (min over corners), from the full
    quadratic shape functions -- so it would catch a displaced midside node."""
    idx = np.zeros(int(nid.max()) + 1, dtype=int); idx[nid] = np.arange(len(nid))
    X = coords[idx[tet10]]                                    # (n, 10, 3)
    out = []
    for (r, s, t) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (.25, .25, .25)]:
        u = 1 - r - s - t
        # dN/d(r,s,t) for the 10-node tet, Abaqus ordering
        dN = np.array([
            [-(4*u-1), -(4*u-1), -(4*u-1)],
            [4*r-1, 0, 0], [0, 4*s-1, 0], [0, 0, 4*t-1],
            [4*(u-r), -4*r, -4*r],
            [4*s, 4*r, 0],
            [-4*s, 4*(u-s), -4*s],
            [-4*t, -4*t, 4*(u-t)],
            [4*t, 0, 4*r],
            [0, 4*t, 4*s]])
        J = np.einsum("nki,kj->nij", X, dN)
        out.append(np.linalg.det(J))
    return np.min(out, axis=0)


# ---------------------------------------------------------------------------
def smooth_ramp(n=41):
    """Tabular smooth step 0->1 over unit time (zero velocity at both ends)."""
    t = np.linspace(0, 1, n)
    return t, t**3 * (10 - 15*t + 6*t**2)                      # C2 smoothstep


def dynamic_step(duration, alpha=-0.05, inc0=None, inc_max=None):
    """*DYNAMIC (implicit HHT) step text. No DIRECT, no EXPLICIT: CalculiX then runs
    the implicit alpha-method with automatic incrementation."""
    inc0 = inc0 or duration / 200; inc_max = inc_max or duration / 50
    return (f"*STEP, NLGEOM, INC=100000\n*DYNAMIC, ALPHA={alpha}\n"
            f"{inc0:.6g}, {duration:.6g}, {duration*1e-7:.6g}, {inc_max:.6g}\n")


def write_deck(path, coords, nid, elems, *, disp, bc="sym", elem="C3D4",
               material=None, step="static", slab_tol=0.02, dyn=None):
    """Write one CalculiX deck. Returns a dict of the node-set sizes and extents."""
    material = material or dict(FE.NEO_HOOKE)
    ext = np.ptp(coords, axis=0); lo = coords.min(0); hi = coords.max(0)
    z = coords[:, 2]
    bot = nid[z <= lo[2] + slab_tol * ext[2]]
    top = nid[z >= hi[2] - slab_tol * ext[2]]
    stol = slab_tol * max(ext[0], ext[1])
    xsym = nid[coords[:, 0] <= lo[0] + stol]
    ysym = nid[coords[:, 1] <= lo[1] + stol]
    sets = dict(BOT=bot, TOP=top)
    lines = []
    if bc == "sym":
        sets.update(XSYM=xsym, YSYM=ysym)
        fix = "BOT, 3, 3, 0\nXSYM, 1, 1, 0\nYSYM, 2, 2, 0\n"
        load = f"TOP, 3, 3, {disp:.6f}\n"
        rf_dof = 3
    elif bc == "free":
        bmask = np.isin(nid, bot); bc_xy = coords[bmask]
        c = bc_xy.mean(0)
        a = nid[bmask][np.argmin(np.linalg.norm(bc_xy[:, :2] - c[:2], axis=1))]
        b = nid[bmask][np.argmax(np.abs(bc_xy[:, 0] - c[0]))]
        sets.update(RBA=np.array([a]), RBB=np.array([b]))
        fix = "BOT, 3, 3, 0\nRBA, 1, 2, 0\nRBB, 2, 2, 0\n"
        load = f"TOP, 3, 3, {disp:.6f}\n"
        rf_dof = 3
    elif bc == "bonded":
        # Fully rough platens: no lateral slip at either face, sides free. Bounds the
        # real platen friction from above, and blocks a whole-piece sway mode: if a
        # piece still collapses here, the collapse is local (walls), not sway.
        fix = "BOT, 1, 3, 0\nTOP, 1, 2, 0\n"
        load = f"TOP, 3, 3, {disp:.6f}\n"
        rf_dof = 3
    elif bc == "shear":
        fix = "BOT, 1, 3, 0\nTOP, 2, 3, 0\n"
        load = f"TOP, 1, 1, {disp:.6f}\n"
        rf_dof = 1
    elif bc == "contact":
        return _write_contact_deck(path, coords, nid, elems, disp=disp, elem=elem,
                                   material=material, mu=(dyn or {}).get("mu", 0.0),
                                   zone=(dyn or {}).get("zone", 0.15),
                                   penalty=(dyn or {}).get("penalty", 1000.0))
    else:
        raise ValueError(bc)

    etype = elem
    with open(path, "w") as f:
        f.write("*NODE, NSET=NALL\n")
        for i, t in enumerate(nid):
            f.write(f"{int(t)}, {coords[i,0]:.6f}, {coords[i,1]:.6f}, {coords[i,2]:.6f}\n")
        f.write(f"*ELEMENT, TYPE={etype}, ELSET=EALL\n")
        for i, e in enumerate(elems, 1):
            if len(e) <= 8:
                f.write(f"{i}, " + ", ".join(map(str, e)) + "\n")
            else:                                   # max 16 entries per line incl. id
                f.write(f"{i}, " + ", ".join(map(str, e[:8])) + ",\n"
                        + ", ".join(map(str, e[8:])) + "\n")
        for name, ids in sets.items():
            f.write(f"*NSET, NSET={name}\n" + ",\n".join(map(str, sorted(set(map(int, ids))))) + "\n")
        f.write("*MATERIAL, NAME=TPU\n"
                f"*HYPERELASTIC, NEO HOOKE\n{material['C10']}, {material['D1']}\n")
        if step == "dynamic":
            f.write(f"*DENSITY\n{RHO_TONNE_PER_MM3:.6g}\n")
            if dyn and dyn.get("rayleigh_beta"):
                f.write(f"*DAMPING, ALPHA=0, BETA={dyn['rayleigh_beta']:.6g}\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=TPU\n")
        if step == "dynamic":
            T = dyn["duration"]
            t, a = smooth_ramp()
            f.write("*AMPLITUDE, NAME=RAMP\n"
                    + "\n".join(f"{ti*T:.6g}, {ai:.6g}" for ti, ai in zip(t, a)) + "\n")
            f.write(dynamic_step(T, dyn.get("alpha", -0.05), dyn.get("inc0"), dyn.get("inc_max")))
            f.write("*BOUNDARY\n" + fix)
            f.write("*BOUNDARY, AMPLITUDE=RAMP\n" + load)
            f.write("*NODE PRINT, NSET=TOP, TOTALS=ONLY\nRF\n")
            f.write("*EL PRINT, ELSET=EALL, TOTALS=ONLY\nELSE, ELKE\n")
        else:
            f.write("*STEP, NLGEOM, INC=2000\n*STATIC\n0.01, 1.0, 1e-5, 0.05\n")
            f.write("*BOUNDARY\n" + fix + "*BOUNDARY\n" + load)
            f.write("*NODE PRINT, NSET=TOP, TOTALS=ONLY\nRF\n")
        f.write("*END STEP\n")
    return dict(ext=ext.tolist(), rf_dof=rf_dof,
                **{f"n_{k}": int(len(set(v))) for k, v in sets.items()})


_TET_FACES = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))     # CalculiX S1..S4 for C3D4/C3D10


def external_faces(tets):
    """(element index, face number 1-4, corner node ids) of every boundary face."""
    t4 = tets[:, :4]
    f = np.concatenate([t4[:, list(fc)] for fc in _TET_FACES])
    owner = np.tile(np.arange(len(t4)), 4); face_no = np.repeat(np.arange(1, 5), len(t4))
    key = np.sort(f, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    ext = cnt[inv.ravel()] == 1
    return owner[ext], face_no[ext], f[ext]


def _write_contact_deck(path, coords, nid, elems, *, disp, elem, material, mu, zone, penalty):
    """Rigid platens + surface-to-surface contact instead of glued node slabs.

    Each platen is a 4x4 block of C3D8 made rigid (*RIGID BODY) and driven through its
    reference node: bottom fixed, top moves in z only. Slave surfaces are the lattice's
    external faces within `zone` x H of each platen, so walls that rotate or fold onto
    a platen during the crush pick up contact too (self-contact is NOT modelled).
    mu = 0 is frictionless; mu > 0 adds Coulomb friction. In-plane rigid motion of the
    lattice is removed with the same 3 DOF as bc="free".
    """
    material = material or dict(FE.NEO_HOOKE)
    lo, hi = coords.min(0), coords.max(0); ext = hi - lo; H = float(ext[2])
    nmax = int(nid.max())
    # --- platen blocks
    pad = 0.2 * max(ext[0], ext[1]); th = max(1.0, 0.03 * H); n = 4
    gx = np.linspace(lo[0] - pad, hi[0] + pad, n + 1); gy = np.linspace(lo[1] - pad, hi[1] + pad, n + 1)
    pn_coords, pn_ids, hexes = [], [], {}
    nxt = nmax + 1
    for name, z0, z1 in (("PBOT", lo[2] - th, lo[2]), ("PTOP", hi[2], hi[2] + th)):
        ids = {}
        for k, zz in enumerate((z0, z1)):
            for i, xx in enumerate(gx):
                for j, yy in enumerate(gy):
                    ids[(i, j, k)] = nxt; pn_coords.append((xx, yy, zz)); pn_ids.append(nxt); nxt += 1
        hexes[name] = [[ids[(i, j, 0)], ids[(i + 1, j, 0)], ids[(i + 1, j + 1, 0)], ids[(i, j + 1, 0)],
                        ids[(i, j, 1)], ids[(i + 1, j, 1)], ids[(i + 1, j + 1, 1)], ids[(i, j + 1, 1)]]
                       for i in range(n) for j in range(n)]
    rb = {"PBOT": (nxt, nxt + 1), "PTOP": (nxt + 2, nxt + 3)}
    ref_xyz = {"PBOT": (lo[0] + ext[0] / 2, lo[1] + ext[1] / 2, lo[2] - th),
               "PTOP": (lo[0] + ext[0] / 2, lo[1] + ext[1] / 2, hi[2] + th)}
    # --- lattice slave faces near each platen
    owner, fno, fn = external_faces(elems)
    idx = np.zeros(nmax + 1, dtype=int); idx[nid] = np.arange(len(nid))
    fz = coords[idx[fn], 2]
    near_bot = (fz <= lo[2] + zone * H).all(axis=1)
    near_top = (fz >= hi[2] - zone * H).all(axis=1)
    # in-plane rigid-body removal on two bottom nodes
    z = coords[:, 2]; bmask = z <= lo[2] + 1e-6 * H
    if bmask.sum() < 2:
        bmask = z <= lo[2] + 0.01 * H
    bxy = coords[bmask]; c = bxy.mean(0)
    a = nid[bmask][np.argmin(np.linalg.norm(bxy[:, :2] - c[:2], axis=1))]
    b = nid[bmask][np.argmax(np.abs(bxy[:, 0] - c[0]))]

    ne = len(elems)
    with open(path, "w") as f:
        f.write("*NODE, NSET=NALL\n")
        for i, t in enumerate(nid):
            f.write(f"{int(t)}, {coords[i,0]:.6f}, {coords[i,1]:.6f}, {coords[i,2]:.6f}\n")
        for t, (x, y, zz) in zip(pn_ids, pn_coords):
            f.write(f"{t}, {x:.6f}, {y:.6f}, {zz:.6f}\n")
        for name, (r, rr) in rb.items():
            x, y, zz = ref_xyz[name]
            f.write(f"{r}, {x:.6f}, {y:.6f}, {zz:.6f}\n{rr}, {x:.6f}, {y:.6f}, {zz:.6f}\n")
        f.write(f"*ELEMENT, TYPE={elem}, ELSET=EALL\n")
        for i, e in enumerate(elems, 1):
            if len(e) <= 8:
                f.write(f"{i}, " + ", ".join(map(str, e)) + "\n")
            else:
                f.write(f"{i}, " + ", ".join(map(str, e[:8])) + ",\n" + ", ".join(map(str, e[8:])) + "\n")
        eid = ne + 1; pl_elems = {}
        for name, hx in hexes.items():
            f.write(f"*ELEMENT, TYPE=C3D8, ELSET={name}E\n")
            pl_elems[name] = []
            for h in hx:
                f.write(f"{eid}, " + ", ".join(map(str, h)) + "\n"); pl_elems[name].append(eid); eid += 1
        for name, hx in hexes.items():
            f.write(f"*NSET, NSET={name}\n" + ",\n".join(map(str, sorted({v for h in hx for v in h}))) + "\n")
        f.write(f"*NSET, NSET=RBA\n{a}\n*NSET, NSET=RBB\n{b}\n")
        f.write(f"*NSET, NSET=TOP\n{rb['PTOP'][0]}\n")
        # surfaces: platen faces facing the lattice (C3D8: S2 = +z face, S1 = -z face)
        f.write("*SURFACE, NAME=MBOT, TYPE=ELEMENT\n" + "".join(f"{e}, S2\n" for e in pl_elems["PBOT"]))
        f.write("*SURFACE, NAME=MTOP, TYPE=ELEMENT\n" + "".join(f"{e}, S1\n" for e in pl_elems["PTOP"]))
        f.write("*SURFACE, NAME=SBOT, TYPE=ELEMENT\n"
                + "".join(f"{o+1}, S{k}\n" for o, k in zip(owner[near_bot], fno[near_bot])))
        f.write("*SURFACE, NAME=STOP, TYPE=ELEMENT\n"
                + "".join(f"{o+1}, S{k}\n" for o, k in zip(owner[near_top], fno[near_top])))
        f.write("*MATERIAL, NAME=TPU\n"
                f"*HYPERELASTIC, NEO HOOKE\n{material['C10']}, {material['D1']}\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=TPU\n")
        f.write("*MATERIAL, NAME=STEEL\n*ELASTIC\n210000., 0.3\n")
        for name in hexes:
            f.write(f"*SOLID SECTION, ELSET={name}E, MATERIAL=STEEL\n")
            f.write(f"*RIGID BODY, NSET={name}, REF NODE={rb[name][0]}, ROT NODE={rb[name][1]}\n")
        f.write("*SURFACE INTERACTION, NAME=SI\n*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR\n"
                f"{penalty:.6g}, 1e-3, 1e-3\n")
        if mu > 0:
            f.write(f"*FRICTION\n{mu:.4g}, {penalty / 10:.6g}\n")
        f.write("*CONTACT PAIR, INTERACTION=SI, TYPE=SURFACE TO SURFACE\nSBOT, MBOT\n")
        f.write("*CONTACT PAIR, INTERACTION=SI, TYPE=SURFACE TO SURFACE\nSTOP, MTOP\n")
        f.write("*STEP, NLGEOM, INC=4000\n*STATIC\n0.01, 1.0, 1e-6, 0.02\n")
        r0, r1 = rb["PBOT"]; t0, t1 = rb["PTOP"]
        f.write(f"*BOUNDARY\n{r0}, 1, 3, 0\n{r1}, 1, 3, 0\n{t0}, 1, 2, 0\n{t1}, 1, 3, 0\n"
                "RBA, 1, 2, 0\nRBB, 2, 2, 0\n")
        f.write(f"*BOUNDARY\n{t0}, 3, 3, {disp:.6f}\n")
        f.write("*NODE PRINT, NSET=TOP, TOTALS=ONLY\nRF\n*END STEP\n")
    return dict(ext=ext.tolist(), rf_dof=3, n_slave_bot=int(near_bot.sum()),
                n_slave_top=int(near_top.sum()), mu=mu, zone=zone, penalty=penalty)


# ---------------------------------------------------------------------------
def parse_dat(dat_path):
    """-> dict(time, RF (n,3), energy_time, ELSE, ELKE) from a CalculiX .dat."""
    txt = Path(dat_path).read_text()
    rf_t, rf = [], []
    for m in re.finditer(r"total force \(fx,fy,fz\) for set TOP and time\s+([\d.E+-]+)\s*\n\s*\n\s*([^\n]+)", txt):
        v = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", m.group(2))]
        if len(v) >= 3:
            rf_t.append(float(m.group(1))); rf.append(v[:3])
    def tot(name):
        out = {}
        for m in re.finditer(rf"{name}[^\n]*for set EALL and time\s+([\d.E+-]+)\s*\n\s*\n\s*([^\n]+)", txt):
            v = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", m.group(2))
            if v: out[float(m.group(1))] = float(v[0])
        return out
    ie, ke = tot("internal energy"), tot("kinetic energy")
    et = sorted(set(ie) & set(ke))
    return dict(time=np.array(rf_t), RF=np.array(rf).reshape(-1, 3),
                energy_time=np.array(et), ELSE=np.array([ie[t] for t in et]),
                ELKE=np.array([ke[t] for t in et]))


def run_deck(workdir, stem="job", timeout=3600, threads=None):
    env = dict(os.environ)
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
        env["CCX_NPROC_EQUATION_SOLVER"] = str(threads)
    t0 = time.time()
    try:
        r = subprocess.run([FE.CCX, stem], cwd=workdir, capture_output=True, text=True,
                           timeout=timeout, env=env)
        out = r.stdout + r.stderr; timed_out = False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout.decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or ""))
        timed_out = True
    Path(workdir, "ccx_stdout.txt").write_text(out)
    dat = Path(workdir) / f"{stem}.dat"
    res = parse_dat(dat) if dat.exists() else dict(time=np.array([]), RF=np.zeros((0, 3)))
    res.update(sec=round(time.time() - t0, 1), timed_out=timed_out,
               errors=sorted(set(re.findall(r"\*ERROR[^\n]*", out)))[:3])
    return res
