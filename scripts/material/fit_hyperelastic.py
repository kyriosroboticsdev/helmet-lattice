"""
fit_hyperelastic.py  --  Fit + validate TPU hyperelastic cards on MATCHED-RATE lab data.
=======================================================================================
STATUS: BLOCKED PENDING LAB DATA. This harness is ready to run, but it refuses to fit
anything that is not lab-measured, matched-rate data (see Manifest below). It must NOT
be pointed at published or assumed coefficients/curves: a card fitted to someone
else's TPU at someone else's rate is exactly the kind of number the FEA cannot absorb
silently. `--selftest` exercises the code on synthetic data from a known model and
never writes a material card.

Models (incompressible strain-energy forms, Abaqus/CalculiX conventions):
  mooney_rivlin   W = C10(I1-3) + C01(I2-3)                       CalculiX MOONEY-RIVLIN
  yeoh            W = C10(I1-3) + C20(I1-3)^2 + C30(I1-3)^3       CalculiX YEOH
  ogden1, ogden2  W = sum 2 mu_i/alpha_i^2 (l1^a + l2^a + l3^a - 3)  CalculiX OGDEN N=1/2

Deformation modes (nominal stress P along the loaded axis, stretch l):
  uniaxial    (tension AND compression share this form; l<1 is compression)
  planar      pure shear, wide-short strip
  equibiaxial bulge / disk inflation

Compressibility is NOT fitted from these curves. D1 = 2/K comes from the manifest
("bulk_modulus_MPa" or "poisson" measured by DIC lateral strain). There is no default.

Validation, per fitted model:
  1. per-mode RMS and max relative error vs the lab curves (fit AND held-out modes)
  2. Drucker stability: dP/dl > 0 in every mode over l in [0.3, 3.0]
  3. CalculiX single-cube tension + compression: the card must CONVERGE and reproduce
     the analytic curve (the 2026-09 Ogden card fitted fine and then diverged in ccx)
  4. optional --validate-lattice STL: the card must carry the benchmark piece to the
     screening crush (Mooney-Rivlin 2026-09 stalled at ~5%)

Manifest (JSON):
{
  "material": "TPU 95A <brand/lot>", "source": "lab", "date": "YYYY-MM-DD",
  "temperature_C": 23, "bulk_modulus_MPa": null, "poisson": 0.47,
  "tests": [
    {"mode": "uniaxial", "file": "uni_tension.csv", "strain_rate_per_s": 0.01,
     "specimen": "ASTM D412 Die C", "machine": "MTS 880", "replicate": 1},
    {"mode": "planar", "file": "planar.csv", "strain_rate_per_s": 0.01, ...},
    {"mode": "equibiaxial", "file": "bulge.csv", "strain_rate_per_s": 0.01, ...}
  ]
}
Each CSV: columns `stretch,nominal_stress_MPa` (engineering, loading branch only).

Usage:
  python scripts/material/fit_hyperelastic.py data/material/manifest.json --out data/material/fit
  python scripts/material/fit_hyperelastic.py --selftest
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

MODES = ("uniaxial", "planar", "equibiaxial")
RATE_MATCH_FACTOR = 2.0        # all tests within a factor 2 in strain rate
STABILITY_RANGE = (0.3, 3.0)


# ---------------------------------------------------------------------------
# Nominal stress per mode
def _inv(mode, l):
    if mode == "uniaxial":
        return l**2 + 2 / l, 2 * l + 1 / l**2
    if mode == "planar":
        v = l**2 + 1 + l**-2
        return v, v
    if mode == "equibiaxial":
        return 2 * l**2 + l**-4, l**4 + 2 * l**-2
    raise ValueError(mode)


def _from_W(mode, l, W1, W2):
    if mode == "uniaxial":
        return 2 * (l - l**-2) * (W1 + W2 / l)
    if mode == "planar":
        return 2 * (l - l**-3) * (W1 + W2)
    return 2 * (l - l**-5) * (W1 + l**2 * W2)


def nominal_stress(model, p, mode, l):
    l = np.asarray(l, float)
    if model == "mooney_rivlin":
        return _from_W(mode, l, p[0], p[1])
    if model == "yeoh":
        I1, _ = _inv(mode, l); x = I1 - 3
        return _from_W(mode, l, p[0] + 2 * p[1] * x + 3 * p[2] * x**2, 0.0)
    if model.startswith("ogden"):
        mu, al = p[0::2], p[1::2]; out = 0.0
        for m, a in zip(mu, al):
            e = {"uniaxial": -a / 2, "planar": -a, "equibiaxial": -2 * a}[mode]
            out = out + (2 * m / a) * (l**(a - 1) - l**(e - 1))
        return out
    raise ValueError(model)


MODELS = {
    "mooney_rivlin": dict(x0=[1.0, 0.1], names=["C10", "C01"]),
    "yeoh":          dict(x0=[1.0, -0.01, 0.001], names=["C10", "C20", "C30"]),
    "ogden1":        dict(x0=[2.0, 2.0], names=["mu1", "alpha1"]),
    "ogden2":        dict(x0=[2.0, 2.0, 0.1, -2.0], names=["mu1", "alpha1", "mu2", "alpha2"]),
}


def fit(model, data, modes=None):
    """Least squares on RELATIVE error (so low-stress modes count), over `modes`."""
    modes = modes or list(data)
    def resid(p):
        r = []
        for m in modes:
            l, P = data[m]
            scale = np.maximum(np.abs(P), 0.05 * np.abs(P).max())
            r.append((nominal_stress(model, p, m, l) - P) / scale)
        return np.concatenate(r)
    best = None
    rng = np.random.default_rng(0)
    for k in range(12):                         # multistart: Ogden is multimodal
        x0 = np.array(MODELS[model]["x0"], float) * (1 if k == 0 else rng.uniform(0.3, 3, len(MODELS[model]["x0"])))
        if model.startswith("ogden") and k:
            x0[1::2] *= rng.choice([-1, 1], len(x0[1::2]))
        try:
            r = least_squares(resid, x0, max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def errors(model, p, data):
    out = {}
    for m, (l, P) in data.items():
        pr = nominal_stress(model, p, m, l)
        rel = np.abs(pr - P) / np.maximum(np.abs(P), 0.05 * np.abs(P).max())
        out[m] = dict(rms_rel=float(np.sqrt(np.mean(rel**2))), max_rel=float(rel.max()))
    return out


def drucker_stable(model, p, lo=STABILITY_RANGE[0], hi=STABILITY_RANGE[1]):
    """First stretch per mode where dP/dl <= 0 (None = stable over the range)."""
    l = np.linspace(lo, hi, 2000); res = {}
    for m in MODES:
        dP = np.gradient(nominal_stress(model, p, m, l), l)
        bad = np.where(dP <= 0)[0]
        res[m] = None if len(bad) == 0 else float(l[bad[0]])
    return res


def calculix_card(model, p, D1):
    if model == "mooney_rivlin":
        return f"*HYPERELASTIC, MOONEY-RIVLIN\n{p[0]:.6g}, {p[1]:.6g}, {D1:.6g}\n"
    if model == "yeoh":
        return f"*HYPERELASTIC, YEOH\n{p[0]:.6g}, {p[1]:.6g}, {p[2]:.6g}, {D1:.6g}, 0., 0.\n"
    n = len(p) // 2
    vals = ", ".join(f"{v:.6g}" for v in p) + f", {D1:.6g}" + (", 0." if n == 2 else "")
    return f"*HYPERELASTIC, OGDEN, N={n}\n{vals}\n"


# ---------------------------------------------------------------------------
def ccx_cube(card, stretch, ccx=None, n=3):
    """Uniaxial cube (free lateral faces) to `stretch` in CalculiX -> (l, P) or error."""
    ccx = ccx or os.environ.get("CCX") or "ccx"
    L = 1.0; g = np.linspace(0, L, n + 1)
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    nid = np.arange(1, len(X) + 1); I = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k + 1
    els = [[I(i,j,k), I(i+1,j,k), I(i+1,j+1,k), I(i,j+1,k), I(i,j,k+1), I(i+1,j,k+1), I(i+1,j+1,k+1), I(i,j+1,k+1)]
           for i in range(n) for j in range(n) for k in range(n)]
    w = Path(tempfile.mkdtemp(prefix="matcube_"))
    eps = 1e-9
    bot = nid[X[:, 2] < eps]; top = nid[X[:, 2] > L - eps]
    x0 = nid[X[:, 0] < eps]; y0 = nid[X[:, 1] < eps]
    with open(w / "c.inp", "w") as f:
        f.write("*NODE\n" + "".join(f"{i},{a:.6f},{b:.6f},{c:.6f}\n" for i, (a, b, c) in zip(nid, X)))
        f.write("*ELEMENT, TYPE=C3D8, ELSET=E\n" + "".join(f"{k+1}," + ",".join(map(str, e)) + "\n" for k, e in enumerate(els)))
        for nm, s in (("BOT", bot), ("TOP", top), ("X0", x0), ("Y0", y0)):
            f.write(f"*NSET, NSET={nm}\n" + ",\n".join(map(str, s)) + "\n")
        f.write("*MATERIAL, NAME=M\n" + card + "*SOLID SECTION, ELSET=E, MATERIAL=M\n")
        f.write("*STEP, NLGEOM, INC=1000\n*STATIC\n0.02, 1.0, 1e-6, 0.05\n")
        f.write(f"*BOUNDARY\nBOT,3,3,0\nX0,1,1,0\nY0,2,2,0\nTOP,3,3,{(stretch - 1) * L:.6f}\n")
        f.write("*NODE PRINT, NSET=TOP, TOTALS=ONLY\nRF\n*END STEP\n")
    r = subprocess.run([ccx, "c"], cwd=w, capture_output=True, text=True, timeout=600)
    import re
    txt = (w / "c.dat").read_text() if (w / "c.dat").exists() else ""
    t = [float(x) for x in re.findall(r"for set TOP and time\s+([\d.E+-]+)", txt)]
    fz = [float(re.findall(r"[-\d.E+]+", b)[2]) for b in re.findall(r"for set TOP and time[^\n]*\n\n\s*([^\n]+)", txt)]
    k = min(len(t), len(fz))
    l = 1 + (stretch - 1) * np.array(t[:k]); P = np.array(fz[:k]) / (L * L)
    return dict(stretch=l, P=P, converged=bool(k and abs(t[k - 1] - 1) < 1e-6),
                errors=sorted(set(re.findall(r"\*ERROR[^\n]*", r.stdout)))[:3])


# ---------------------------------------------------------------------------
def load_manifest(path):
    """Read and GATE the manifest. Raises on anything that is not matched-rate lab data."""
    path = Path(path); man = json.loads(path.read_text())
    problems = []
    if man.get("source") != "lab":
        problems.append(f'source must be "lab" (got {man.get("source")!r}); published or '
                        f'assumed curves/coefficients are not accepted')
    if man.get("bulk_modulus_MPa") is None and man.get("poisson") is None:
        problems.append("need measured bulk_modulus_MPa or poisson for D1 (no default)")
    tests = man.get("tests", [])
    modes = {t["mode"] for t in tests}
    missing = [m for m in MODES if m not in modes]
    if missing:
        problems.append(f"missing deformation modes {missing} (need all three independent modes)")
    rates = [t.get("strain_rate_per_s") for t in tests]
    if any(r is None or r <= 0 for r in rates):
        problems.append("every test needs strain_rate_per_s")
    elif max(rates) / min(rates) > RATE_MATCH_FACTOR:
        problems.append(f"strain rates not matched: {min(rates)}..{max(rates)} /s "
                        f"(> factor {RATE_MATCH_FACTOR})")
    if problems:
        raise SystemExit("BLOCKED -- manifest rejected:\n  - " + "\n  - ".join(problems))
    data = {}
    for t in tests:
        import pandas as pd
        d = pd.read_csv(path.parent / t["file"])
        l, P = d["stretch"].to_numpy(float), d["nominal_stress_MPa"].to_numpy(float)
        if t["mode"] in data:            # replicates: pool
            l = np.r_[data[t["mode"]][0], l]; P = np.r_[data[t["mode"]][1], P]
        data[t["mode"]] = (l, P)
    K = man.get("bulk_modulus_MPa")
    return man, data, K


def run(manifest, out, models, validate_lattice=None):
    man, data, K = load_manifest(manifest)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    report = dict(material=man.get("material"), manifest=str(manifest), models={})
    for model in models:
        p = fit(model, data)
        # shear modulus from the fit -> D1 from the measured K (or nu)
        G = float(np.gradient(nominal_stress(model, p, "uniaxial", np.array([0.999, 1.0, 1.001])),
                              [0.999, 1.0, 1.001])[1] / 3)
        Kd = K if K is not None else 2 * G * (1 + man["poisson"]) / (3 * (1 - 2 * man["poisson"]))
        D1 = 2 / Kd
        # leave-one-mode-out: fit on two modes, score the third (predictive check)
        lomo = {}
        for held in data:
            p2 = fit(model, data, modes=[m for m in data if m != held])
            lomo[held] = errors(model, p2, {held: data[held]})[held]
        card = calculix_card(model, p, D1)
        rec = dict(params=dict(zip(MODELS[model]["names"], map(float, p))), G0_MPa=G, K_MPa=Kd,
                   D1=D1, fit_errors=errors(model, p, data), leave_one_mode_out=lomo,
                   drucker_unstable_from=drucker_stable(model, p), card=card)
        lmax = max(float(data["uniaxial"][0].max()), 1.1); lmin = min(float(data["uniaxial"][0].min()), 0.9)
        rec["ccx_tension"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                              for k, v in ccx_cube(card, lmax).items()}
        rec["ccx_compression"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                                  for k, v in ccx_cube(card, lmin).items()}
        if validate_lattice:
            rec["lattice"] = lattice_check(card, validate_lattice)
        report["models"][model] = rec
        (out / f"{model}.card").write_text(card)
        print(f"{model}: {rec['params']}  fit {rec['fit_errors']}  "
              f"ccx tension converged={rec['ccx_tension']['converged']} "
              f"compression converged={rec['ccx_compression']['converged']}", flush=True)
    (out / "fit_report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def lattice_check(card, stl, crush=0.20):
    """Can this card carry the benchmark lattice piece to the screening crush?"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fea"))
    import fea_evaluator as FE
    coords, nid, tets, _ = FE.mesh_for_fea(stl)
    w = Path(tempfile.mkdtemp(prefix="matlat_")); H = float(np.ptp(coords[:, 2]))
    FE._write_inp(w / "job.inp", coords, nid, tets, disp=-crush * H)
    txt = (w / "job.inp").read_text()
    i = txt.index("*HYPERELASTIC"); j = txt.index("*SOLID SECTION")
    (w / "job.inp").write_text(txt[:i] + card + txt[j:])
    t, F = FE._run_ccx("job", str(w), timeout=FE.FEA_TIMEOUT_S)
    return dict(reached_strain=float(crush * t[-1]), converged=bool(abs(t[-1] - 1) < 1e-3))


# ---------------------------------------------------------------------------
def selftest():
    """Recover known parameters from synthetic 3-mode data. Writes NO material card."""
    truth = dict(mooney_rivlin=[3.0, 0.4], yeoh=[2.5, -0.05, 0.004], ogden2=[2.0, 2.5, 0.3, -1.5])
    rng = np.random.default_rng(1); ok = True
    rngs = dict(uniaxial=(0.6, 2.5), planar=(1.0, 2.0), equibiaxial=(1.0, 1.6))
    for model, p in truth.items():
        data = {}
        for m, (a, b) in rngs.items():
            l = np.linspace(a, b, 40); l = l[np.abs(l - 1) > 1e-3]
            P = nominal_stress(model, p, m, l); data[m] = (l, P * (1 + 0.01 * rng.standard_normal(len(l))))
        q = fit(model, data); e = errors(model, q, data)
        worst = max(v["rms_rel"] for v in e.values())
        print(f"[selftest] {model}: truth {p} -> fit {np.round(q, 4).tolist()}  worst mode RMS rel {worst:.3%}"
              f"  stable={drucker_stable(model, q)}")
        ok &= worst < 0.03
    if os.environ.get("CCX"):
        # CalculiX analytic check: the uniaxial cube must reproduce the closed form
        r = ccx_cube(calculix_card("mooney_rivlin", truth["mooney_rivlin"], 1e-4), 1.5)
        an = nominal_stress("mooney_rivlin", truth["mooney_rivlin"], "uniaxial", r["stretch"][1:])
        dev = float(np.max(np.abs(r["P"][1:] - an) / np.abs(an)))
        print(f"[selftest] ccx cube MR (D1=1e-4) to stretch 1.5: converged={r['converged']}, "
              f"max deviation from incompressible closed form {dev:.2%}")
        ok &= r["converged"] and dev < 0.03
    print("[selftest]", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("manifest", nargs="?")
    ap.add_argument("--out", default="data/material/fit")
    ap.add_argument("--models", nargs="+", default=["mooney_rivlin", "yeoh", "ogden1", "ogden2"],
                    choices=list(MODELS))
    ap.add_argument("--validate-lattice", help="benchmark piece STL the card must carry to 20%%")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.manifest:
        ap.error("manifest required (or --selftest)")
    run(a.manifest, a.out, a.models, a.validate_lattice)
