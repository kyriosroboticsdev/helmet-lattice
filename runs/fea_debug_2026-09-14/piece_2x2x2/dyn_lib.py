"""Static vs implicit-dynamic compression inputs built from fea_evaluator's writer (BCs unchanged)."""
import os, re, sys, time, subprocess, tempfile
import numpy as np
sys.path.insert(0, r"<repo>\scripts\fea")
import fea_evaluator as E

STATIC_STEP = "*STEP, NLGEOM, INC=2000\n*STATIC\n0.01, 1.0, 1e-5, 0.05\n"

def write(wd, coords, nid, tets, crush, mode, T=0.05, alpha=-0.3):
    inp = os.path.join(wd, "job.inp")
    H = float(np.ptp(coords[:, 2]))
    E._write_inp(inp, coords, nid, tets, disp=-crush * H, material="neo_hooke")
    s = open(inp).read()
    assert STATIC_STEP in s
    if mode == "dynamic":
        # density in tonne/mm^3 for the N-mm-s system; displacement ramped over T
        s = s.replace("*SOLID SECTION", f"*DENSITY\n{E.TPU_DENSITY_G_PER_MM3 * 1e-6:.4e}\n*SOLID SECTION", 1)
        s = s.replace(STATIC_STEP,
                      f"*AMPLITUDE, NAME=RAMP\n0., 0., {T}, 1.\n"
                      f"*STEP, NLGEOM, INC=100000\n*DYNAMIC, ALPHA={alpha}\n{T/100}, {T}, {T*1e-6}, {T/50}\n")
        s = re.sub(r"\*BOUNDARY\nTOP, 3, 3,", "*BOUNDARY, AMPLITUDE=RAMP\nTOP, 3, 3,", s)
        s = s.replace("*END STEP", "*EL PRINT, ELSET=EALL, TOTALS=ONLY\nELKE\n*END STEP")
    open(inp, "w").write(s)
    return H

def run(label, coords, nid, tets, crush, mode, timeout, **kw):
    wd = tempfile.mkdtemp(prefix=f"dyn_{mode}_")
    H = write(wd, coords, nid, tets, crush, mode, **kw)
    t0 = time.time()
    try:
        r = subprocess.run([E.CCX, "job"], cwd=wd, capture_output=True, text=True, timeout=timeout)
        open(os.path.join(wd, "stdout.txt"), "w").write(r.stdout)
        status = "finished"
    except subprocess.TimeoutExpired:
        status = f"TIMEOUT {timeout}s"
    dat = open(os.path.join(wd, "job.dat")).read() if os.path.exists(os.path.join(wd, "job.dat")) else ""
    times = [float(x) for x in re.findall(r"total force \(fx,fy,fz\) for set TOP and time\s+([\d.E+-]+)", dat)]
    fz = [float(re.findall(r"[-\d.E+]+", b)[2]) for b in re.findall(r"total force \(fx,fy,fz\) for set TOP and time[^\n]*\n\n\s*([^\n]+)", dat)]
    T = kw.get("T", 0.05) if mode == "dynamic" else 1.0
    frac = np.array(times) / T
    area = float(np.ptp(coords[:, 0]) * np.ptp(coords[:, 1]))
    strain = frac * crush; stress_kpa = -np.array(fz) / area * 1000
    ke = [float(x) for x in re.findall(r"total kinetic energy[^\n]*\n\n\s*([-\d.E+]+)", dat)]
    ie = [float(x) for x in re.findall(r"total internal energy[^\n]*\n\n\s*([-\d.E+]+)", dat)]
    errs = sorted(set(re.findall(r"\*ERROR[^\n]*", open(os.path.join(wd, "stdout.txt")).read())))[:3] \
        if os.path.exists(os.path.join(wd, "stdout.txt")) else []
    reached = strain[-1] if len(strain) else 0.0
    print(f"{label}: {status} in {time.time()-t0:.0f}s | reached {reached*100:.2f}% of {crush*100:.0f}% crush "
          f"| {len(times)} points | errors={errs} | {wd}", flush=True)
    return dict(wd=wd, strain=strain, stress_kpa=stress_kpa, ke=np.array(ke), ie=np.array(ie), reached=reached)

def ke_ratio(res, coords):
    """Peak kinetic energy / external work done (N*mm). Quasi-static if this is small (<~5%)."""
    if len(res["strain"]) < 2 or len(res["ke"]) == 0:
        return float("nan")
    H = float(np.ptp(coords[:, 2])); area = float(np.ptp(coords[:, 0]) * np.ptp(coords[:, 1]))
    F = res["stress_kpa"] / 1000 * area; u = res["strain"] * H
    work = float(np.trapezoid(np.r_[0, F], np.r_[0, u]))
    return float(res["ke"].max() / work) if work > 0 else float("nan")
