"""
run_loop.py  --  Self-running FEA design loop, until diminishing returns.
========================================================================
  seed (Latin hypercube)
    -> each design: realize coarse FEA mesh -> printability screen -> FEA gate
       (meshes once) -> CalculiX FEA -> row in master_table.csv
  -> iterate: fit GPs on finished designs -> log-qEHVI proposes a ranked, diverse
     shortlist -> evaluate in rank order until `batch` designs reach FEA
     -> hypervolume vs a FIXED reference -> stopping controller
  -> confirm: re-run the Pareto-front designs on the full tile

Screening model: a 2x2x2-cell piece of each design crushed 20% (~3-10 min per
design; reaches the plateau onset). Confirm model: the full 50.8x50.8x38.1 mm
coupon crushed 10% (~30 min), which the static solve can reliably finish. The
confirm report gives the piece-vs-tile offset at 10% crush for those designs.

Objectives (scripts/loop/fea_optimize.py), both at a fixed 10% strain: maximize
SEA, minimize PEAK stress. Stops when ANY of the stopping.py criteria fires: no hypervolume
improvement for `patience` iterations, expected improvement below a fraction of the
first iteration's, or the hard cap.

Usage (Git Bash, ccx on PATH or CCX=<path to ccx.exe>):
  python scripts/loop/run_loop.py --out runs/exp1
  python scripts/loop/run_loop.py --out runs/exp1 --resume            # after a pause/crash
  python scripts/loop/run_loop.py --out runs/exp1 --resume --extra 20 # after a hard cap
  python scripts/loop/run_loop.py --out runs/exp1 --resume --confirm-only

Pause: create an empty file <out>/PAUSE. The run stops cleanly before the next design
(the design in progress, including its solve, finishes first). Delete PAUSE and re-run
with --resume; finished designs are skipped.

Outputs in <out>/: master_table.csv (one row per design attempt), curves/<id>.csv
(stress-strain), loop_log.jsonl, stop_state.json, hv_reference.json, seed_plan.json,
confirm_report.md.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for sub in ("scripts/ml", "scripts/fea", "scripts/loop"):
    sys.path.insert(0, str(ROOT / sub))

from fea_readiness_gate import check_fea_ready, GateConfig
from fea_evaluator import safe_schema_row, append_to_master
from stopping import StoppingController, StopConfig, StopState

# Proposal fields needed to rebuild a design exactly (stored per row as JSON).
PARAM_KEYS = ["rel_density_measured", "cell_size_mm", "cell_aspect_y", "cell_aspect_z",
              "liner_thickness_mm", "gradient_param", "shear_anisotropy_ratio",
              "topology", "blend", "blend_seed"]

MODELS = {   # fea_model -> realize/evaluate settings
    "piece": dict(piece_cells=2, crush=0.20, timeout=900),    # healthy solves take 2-5 min
    "full":  dict(piece_cells=None, crush=0.10, timeout=5400),
}


def evaluate_design(params, design_id, out, gate_cfg, master_csv, log, realize, schema, it,
                    fea_res=60, fea_model="piece"):
    """realize (coarse FEA mesh) -> printability screen -> FEA-readiness gate (meshes
    once) -> FEA on that same mesh -> master row. Every outcome is logged and (except
    Returns the outcome: "ok", "failed", "print_rejected", "gate_rejected" or
    "realize_rejected"."""
    import features as F
    cfg = MODELS[fea_model]
    t0 = time.time()
    info = realize.realize_proposal(params, out / "stl", fea_res=fea_res,
                                    piece_cells=cfg["piece_cells"])
    stl_path = info.get("stl_path")
    # Contract geometry + blend_* columns so the surrogate has its features on every row.
    geom = F.derive_geometry(params)
    geom["topology"] = params.get("topology", "blend")
    extra = {k: v for k, v in geom.items() if k.startswith("blend_")}
    extra["blend_seed"] = params.get("blend_seed")
    extra["fea_model"] = fea_model
    extra["proposal_json"] = json.dumps({k: params[k] for k in PARAM_KEYS if k in params})
    extra["iteration"] = it
    # realize enlarges the cell for wall families; the GP features use the requested
    # cell_size_mm, so keep what was actually built for analysis.
    extra["cell_size_built_mm"] = info.get("cell_size_built_mm")
    if info.get("rel_density_as_built") is not None:
        geom["rel_density_measured"] = info["rel_density_as_built"]

    def reject(stage, reason):
        row = {c: np.nan for c in schema.ALL_COLUMNS}
        row.update({k: v for k, v in geom.items() if k in row})
        row.update(extra)
        row.update(dict(design_id=design_id, replicate_id=1, data_source="fea",
                        raw_file_path=str(stl_path), notes=f"{stage.upper()} REJECT: {reason}"))
        row["fea_status"] = f"{stage}_rejected"
        append_to_master(row, str(master_csv))
        log(dict(iter=it, design=design_id, stage=stage, status="rejected", reason=reason))
        print(f"  [{design_id}] {stage} rejected: {reason}", flush=True)
        return f"{stage}_rejected"

    if not stl_path:
        # Recorded as a row: a resume must not retry it, and the feasibility model
        # should learn to avoid it (e.g. a generator error for that blend/density).
        return reject("realize", info.get("status"))
    if not info.get("printable", True):
        return reject("print", info.get("status"))

    gate = check_fea_ready(str(stl_path), gate_cfg)
    if not gate.ok:
        return reject("gate", gate.reason)

    meta = {k: v for k, v in geom.items() if k in schema.ALL_COLUMNS and k != "rel_density_measured"}
    t_fea = time.time()
    row = safe_schema_row(str(stl_path), design_id,
                          topology=params.get("topology", "blend"), design_meta=meta,
                          mesh=getattr(gate, "mesh", None), crush=cfg["crush"],
                          timeout=cfg["timeout"])
    curve = row.pop("_stress_strain", None)
    if curve:
        (out / "curves").mkdir(exist_ok=True)
        pd.DataFrame(curve, columns=["strain", "stress_kPa"]).to_csv(
            out / "curves" / f"{design_id}.csv", index=False)
    if pd.isna(row.get("rel_density_measured")):   # failed solve: keep the as-built density
        row["rel_density_measured"] = geom.get("rel_density_measured")
    row.update(extra)
    row["sec_prep"] = round(t_fea - t0)
    row["sec_fea"] = round(time.time() - t_fea)
    append_to_master(row, str(master_csv))
    log(dict(iter=it, design=design_id, stage="fea", model=fea_model, status=row.get("fea_status"),
             SEA=row.get("SEA_J_per_kg"), stress=row.get("stress_at_max_strain_kPa"),
             reached=row.get("fea_max_strain"), sec_prep=row["sec_prep"], sec_fea=row["sec_fea"]))
    print(f"  [{design_id}] FEA({fea_model}) {row.get('fea_status')}  "
          f"SEA={row.get('SEA_J_per_kg')} J/kg  stress={row.get('stress_at_max_strain_kPa')} kPa  "
          f"(prep {row['sec_prep']}s, FEA {row['sec_fea']}s)", flush=True)
    return row.get("fea_status", "failed")


def confirm_front(out, master_csv, gate_cfg, log, realize, schema, fea_res, top_n, crush, ref):
    """Re-run the screening Pareto front on the full tile; write the offset report."""
    import fea_optimize as FO
    df = pd.read_csv(master_csv)
    rows = FO.training_rows(df, crush)
    if rows.empty:
        print("[confirm] no finished screening designs to confirm"); return
    Y = FO.objective_matrix(rows)
    front = rows[FO.pareto_mask(Y)].copy()
    front["_score"] = FO.objective_matrix(front)[:, 0]          # best SEA first
    front = front.sort_values("_score", ascending=False).head(top_n)
    done = set(df["design_id"])
    print(f"\n===== CONFIRM: {len(front)} Pareto designs on the full tile =====", flush=True)
    for _, r in front.iterrows():
        fid = f"{r['design_id']}_full"
        if fid in done:
            continue
        if (out / "PAUSE").exists():
            print("[run_loop] PAUSE file found -> stopped before confirm", fid); return
        params = json.loads(r["proposal_json"])
        evaluate_design(params, fid, out, gate_cfg, master_csv, log, realize, schema,
                        it="confirm", fea_res=fea_res, fea_model="full")

    df = pd.read_csv(master_csv)
    lines = ["# Confirm report: 2x2x2 piece vs full tile", "",
             "Both at a fixed 10% crush. Offset = (piece - full) / full.", "",
             "| design | piece stress@10% kPa | full stress@10% kPa | offset | piece SEA@10% | full SEA@10% | offset | full status |",
             "|---|---|---|---|---|---|---|---|"]
    offs = []
    for _, r in front.iterrows():
        f = df[df["design_id"] == f"{r['design_id']}_full"]
        if f.empty:
            continue
        f = f.iloc[-1]
        ps, fs = r.get("stress_at_10pct_kPa"), f.get("stress_at_10pct_kPa")
        pe, fe = r.get("SEA_to_10pct_J_per_kg"), f.get("SEA_to_10pct_J_per_kg")
        o1 = (ps - fs) / fs if pd.notna(fs) and fs else np.nan
        o2 = (pe - fe) / fe if pd.notna(fe) and fe else np.nan
        if pd.notna(o1):
            offs.append((o1, o2))
        lines.append(f"| {r['design_id']} | {ps:.1f} | {fs:.1f} | {o1:+.1%} | {pe:.1f} | {fe:.1f} | {o2:+.1%} | {f['fea_status']} |")
    if offs:
        a = np.array(offs, dtype=float)
        lines += ["", f"Mean stress offset {np.nanmean(a[:,0]):+.1%} (range {np.nanmin(a[:,0]):+.1%} to {np.nanmax(a[:,0]):+.1%}); "
                      f"mean SEA offset {np.nanmean(a[:,1]):+.1%}; n={len(a)}."]
    lines += ["", f"Final screening hypervolume (fixed reference): {FO.hypervolume(FO.training_rows(df, crush), ref):.4g}"]
    (out / "confirm_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def run(out_dir: str, max_iters: int, batch: int, resume: bool, extra: int,
        seed_n: int = 30, seed_only: bool = False, fea_res: int = 60, shortlist: int = 3,
        confirm_top: int = 5, confirm_only: bool = False, n_candidates: int = 300):
    import realize, schema
    import fea_optimize as FO
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    master_csv = out / "master_table.csv"
    state_json = out / "stop_state.json"
    ref_json   = out / "hv_reference.json"
    log_path   = out / "loop_log.jsonl"
    crush = MODELS["piece"]["crush"]
    FO.CURVES_DIR = out / "curves"    # objectives are computed from the saved curves

    cfg = StopConfig(max_iterations=max_iters)
    if resume and state_json.exists():
        ctrl = StoppingController(cfg, StopState.load(state_json))
        ctrl.resume(extra_iterations=extra or 20)
        print(f"[run_loop] RESUMED at iter {ctrl.state.iteration}, cap {ctrl.cfg.max_iterations}")
    else:
        ctrl = StoppingController(cfg)

    gate_cfg = GateConfig(min_wall_mm=0.8, nozzle_mm=0.4)

    def log(ev):
        ev = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), **ev}
        with open(log_path, "a") as f: f.write(json.dumps(ev, default=str) + "\n")

    pause_file = out / "PAUSE"
    def paused(where):
        if not pause_file.exists():
            return False
        log(dict(event="paused", where=where))
        print(f"\n[run_loop] PAUSE file found -> stopped cleanly {where}.\n"
              f"           Delete {pause_file} and re-run with --resume to continue.")
        return True

    def load():
        return pd.read_csv(master_csv) if master_csv.exists() else pd.DataFrame()

    if confirm_only:
        df = load()
        ref = FO.reference_point(FO.training_rows(df, crush), ref_json)
        return confirm_front(out, master_csv, gate_cfg, log, realize, schema, fea_res,
                             confirm_top, crush, ref)

    # ---- 0) seed: Latin hypercube, recorded so a resumed run regenerates the SAME
    # seeds and skips the ones already in the master table.
    plan_json = out / "seed_plan.json"
    if plan_json.exists():
        seed_n = json.loads(plan_json.read_text())["seed_n"]
    elif not master_csv.exists():
        plan_json.write_text(json.dumps({"seed_n": seed_n}))
    if seed_n > 0 and plan_json.exists():
        import seed_lhs
        seeds = seed_lhs.proposals_for_realize(n=seed_n)
        done = set(load().get("design_id", []))
        todo = [(j, p) for j, p in enumerate(seeds) if f"seed{j:03d}" not in done]
        if todo:
            print(f"\n===== SEED: {len(todo)} of {len(seeds)} LHS designs to run =====", flush=True)
        for j, params in todo:
            if paused(f"before seed{j:03d}"):
                return
            evaluate_design(params, f"seed{j:03d}", out, gate_cfg, master_csv, log,
                            realize, schema, it=0, fea_res=fea_res)
    if seed_only:
        print(f"\n--seed-only: stopping after the seed phase. Results: {master_csv}")
        return

    rows = FO.training_rows(load(), crush)
    if len(rows) < FO.MIN_TRAIN:
        print(f"\n[run_loop] Only {len(rows)} seed designs finished FEA (need {FO.MIN_TRAIN}). "
              f"Re-run with a larger --seed-n in a new --out directory.")
        log(dict(event="stopped", reason="too_few_seed_results", n=len(rows)))
        return
    ref = FO.reference_point(rows, ref_json)

    # ---- 1) optimize until diminishing returns ---------------------------------
    stop = bool(ctrl.state.stopped)
    while not stop:
        if paused(f"before iteration {ctrl.state.iteration + 1}"):
            return
        it = ctrl.state.iteration + 1
        df = load()
        print(f"\n===== ITERATION {it} =====", flush=True)
        t_it = time.time()
        try:
            props, top_acq, fit = FO.propose(df, crush, ref, n_propose=batch * shortlist,
                                             n_candidates=n_candidates, seed=1000 + it)
        except Exception as e:
            log(dict(iter=it, event="propose_failed", err=f"{type(e).__name__}: {e}"[:300]))
            print(f"[run_loop] proposal step failed: {type(e).__name__}: {e}")
            raise
        print(f"  GP on {fit['n_train']} designs x {fit['n_features']} features; "
              f"top expected HV improvement {top_acq:.4g}; feasibility model on "
              f"{fit['feas_labels']} attempts ({fit['feas_rate']:.0%} reached FEA), "
              f"mean P(feasible) shortlist {fit['p_feasible_chosen']:.2f} vs pool "
              f"{fit['p_feasible_pool']:.2f}", flush=True)
        log(dict(iter=it, event="proposed", n=len(props), **fit))

        prefix = f"it{it:03d}_"
        taken = sum(str(d).startswith(prefix) for d in df.get("design_id", []))  # crash-resume safe
        reached_fea = 0
        for j, params in enumerate(props[taken:], start=taken):
            if reached_fea >= batch:
                break
            status = evaluate_design(params, f"{prefix}{j}", out, gate_cfg, master_csv, log,
                                     realize, schema, it, fea_res=fea_res)
            reached_fea += status in ("ok", "failed")
        if reached_fea == 0:
            print(f"  !! no proposal of iteration {it} reached FEA (all rejected by the screens)")

        hv = FO.hypervolume(FO.training_rows(load(), crush), ref)
        if reached_fea:
            stop, reason = ctrl.update(best_metric=hv, top_ei=top_acq,
                                       note=f"designs_to_fea={reached_fea}")
        else:
            # Nothing was measured, so "no improvement" would be a screening artifact,
            # not diminishing returns: count the iteration toward the hard cap only.
            s = ctrl.state
            s.iteration += 1
            s.history.append(dict(iteration=s.iteration, best_metric=round(float(hv), 6),
                                  improved=False, stale_rounds=s.stale_rounds, top_ei=top_acq,
                                  ei_frac=None, note="no design reached FEA; not counted as stale"))
            stop = s.iteration >= ctrl.cfg.max_iterations
            reason = f"HARD_CAP: reached max_iterations={ctrl.cfg.max_iterations}" if stop else None
            s.stopped, s.stop_reason = stop, reason
        ctrl.state.save(state_json)
        log(dict(iter=it, event="iteration_done", hv=hv, top_acq=top_acq, stop=stop,
                 reason=reason, sec=round(time.time() - t_it)))
        print(f"  hypervolume={hv:.4g}  top_acq={top_acq:.4g}  ({time.time()-t_it:.0f}s)"
              f"  -> stop={stop} {reason or ''}", flush=True)

    print("\n" + ctrl.summary(), flush=True)
    # ---- 2) confirm the front on the full tile ---------------------------------
    if confirm_top > 0:
        confirm_front(out, master_csv, gate_cfg, log, realize, schema, fea_res,
                      confirm_top, crush, ref)
    print(f"\nResults: {master_csv}\nState:   {state_json}  "
          f"(after a HARD_CAP stop: --resume --extra N)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Self-running FEA design loop")
    ap.add_argument("--out", required=True, help="output run directory")
    ap.add_argument("--max-iters", type=int, default=40, help="hard cap on BO iterations")
    ap.add_argument("--batch", type=int, default=2, help="designs sent to FEA per iteration")
    ap.add_argument("--shortlist", type=int, default=3,
                    help="proposals ranked per FEA slot, to replace screen rejections")
    ap.add_argument("--resume", action="store_true", help="continue a paused, crashed or hard-capped run in --out")
    ap.add_argument("--extra", type=int, default=20, help="extra iterations when resuming a hard cap")
    ap.add_argument("--seed-n", type=int, default=30, help="LHS seed designs on a fresh run (0 = none)")
    ap.add_argument("--seed-only", action="store_true", help="run the seed phase and stop")
    ap.add_argument("--fea-res", type=int, default=60,
                    help="marching-cubes samples across the full tile (the piece uses the same pitch)")
    ap.add_argument("--candidates", type=int, default=300,
                    help="candidate pool scored per iteration (~0.4 s each: sa_to_v is measured)")
    ap.add_argument("--confirm-top", type=int, default=5, help="Pareto designs re-run on the full tile (0 = skip)")
    ap.add_argument("--confirm-only", action="store_true", help="only run/refresh the confirm phase")
    a = ap.parse_args()
    run(a.out, a.max_iters, a.batch, a.resume, a.extra, a.seed_n, a.seed_only, a.fea_res,
        a.shortlist, a.confirm_top, a.confirm_only, a.candidates)
