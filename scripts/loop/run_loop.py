"""
run_loop.py  --  Top-level autonomous design loop.
==================================================
Chains every component into one command:

  optimizer.propose  ->  realize (params->STL)  ->  FEA-readiness GATE
      ->  validated FEA (safe/graceful)  ->  ingest into master table
      ->  surrogate retrain  ->  STOPPING controller  ->  repeat / stop

Design goals baked in:
  * GRACEFUL: a design that fails the gate or the FEA is logged & SKIPPED,
    never crashes the loop (safe_schema_row + gate rejection rows).
  * RESUMABLE: stopping state is saved to JSON; a HARD_CAP stop can be
    manually reviewed and resumed.
  * HONEST: this is the integration harness. It WILL need a shakedown run on
    real hardware -- paths, imports, and a few data types are where surprises
    live. Run it once locally on a small max_iterations before scaling.

Usage:
  python scripts/loop/run_loop.py --out runs/exp1 --max-iters 40 --batch 2
  python scripts/loop/run_loop.py --out runs/exp1 --resume --extra 20   # after a hard cap

Pause: create an empty file runs/exp1/PAUSE. The run stops cleanly before the next
design (the current design and its FEA solve finish first). Delete PAUSE and re-run
with --resume; finished designs are skipped.

Requires the sibling modules:
  scripts/ml/{optimize,realize,surrogate_gp,data,schema}.py   (friend's)
  scripts/fea/{fea_evaluator,fea_readiness_gate}.py           (ours)
  scripts/loop/stopping.py                                    (ours)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for sub in ("scripts/ml", "scripts/fea", "scripts/loop"):
    sys.path.insert(0, str(ROOT / sub))

# --- our modules (always present) ---
from fea_readiness_gate import check_fea_ready, GateConfig
from fea_evaluator import safe_schema_row, append_to_master
from stopping import StoppingController, StopConfig, StopState, pareto_hypervolume

# --- friend's modules: imported lazily with a clear message if missing ---
def _import_friend():
    mods = {}
    try:
        import optimize, realize, data, schema
        mods.update(optimize=optimize, realize=realize, data=data, schema=schema)
    except Exception as e:
        print(f"[run_loop] Could not import the ML modules ({e}).\n"
              f"           This harness expects scripts/ml/{{optimize,realize,data,schema}}.py.\n"
              f"           Wire the two call sites marked TODO(friend) to your actual API names.")
        raise
    return mods


# ---- the two objectives the surrogate optimizes (minimized injury metrics) ----
# For FEA-only screening we use quasi-static proxies. Swap to HIC/BrIC once impact
# data exists. Both are 'lower is better' -> pareto_hypervolume negates internally.
OBJECTIVES = ["plateau_stress_kPa", "SEA_J_per_kg"]   # example: tune to your goal


def evaluate_design(params, design_id, out, gate_cfg, master_csv, log, realize, schema, it,
                    fea_res=60):
    """realize (coarse FEA mesh) -> printability screen -> FEA-readiness gate (meshes
    once) -> FEA on that same mesh -> master row. Every outcome is logged and (except
    realize failures) appended. Designs that pass get re-built at print resolution
    only when chosen for printing, not here."""
    import features as F
    t0 = time.time()
    info = realize.realize_proposal(params, out / "stl", fea_res=fea_res)
    stl_path = info.get("stl_path")
    # Contract geometry + blend_* columns so the surrogate has its features on every row.
    geom = F.derive_geometry(params)
    geom["topology"] = params.get("topology", "blend")
    extra = {k: v for k, v in geom.items() if k.startswith("blend_")}
    extra["blend_seed"] = params.get("blend_seed")
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
        print(f"  [{design_id}] {stage} rejected: {reason}")

    if not stl_path:
        log(dict(iter=it, design=design_id, stage="realize", status="failed", err=info.get("status")))
        print(f"  [{design_id}] realize failed -> skip ({info.get('status')})"); return
    if not info.get("printable", True):
        return reject("print", info.get("status"))

    gate = check_fea_ready(str(stl_path), gate_cfg)
    if not gate.ok:
        return reject("gate", gate.reason)

    meta = {k: v for k, v in geom.items() if k in schema.ALL_COLUMNS and k != "rel_density_measured"}
    t_fea = time.time()
    row = safe_schema_row(str(stl_path), design_id,
                          topology=params.get("topology", "blend"), design_meta=meta,
                          mesh=getattr(gate, "mesh", None))
    if pd.isna(row.get("rel_density_measured")):   # failed solve: keep the as-built density
        row["rel_density_measured"] = geom.get("rel_density_measured")
    row.update(extra)
    append_to_master(row, str(master_csv))
    log(dict(iter=it, design=design_id, stage="fea", status=row.get("fea_status"),
             SEA=row.get("SEA_J_per_kg"), plateau=row.get("plateau_stress_kPa"),
             sec_prep=round(t_fea - t0), sec_fea=round(time.time() - t_fea)))
    print(f"  [{design_id}] FEA {row.get('fea_status')}  "
          f"SEA={row.get('SEA_J_per_kg')} plateau={row.get('plateau_stress_kPa')}  "
          f"(prep {t_fea - t0:.0f}s, FEA {time.time() - t_fea:.0f}s)")


def run(out_dir: str, max_iters: int, batch: int, resume: bool, extra: int,
        seed_n: int = 20, seed_only: bool = False, fea_res: int = 60):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    master_csv = out / "master_table.csv"
    state_json = out / "stop_state.json"
    log_path   = out / "loop_log.jsonl"

    mods = _import_friend()
    optimize, realize, data, schema = (mods[k] for k in ("optimize","realize","data","schema"))

    # ---- stopping controller (fresh or resumed) ----
    cfg = StopConfig(max_iterations=max_iters)
    if resume and state_json.exists():
        ctrl = StoppingController(cfg, StopState.load(state_json))
        ctrl.resume(extra_iterations=extra or 20)
        print(f"[run_loop] RESUMED at iter {ctrl.state.iteration}, new cap {ctrl.cfg.max_iterations}")
    else:
        ctrl = StoppingController(cfg)

    gate_cfg = GateConfig(min_wall_mm=0.8, nozzle_mm=0.4)

    def log(ev):
        with open(log_path, "a") as f: f.write(json.dumps(ev, default=str) + "\n")

    # ---- pause: create <out>/PAUSE to stop cleanly at the next design boundary ----
    # The design in progress (including its FEA solve) finishes first; nothing is
    # lost. Delete the file and re-run with --resume to continue where it stopped.
    pause_file = out / "PAUSE"
    def paused(where):
        if not pause_file.exists():
            return False
        log(dict(event="paused", where=where))
        print(f"\n[run_loop] PAUSE file found -> stopped cleanly {where}.\n"
              f"           Delete {pause_file} and re-run with --resume to continue.")
        return True

    # ---- 0) initial dataset: Latin Hypercube seed --------------------------------
    # The plan (n) is recorded so a resumed run regenerates the SAME seeds, then
    # skips every seed already in the master table -- a pause or crash mid-seed
    # picks up at the next unfinished design instead of silently dropping the rest.
    plan_json = out / "seed_plan.json"
    if plan_json.exists():
        seed_n = json.loads(plan_json.read_text())["seed_n"]
    elif not master_csv.exists():
        plan_json.write_text(json.dumps({"seed_n": seed_n}))
    if seed_n > 0 and plan_json.exists():
        import seed_lhs
        seeds = seed_lhs.proposals_for_realize(n=seed_n)
        done = set(pd.read_csv(master_csv)["design_id"]) if master_csv.exists() else set()
        todo = [(j, p) for j, p in enumerate(seeds) if f"seed{j:03d}" not in done]
        if todo:
            print(f"\n===== SEED: {len(todo)} of {len(seeds)} LHS designs to run =====")
        for j, params in todo:
            if paused(f"before seed{j:03d}"):
                return
            evaluate_design(params, f"seed{j:03d}", out, gate_cfg, master_csv, log,
                            realize, schema, it=0, fea_res=fea_res)
    if seed_only:
        print(f"\n--seed-only: stopping after the seed phase. Results: {master_csv}")
        return

    while True:
        # Between iterations only: the stopping state is saved after each iteration,
        # so a resumed run re-proposes from the full data with consistent design ids.
        if paused(f"before iteration {ctrl.state.iteration + 1}"):
            return
        it = ctrl.state.iteration + 1
        print(f"\n===== ITERATION {it} =====")

        # 1) train surrogate on current data + propose next designs -----------
        # TODO(friend): match these to your real API (optimize.propose_next / surrogate)
        df = pd.read_csv(master_csv) if master_csv.exists() else data.empty_frame()
        proposals = optimize.propose_next(df, batch_size=batch)   # -> list of param dicts + scores
        params_list = proposals["params"] if isinstance(proposals, dict) else proposals
        top_ei = float(proposals.get("top_score", 1.0)) if isinstance(proposals, dict) else 1.0

        # 2) realize + gate + FEA each proposal (graceful) --------------------
        for j, params in enumerate(params_list):
            evaluate_design(params, f"it{it:03d}_{j}", out, gate_cfg, master_csv, log,
                            realize, schema, it, fea_res=fea_res)

        # 3) compute best metric (Pareto hypervolume over successful rows) -----
        df = pd.read_csv(master_csv)
        ok_df = df[df.get("fea_status","ok") == "ok"] if "fea_status" in df else df
        best_metric = pareto_hypervolume(ok_df, OBJECTIVES) if len(ok_df) else -1e18

        # 4) stopping decision -------------------------------------------------
        stop, reason = ctrl.update(best_metric=best_metric, top_ei=top_ei,
                                   note=f"n_ok={len(ok_df)}")
        ctrl.state.save(state_json)
        print(f"  best_metric(HV)={best_metric:.4g}  EI={top_ei:.4g}  -> stop={stop} {reason or ''}")
        if stop:
            print("\n" + ctrl.summary())
            print(f"\nSTOPPED: {reason}")
            print(f"Results: {master_csv}\nState:   {state_json}  (resume with --resume --extra N)")
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Autonomous lattice design loop")
    ap.add_argument("--out", required=True, help="output run directory")
    ap.add_argument("--max-iters", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2, help="designs proposed per iteration")
    ap.add_argument("--resume", action="store_true", help="continue a paused, crashed or hard-capped run in --out")
    ap.add_argument("--extra", type=int, default=20, help="extra iterations on resume")
    ap.add_argument("--seed-n", type=int, default=20,
                    help="LHS seed designs evaluated before BO on a fresh run (0 = none)")
    ap.add_argument("--seed-only", action="store_true",
                    help="run the seed phase and stop (BO step is not wired for FEA objectives yet)")
    ap.add_argument("--fea-res", type=int, default=60,
                    help="marching-cubes samples for the FEA mesh (print mesh is built separately)")
    a = ap.parse_args()
    run(a.out, a.max_iters, a.batch, a.resume, a.extra, a.seed_n, a.seed_only, a.fea_res)
