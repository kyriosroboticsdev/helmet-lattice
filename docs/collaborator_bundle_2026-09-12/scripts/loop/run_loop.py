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


def run(out_dir: str, max_iters: int, batch: int, resume: bool, extra: int):
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

    gate_cfg = GateConfig(min_wall_mm=0.8, nozzle_mm=0.4, run_linear_presolve=True)

    def log(ev): 
        with open(log_path, "a") as f: f.write(json.dumps(ev, default=str) + "\n")

    while True:
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
            design_id = f"it{it:03d}_{j}"
            try:
                stl_path = realize.params_to_stl(params, out / "stl" / f"{design_id}.stl")  # TODO(friend)
            except Exception as e:
                log(dict(iter=it, design=design_id, stage="realize", status="failed", err=str(e)[:150]))
                print(f"  [{design_id}] realize failed -> skip"); continue

            gate = check_fea_ready(str(stl_path), gate_cfg)
            if not gate.ok:
                row = {c: np.nan for c in schema.ALL_COLUMNS}
                row.update(dict(design_id=design_id, data_source="fea", topology=params.get("topology","blend"),
                                raw_file_path=str(stl_path), notes=f"GATE REJECT: {gate.reason}"))
                row["fea_status"] = "gate_rejected"
                append_to_master(row, str(master_csv))
                log(dict(iter=it, design=design_id, stage="gate", status="rejected", reason=gate.reason))
                print(f"  [{design_id}] gate rejected: {gate.reason}"); continue

            row = safe_schema_row(str(stl_path), design_id,
                                  topology=params.get("topology","blend"), design_meta=params)
            append_to_master(row, str(master_csv))
            log(dict(iter=it, design=design_id, stage="fea", status=row.get("fea_status"),
                     SEA=row.get("SEA_J_per_kg"), plateau=row.get("plateau_stress_kPa")))
            print(f"  [{design_id}] FEA {row.get('fea_status')}  "
                  f"SEA={row.get('SEA_J_per_kg')} plateau={row.get('plateau_stress_kPa')}")

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
    ap.add_argument("--resume", action="store_true", help="resume after a hard-cap stop")
    ap.add_argument("--extra", type=int, default=20, help="extra iterations on resume")
    a = ap.parse_args()
    run(a.out, a.max_iters, a.batch, a.resume, a.extra)
