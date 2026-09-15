"""Seed 11 end to end through the REAL loop code path (run_loop.evaluate_design)."""
import sys, json, time, shutil
from pathlib import Path
sys.argv = ["run_loop.py"]
sys.path.insert(0, r"<repo>\scripts\loop")
import run_loop as RL
import seed_lhs, realize, schema
from fea_readiness_gate import GateConfig
import fea_evaluator as E
print("CCX:", E.CCX, flush=True)

out = Path(r"<scratchpad>\e2e")
shutil.rmtree(out, ignore_errors=True); out.mkdir()
master, logp = out / "master_table.csv", out / "loop_log.jsonl"
def log(ev):
    print("   log:", json.dumps(ev, default=str), flush=True)
    with open(logp, "a") as f: f.write(json.dumps(ev, default=str) + "\n")

# keep the CalculiX working dir so the solve can be inspected
_orig = E.evaluate_stl
def keep(*a, **k):
    k["keep_files"] = True; k["timeout"] = 5400; r = _orig(*a, **k); print("   evaluate_stl ->", json.dumps(r, default=str), flush=True); return r
E.evaluate_stl = keep

p = seed_lhs.proposals_for_realize(20)[11]
t = time.time()
RL.evaluate_design(p, "seed011", out, GateConfig(), master, log, realize, schema, it=0, fea_res=60)
print(f"TOTAL {time.time()-t:.0f}s", flush=True)
if master.exists():
    import pandas as pd
    print(pd.read_csv(master).T.dropna().to_string())
