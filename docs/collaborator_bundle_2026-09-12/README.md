# Helmet Lattice — FEA + Loop Bundle
### The pieces that close the generative → FEA → ML loop

This bundle contains the four modules built to connect your friend's ML/generator
pipeline to a validated FEA engine, plus the autonomous loop harness that chains
everything. Drop these into your friend's repo (`helmet-lattice/scripts/...`).

---

## 0. Honest status (read this first)

Every component here is **built and individually tested**. What has NOT happened yet:
- The whole chain has never run **connected, end-to-end, on real hardware**.
- Two call sites in `run_loop.py` are marked `TODO(friend)` — they must be wired to
  your friend's actual function names (`propose_next`, `params_to_stl`, etc.).
- A shakedown run WILL surface path/import/dtype hiccups. That's normal integration
  work, not a defect. Run it once with `--max-iters 3` on a laptop before scaling.

So: this is a **complete, tested toolkit + integration harness** — not a
press-button-and-walk-away product until you've done one supervised local run.

---

## 1. What each file does

| File | Role |
|---|---|
| `scripts/fea/fea_evaluator.py`      | Runs validated CalculiX quasi-static compression on an STL using the calibrated TPU-95A Ogden model + frictionless BCs. Outputs a schema-perfect row. `safe_schema_row()` = graceful (never crashes the loop). |
| `scripts/fea/fea_readiness_gate.py` | Rejects un-simulatable geometry BEFORE the expensive FEA. 4 checks: watertight+single-body, min wall thickness, mesh-ability, linear pre-solve. |
| `scripts/loop/stopping.py`          | Stopping controller: (1) hard cap, (2) no-improvement, (3) low expected-improvement. Resumable after a hard-cap stop. |
| `scripts/loop/run_loop.py`          | Top-level harness chaining propose → gate → FEA → ingest → stop → repeat. |

---

## 2. Requirements

- **CalculiX** (`ccx`) on PATH. (Bundled with PrePoMax, or `apt install calculix-ccx`.)
- Python: `numpy pandas gmsh trimesh scipy` and optionally `pymeshlab` (for STL
  decimation) and `botorch/torch` (for exact hypervolume; a fallback exists without it).
- Your friend's ML modules: `scripts/ml/{optimize,realize,surrogate_gp,data,schema}.py`.

```bash
pip install numpy pandas gmsh trimesh scipy pymeshlab
```

---

## 3. The validated material model (baked into fea_evaluator.py)

Ogden N=3, mu>0 (CalculiX/SimScale-compatible), tension+compression consistent
(Option-B: compression anchored to YOUR measured samples, tension shape from
cross-validated 2024-25 TPU-95A literature). Validated against your real
compression-button test: **~21% mean error, 3% at 30% strain.**

    mu    = [0.25246, 0.01695, 11.30314]  MPa
    alpha = [1.9058,  5.9637,  1.9056]
    D1    = 0.0435  (1/MPa, near-incompressible)
    density = 1.22e-3 g/mm^3

**Critical boundary condition:** FRICTIONLESS platens (bottom z-fixed, symmetry
planes kill rigid body, top pushed in z). DO NOT constrain faces laterally — that
caused a 4x-too-stiff blow-up. This is already correct in the code.

To swap in better data later (e.g. matched-rate lab tests): edit the OGDEN_MU /
OGDEN_ALPHA constants at the top of `fea_evaluator.py`. Nothing else changes.

---

## 4. How to run

### One design (test the FEA engine)
```bash
python scripts/fea/fea_evaluator.py path/to/lattice.stl --design-id D01 \
       --topology blend --append-to runs/test/master_table.csv
```

### Gate one design (test the readiness filter)
```bash
python scripts/fea/fea_readiness_gate.py path/to/lattice.stl --min-wall 0.8
```

### The full loop (after wiring the 2 TODO call sites)
```bash
# small supervised shakedown FIRST:
python scripts/loop/run_loop.py --out runs/shakedown --max-iters 3 --batch 1

# real run:
python scripts/loop/run_loop.py --out runs/exp1 --max-iters 40 --batch 2

# after a HARD_CAP stop + manual review, continue:
python scripts/loop/run_loop.py --out runs/exp1 --resume --extra 20
```

---

## 5. Wiring the two integration points (for your friend)

In `run_loop.py`, search for `TODO(friend)` — two spots:

1. **Propose:** `optimize.propose_next(df, batch_size=batch)` must return the
   proposed parameter dicts and, ideally, the top acquisition (qEHVI) score as
   `top_score` (used for stopping criterion 3). Adjust to your real function.
2. **Realize:** `realize.params_to_stl(params, out_path)` must turn a proposal into
   a **watertight STL** (ideally the generator's DIRECT export, not via Fusion).

Also set `OBJECTIVES` (top of run_loop.py) to the metrics you optimize. For FEA-only
screening use `plateau_stress_kPa` + `SEA_J_per_kg`; switch to `HIC`/`BrIC` once you
have physical-impact data.

---

## 6. How the loop stays reliable (the design that makes it autonomous)

- **Gate before FEA:** un-meshable / too-thin / numerically-broken designs are
  rejected in seconds and logged, never reaching the expensive solve.
- **Graceful FEA:** `safe_schema_row()` turns any mesh/solve failure into a logged
  `fea_status="failed"` row and continues. One bad design cannot halt the run.
- **Resumable stop:** hard-cap stops save state; review then `--resume`.
- **Everything logged:** `runs/<exp>/loop_log.jsonl` records every design's fate
  (realized / gate-rejected / fea-ok / fea-failed) — including WHY it failed, which
  you can feed back to the optimizer as "avoid this region."

---

## 7. Two fixes owed on the generator side (see Generator_Fixes_For_Friend.pdf)

1. **Density bug (1 line):** flip gyroid & schwartz to `iso="quantile"` in
   `generate_tiles.py` FIELD_TOPOLOGIES, so requested density is honored.
   (Training data is fine — the surrogate uses `rel_density_measured`.)
2. **Direct watertight export:** feed the loop the generator's watertight mesh, NOT
   the Fusion-exported STL. This makes the gate pass and FEA converge far more often.

---

## 8. What still needs YOU (the honest to-do)

- [ ] Friend: apply the density fix + wire the 2 TODO call sites.
- [ ] Friend: enable direct watertight mesh export.
- [ ] You: `ccx` installed, run the 3-iteration shakedown locally, fix path/import hiccups.
- [ ] You: tune stopping thresholds after watching one real run.
- [ ] Optional: swap in matched-rate lab material data when it arrives (edit 6 numbers).

Once the shakedown run completes clean, the loop runs autonomously within its
stopping rules. That final integration pass is the one thing no amount of prior
testing can replace — but every piece under it is proven.
