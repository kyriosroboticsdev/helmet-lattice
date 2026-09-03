# ML pipeline — TPU lattice helmet liner (geometry-only)

Implements the workflow in `ML_Computational_Workflow_TEAM.md`. Everything runs
end-to-end on **synthetic dummy data** first (spec §10 step 2), so the code is
proven before real Virginia Tech / rig data arrives. Swap in the real handoff
CSV and nothing else changes.

## Quick start

```bash
pip install numpy pandas scipy scikit-learn torch gpytorch botorch matplotlib pyarrow
pip install xgboost shap statsmodels        # optional — auto-upgrades importance/ANOVA

python scripts/ml/make_dummy_data.py        # writes data/raw/dummy_master.csv
python scripts/run_pipeline.py              # runs the whole pipeline -> outputs/
```

Point at real data once it exists:

```bash
python scripts/run_pipeline.py --data path/to/handoff.csv
```

## Modules (each runs standalone for debugging)

| File | Spec § | What it does |
|---|---|---|
| `schema.py` | 2 | **The Data Handoff Contract as code** — the single source of truth for columns, units, categories, and the two headline objectives. Run it to emit a blank CSV template. |
| `make_dummy_data.py` | 10 | Synthetic master table with plausible physics and a real linear-vs-rotational tradeoff; correlated replicates per `design_id`. |
| `data.py` | 2.4 | Load + validate against the contract; exclude clipped rows; train with/without FEA via `data_source`. |
| `features.py` | 3 | Physics-informed engineered features (ρ, ρ², topology×ρ, …) **and** the `params → feature_row` interface the optimizer/CAD share. |
| `importance.py` | 4.3 #1 | RF + gradient boosting (XGBoost if installed), **grouped CV by `design_id`**, SHAP (or permutation) importance, RMSE/MAE with 95% CIs. |
| `surrogate_gp.py` | 4.3 #2 | BoTorch GP per objective, **leave-one-design-out CV**, **uncertainty calibration** (95% interval coverage). |
| `optimize.py` | 4.4 | Multi-objective **qEHVI** Pareto front of linear vs. rotational injury + the **active-learning** next-design proposal. |
| `realize.py` | 6 | Turns the proposed design into a printable **STL** (`params → STL`); gyroid/schwartz/graded/auxetic supported, others flagged for FreeCAD/nTop. |
| `../run_pipeline.py` | 10 | Driver that runs all of the above and writes `outputs/pipeline_report.md`. |

## The two rules that keep the results honest

1. **Leave-one-group-out by `design_id`** — replicates of one design never
   straddle train/test (spec §4.2; the single most important methodological
   rule). Enforced in both `importance.py` (`GroupKFold`) and
   `surrogate_gp.py` (`LeaveOneGroupOut`).
2. **Uncertainty calibration** — we report whether ~95% of held-out truth falls
   in the 95% predictive interval, so point predictions are never quoted naked.

## Data Handoff Contract

Lock `schema.py` with the testing team **before any testing starts** (§10 #1).
Both teams keep an identical copy. `python scripts/ml/schema.py` writes
`data/handoff_template.csv` — the exact blank table the testing team fills in,
one row per test.

## Outputs

- `outputs/importance/<target>_importance.{csv,png}` — ranked drivers per metric.
- `outputs/pareto_front.png` — tested designs, GP-predicted Pareto front, and the
  starred next design to print.
- `outputs/proposed_designs.csv` — the proposed design(s) + predicted metrics + STL path.
- `stl_output/PROP_*.stl` — printable STL of the proposed design (skip with `--no-stl`).
- `outputs/pipeline_report.md` — human-readable summary of every stage.

## Optional dependencies & fallbacks

| Missing package | Fallback used | Install to upgrade |
|---|---|---|
| `xgboost` | sklearn `GradientBoostingRegressor` | `pip install xgboost` |
| `shap` | sklearn permutation importance | `pip install shap` |
| `statsmodels` | (only needed for the planned §5 ANOVA module) | `pip install statsmodels` |
