# ML & Computational Workflow — Team Document
### TPU Lattice Helmet Liner Project (geometry-only study)

*This is the complete computational spec for the ML/code team. It is self-contained: you do not need the hardware document to do your work. The one thing you **must** coordinate on with the testing team is the **Data Handoff Contract (Section 2)** — that's the interface between the two halves of the project. Get that agreed and locked early.*

---

## 1. Project context (what you're modeling and why)

**Goal:** discover and optimize 3D-printed TPU lattice helmet-liner geometries that reduce both **linear** and **rotational** head-acceleration concussion metrics, validated against the Virginia Tech (VT) STAR system.

**Critical scope rule:** this is a **geometry-only** study. The material (one TPU, one durometer, one printer profile) is held *constant*. Every input feature you use is **geometric** — there are no material features. This is deliberate: it means any performance difference the model learns is attributable to architecture alone, which keeps the causal story clean.

**Where ML sits — the multi-fidelity pyramid:**

| Tier | Source of data | Volume | Your relationship to it |
|---|---|---|---|
| 0 | Dynamic FEA (SimScale, Ogden TPU model) | ~Hundreds | Cheap data to expand training set (once validated) |
| 1 | Physical rig (quasi-static + impact tests) | ~Dozens | Your **ground-truth** training data |
| 2 | Virginia Tech STAR | 2–4 designs | Gold-standard external validation target |

**What ML must deliver:**
1. **Feature importance** — quantitatively answer "which geometric parameters most affect linear vs. rotational performance."
2. **A surrogate model** that predicts impact metrics from geometry, with **uncertainty**.
3. **Multi-objective optimization** that proposes improved geometries (the Pareto front of linear vs. rotational injury).
4. **An active-learning loop** that tells the testing team which design to print next.
5. **Blind validation** — predict an untested design, have it printed/tested, report hit/miss.

Realistic expectation: you will have **tens** of physical data points, not thousands. The deliverable is **not** a deep neural net. It is a rigorous **surrogate + Bayesian optimization + feature-importance** pipeline with honest uncertainty — which is exactly what professional labs use when experiments are expensive.

---

## 2. DATA HANDOFF CONTRACT (the interface — agree on this first)

The testing team produces one row per test into a single master table (CSV/Parquet). **Lock this schema before any testing starts.** Both teams keep an identical copy of this section.

### 2.1 Identifier & design (geometry) columns
| Column | Type | Notes |
|---|---|---|
| `design_id` | string | Unique per geometry configuration |
| `replicate_id` | int | 1, 2, 3… (which printed copy) |
| `data_source` | enum | `fea` / `quasistatic` / `linear_impact` / `oblique_impact` / `vt` |
| `topology` | categorical | bcc / kelvin / gyroid / octet / hybrid / alternating / graded |
| `region_encoding` | string/JSON | For hybrid/alternating: which topology in which region/layer |
| `rel_density_designed` | float | As designed in CAD |
| `rel_density_measured` | float | **From mass/volume of the printed part** (use this as the real feature) |
| `cell_size_mm` | float | |
| `strut_thickness_mm` | float | Derived from density+cell size |
| `liner_thickness_mm` | float | |
| `gradient_param` | float | Density slope through thickness (0 = uniform) |
| `shear_anisotropy_ratio` | float | Normal stiffness ÷ tangential stiffness |
| `sa_to_v` | float | Surface-area-to-volume (from CAD) |
| `connectivity_z` | float | Nodal connectivity (from CAD) |

### 2.2 Test-condition columns
| Column | Type | Notes |
|---|---|---|
| `impact_zone` | categorical | front / boss / side / back (impact tests) |
| `impact_speed_mps` | float | |
| `impact_energy_J` | float | |
| `temperature_C` | float | Should be ~23; log it |

### 2.3 Target (output) columns — units fixed, never mixed
| Column | Unit | From which test |
|---|---|---|
| `peak_lin_accel_g` | g | linear/oblique impact |
| `HIC` | — | linear impact |
| `peak_ang_accel_rad_s2` | rad/s² | oblique impact |
| `peak_ang_vel_rad_s` | rad/s | oblique impact |
| `BrIC` | — | oblique impact |
| `SEA_J_per_kg` | J/kg | quasi-static + impact |
| `plateau_stress_kPa` | kPa | quasi-static |
| `densification_strain` | — | quasi-static |
| `restitution` | — | impact |
| `raw_file_path` | string | Pointer to the immutable raw signal file |
| `notes` | string | Anything unusual (clipping, print defect, etc.) |

### 2.4 Format conventions (non-negotiable)
- **One row per test.** Replicates are separate rows sharing a `design_id`.
- **Fixed units** per column (see tables); never mix g and m/s².
- **Missing values** are empty/`NaN`, never `0` (0 is a real measurement).
- **Flag clipped sensor data** in `notes` and exclude from training (a clipped peak is not a real peak).
- The testing team keeps an **immutable raw archive**; this table is the processed layer.

---

## 3. Feature reference (geometry-only) — context for importance analysis

Prior engineering ranking of which geometry variables should matter most (your SHAP/importance analysis will *test* this — confirming or overturning it is a result either way):

1. **Relative density** (use `rel_density_measured`) — expected dominant driver.
2. **Unit-cell topology** — sets plateau shape and isotropy.
3. **Gradient geometry** (`gradient_param`) — core novelty; soft-entry-then-stiffen.
4. **Hybrid / alternating architecture** (`region_encoding`) — core novelty; may decouple linear vs. rotational.
5. **Shear-tuned anisotropy** (`shear_anisotropy_ratio`) — main geometric lever on the *rotational* metric.
6. **Cell size**, then **liner thickness** (often a fixed envelope), then **strut thickness** (derived).

**Physics-informed engineered features** (add these — they reduce overfitting on small data):
- `(rel_density)` and `(rel_density)^2` — Gibson–Ashby scaling for stiffness/plateau.
- Interaction terms: `topology × rel_density` (the best density depends on topology).
- `sa_to_v`, `connectivity_z`.

---

## 4. ML framework

### 4.1 Inputs / targets
- **Inputs:** all geometry columns (Section 2.1) + engineered features (Section 3) + test conditions (`impact_zone`, `impact_speed_mps`).
- **Targets:** model each output in Section 2.3 separately (or multi-output). The two *headline* objectives for optimization are a **linear** metric (HIC or `peak_lin_accel_g`) and a **rotational** metric (BrIC or `peak_ang_accel_rad_s2`).

### 4.2 Training & validation procedure
- **Scale features** (standardize continuous; one-hot categoricals).
- **Leave-one-group-out cross-validation grouped by `design_id`** — replicates of one design must **never** straddle train/test, or you get leakage and inflated scores. *This is the single most important methodological rule in the whole ML workflow.*
- **Nested CV** for honest hyperparameter tuning.
- Report **RMSE/MAE with confidence intervals**, plus **uncertainty calibration** (for the GP: do ~95% of true values fall in the 95% predictive interval?).

### 4.3 Model selection
1. **Random Forest + Gradient Boosting (XGBoost):** robust on small tabular data; run **SHAP / permutation importance** to quantitatively rank features (delivers Section 1 item #1).
2. **Gaussian Process Regression (GPR):** small-data friendly, gives calibrated uncertainty → this is what powers Bayesian optimization. This is your hero model.

### 4.4 Optimization & the active-learning loop (your headline contribution)
- **Multi-objective Bayesian optimization** on the GPR surrogate. Acquisition: **qEHVI** or **ParEGO** to map the **Pareto front** of linear vs. rotational injury metrics. There is no single "best" design — *the front itself is the result.*
- **Active-learning loop:**
  1. Train GPR on all data so far.
  2. Acquisition function proposes the next most-informative design.
  3. Hand that design to the testing team (via the parametric pipeline, Section 6) → they print & test it.
  4. Add the result, retrain, repeat.
  Each experiment is maximally informative — this saves the testing team print time and material.
- **Blind validation:** before printing, the model *predicts* a held-out design's metrics; the testing team then prints and tests it; you report predicted-vs-actual. A successful blind prediction is the strongest possible proof the model learned real physics, not noise.
- The **final 1–2 designs** from the Pareto front go to Virginia Tech.

---

## 5. Statistical analysis (your responsibility too)

- **Screening DOE:** the testing team prints ~42 configurations × 3 replicates (~126 coupons). Help them confirm the design is space-filling (good coverage for the surrogate).
- **ANOVA** (two-way: topology × density, plus cell size) on each quasi-static metric — report **effect sizes (partial η²)** and **95% CIs**, not just p-values.
- **Pareto analysis** for the linear-vs-rotational tradeoff (also feeds optimization).
- **Multiple-comparison correction** (Benjamini–Hochberg) across the many metrics — you'll test many outputs, so control the false-discovery rate.
- **Agreement statistics** for the validation work:
  - **Bland–Altman** plots comparing the three rotational-velocity methods (accelerometer-array vs. gyro vs. video).
  - **Spearman/Pearson correlation** for the rig-vs-VT analysis (Section 7).
  - **ICC** for rig repeatability (target > 0.9).

---

## 6. Parametric CAD → STL pipeline (code — your build)

The active-learning loop only works if a proposed parameter vector can be turned into a printable/simulatable geometry **automatically**. Build a scriptable generator.

- **Recommended:** Rhino + **Grasshopper** (scriptable, excellent for parametric trusses) or **nTop** (academic access; implicit modeling for TPMS/graded/hybrid). For pure-code routes, Python implicit-geometry libraries can generate gyroid/TPMS via signed-distance fields.
- **Must generate, parametrically:** the baseline families (BCC, Kelvin, gyroid, octet), **graded** (density slope through thickness), and **hybrid/alternating** (region/layer topology mixing) — these last two are the project's novelty, so the generator must handle them.
- **Output per design:** an STL (for slicing/printing) **and** a clean solid for SimScale, plus an auto-computed feature record (`rel_density_designed`, `sa_to_v`, `connectivity_z`, `gradient_param`, `shear_anisotropy_ratio`) written straight into the handoff table.
- **Interface to the loop:** a function `params → STL + feature_row`. The Bayesian optimizer outputs `params`; this function realizes them.

---

## 7. ML-relevant validation

- **Blind prediction** (Section 4.4) — the headline internal proof.
- **Uncertainty calibration** — coverage of predictive intervals.
- **FEA-vs-experiment** (support the testing team): once dynamic FEA agrees with rig data on tested designs, FEA-generated points can augment the training set — but **only after** that agreement is demonstrated; flag FEA-sourced rows via `data_source` so you can train with/without them and report the difference.
- **Ultimate: rig/model vs. Virginia Tech.** Regress your predicted (or rig-measured) metric against VT's actual STAR scores for the 2–4 designs VT tests. Pull the stock helmet's **public** VT baseline for the same shell. A strong correlation here validates the entire cheap pipeline against the world standard — this is the figure that wins the project.

---

## 8. ML pitfalls a judge will probe (and your defenses)

| Pitfall | Defense |
|---|---|
| "You overfit 20 rows." | Physics-informed features; GPR with uncertainty; grouped CV; **blind prediction** on a freshly printed design |
| Data leakage via replicates | **Leave-one-group-out by `design_id`** — never split replicates |
| "Did the ML actually do anything?" | The active-learning loop *proposed* designs that were then printed and beat the baseline; show the Pareto front improving across iterations |
| Overclaiming a single magic design | Report the **Pareto front**, not one number; there's a linear-vs-rotational tradeoff |
| Mixing simulation and experiment silently | `data_source` flag; report results with and without FEA-augmented data |
| Uncalibrated confidence | Report interval coverage; don't quote point predictions without uncertainty |

---

## 9. Suggested tools & a realistic learning path

- **Language:** Python.
- **Core libraries:** `pandas` (data), `scikit-learn` (RF, CV, metrics), `xgboost` (gradient boosting), `shap` (feature importance), `scipy.stats` / `statsmodels` (ANOVA, correlation), `matplotlib` (figures).
- **Gaussian process + Bayesian optimization:** `scikit-optimize` (gentler entry) or **`BoTorch` + `Ax`** (proper multi-objective qEHVI; steeper but the right tool). `GPyTorch` underlies BoTorch.
- **Suggested order to build:** (1) load + validate the handoff table; (2) RF/XGBoost + SHAP for importance; (3) GPR single-objective; (4) multi-objective Bayesian optimization + the active-learning loop; (5) blind-validation reporting; (6) the rig-vs-VT correlation figure.
- Keep everything in a version-controlled repo with a clear `data/ raw + processed`, `notebooks/`, `src/` structure, and a README documenting the handoff schema.

---

## 10. ML team — first steps
1. **Agree and lock the Data Handoff Contract (Section 2) with the testing team.** Nothing else matters until this is done.
2. Stand up the repo + load a small dummy dataset matching the schema, so your whole pipeline runs end-to-end on fake data before real data arrives.
3. Build the parametric `params → STL + feature_row` generator (Section 6) — the testing team needs it to print, and the loop needs it to function.
4. Implement RF/XGBoost + SHAP so that, the moment the screening data lands, you can report feature importance.
5. Build the GPR + multi-objective Bayesian optimization scaffold so the active-learning loop is ready when impact data arrives.

*Coordinate with the testing team on cadence: the active-learning loop is print-rate-limited, so plan around how fast they can produce and test a proposed design.*
