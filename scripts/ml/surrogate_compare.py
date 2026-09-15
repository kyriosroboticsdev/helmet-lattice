"""
surrogate_compare.py  --  Which surrogate predicts our FEA best?
================================================================
Runs AFTER the loop has produced real FEA data. Answers, with evidence rather
than assumption, the question: "for OUR dataset size and shape, which surrogate
model most accurately predicts FEA results?"

Compares (on the SAME data, same splits):
  * Gaussian Process (GP)         -- current surrogate; small-data + uncertainty
  * Random Forest                 -- robust, handles bumpy responses
  * Gradient Boosting (XGBoost/HGB) -- scales better as data grows
  * (optional) Deep-Kernel GP     -- small NN feature-extractor + GP on top
                                     [only if torch is installed]

Produces:
  1. A LEARNING CURVE  (error vs. training-set size) for each model -> shows
     which model needs how much data, and whether an NN would ever catch up.
  2. A held-out ACCURACY TABLE (RMSE / MAE / R2) per model per target.
  3. A GROUPED cross-validation option (leave-one-design-family-out) to test
     genuine generalization, not just interpolation.

WHY grouped CV: random splits let near-duplicate replicates leak across train/test
and inflate scores. Grouping by design (or topology family) is the honest test.

Usage:
  python scripts/ml/surrogate_compare.py --data data/master_table.csv \
         --targets plateau_stress_kPa SEA_J_per_kg --group-col design_id
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- feature columns (geometry inputs the surrogate learns from) ---
DEFAULT_FEATURES = ["rel_density_measured", "cell_size_mm", "gradient_param",
                    "shear_anisotropy_ratio", "sa_to_v", "connectivity_z",
                    "strut_thickness_mm"]

def _models():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    kern = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
    return {
        "GaussianProcess": make_pipeline(StandardScaler(),
            GaussianProcessRegressor(kernel=kern, normalize_y=True, alpha=1e-6,
                                     n_restarts_optimizer=2, random_state=0)),
        "RandomForest": make_pipeline(StandardScaler(),
            RandomForestRegressor(n_estimators=300, random_state=0)),
        "GradientBoosting": make_pipeline(StandardScaler(),
            HistGradientBoostingRegressor(random_state=0)),
    }

def _try_add_deep_kernel(models):
    """Add a deep-kernel GP only if torch/gpytorch are present (optional)."""
    try:
        import torch, gpytorch  # noqa
    except Exception:
        return models, False
    # kept minimal & optional; documented for the student to enable if they have torch
    return models, True   # (implementation stub left to enable locally)

def _load(data, features, targets, group_col):
    df = pd.read_csv(data)
    if "fea_status" in df:
        df = df[df["fea_status"].astype(str).isin(["ok", "nan", ""] ) | df["fea_status"].isna()]
    feats = [f for f in features if f in df.columns]
    keep = df.dropna(subset=[c for c in feats + targets if c in df.columns])
    X = keep[feats].astype(float).values
    Ys = {t: keep[t].astype(float).values for t in targets if t in keep.columns}
    groups = keep[group_col].values if group_col in keep.columns else np.arange(len(keep))
    return X, Ys, groups, feats

def _metrics(y, p):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return dict(RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)),
                R2=float(r2_score(y, p)))

def accuracy_table(X, Ys, groups, n_splits=5):
    """Grouped CV accuracy per model per target."""
    from sklearn.model_selection import GroupKFold
    out = {}
    n_groups = len(np.unique(groups))
    k = max(2, min(n_splits, n_groups))
    for tgt, y in Ys.items():
        out[tgt] = {}
        for name, mdl in _models().items():
            preds = np.zeros_like(y, dtype=float); gkf = GroupKFold(n_splits=k)
            for tr, te in gkf.split(X, y, groups):
                from sklearn.base import clone
                m = clone(mdl); m.fit(X[tr], y[tr]); preds[te] = m.predict(X[te])
            out[tgt][name] = _metrics(y, preds)
    return out

def learning_curve(X, Ys, groups, fractions=(0.2,0.4,0.6,0.8,1.0), reps=3, test_frac=0.2):
    """RMSE vs. training-set size for each model/target -> the 'do we ever need an NN' plot.

    Split by GROUP, same as accuracy_table: each rep holds out ~test_frac of the
    designs, then trains on `frac` of the remaining designs. A row-level split puts
    replicates of one design on both sides and flatters every model."""
    from sklearn.base import clone
    rng = np.random.default_rng(0); curves = {}
    ug = np.unique(groups); n_g = len(ug)
    n_test = max(1, int(round(test_frac * n_g)))
    for tgt, y in Ys.items():
        curves[tgt] = {name: {"frac": [], "n_train_designs": [], "rmse": []} for name in _models()}
        for frac in fractions:
            n_tr = max(2, int(frac * (n_g - n_test)))
            for name, mdl in _models().items():
                errs = []
                for rep in range(reps):
                    perm = np.random.default_rng(rep).permutation(ug)  # same splits for every model
                    te_g, tr_g = perm[:n_test], perm[n_test:n_test + n_tr]
                    tr, te = np.isin(groups, tr_g), np.isin(groups, te_g)
                    if tr.sum() < 4 or te.sum() < 2: continue
                    mm = clone(mdl); mm.fit(X[tr], y[tr])
                    errs.append(np.sqrt(np.mean((mm.predict(X[te]) - y[te])**2)))
                if errs:
                    curves[tgt][name]["frac"].append(frac)
                    curves[tgt][name]["n_train_designs"].append(int(n_tr))
                    curves[tgt][name]["rmse"].append(float(np.mean(errs)))
    return curves

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", nargs="+", default=["plateau_stress_kPa", "SEA_J_per_kg"])
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    ap.add_argument("--group-col", default="design_id")
    ap.add_argument("--out", default="runs/surrogate_compare")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    X, Ys, groups, feats = _load(a.data, a.features, a.targets, a.group_col)
    print(f"loaded {len(X)} rows, {len(feats)} features: {feats}")
    if len(X) < 8:
        print("NOTE: <8 usable rows -- results are indicative only. Re-run after "
              "the loop has produced more FEA data."); 
    tab = accuracy_table(X, Ys, groups)
    print("\n=== grouped-CV accuracy (lower RMSE / higher R2 = better) ===")
    for tgt, models in tab.items():
        print(f"\n[{tgt}]")
        for name, m in sorted(models.items(), key=lambda kv: kv[1]["RMSE"]):
            print(f"  {name:16} RMSE={m['RMSE']:.4g}  MAE={m['MAE']:.4g}  R2={m['R2']:.3f}")
    json.dump(tab, open(out/"accuracy_table.json","w"), indent=2)

    curves = learning_curve(X, Ys, groups)
    json.dump(curves, open(out/"learning_curves.json","w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for tgt, models in curves.items():
            plt.figure(figsize=(6,4))
            for name, c in models.items():
                if c["frac"]: plt.plot(c["n_train_designs"], c["rmse"], "o-", label=name)
            plt.xlabel("training-set size (designs)"); plt.ylabel("held-out RMSE")
            plt.title(f"Learning curve: {tgt}"); plt.legend(); plt.grid(alpha=.3)
            plt.tight_layout(); plt.savefig(out/f"learning_curve_{tgt}.png", dpi=130); plt.close()
        print(f"\nsaved plots + tables -> {out}")
    except Exception as e:
        print(f"(plotting skipped: {e})")

    # pick the winner per target
    print("\n=== RECOMMENDED surrogate per target (lowest CV RMSE) ===")
    for tgt, models in tab.items():
        best = min(models.items(), key=lambda kv: kv[1]["RMSE"])
        print(f"  {tgt:20} -> {best[0]}  (RMSE={best[1]['RMSE']:.4g}, R2={best[1]['R2']:.3f})")

if __name__ == "__main__":
    main()
