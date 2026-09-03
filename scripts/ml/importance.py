"""
importance.py
=============
Feature-importance analysis (spec 4.3 #1 / deliverable #1): quantitatively rank
which geometric parameters most affect linear vs. rotational performance.

  * Models: Random Forest + gradient boosting (XGBoost if installed, else
    sklearn GradientBoosting).
  * Honest accuracy: GroupKFold CV grouped by `design_id` so replicates of one
    design never straddle train/test (spec 4.2 -- the single most important rule).
  * Importance: SHAP mean-|value| if `shap` is installed, else sklearn
    permutation importance (model-agnostic) as a fallback.
  * Reports RMSE / MAE with a 95% CI across folds.

Gracefully degrades when xgboost / shap are absent (prints which engine ran).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error

import schema as S
import data as D
import features as F

# Optional, better tools — used automatically when present.
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


def make_gbm():
    """Gradient-boosting regressor: XGBoost if available, else sklearn."""
    if _HAS_XGB:
        return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                            random_state=0, n_jobs=0)
    return GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                     learning_rate=0.05, subsample=0.9, random_state=0)


def make_rf():
    return RandomForestRegressor(n_estimators=400, max_depth=None,
                                 min_samples_leaf=2, random_state=0, n_jobs=-1)


def grouped_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, model_factory):
    """GroupKFold CV by design_id. Returns dict with per-fold RMSE/MAE + 95% CI."""
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    rmses, maes = [], []
    for tr, te in gkf.split(X, y, groups):
        m = model_factory()
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        maes.append(mean_absolute_error(y[te], p))

    rmses, maes = np.array(rmses), np.array(maes)

    def ci(a):
        m, sd = a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0
        half = 1.96 * sd / np.sqrt(len(a))
        return m, (m - half, m + half)

    rmse_m, rmse_ci = ci(rmses)
    mae_m, mae_ci = ci(maes)
    return {"n_splits": n_splits, "rmse": rmse_m, "rmse_ci": rmse_ci,
            "mae": mae_m, "mae_ci": mae_ci}


def importance_scores(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Rank features. Averages RF and GBM; uses SHAP if available else permutation."""
    rankings = {}
    for name, factory in [("rf", make_rf), ("gbm", make_gbm)]:
        m = factory()
        m.fit(X, y)
        if _HAS_SHAP:
            try:
                expl = shap.TreeExplainer(m)
                sv = expl.shap_values(X)
                imp = np.abs(np.asarray(sv)).mean(axis=0)
            except Exception:
                imp = permutation_importance(m, X, y, n_repeats=20,
                                             random_state=0).importances_mean
        else:
            imp = permutation_importance(m, X, y, n_repeats=20,
                                         random_state=0).importances_mean
        imp = np.clip(imp, 0, None)
        s = imp.sum()
        rankings[name] = imp / s if s > 0 else imp

    df = pd.DataFrame(rankings, index=X.columns)
    df["mean_importance"] = df.mean(axis=1)
    return df.sort_values("mean_importance", ascending=False)


def analyze_target(df: pd.DataFrame, target: str, out_dir: Path,
                   include_conditions: bool = True) -> dict:
    """Full importance + grouped-CV report for one target column."""
    rows = D.labeled(df, target)
    if len(rows) < 6 or rows["design_id"].nunique() < 3:
        return {"target": target, "skipped": True, "n": len(rows)}

    X, names = F.build_features(rows, include_conditions=include_conditions)
    y = pd.to_numeric(rows[target], errors="coerce").values
    groups = rows["design_id"].values

    cv_rf = grouped_cv(X, y, groups, make_rf)
    cv_gbm = grouped_cv(X, y, groups, make_gbm)
    ranking = importance_scores(X, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out_dir / f"{target}_importance.csv")
    _plot(ranking, target, out_dir / f"{target}_importance.png")

    return {"target": target, "skipped": False, "n": len(rows),
            "n_designs": int(rows["design_id"].nunique()),
            "cv_rf": cv_rf, "cv_gbm": cv_gbm,
            "top_features": ranking.index[:6].tolist(),
            "ranking": ranking}


def _plot(ranking: pd.DataFrame, target: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = ranking.head(12).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top.index, top["mean_importance"], color="steelblue")
    plt.xlabel("Relative importance (RF+GBM avg)")
    plt.title(f"Feature importance -> {target}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(df: pd.DataFrame, out_dir: Path) -> list[dict]:
    """Run importance for the headline objectives + key quasi-static metrics."""
    engine = f"GBM={'xgboost' if _HAS_XGB else 'sklearn'}, " \
             f"importance={'SHAP' if _HAS_SHAP else 'permutation'}"
    print(f"[importance] engine: {engine}")
    if not (_HAS_XGB and _HAS_SHAP):
        print("[importance] tip: `pip install xgboost shap` upgrades this module.")

    results = []
    # Impact targets use impact rows (need conditions); QS targets use QS rows.
    plan = [
        (["oblique_impact"], S.LINEAR_OBJECTIVE, True),
        (["oblique_impact"], S.ROTATIONAL_OBJECTIVE, True),
        (["oblique_impact"], "peak_lin_accel_g", True),
        (["oblique_impact"], "peak_ang_accel_rad_s2", True),
        (["quasistatic"], "SEA_J_per_kg", False),
        (["quasistatic"], "plateau_stress_kPa", False),
    ]
    for sources, target, cond in plan:
        view = D.training_view(df, data_sources=sources)
        res = analyze_target(view, target, out_dir, include_conditions=cond)
        results.append(res)
        if res.get("skipped"):
            print(f"  {target:24s}: skipped (n={res['n']})")
        else:
            cv = res["cv_gbm"]
            print(f"  {target:24s}: GBM RMSE={cv['rmse']:.3g} "
                  f"[{cv['rmse_ci'][0]:.3g},{cv['rmse_ci'][1]:.3g}] "
                  f"| top: {', '.join(res['top_features'][:3])}")
    return results


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    root = Path(__file__).resolve().parents[2]
    df = D.load_master(root / "data" / "raw" / "dummy_master.csv")
    D.validate(df)
    run(df, root / "outputs" / "importance")
