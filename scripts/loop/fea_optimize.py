"""
fea_optimize.py  --  Multi-objective BO on FEA results (no impact data needed).
============================================================================
Objectives, both from the quasi-static compression FEA of the 2x2x2-cell piece,
evaluated at a FIXED strain E_EVAL (10%) from the saved stress-strain curve:
  * SEA_to_eval         energy absorbed up to E_EVAL per mass       -> MAXIMIZE
  * peak_stress_to_eval highest stress reached up to E_EVAL (the
                        force a liner would transmit)               -> MINIMIZE
They trade off (stiffer lattices absorb more energy but transmit more force), so
the result is a Pareto front, searched with log-qEHVI over one GP per objective.

Why a fixed strain and PEAK stress: the piece is crushed 20%, but lattices peak and
then soften as walls buckle, and the static solve often ends a little short (the
first real seed stopped at 18.6%, with stress at 10% above stress at 18.6%). SEA to
"wherever the solve stopped" is not comparable between designs, and the stress at
the last point understates the force after a peak. Evaluating at 10% keeps nearly
every finished solve usable; designs that do not reach 10% are left out.

Reuses the existing pipeline pieces rather than re-implementing them:
  optimize.sample_candidates   generative blend candidates (cap, family count, periodic)
  optimize.select_diverse      descriptor-space diversity for the batch
  features.build_features      the GP's dense feature set
  surrogate_gp.fit_gp          GP with the Gibson-Ashby physics mean

The hypervolume reference point is FIXED once (from the seed data) and stored in
the run directory. A reference recomputed from the data every iteration moves
whenever a new worst design appears, and then "hypervolume improved" stops
meaning anything -- which is exactly what the stopping rule needs to measure.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import features as F
import optimize as O
import surrogate_gp as G

torch.set_default_dtype(torch.float64)

# Strain both objectives are evaluated at. Was 0.15; lowered after the first real
# seeds: of 5 finished piece solves, 3 stalled below 15% (at 12.7%, 14.4% and 3.8% --
# the denser designs stall earliest), leaving 2 usable rows. 4 of the 5 reach 10%.
# The price is that the objectives no longer see plateau onset.
E_EVAL = 0.10
TPU_DENSITY_G_PER_MM3 = 1.22e-3    # must match fea_evaluator

# (column, sign): sign +1 maximizes, -1 minimizes. BoTorch maximizes sign*value.
OBJECTIVES = [("SEA_to_eval", +1.0), ("peak_stress_to_eval", -1.0)]
MIN_TRAIN = 5          # fewer finished designs than this -> no GP fit, keep seeding
CURVES_DIR: Path | None = None     # set by run_loop to <out>/curves


def curve_objectives(curve: pd.DataFrame, rho: float, e: float = E_EVAL):
    """(SEA to strain e in J/kg, peak stress up to e in kPa) from one saved curve.

    SEA = absorbed energy density / solid mass density; the tile volume cancels.
    Returns (nan, nan) if the solve did not reach e.
    """
    s = curve["strain"].to_numpy(float); p = curve["stress_kPa"].to_numpy(float)
    if len(s) < 2 or s.max() < e - 1e-9 or not rho or rho <= 0:
        return float("nan"), float("nan")
    keep = s <= e + 1e-12
    se = np.r_[s[keep], e]; pe = np.r_[p[keep], np.interp(e, s, p)]
    W_mJ_per_mm3 = float(np.trapezoid(pe, se)) / 1000.0          # kPa -> MPa = mJ/mm^3
    sea = W_mJ_per_mm3 / (rho * TPU_DENSITY_G_PER_MM3)            # mJ/g == J/kg
    return sea, float(pe.max())


def training_rows(df: pd.DataFrame, crush: float = None, fea_model: str = "piece",
                  curves_dir: Path | None = None) -> pd.DataFrame:
    """Finished screening rows that reached E_EVAL, with both objectives attached.

    The objectives are computed from the saved curves (curves_dir/<design_id>.csv),
    so the definition can change without re-running any FEA. `crush` is unused and
    kept for call compatibility.
    """
    curves_dir = curves_dir or CURVES_DIR
    if df.empty or "fea_status" not in df or curves_dir is None:
        return df.iloc[0:0]
    rows = df[(df["fea_status"] == "ok") & (df.get("fea_model") == fea_model)].copy()
    sea, peak = [], []
    for _, r in rows.iterrows():
        f = Path(curves_dir) / f"{r['design_id']}.csv"
        if f.exists():
            a, b = curve_objectives(pd.read_csv(f), float(r["rel_density_measured"]))
        else:
            a = b = float("nan")
        sea.append(a); peak.append(b)
    rows["SEA_to_eval"] = sea
    rows["peak_stress_to_eval"] = peak
    return rows[rows["SEA_to_eval"].notna() & rows["peak_stress_to_eval"].notna()].reset_index(drop=True)


def objective_matrix(rows: pd.DataFrame) -> np.ndarray:
    """Objectives in MAXIMIZATION form, one column per objective."""
    return np.column_stack([s * pd.to_numeric(rows[c], errors="coerce").values
                            for c, s in OBJECTIVES])


def reference_point(rows: pd.DataFrame, path: Path) -> np.ndarray:
    """Load the run's fixed hypervolume reference, creating it from `rows` once.

    10% worse than the worst seed on each objective (in maximization space).
    """
    if path.exists():
        return np.array(json.loads(path.read_text())["ref_max_space"])
    Y = objective_matrix(rows)
    span = np.maximum(Y.max(0) - Y.min(0), 1e-9)
    ref = Y.min(0) - 0.1 * np.maximum(np.abs(Y.min(0)), span)
    path.write_text(json.dumps({"ref_max_space": ref.tolist(),
                                "objectives": OBJECTIVES,
                                "n_rows_when_fixed": int(len(rows))}, indent=2))
    return ref


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    from botorch.utils.multi_objective.pareto import is_non_dominated
    return is_non_dominated(torch.as_tensor(Y)).numpy()


def hypervolume(rows: pd.DataFrame, ref: np.ndarray) -> float:
    """Hypervolume of the observed front against the FIXED reference (bigger = better)."""
    if len(rows) == 0:
        return 0.0
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    Y = objective_matrix(rows)
    Y = Y[(Y > ref).all(axis=1)]           # points not beating the reference add nothing
    if len(Y) == 0:
        return 0.0
    front = torch.as_tensor(Y[pareto_mask(Y)])
    return float(Hypervolume(ref_point=torch.as_tensor(ref)).compute(front))


REACHED_FEA = ("ok", "failed")
ATTEMPTED = REACHED_FEA + ("print_rejected", "gate_rejected", "realize_rejected")


def feasibility_model(df: pd.DataFrame, names: list[str], fea_model: str = "piece"):
    """P(a design passes the screens and reaches FEA), from every attempt so far.

    Without it, qEHVI keeps proposing designs that cannot be built: minimizing peak
    stress pulls toward low density, where walls get too thin and the lattice falls
    apart (iteration 1 of the first real run: 4 of 6 proposals rejected, all at
    rho 0.175-0.199, plus one generator failure). Returns a callable X -> p, or None
    when there are not yet both outcomes to learn from.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    att = df[(df.get("fea_model") == fea_model) & df["fea_status"].isin(ATTEMPTED)]
    if att.empty:
        return None, {}
    y = att["fea_status"].isin(REACHED_FEA).astype(int).values
    if y.min() == y.max():
        return None, dict(n_labels=int(len(y)), feasible_rate=float(y.mean()))
    X, _ = F.build_features(att, include_conditions=False, drop_sparse_topology=True)
    X = X.apply(pd.to_numeric, errors="coerce").reindex(columns=names).fillna(0.0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced",
                                                             max_iter=2000))
    clf.fit(X.values, y)
    return (lambda Xc: clf.predict_proba(np.asarray(Xc))[:, 1]), \
        dict(n_labels=int(len(y)), feasible_rate=float(y.mean()))


def _features(rows: pd.DataFrame):
    X, names = F.build_features(rows, include_conditions=False, drop_sparse_topology=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    keep = [c for c in X.columns if X[c].notna().all() and X[c].std() > 1e-12]
    return X[keep], keep


def propose(df: pd.DataFrame, crush: float, ref: np.ndarray, n_propose: int,
            n_candidates: int = 800, seed: int = 0,
            min_distance: float = O.MIN_DESIGN_DISTANCE) -> tuple[list[dict], float, dict]:
    """Rank generative candidates by log-qEHVI; return a diverse shortlist.

    Returns (proposals best-first, top acquisition value in linear units, fit info).
    """
    from botorch.models import ModelListGP
    from botorch.utils.multi_objective.box_decompositions.non_dominated import \
        FastNondominatedPartitioning
    from botorch.sampling.normal import SobolQMCNormalSampler
    try:
        from botorch.acquisition.multi_objective.logei import \
            qLogExpectedHypervolumeImprovement as ACQ
        log_acq = True
    except Exception:
        from botorch.acquisition.multi_objective.monte_carlo import \
            qExpectedHypervolumeImprovement as ACQ
        log_acq = False

    rows = training_rows(df, crush)
    if len(rows) < MIN_TRAIN:
        raise RuntimeError(f"only {len(rows)} finished designs; need {MIN_TRAIN} to fit the GPs")
    X, names = _features(rows)
    Y = objective_matrix(rows)
    mean_idx = G.density_term_indices(names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = [G.fit_gp(X.values, Y[:, j], mean_idx=mean_idx) for j in range(Y.shape[1])]
    model = ModelListGP(*models)

    cands = O.sample_candidates(n_candidates, seed=seed, generative=True)
    Xc = np.vstack([F.design_to_feature_row(c, None, template_columns=names).values[0]
                    for c in cands])
    partitioning = FastNondominatedPartitioning(ref_point=torch.as_tensor(ref),
                                                Y=torch.as_tensor(Y))
    acqf = ACQ(model=model, ref_point=ref.tolist(), partitioning=partitioning,
               sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])))
    with torch.no_grad():
        scores = acqf(torch.as_tensor(Xc).unsqueeze(1)).numpy()
    # Stopping still sees the raw expected improvement, so its ratio to the first
    # iteration's value keeps one definition for the whole run.
    top = float(np.exp(scores.max())) if log_acq else float(scores.max())

    # Constrained BO: rank by expected improvement x P(reaches FEA).
    feas, feas_info = feasibility_model(df, names)
    p_feas = feas(Xc) if feas is not None else np.ones(len(Xc))
    ranked = scores + np.log(np.clip(p_feas, 1e-3, 1.0)) if log_acq else scores * p_feas

    order = O.select_diverse(ranked, Xc, names, q=n_propose, min_distance=min_distance)
    mu = np.column_stack([G.predict(m, Xc[order])[0] for m in models])
    props = []
    for rank, i in enumerate(order):
        d = dict(cands[int(i)])
        d["topology"] = "blend"
        d["blend"] = {b: round(float(v), 3) for b, v in F.normalize_blend(d["blend"]).items()}
        d["acq"] = float(scores[i])
        d["p_feasible"] = float(p_feas[i])
        for j, (col, s) in enumerate(OBJECTIVES):
            d[f"pred_{col}"] = float(s * mu[rank, j])
        props.append(d)
    info = dict(n_train=len(rows), n_features=len(names), top_acq=top,
                feas_labels=feas_info.get("n_labels", 0),
                feas_rate=round(feas_info.get("feasible_rate", float("nan")), 3),
                p_feasible_chosen=round(float(np.mean(p_feas[order])), 3),
                p_feasible_pool=round(float(np.mean(p_feas)), 3))
    return props, top, info
