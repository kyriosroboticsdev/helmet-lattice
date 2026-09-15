"""
fea_optimize.py  --  Multi-objective BO on FEA results (no impact data needed).
============================================================================
Objectives, both from the quasi-static compression FEA of the 2x2x2-cell piece and
both evaluated at ONE fixed screening strain E_EVAL (20%, what the piece is crushed
to), read off the saved stress-strain curve:
  * SEA_to_eval             energy absorbed up to E_EVAL per mass  -> MAXIMIZE
  * plateau_stress_to_eval  mean stress over the upper half of the
                            crush -- the load passed to the head   -> MINIMIZE
They trade off (stiffer lattices absorb more energy but transmit more force), so
the result is a Pareto front, searched with log-qEHVI over one GP per objective.
Set STRESS_METRIC = "peak" to minimize the highest stress instead.

DEPTH IS PART OF THE NUMBER. SEA is an integral to a strain, so SEA-to-20% (this
screening depth) and SEA measured deeper on finalists are different quantities and
must never be ranked against each other. Everything here is computed at E_EVAL from
the curve, every row carries eval_strain, and a solve that stalls short of E_EVAL is
dropped rather than compared. schema.check_sea_depths guards the master table.

Why read the curve instead of the row's SEA_J_per_kg: a solve stops wherever it
stops (the first real run holds depths from 3.7% to 20%), so that column mixes
depths by construction and is only meaningful next to fea_max_strain.

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

# Strain BOTH objectives are evaluated at -- the screening depth. 0.20 is what the
# 2x2x2 piece is crushed to; SEA and plateau stress are therefore "to 20%", and are
# NOT comparable to values measured deeper on finalists (see schema.check_sea_depths).
# Measured cost of this depth on the first real run: 11 of 29 finished piece solves
# reach 20%, 23 reach 10% -- designs either make 20% or buckle well before it.
E_EVAL = 0.20
# Stress objective: "plateau" = mean stress over the upper half of the crush
# (PLATEAU_BAND x E_EVAL), the load a liner would transmit while crushing;
# "peak" = the highest stress up to E_EVAL.
STRESS_METRIC = "plateau"
PLATEAU_BAND = (0.5, 1.0)
# Energy objective. "SEA" = energy absorbed per mass (the decided default).
# "efficiency" = energy absorbed / (peak stress x E_EVAL): 1.0 is a perfectly flat
# plateau, and it is the metric that actually trades off against stress. MEASURED on
# the first run's 11 designs at 20%: corr(SEA, plateau) = +0.96 with 7 of 11 designs
# "Pareto-optimal" -- SEA is the same stress integral divided by density, so that pair
# is nearly one objective and the search drifts to low density. corr(efficiency,
# plateau) = -0.52 with 2 of 11 on the front: a real trade-off.
ENERGY_METRIC = "SEA"
TPU_DENSITY_G_PER_MM3 = 1.22e-3    # must match fea_evaluator

MIN_TRAIN = 5          # fewer finished designs than this -> no GP fit, keep seeding
CURVES_DIR: Path | None = None     # set by run_loop to <out>/curves


def objectives() -> list[tuple[str, float]]:
    """(column, sign) pairs; sign +1 maximizes, -1 minimizes (BoTorch maximizes sign*value).

    SEA is MAXIMIZED (more energy absorbed per gram is better) and the stress
    objective is MINIMIZED (less force transmitted to the head is better).
    """
    energy = "SEA_to_eval" if ENERGY_METRIC == "SEA" else "efficiency_to_eval"
    return [(energy, +1.0), (f"{STRESS_METRIC}_stress_to_eval", -1.0)]


# Back-compat for callers that read the module constant.
OBJECTIVES = objectives()


def curve_objectives(curve: pd.DataFrame, rho: float, e: float | None = None) -> dict:
    """Metrics from one saved stress-strain curve, all evaluated AT STRAIN e.

    Returns SEA_to_eval (J/kg), peak_stress_to_eval and plateau_stress_to_eval (kPa),
    plus eval_strain so the depth travels with the numbers. All NaN if the solve did
    not reach e -- a shallower curve gives a different quantity, never a comparable one.
    """
    e = E_EVAL if e is None else e
    s = curve["strain"].to_numpy(float); p = curve["stress_kPa"].to_numpy(float)
    nan = dict(SEA_to_eval=float("nan"), peak_stress_to_eval=float("nan"),
               plateau_stress_to_eval=float("nan"), efficiency_to_eval=float("nan"),
               eval_strain=e)
    if len(s) < 2 or s.max() < e - 1e-9 or not rho or rho <= 0:
        return nan
    keep = s <= e + 1e-12
    se = np.r_[s[keep], e]; pe = np.r_[p[keep], np.interp(e, s, p)]
    W_mJ_per_mm3 = float(np.trapezoid(pe, se)) / 1000.0          # kPa -> MPa = mJ/mm^3
    lo, hi = PLATEAU_BAND[0] * e, PLATEAU_BAND[1] * e
    band = (se >= lo - 1e-12) & (se <= hi + 1e-12)
    if band.sum() >= 2:
        plateau = float(np.trapezoid(pe[band], se[band]) / (se[band].max() - se[band].min()))
    else:
        plateau = float(np.interp((lo + hi) / 2, se, pe))
    peak = float(pe.max())
    return dict(SEA_to_eval=W_mJ_per_mm3 / (rho * TPU_DENSITY_G_PER_MM3),   # mJ/g == J/kg
                peak_stress_to_eval=peak,
                plateau_stress_to_eval=plateau,
                # cushioning efficiency: 1.0 would be a perfectly flat plateau
                efficiency_to_eval=float(np.trapezoid(pe, se) / (peak * e)) if peak > 0 else float("nan"),
                eval_strain=e)


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
    metrics = []
    for _, r in rows.iterrows():
        f = Path(curves_dir) / f"{r['design_id']}.csv"
        metrics.append(curve_objectives(pd.read_csv(f), float(r["rel_density_measured"]))
                       if f.exists() else curve_objectives(pd.DataFrame({"strain": [], "stress_kPa": []}), 0))
    for k in ("SEA_to_eval", "peak_stress_to_eval", "plateau_stress_to_eval",
              "efficiency_to_eval", "eval_strain"):
        rows[k] = [m[k] for m in metrics]
    cols = [c for c, _ in objectives()]
    return rows[rows[cols].notna().all(axis=1)].reset_index(drop=True)


def objective_matrix(rows: pd.DataFrame) -> np.ndarray:
    """Objectives in MAXIMIZATION form, one column per objective."""
    return np.column_stack([sign * pd.to_numeric(rows[c], errors="coerce").values
                            for c, sign in objectives()])


def reference_point(rows: pd.DataFrame, path: Path) -> np.ndarray:
    """Load the run's fixed hypervolume reference, creating it from `rows` once.

    10% worse than the worst seed on each objective (in maximization space).
    """
    spec = dict(objectives=objectives(), eval_strain=E_EVAL, stress_metric=STRESS_METRIC,
                energy_metric=ENERGY_METRIC)
    if path.exists():
        saved = json.loads(path.read_text())
        # The reference anchors every hypervolume in the run. If the objective
        # DEFINITION changed (depth or stress metric), old and new hypervolumes are
        # different quantities and must not be compared.
        if (saved.get("eval_strain") != E_EVAL or saved.get("stress_metric") != STRESS_METRIC
                or saved.get("energy_metric", "SEA") != ENERGY_METRIC):
            raise ValueError(
                f"{path.name} was fixed for eval_strain={saved.get('eval_strain')} / "
                f"{saved.get('stress_metric')} stress, but this run uses {E_EVAL} / "
                f"{STRESS_METRIC}. Hypervolumes across the two are not comparable: "
                f"re-run with --rebuild-hv-reference (restarts the stopping history) "
                f"or use a fresh --out directory.")
        return np.array(saved["ref_max_space"])
    Y = objective_matrix(rows)
    span = np.maximum(Y.max(0) - Y.min(0), 1e-9)
    ref = Y.min(0) - 0.1 * np.maximum(np.abs(Y.min(0)), span)
    path.write_text(json.dumps({"ref_max_space": ref.tolist(),
                                **spec,
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
    att = df[(df.get("fea_model") == fea_model) & df["fea_status"].isin(ATTEMPTED)].copy()
    if att.empty:
        return None, {}
    # Usable = produced objectives at E_EVAL. A solve that stalls short of the
    # screening depth is as useless to the search as a design that cannot be built,
    # so the model learns to avoid both.
    usable = set(training_rows(df, fea_model=fea_model)["design_id"])
    y = att["design_id"].isin(usable).astype(int).values
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
