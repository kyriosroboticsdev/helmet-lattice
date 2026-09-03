"""
optimize.py
===========
Multi-objective Bayesian optimization + the active-learning loop (spec 4.4 --
the headline contribution).

  * Objectives: minimize a LINEAR metric (HIC) and a ROTATIONAL metric (BrIC)
    simultaneously. There is no single "best" design -- the PARETO FRONT is the
    result.
  * Acquisition: qNEHVI (noisy expected hypervolume improvement) over the GP
    surrogate, evaluated on a Sobol candidate pool across the geometry design
    space (topology enumerated, continuous knobs space-filled). qNEHVI is the
    modern, noise-robust form of qEHVI.
  * Active learning: propose the single most-informative next design to print
    and test, then (re)train and repeat.

Both metrics are MINIMIZED; BoTorch maximizes, so we optimize their negatives.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
try:  # modern log-domain acquisition (better numerics); fall back if unavailable
    from botorch.acquisition.multi_objective.logei import (
        qLogExpectedHypervolumeImprovement as _qEHVI,
    )
except Exception:
    from botorch.acquisition.multi_objective.monte_carlo import (
        qExpectedHypervolumeImprovement as _qEHVI,
    )
from botorch.sampling.normal import SobolQMCNormalSampler
from scipy.stats import qmc

import schema as S
import data as D
import features as F
import surrogate_gp as G

torch.set_default_dtype(torch.float64)

# Continuous design-space bounds (must match what's printable / the CAD generator).
CONT_BOUNDS = {
    "rel_density_measured": (0.18, 0.42),
    "cell_size_mm":         (5.0, 12.0),
    "cell_aspect_y":        (0.6, 1.6),   # non-cubic cells: y/x spacing
    "cell_aspect_z":        (0.6, 1.6),   # z/x spacing -> directional compression
    "liner_thickness_mm":   (22.0, 30.0),
    "gradient_param":       (0.0, 0.30),
    "shear_anisotropy_ratio": (0.6, 1.8),
}
CONT_KEYS = list(CONT_BOUNDS.keys())

# Per-family ceiling in generative mode: no single family may exceed this share of a
# design. This is the whole "never pure, always a genuine mix" rule, and it carries a
# floor on family COUNT with it -- a design needs at least ceil(1/cap) families to
# reach a total of 1, so at 0.45 every proposal uses >= 3. Set 1.0 to allow pure
# corners again.
MAX_FAMILY_WEIGHT = 0.45

# Legacy name: the old scheme forced EVERY basis above a floor, which pinned the
# family count at len(basis). The cap replaces it (bound from above, not below), so
# the count is free to vary.
BLEND_MIN_WEIGHT = 0.2

# How many families a single design may combine. The low end is set by the cap; the
# high end keeps a tile from turning into featureless mush.
FAMILIES_PER_DESIGN = (3, 5)

# Descriptors that define "these two designs look different". Geometry, not category:
# two different weight vectors can mesh into near-identical tiles, so diversity is
# enforced here rather than in weight space.
DIVERSITY_KEYS = ["sa_to_v", "connectivity_z", "shear_anisotropy_ratio",
                  "pore_size_mm", "irregularity"]

# Minimum separation between accepted proposals, in standard deviations of the
# candidate pool's own descriptor spread (so it is scale-free). 1.0 measured as
# non-binding -- the qEHVI top-q on this pool was already that far apart, so the
# filter never fired. 2.5 is the point where it starts genuinely rejecting
# near-duplicates and the batch spans different families; raise it for more contrast
# at some cost in acquisition value, set 0 to disable.
MIN_DESIGN_DISTANCE = 2.5


def sample_candidates(n: int, seed: int = 0, generative: bool = False,
                      blend_floor: float = BLEND_MIN_WEIGHT,
                      cap: float = MAX_FAMILY_WEIGHT,
                      families: tuple = FAMILIES_PER_DESIGN,
                      periodic: bool = True) -> list[dict]:
    """Space-filling candidate designs over the continuous knobs.

    generative=False: enumerate the discrete topology categories (original mode).
    generative=True : search the CONTINUOUS cross-family blend simplex, so candidates
                      are new in-between geometries the BO can optimize over smoothly.
                      Each candidate draws a random SUBSET of families (size in
                      `families`) and a Dirichlet mix over it, then projects onto the
                      capped simplex so no family exceeds `cap`.

    Sampling a subset rather than weighting all nine is deliberate: a full-basis
    Dirichlet puts weight on every family in every design, which makes each tile a
    blur of all nine and all of them resemble each other -- the opposite of the goal.
    """
    lo = np.array([CONT_BOUNDS[k][0] for k in CONT_KEYS])
    hi = np.array([CONT_BOUNDS[k][1] for k in CONT_KEYS])
    pts = lo + qmc.LatinHypercube(d=len(CONT_KEYS), seed=seed).random(n) * (hi - lo)
    rng = np.random.default_rng(seed)
    cands = []
    if generative:
        # periodic=True keeps only families whose field repeats exactly with the unit
        # cell. A blend is periodic iff every member is, so one aperiodic member (a
        # jittered voronoi, an unquantised spinodoid, any bouligand) makes the whole
        # tile a one-off blob rather than the repeating pattern a coupon needs.
        basis = [b for b in F.BLEND_BASIS if b in F.PERIODIC_FAMILIES] \
            if periodic else list(F.BLEND_BASIS)
        k_min = max(int(families[0]), int(np.ceil(1.0 / max(cap, 1e-9))))
        k_max = max(k_min, min(int(families[1]), len(basis)))
        for i in range(n):
            d = {k: float(pts[i, j]) for j, k in enumerate(CONT_KEYS)}
            k = int(rng.integers(k_min, k_max + 1))
            members = rng.choice(len(basis), size=k, replace=False)
            raw = rng.dirichlet(np.ones(k))
            d["blend"] = _cap_blend({basis[m]: float(w)
                                     for m, w in zip(members, raw)}, cap)
            # Stochastic members (voronoi/spinodoid) need a seed to be reproducible.
            d["blend_seed"] = int(rng.integers(0, 2**31 - 1))
            cands.append(d)
    else:
        topos = rng.choice(S.TOPOLOGIES, size=n)
        for i in range(n):
            d = {k: float(pts[i, j]) for j, k in enumerate(CONT_KEYS)}
            d["topology"] = str(topos[i])
            cands.append(d)
    return cands


def _cap_blend(weights: dict, cap: float = MAX_FAMILY_WEIGHT) -> dict:
    """L1-normalise `weights` with every entry <= cap (water-filling projection).

    Clip over-cap entries to `cap` and redistribute the freed mass in proportion to
    the others' remaining headroom -- proportional-to-headroom is what keeps the fix
    from pushing a different family over the ceiling. Mirrors
    generate_tiles.project_capped_simplex so the optimizer and the mesher agree on
    what a blend means without the ML layer importing the meshing stack.
    """
    names = list(weights)
    w = np.array([max(0.0, float(weights[k])) for k in names], dtype=float)
    if w.sum() <= 0:
        raise ValueError("blend has no positive weight")
    if len(names) * cap < 1.0 - 1e-9:
        raise ValueError(f"{len(names)} families cannot sum to 1 under cap {cap:g}")
    w /= w.sum()
    for _ in range(64):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        room = np.where(over, 0.0, cap - w)
        total = float(room.sum())
        if total <= 1e-12:
            break
        w += excess * room / total
    return {k: float(v) for k, v in zip(names, w)}


def _feature_block(designs: list[dict], conditions: dict, template: list[str]) -> np.ndarray:
    """Stack one feature row per design, aligned to the training feature template."""
    return np.vstack([
        F.design_to_feature_row(d, conditions, template_columns=template).values[0]
        for d in designs
    ])


def _objective_matrix(gps: dict, Xc: np.ndarray):
    """Vectorized GP predictions: means[n, m], stds[n, m] over objectives."""
    means = np.zeros((Xc.shape[0], len(S.OBJECTIVES)))
    stds = np.zeros_like(means)
    for j, obj in enumerate(S.OBJECTIVES):
        mu, sd = G.predict(gps[obj].model, Xc)
        means[:, j] = mu; stds[:, j] = sd
    return means, stds


def observed_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Pareto-optimal TESTED designs on (HIC, BrIC), both minimized."""
    view = D.training_view(df, data_sources=["oblique_impact"])
    rows = view.dropna(subset=S.OBJECTIVES)
    # aggregate replicates to one point per design (mean)
    agg = rows.groupby("design_id")[S.OBJECTIVES].mean().reset_index()
    Y = torch.as_tensor(agg[S.OBJECTIVES].values)
    mask = is_non_dominated(-Y)  # minimize -> non-dominated of negative
    agg["pareto"] = mask.numpy()
    return agg


def propose_next(gps: dict, df: pd.DataFrame, conditions: dict,
                 n_candidates: int = 2000, q: int = 1, seed: int = 0,
                 generative: bool = False, blend_floor: float = BLEND_MIN_WEIGHT,
                 cap: float = MAX_FAMILY_WEIGHT,
                 families: tuple = FAMILIES_PER_DESIGN,
                 min_distance: float = MIN_DESIGN_DISTANCE,
                 periodic: bool = True):
    """qNEHVI active-learning step: rank candidates, return the top-q to test next.

    generative=True searches the continuous cross-family blend simplex and returns
    blended (new-geometry) proposals instead of picking a discrete topology. `cap` is
    the per-family ceiling (0.45 => no family dominates, and >= 3 families per design
    follows from it); `families` is the (min, max) family count; `min_distance` is the
    descriptor-space separation the returned batch must respect, so a q>1 batch is a
    set of visibly different tiles rather than q variations of one.
    """
    # Reference point: slightly worse than the worst observed objective (in min space).
    obs = observed_pareto(df)
    ref_min = obs[S.OBJECTIVES].max().values * 1.05            # nadir-ish in min space
    ref_point = torch.as_tensor(-ref_min)                      # negate -> maximization

    # Incumbent Pareto Y in maximization space (negated objectives).
    Y_obs = torch.as_tensor(-obs[S.OBJECTIVES].values)

    cands = sample_candidates(n_candidates, seed=seed, generative=generative,
                              blend_floor=blend_floor, cap=cap, families=families,
                              periodic=periodic)

    # Both objectives are modeled from the same (oblique-impact) feature columns,
    # so they share one feature template -> one feature block scores all of them.
    template = gps[S.OBJECTIVES[0]].feature_names
    assert template == gps[S.OBJECTIVES[1]].feature_names, \
        "objectives must share a feature template for a joint ModelListGP"
    Xc = _feature_block(cands, conditions, template)
    Xc_t = torch.as_tensor(Xc)

    means, stds = _objective_matrix(gps, Xc)

    from botorch.models import ModelListGP
    model = ModelListGP(*[gps[o].model for o in S.OBJECTIVES])

    # qNEHVI wants outcomes as maximization; flip sign via a scalarized objective.
    from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
    neg_obj = WeightedMCMultiOutputObjective(weights=torch.tensor([-1.0, -1.0]))

    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=Y_obs)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]))
    acqf = _qEHVI(
        model=model, ref_point=ref_point.tolist(),
        partitioning=partitioning, sampler=sampler, objective=neg_obj,
    )
    with torch.no_grad():
        scores = acqf(Xc_t.unsqueeze(1)).numpy()   # q=1 evaluation per candidate

    order = select_diverse(scores, Xc, template, q=q, min_distance=min_distance)
    proposals = []
    for rank, idx in enumerate(order):
        d = dict(cands[int(idx)])
        if "blend" in d:                       # generative: tag as a blended geometry
            d["topology"] = "blend"
            d["blend"] = {b: round(float(v), 3)
                          for b, v in F.normalize_blend(d["blend"]).items()}
            d["n_families"] = len(d["blend"])
            d["max_family_weight"] = max(d["blend"].values())
            # Cheap printability pre-check: a weighted sum of z-invariant fields is
            # still z-invariant, so a blend of support-free families is support-free
            # by construction and never needs measuring. Mixes containing spinodoid
            # or bouligand do, and realize.py measures those.
            d["support_free_by_construction"] = all(
                b in F.SUPPORT_FREE_FAMILIES for b in d["blend"])
        proposals.append({
            **d,
            "acq_qEHVI": float(scores[idx]),
            "pred_HIC": float(means[idx, 0]), "pred_HIC_sd": float(stds[idx, 0]),
            "pred_BrIC": float(means[idx, 1]), "pred_BrIC_sd": float(stds[idx, 1]),
        })
    return proposals, (cands, means, stds)


def select_diverse(scores, Xc, template, q=1, min_distance=MIN_DESIGN_DISTANCE,
                   keys=DIVERSITY_KEYS):
    """Greedy acquisition-ranked selection that also forces the batch to look different.

    qEHVI ranks candidates on predicted hypervolume gain and nothing else, so the top-q
    of a smooth space are typically near-duplicates of one another -- fine for pure
    exploitation, useless when the point is a set of visibly distinct tiles to print.
    This walks down the acquisition order and accepts a candidate only if it is at
    least `min_distance` away from every already-accepted one in DESCRIPTOR space
    (standardised by the pool's own spread, so the threshold is unitless).

    Descriptor space rather than blend-weight space on purpose: (0.45, 0.45, 0.10) and
    (0.44, 0.44, 0.12) are far apart in neither, but two quite different weight vectors
    can still mesh into the same-looking tile. What matters is the geometry that comes
    out. Falls back to plain top-q if none of the descriptor columns are in the
    surrogate's feature template.
    """
    order = np.argsort(-np.asarray(scores))
    q = int(q)
    idx_map = {name: i for i, name in enumerate(template)}
    cols = [idx_map[k] for k in keys if k in idx_map]
    if not cols or q <= 1:
        return order[:q]

    Z = np.asarray(Xc)[:, cols].astype(float)
    sd = Z.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (Z - Z.mean(axis=0)) / sd            # unitless, comparable across descriptors

    chosen = []
    for idx in order:
        if not chosen:
            chosen.append(int(idx))
        else:
            d = np.linalg.norm(Z[list(chosen)] - Z[idx], axis=1).min()
            if d >= min_distance:
                chosen.append(int(idx))
        if len(chosen) >= q:
            break
    if len(chosen) < q:
        # Space is too tight for the requested spread; top up by acquisition rank
        # rather than silently returning a short batch.
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
            if len(chosen) >= q:
                break
    return np.array(chosen[:q])


def predicted_pareto(cands, means):
    """Pareto-optimal designs in the GP-PREDICTED objective space (min, min)."""
    Y = torch.as_tensor(-means)
    mask = is_non_dominated(Y).numpy()
    return mask


def plot_front(df, cands, means, mask, proposals, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    obs = observed_pareto(df)

    plt.figure(figsize=(8, 6))
    # Predicted cloud + predicted Pareto front
    plt.scatter(means[:, 0], means[:, 1], s=8, c="lightgray", label="predicted candidates")
    pf = means[mask]
    pf = pf[np.argsort(pf[:, 0])]
    plt.plot(pf[:, 0], pf[:, 1], "-o", c="crimson", ms=4, label="predicted Pareto front")
    # Tested designs
    plt.scatter(obs["HIC"], obs["BrIC"], s=70,
                c=np.where(obs["pareto"], "black", "steelblue"), zorder=5,
                label="tested designs")
    # Next proposal(s)
    for p in proposals:
        plt.scatter([p["pred_HIC"]], [p["pred_BrIC"]], marker="*", s=320,
                    c="gold", edgecolors="k", zorder=6, label="proposed next")
    plt.xlabel(f"{S.LINEAR_OBJECTIVE}  (linear injury, lower=better)")
    plt.ylabel(f"{S.ROTATIONAL_OBJECTIVE}  (rotational injury, lower=better)")
    plt.title("Linear vs. rotational Pareto front (GP surrogate)")
    # de-duplicate legend
    h, l = plt.gca().get_legend_handles_labels()
    seen = dict(zip(l, h))
    plt.legend(seen.values(), seen.keys(), fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    root = Path(__file__).resolve().parents[2]
    df = D.load_master(root / "data" / "raw" / "dummy_master.csv")
    conditions = {"impact_speed_mps": 5.0}
    gps = G.fit_objectives(df, conditions)
    props, (cands, means, stds) = propose_next(gps, df, conditions)
    mask = predicted_pareto(cands, means)
    plot_front(df, cands, means, mask, props, root / "outputs" / "pareto_front.png")
    print("\nProposed next design to print & test:")
    for p in props:
        print(f"  topology={p['topology']}, rho={p['rel_density_measured']:.3f}, "
              f"cell={p['cell_size_mm']:.1f}mm, gradient={p['gradient_param']:.3f}, "
              f"shear={p['shear_anisotropy_ratio']:.2f}")
        print(f"    predicted HIC={p['pred_HIC']:.0f}+/-{p['pred_HIC_sd']:.0f}, "
              f"BrIC={p['pred_BrIC']:.3f}+/-{p['pred_BrIC_sd']:.3f}")
