"""
seed_lhs.py  --  Space-filling initial seed for the active-learning loop.
========================================================================
Replaces the "5 hand-picked designs (D1-D5)" start with a SYSTEMATIC seed that
evenly covers the real design space, so Bayesian optimization begins with a good
map of the whole space instead of 5 arbitrary points.

WHAT it does
------------
Latin Hypercube Sampling (LHS) over the SAME continuous bounds the optimizer uses
(CONT_BOUNDS in optimize.py), PLUS a valid blend over BLEND_BASIS that respects the
per-family cap (MAX_FAMILY_WEIGHT) and family-count range (FAMILIES_PER_DESIGN).
Output rows are in exactly the proposal format realize.py consumes, so seeds flow
straight into generate -> gate -> FEA with no adapter.

WHY LHS (not random, not 5 hand-picks)
--------------------------------------
Random points clump and leave gaps; 5 hand-picks cover almost nothing. LHS
guarantees each parameter is sampled across its full range with no clustering, so
the surrogate's first fit sees the whole space. Better start -> faster convergence
-> better designs for the same number of expensive FEA runs.

Optionally KEEP the D1-D5 anchors alongside the LHS points (include_anchors=True):
you lose nothing and you still have your known baselines in the dataset.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import schema as S
import features as F
# Pull constants from optimize.py by parsing it, so we don't import torch here.
class _OPT: pass
OPT = _OPT()
def _load_opt_constants():
    import ast, pathlib
    src = (pathlib.Path(__file__).parent / "optimize.py").read_text()
    tree = ast.parse(src)
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in (
                    "CONT_BOUNDS","MAX_FAMILY_WEIGHT","BLEND_MIN_WEIGHT","FAMILIES_PER_DESIGN"):
                    try: ns[t.id] = ast.literal_eval(node.value)
                    except Exception: pass
    return ns
_c = _load_opt_constants()
OPT.CONT_BOUNDS = _c["CONT_BOUNDS"]
OPT.CONT_KEYS = list(OPT.CONT_BOUNDS.keys())
OPT.MAX_FAMILY_WEIGHT = _c["MAX_FAMILY_WEIGHT"]
OPT.FAMILIES_PER_DESIGN = _c["FAMILIES_PER_DESIGN"]


def _lhs_unit(n: int, d: int, rng) -> np.ndarray:
    """Latin Hypercube in the unit cube [0,1]^d (scipy if available, else manual)."""
    try:
        from scipy.stats.qmc import LatinHypercube
        return LatinHypercube(d=d, seed=int(rng.integers(0, 2**31 - 1))).random(n)
    except Exception:
        # manual LHS: one stratified sample per parameter per row, then shuffle cols
        cut = np.linspace(0, 1, n + 1)
        u = rng.random((n, d))
        pts = cut[:n, None] + u * (1.0 / n)
        for j in range(d):
            rng.shuffle(pts[:, j])
        return pts


def _sample_blend(rng, periodic: bool = True) -> dict:
    """A valid blend: pick k families in FAMILIES_PER_DESIGN, weights sum to 1,
    each <= MAX_FAMILY_WEIGHT (so no family dominates -> always a genuine mix).

    periodic=True draws only from F.PERIODIC_FAMILIES, same as
    optimize.sample_candidates: a blend repeats with the unit cell only if every
    member does, and bouligand never does, so it would fail the coupon rule."""
    basis = [b for b in F.BLEND_BASIS if b in F.PERIODIC_FAMILIES] \
        if periodic else list(F.BLEND_BASIS)
    kmin, kmax = OPT.FAMILIES_PER_DESIGN
    kmin = max(kmin, int(np.ceil(1.0 / OPT.MAX_FAMILY_WEIGHT)))  # feasibility floor
    k = int(rng.integers(kmin, kmax + 1))
    members = rng.choice(len(basis), size=k, replace=False)
    # Dirichlet gives weights summing to 1; rejection-sample until under the cap
    for _ in range(200):
        w = rng.dirichlet(np.ones(k))
        if w.max() <= OPT.MAX_FAMILY_WEIGHT + 1e-9:
            break
    else:
        w = np.full(k, 1.0 / k)  # fallback: uniform (always <= cap for k>=ceil(1/cap))
    return {basis[m]: float(round(wi, 3)) for m, wi in zip(members, w)}


def build_seed(n: int = 20, seed: int = 0, include_anchors: bool = True,
               periodic: bool = True) -> pd.DataFrame:
    """Return n LHS seed designs (+ optional D1-D5 anchors) as proposal rows.

    Built FROM proposals_for_realize(n, seed) so the CSV describes exactly the
    tiles the loop realizes (same blends, same blend_seed) for a given seed."""
    keys = OPT.CONT_KEYS
    rows = []
    for i, prop in enumerate(proposals_for_realize(n, seed, periodic=periodic)):
        p = {k: prop[k] for k in keys}
        blend = prop["blend"]
        tag = "-".join(f"{b[:3]}{int(round(w*100))}"
                       for b, w in sorted(blend.items(), key=lambda kv: -kv[1]))
        row = {c: "" for c in S.ALL_COLUMNS}
        row.update({
            "design_id": f"SEED_lhs_{i:03d}_{tag}",
            "replicate_id": 1,
            "data_source": "fea",                # seeds are evaluated by FEA first
            "topology": "blend",
            "region_encoding": str(blend),
            "rel_density_designed": round(p["rel_density_measured"], 4),
            "cell_size_mm": round(p["cell_size_mm"], 3),
            "cell_aspect_y": round(p["cell_aspect_y"], 4),
            "cell_aspect_z": round(p["cell_aspect_z"], 4),
            "liner_thickness_mm": round(p["liner_thickness_mm"], 2),
            "gradient_param": round(p["gradient_param"], 4),
            "shear_anisotropy_ratio": round(p["shear_anisotropy_ratio"], 4),
            "temperature_C": 23.0,
            "notes": f"LHS space-filling seed; blend_seed={prop['blend_seed']} "
                     "(measured density + targets pending)",
        })
        # carry the machine-usable proposal fields too (realize.py reads these)
        row["_proposal_blend"] = blend
        row["_proposal_params"] = {k: p[k] for k in keys}
        row["_proposal_blend_seed"] = prop["blend_seed"]
        rows.append(row)

    df = pd.DataFrame(rows)
    if include_anchors:
        try:
            import seed_designs as anchors
            adf = anchors.build()
            df = pd.concat([df, adf], ignore_index=True)
        except Exception as e:
            print(f"[seed_lhs] anchors skipped ({e})")
    return df


def proposals_for_realize(n: int = 20, seed: int = 0, periodic: bool = True) -> list[dict]:
    """The seeds as plain proposal dicts, ready to hand to realize.realize_proposal()."""
    rng = np.random.default_rng(seed)
    keys = OPT.CONT_KEYS
    lo = np.array([OPT.CONT_BOUNDS[k][0] for k in keys])
    hi = np.array([OPT.CONT_BOUNDS[k][1] for k in keys])
    X = lo + _lhs_unit(n, len(keys), rng) * (hi - lo)
    out = []
    for i in range(n):
        p = {k: float(X[i, j]) for j, k in enumerate(keys)}
        p.update({"topology": "blend", "blend": _sample_blend(rng, periodic),
                  "rel_density_measured": p["rel_density_measured"],
                  "blend_seed": int(rng.integers(0, 2**31 - 1))})
        out.append(p)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Latin Hypercube seed for the loop")
    ap.add_argument("-n", type=int, default=20, help="number of LHS seed designs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-anchors", action="store_true", help="omit D1-D5 anchors")
    ap.add_argument("--out", default="data/seed_lhs.csv")
    a = ap.parse_args()
    df = build_seed(a.n, a.seed, include_anchors=not a.no_anchors)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in df.columns if c.startswith("_proposal")], errors="ignore")\
      .to_csv(a.out, index=False)
    print(f"wrote {len(df)} seed rows -> {a.out}")
    print(f"  LHS points: {a.n}   anchors: {'no' if a.no_anchors else 'D1-D5'}")
    print("  density range covered:",
          round(float(df['rel_density_designed'].replace('', np.nan).astype(float).min()),3),
          "->", round(float(df['rel_density_designed'].replace('', np.nan).astype(float).max()),3))
