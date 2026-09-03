"""
surrogate.py
GP surrogate model + auto-STL generation for ML-recommended designs.

Usage:
    1. Fill vt_results/results.csv with VT test data after each trip.
    2. Run: python scripts/surrogate.py
    3. Script fits the surrogate, shows Pareto front, and automatically
       generates STL files for the top recommended next designs.

Requirements:
    pip install torch gpytorch botorch pymoo scikit-learn matplotlib pandas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

import generate_tiles as gen

torch.set_default_dtype(torch.float64)

RESULTS_CSV = Path(__file__).parent.parent / "vt_results" / "results.csv"

# Maps topology name → integer code for GP input
TOPOLOGY_MAP = {
    "gyroid":    0,
    "schwartz":  1,
    "auxetic":   2,
    "graded":    3,
}

# Design space bounds: [cell_size_mm, relative_density, topology_code]
BOUNDS_RAW = torch.tensor([
    [5.0,  0.18, 0.0],
    [12.0, 0.42, float(len(TOPOLOGY_MAP) - 1)],
])


# ── Data loading ───────────────────────────────────────────────────────────────

def load_results():
    if not RESULTS_CSV.exists():
        print(f"No results file at {RESULTS_CSV}")
        print("Add columns: design, topology, cell_size, relative_density, peak_G, BrIC, mass_g")
        return None
    df = pd.read_csv(RESULTS_CSV)
    if df.empty or len(df.dropna()) == 0:
        print("results.csv is empty. Add VT test data first.")
        return None
    df = df.dropna(subset=["peak_G", "BrIC", "mass_g"])
    df["topology_code"] = df["topology"].map(TOPOLOGY_MAP).fillna(0)
    return df


# ── Surrogate model ────────────────────────────────────────────────────────────

def normalize(X, bounds=BOUNDS_RAW):
    return (X - bounds[0]) / (bounds[1] - bounds[0])

def denormalize(X_norm, bounds=BOUNDS_RAW):
    return X_norm * (bounds[1] - bounds[0]) + bounds[0]

def fit_surrogate(X_norm, Y_neg):
    model = SingleTaskGP(X_norm, Y_neg)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

def validate_surrogate(model, X_norm, Y_neg):
    """
    Leave-one-out cross-validation — tells you how accurate the surrogate is.
    R^2 > 0.8 means the GP is reliable enough to trust its recommendations.
    R^2 < 0.5 means you need more VT data before trusting the suggestions.
    """
    n = len(X_norm)
    preds = []
    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool)
        mask[i] = False
        m = SingleTaskGP(X_norm[mask], Y_neg[mask])
        fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
        with torch.no_grad():
            p = m.posterior(X_norm[i:i+1]).mean
        preds.append(p)

    Y_pred = -torch.cat(preds, dim=0)
    Y_true = -Y_neg

    ss_res = ((Y_true - Y_pred) ** 2).sum(0)
    ss_tot = ((Y_true - Y_true.mean(0)) ** 2).sum(0)
    r2 = (1 - ss_res / ss_tot).numpy()

    print("Surrogate accuracy (leave-one-out R^2):")
    labels = ["peak_G", "BrIC", "mass_g"]
    for label, score in zip(labels, r2):
        quality = "reliable" if score > 0.8 else ("marginal" if score > 0.5 else "needs more data")
        print(f"  {label:12s}: R^2 = {score:.3f}  ({quality})")
    print()
    return r2


# ── Recommendations + STL generation ──────────────────────────────────────────

def suggest_and_generate(model, df, n_candidates=1000, n_suggest=4):
    """
    Finds the top N predicted-best designs and immediately generates
    their STL files. These are ML-optimized recommendations.
    """
    X_cand = draw_sobol_samples(bounds=torch.zeros(2, 3), n=n_candidates, q=1).squeeze(1)

    with torch.no_grad():
        posterior = model.posterior(X_cand)
        Y_pred    = -posterior.mean        # un-negate: lower = better
        Y_std     = posterior.variance.sqrt()

    # Score by predicted peak_G (primary objective)
    score     = Y_pred[:, 0]
    best_idx  = score.argsort()[:n_suggest]
    best_X    = denormalize(X_cand[best_idx])

    inv_topo  = {v: k for k, v in TOPOLOGY_MAP.items()}

    # Determine current round number from results
    round_num = int(len(df) / 5) + 2  # rough: round 1 = 5 manual, round 2+ = ML

    print(f"\nTop {n_suggest} ML-recommended designs [ML-OPTIMIZED]:")
    print(f"Generating STL files automatically...\n")

    generated = []
    for rank, idx in enumerate(best_idx):
        cs    = float(best_X[rank, 0])
        rd    = float(best_X[rank, 1])
        topo  = inv_topo[round(float(best_X[rank, 2]))]
        pg    = float(Y_pred[idx, 0])
        br    = float(Y_pred[idx, 1])
        pg_sd = float(Y_std[idx, 0])

        rec_id = f"R{round_num}_{rank+1:02d}"
        print(f"  {rec_id}: topology={topo}, cell={cs:.1f}mm, density={rd:.3f}")
        print(f"    Predicted peak G: {pg:.1f} +/- {pg_sd:.1f}g  |  Predicted BrIC: {br:.4f}")

        # Skip if topology doesn't support auto-generation (auxetic/graded need special handling)
        if topo not in gen.TOPOLOGY_FN:
            print(f"    Skipping STL generation: '{topo}' requires manual design in Fusion 360.")
            print()
            continue

        try:
            mesh, path, mass = gen.generate_from_recommendation(rec_id, topo, cs, rd)
            print(f"    STL saved: {path}")
            print(f"    Est. mass: ~{mass:.1f}g  |  Watertight: {mesh.is_watertight}")
            generated.append({
                "design": rec_id, "topology": topo,
                "cell_size": cs, "relative_density": rd,
                "pred_peak_G": pg, "pred_BrIC": br,
                "stl_path": path
            })
        except Exception as e:
            print(f"    STL generation failed: {e}")
        print()

    # Save recommendation log
    if generated:
        log_path = Path(__file__).parent.parent / "vt_results" / f"recommendations_round{round_num}.csv"
        pd.DataFrame(generated).to_csv(log_path, index=False)
        print(f"Recommendations logged to {log_path}")
        print("Print these tiles, test at VT, then add results to results.csv and rerun.")

    return generated


# ── Pareto plot ────────────────────────────────────────────────────────────────

def plot_pareto(Y, labels, round_num):
    pareto = is_non_dominated(-Y)
    colors = ["red" if p else "steelblue" for p in pareto.numpy()]

    plt.figure(figsize=(8, 6))
    plt.scatter(Y[:, 0].numpy(), Y[:, 1].numpy(), c=colors, s=80, zorder=3)
    for i, label in enumerate(labels):
        plt.annotate(label, (Y[i, 0].item(), Y[i, 1].item()),
                     textcoords="offset points", xytext=(6, 4), fontsize=7)

    plt.xlabel("Peak G (lower = better)")
    plt.ylabel("BrIC (lower = better)")
    plt.title(f"Pareto Front after Round {round_num - 1} VT Testing")
    plt.grid(True, alpha=0.3)

    red_p  = plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='red',   label='Pareto-optimal')
    blue_p = plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue', label='Dominated')
    plt.legend(handles=[red_p, blue_p])
    plt.tight_layout()

    out = Path(__file__).parent.parent / "vt_results" / "pareto_front.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Pareto plot saved to {out}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = load_results()
    if df is None or len(df) < 3:
        count = len(df) if df is not None else 0
        print(f"Need at least 3 VT test results to fit surrogate. You have {count}.")
        print("Keep testing and adding results to vt_results/results.csv.")
        return

    n = len(df)
    round_num = int(n / 5) + 2
    print(f"Loaded {n} test results (Round {round_num - 1} data).\n")

    X_raw = torch.tensor(df[["cell_size", "relative_density", "topology_code"]].values)
    Y_raw = torch.tensor(df[["peak_G", "BrIC", "mass_g"]].values)

    X_norm = normalize(X_raw)
    Y_neg  = -Y_raw

    print("Fitting GP surrogate...")
    model = fit_surrogate(X_norm, Y_neg)
    print("Done.\n")

    # Validate before trusting recommendations
    if n >= 5:
        r2 = validate_surrogate(model, X_norm, Y_neg)
        min_r2 = r2[:2].min()  # peak_G and BrIC only
        if min_r2 < 0.5:
            print("WARNING: Surrogate accuracy is low (R^2 < 0.5).")
            print("Recommendations below may not be reliable yet.")
            print("Collect more VT data before acting on these suggestions.\n")
    else:
        print("(Skipping cross-validation — need 5+ points for reliable LOO.)\n")

    # Show best tested designs so far
    pareto_mask = is_non_dominated(Y_neg)
    print("Best designs tested so far (Pareto-optimal):")
    print(df[pareto_mask.numpy()][
        ["design", "topology", "cell_size", "relative_density", "peak_G", "BrIC", "mass_g"]
    ].to_string(index=False))
    print()

    plot_pareto(Y_raw, df["design"].tolist(), round_num)

    # Generate recommended STLs
    suggest_and_generate(model, df)


if __name__ == "__main__":
    main()
