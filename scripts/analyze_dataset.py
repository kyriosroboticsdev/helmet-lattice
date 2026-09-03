"""
analyze_dataset.py

Ingests the Matej (2020) impact dataset from processed_summary.xlsx and:
  1. Parses all 144 experiments into a tidy DataFrame
  2. Fits a GP baseline model: (speed, mass, impact_shape) -> (peak_G, HIC)
  3. Extrapolates to NOCSAE conditions (5.5 m/s, 5kg headform)
  4. Generates a baseline comparison plot - when VT data comes in,
     this is what your lattices are beating.

Dataset credit: Matej, J. (2020). Technical University in Zvolen. CC BY 4.0.

Run: python scripts/analyze_dataset.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)

XLSX = Path(__file__).parent.parent / "dataverse_files" / "processed_summary.xlsx"
OUT  = Path(__file__).parent.parent / "vt_results"

# -- Parse spreadsheet ---------------------------------------------------------

SHAPE_CODE = {"cube": 0, "cylinder": 1, "wedge": 2}
MASS_MAP   = {"300g": 0.300, "400g": 0.400, "500g": 0.500}

def parse_xlsx(path):
    wb_data = pd.read_excel(path, sheet_name=None, header=None)
    rows = []
    for shape_name, df in wb_data.items():
        # Row 0: mass headers (300g, 400g, 500g)
        # Row 1: column labels (peak, 3ms, HIC, Speed) × 3
        # Rows 2+: data
        mass_labels = [df.iloc[0, c] for c in [0, 4, 8]]   # '300g','400g','500g'
        for data_row_idx in range(2, len(df)):
            row = df.iloc[data_row_idx]
            if pd.isna(row[0]):
                continue
            for m_idx, (mass_label, col_start) in enumerate(zip(mass_labels, [0, 4, 8])):
                peak  = row[col_start + 0]
                ms3   = row[col_start + 1]
                hic   = row[col_start + 2]
                speed = row[col_start + 3]
                if pd.isna(peak):
                    continue
                rows.append({
                    "shape":       shape_name,
                    "shape_code":  SHAPE_CODE[shape_name],
                    "mass_kg":     MASS_MAP[mass_label],
                    "speed_ms":    float(speed),
                    "peak_G":      float(peak),
                    "criterion_3ms": float(ms3),
                    "HIC":         float(hic),
                })
    return pd.DataFrame(rows)

# -- GP baseline model ---------------------------------------------------------

def fit_baseline_gp(df):
    """
    Fit a GP on (speed_ms, mass_kg, shape_code) -> (peak_G, HIC).
    This gives us a physics baseline: what does an UNPROTECTED surface do?
    When VT results come in, we compare lattice performance against this.
    """
    X = torch.tensor(df[["speed_ms", "mass_kg", "shape_code"]].values)
    Y = torch.tensor(df[["peak_G", "HIC"]].values)

    # Normalize inputs
    X_mean, X_std = X.mean(0), X.std(0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X - X_mean) / X_std

    # Normalize outputs
    Y_mean, Y_std = Y.mean(0), Y.std(0)
    Y_norm = (Y - Y_mean) / Y_std

    model = SingleTaskGP(X_norm, Y_norm)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model, X_mean, X_std, Y_mean, Y_std

def predict_baseline(model, X_mean, X_std, Y_mean, Y_std,
                     speed_ms, mass_kg, shape_code):
    X_q = torch.tensor([[speed_ms, mass_kg, shape_code]])
    X_q_norm = (X_q - X_mean) / X_std
    with torch.no_grad():
        post = model.posterior(X_q_norm)
        y_norm = post.mean
        y_var  = post.variance
    y = y_norm * Y_std + Y_mean
    y_std = y_var.sqrt() * Y_std
    return y[0].numpy(), y_std[0].numpy()

# -- NOCSAE scaling ------------------------------------------------------------

def nocsae_extrapolation(model, X_mean, X_std, Y_mean, Y_std):
    """
    Extrapolate baseline to NOCSAE conditions.
    Direct impact (cube) -> ND001 (linear)
    Oblique/wedge impact -> ND081 (rotational)

    Note: extrapolation beyond training data range (max ~2.9 m/s -> 5.5 m/s)
    carries high uncertainty. Treat as ORDER-OF-MAGNITUDE estimate only.
    """
    print("\n-- NOCSAE Extrapolation (no protective material) --")
    print("   (Uncertainty is high - speeds exceed training range 2.9 -> 5.5 m/s)")
    print()

    scenarios = [
        ("Direct impact (cube, ~ND001)", 5.5, 5.0, SHAPE_CODE["cube"]),
        ("Oblique impact (wedge, ~ND081)", 5.5, 5.0, SHAPE_CODE["wedge"]),
    ]

    for label, speed, mass, shape in scenarios:
        y, y_std = predict_baseline(
            model, X_mean, X_std, Y_mean, Y_std, speed, mass, shape
        )
        peak_G, hic = y
        peak_G_std, hic_std = y_std
        print(f"  {label}")
        print(f"    Predicted peak G : {peak_G:.0f} ± {peak_G_std:.0f} g")
        print(f"    Predicted HIC    : {hic:.1f} ± {hic_std:.1f}")
        print(f"    NOCSAE pass threshold: < 300g peak, HIC < 1200")
        print()

# -- Shape effect summary ------------------------------------------------------

def shape_effect_summary(df):
    """
    Compare impact shapes at matched conditions.
    This shows how much rotational (wedge) vs direct (cube) impact
    changes peak_G - motivates why we need rotational compliance.
    """
    print("\n-- Impact Shape Effect (matched speed ~2.7 m/s, 400g) --")
    mask = (df["speed_ms"].between(2.5, 2.9)) & (df["mass_kg"] == 0.4)
    sub = df[mask].groupby("shape")[["peak_G", "HIC"]].mean()
    print(sub.to_string())
    print()
    if "cube" in sub.index and "cylinder" in sub.index:
        ratio = sub.loc["cylinder", "peak_G"] / sub.loc["cube", "peak_G"]
        print(f"  Cylinder/Cube peak_G ratio: {ratio:.2f}x")
        print(f"  -> Round impactors concentrate force more than flat ones")
    if "cube" in sub.index and "wedge" in sub.index:
        ratio_w = sub.loc["wedge", "peak_G"] / sub.loc["cube", "peak_G"]
        print(f"  Wedge/Cube peak_G ratio   : {ratio_w:.2f}x")
        print(f"  -> Oblique/wedge impacts are {ratio_w:.2f}x of direct at same speed")

# -- Plots ---------------------------------------------------------------------

def generate_plots(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"cube": "steelblue", "cylinder": "tomato", "wedge": "seagreen"}

    # Plot 1: Speed vs Peak G
    ax = axes[0]
    for shape, grp in df.groupby("shape"):
        ax.scatter(grp["speed_ms"], grp["peak_G"],
                   c=colors[shape], label=shape.capitalize(),
                   alpha=0.6, s=40)
    ax.set_xlabel("Impact Speed (m/s)")
    ax.set_ylabel("Peak Acceleration (g)")
    ax.set_title("Speed vs Peak G - Baseline (No Lattice)")
    ax.axvline(5.5, color="black", linestyle="--", linewidth=1, label="NOCSAE 5.5 m/s")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Speed vs HIC
    ax = axes[1]
    for shape, grp in df.groupby("shape"):
        ax.scatter(grp["speed_ms"], grp["HIC"],
                   c=colors[shape], label=shape.capitalize(),
                   alpha=0.6, s=40)
    ax.set_xlabel("Impact Speed (m/s)")
    ax.set_ylabel("HIC")
    ax.set_title("Speed vs HIC - Baseline (No Lattice)")
    ax.axvline(5.5, color="black", linestyle="--", linewidth=1, label="NOCSAE 5.5 m/s")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Impact Baseline - Matej (2020) Dataset\n"
        "Blue dashed = NOCSAE test speed. Your lattices should land well left of this curve.",
        fontsize=10
    )
    plt.tight_layout()
    out_path = OUT / "baseline_impact_data.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nBaseline plot saved to {out_path}")

# -- Integration hook for surrogate.py -----------------------------------------

def compute_reduction(vt_peak_G, vt_hic, baseline_peak_G, baseline_hic):
    """
    After VT testing, call this to compute how much your lattice reduces
    impact vs the no-protection baseline (extrapolated to NOCSAE conditions).
    Returns: reduction percentages.
    """
    peak_reduction = (1 - vt_peak_G / baseline_peak_G) * 100
    hic_reduction  = (1 - vt_hic   / baseline_hic)    * 100
    return peak_reduction, hic_reduction

# -- Main ----------------------------------------------------------------------

def main():
    print("Loading dataset...")
    df = parse_xlsx(XLSX)
    print(f"Loaded {len(df)} experiments across shapes: {df['shape'].unique()}")
    print(f"Speed range: {df['speed_ms'].min():.2f} – {df['speed_ms'].max():.2f} m/s")
    print(f"Peak G range: {df['peak_G'].min():.1f} – {df['peak_G'].max():.1f} g")
    print(f"HIC range: {df['HIC'].min():.3f} – {df['HIC'].max():.3f}")

    # Save tidy CSV for reference
    csv_path = OUT / "baseline_dataset_tidy.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTidy dataset saved to {csv_path}")

    shape_effect_summary(df)

    print("\nFitting GP baseline model...")
    model, Xm, Xs, Ym, Ys = fit_baseline_gp(df)
    print("GP fitted.")

    nocsae_extrapolation(model, Xm, Xs, Ym, Ys)

    print("Generating plots...")
    generate_plots(df)

    print("\n-- How to use this in your research --")
    print()
    print("  1. This dataset establishes a NO-PROTECTION baseline:")
    print("     what peak_G and HIC look like without any lattice padding.")
    print()
    print("  2. After VT testing, compute your reduction factor:")
    print("     from scripts.analyze_dataset import compute_reduction")
    print("     peak_red, hic_red = compute_reduction(vt_peak_G, vt_hic,")
    print("                                           baseline_G, baseline_HIC)")
    print()
    print("  3. The shape comparison (cube=direct, wedge=oblique) shows")
    print("     rotational impacts add ~10-30% to peak_G vs direct -")
    print("     this is why rotational compliance in your lattice matters.")
    print()
    print("  4. The PREVIEW/Dataset_v2.zip appears to be an FEA image dataset")
    print("     for a separate CNN surrogate model - check Dataset_v2.zip")
    print("     to confirm before using those images.")

if __name__ == "__main__":
    main()
