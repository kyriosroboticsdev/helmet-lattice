"""Re-score a finished loop run from its saved stress-strain curves.

No FEA is re-run. The point is that the screening objectives stored in
master_table.csv cannot answer the question the study actually asks:

  * `plateau_stress_kPa` is NaN in every row. fea_evaluator.PLATEAU_BAND is
    (0.20, 0.40) and the plateau is only reported if the solve reached 0.40,
    but screening crushes to 0.20. So the column is structurally empty.
  * What the loop minimised instead is `stress_at_max_strain_kPa` -- stress at
    20% strain, i.e. a stiffness proxy. It tracks density, and so does SEA,
    which is why the two objectives collapse onto one.

This recomputes, over a band the 20% solve actually covers:

  plateau_stress_kPa_10_20  mean nominal stress over strain 0.10-0.20
  plateau_cv                std/mean over that band -- how flat the plateau is
  cushioning_efficiency     eta = (integral of sigma de) / (sigma(e) * e) at
                            e=0.20 -- energy absorbed as a fraction of the
                            ideal-absorber rectangle. 1.0 is a perfectly flat
                            plateau; a linear-elastic curve gives 0.5.

Usage:  python scripts/loop/rescore_curves.py runs/loop_2026-09-14
"""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd

BAND = (0.10, 0.20)
EVAL_STRAIN = 0.20


def _pareto(pts: np.ndarray, maximize: tuple[bool, ...]) -> list[int]:
    """Indices of non-dominated rows. pts is (n, 2)."""
    sign = np.array([1.0 if m else -1.0 for m in maximize])
    v = pts * sign                                   # now all maximised
    keep = []
    for i, p in enumerate(v):
        dominated = ((v >= p).all(axis=1) & (v > p).any(axis=1)).any()
        if not dominated:
            keep.append(i)
    return keep


def rescore(run_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(run_dir, "master_table.csv"))
    # Only the clean screening set: solved, crushed to 20%, and actually got there.
    ok = df[(df["fea_status"] == "ok")
            & (df["fea_crush"] == 0.2)
            & (df["fea_max_strain"] >= EVAL_STRAIN - 1e-3)].copy()

    out = []
    for _, r in ok.iterrows():
        p = os.path.join(run_dir, "curves", f"{r['design_id']}.csv")
        if not os.path.exists(p):
            continue
        c = pd.read_csv(p)
        e, s = c["strain"].to_numpy(), c["stress_kPa"].to_numpy()
        band = (e >= BAND[0]) & (e <= BAND[1])
        if band.sum() < 2:
            continue
        plat = float(np.mean(s[band]))
        w = float(np.trapezoid(s, e))                # kPa == mJ/mm^3
        s_at = float(np.interp(EVAL_STRAIN, e, s))
        out.append(dict(
            design_id=r["design_id"],
            iteration=r["iteration"],
            topology=r["topology"],
            rel_density=r["rel_density_measured"],
            SEA_J_per_kg=r["SEA_J_per_kg"],
            stress_at_20pct_kPa=r["stress_at_max_strain_kPa"],
            plateau_stress_kPa_10_20=round(plat, 1),
            plateau_cv=round(float(np.std(s[band]) / plat), 3),
            cushioning_efficiency=round(w / (s_at * EVAL_STRAIN) if s_at > 0 else np.nan, 3),
        ))
    d = pd.DataFrame(out)

    # Rank under each objective pair.
    screen = _pareto(d[["SEA_J_per_kg", "stress_at_20pct_kPa"]].to_numpy(), (True, False))
    effic = _pareto(d[["cushioning_efficiency", "plateau_stress_kPa_10_20"]].to_numpy(), (True, False))
    d["front_as_screened"] = d.index.isin(screen)
    d["front_efficiency"] = d.index.isin(effic)
    return d


def main() -> None:
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/loop_2026-09-14"
    d = rescore(run_dir)
    dest = os.path.join(run_dir, "rescore.csv")
    d.to_csv(dest, index=False)

    pairs = [("SEA_J_per_kg", "stress_at_20pct_kPa"),
             ("SEA_J_per_kg", "plateau_stress_kPa_10_20"),
             ("SEA_J_per_kg", "rel_density"),
             ("cushioning_efficiency", "plateau_stress_kPa_10_20"),
             ("cushioning_efficiency", "rel_density")]
    print(f"{len(d)} designs re-scored from saved curves -> {dest}\n")
    print("correlations")
    for a, b in pairs:
        print(f"  {a:>22} vs {b:<26} {d[a].corr(d[b]):+.3f}")

    for col, label in [("front_as_screened", "AS SCREENED  (max SEA, min stress@20%)"),
                       ("front_efficiency", "RE-SCORED    (max cushioning eff, min plateau)")]:
        f = d[d[col]].sort_values("plateau_stress_kPa_10_20")
        print(f"\n{label} -- {len(f)} designs")
        print(f[["design_id", "rel_density", "SEA_J_per_kg", "stress_at_20pct_kPa",
                 "plateau_stress_kPa_10_20", "plateau_cv",
                 "cushioning_efficiency"]].to_string(index=False))


if __name__ == "__main__":
    main()
