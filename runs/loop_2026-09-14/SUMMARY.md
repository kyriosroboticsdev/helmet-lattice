# FEA design loop — run `loop_2026-09-14`

Branch `fea-loop`. Screening model: 2×2×2-cell piece, Neo-Hooke C10=5.005 D1=0.0435,
CalculiX, crushed to 20% nominal strain. Reproduce the tables below with:

    python scripts/loop/rescore_curves.py runs/loop_2026-09-14

## Run outcome

The loop terminated on its own — `NO_IMPROVEMENT: best metric improved < 1% for
5 rounds`, at iteration 25 of a 40 cap. Not a crash, not a timeout.

| | |
|---|---|
| Designs attempted | 133 |
| Rejected on printability / realization | 69 |
| Solver failures | 2 |
| Solved | 62 |
| **Reached a clean 20% crush (usable)** | **27** |
| Solver wall time | ~9.5 h |

The 35 solved-but-unusable designs stalled between 1.7% and 19% crush. SEA is an
integral *to a depth*, so those cannot be pooled with the 20% set — `schema.check_sea_depths`
enforces this and we should keep it that way.

Five Pareto designs were re-run on the full tile. Piece-vs-tile agreement is good
and now consistent in sign: **−2.0% on stress, −1.2% on SEA** (mean, n=4). `it020_0`
is excluded — its full-tile solve reached 0.094 strain and the comparison reads at
0.100, so it reports `nan`.

## The finding that matters

**`plateau_stress_kPa` is NaN in all 133 rows, and structurally so.**
`fea_evaluator.py:80` sets `PLATEAU_BAND = (0.20, 0.40)` and only reports a plateau
if the solve reached 0.40. Screening crushes to 0.20, so the band is never covered.
`densification_strain` is empty for the same reason.

What the loop actually minimized is `stress_at_max_strain_kPa` — stress at 20%
strain. That is a stiffness proxy, not a cushioning metric, and it tracks density.
So does SEA. Hence the objectives collapse:

| pair | r |
|---|---|
| SEA vs stress@20% | **+0.913** |
| SEA vs true plateau stress (0.10–0.20 band) | **+0.948** |
| SEA vs relative density | +0.497 |
| **cushioning efficiency vs plateau stress** | **−0.291** |
| cushioning efficiency vs relative density | −0.417 |

Re-defining the plateau over a band the solve *does* cover makes the collinearity
worse, not better (+0.948). So this is not a metric-definition artifact — for this
design space SEA and plateau stress genuinely are one objective, and the qEHVI
front was largely ranking density. Replacing the energy term with **cushioning
efficiency** (η = ∫σdε ⁄ σ(ε)·ε at ε=0.20; 1.0 = ideal flat plateau, 0.5 = linear
elastic) is what separates them.

## Both rankings, side by side

**As screened — maximize SEA, minimize stress@20%. 11 of 27 on the front.**

| design | ρ | SEA J/kg | σ@20% kPa | plateau kPa | plateau CV | η |
|---|---|---|---|---|---|---|
| it015_0 | 0.273 | 17.7 | 51.0 | 43.3 | 0.135 | 0.576 |
| it004_0 | 0.196 | 79.5 | 116.4 | 114.8 | 0.020 | 0.816 |
| it005_0 | 0.195 | 129.4 | 176.1 | 179.0 | 0.012 | 0.874 |
| it017_0 | 0.187 | 201.1 | 302.9 | 279.5 | 0.057 | 0.756 |
| it014_1 | 0.270 | 267.1 | 589.2 | 569.3 | 0.027 | 0.747 |
| it019_3 | 0.271 | 331.6 | 739.2 | 694.2 | 0.028 | 0.741 |
| it022_3 | 0.272 | 359.7 | 819.4 | 793.3 | 0.045 | 0.728 |
| it020_0 | 0.266 | 378.7 | 826.0 | 839.7 | 0.041 | 0.743 |
| it014_0 | 0.268 | 439.6 | 1134.7 | 1067.9 | 0.083 | 0.633 |
| it020_1 | 0.265 | 479.3 | 1208.2 | 1207.2 | 0.090 | 0.641 |
| it018_1 | 0.341 | 493.4 | 1773.2 | 1619.5 | 0.141 | 0.578 |

**Re-scored — maximize cushioning efficiency, minimize plateau stress. 3 of 27.**

| design | ρ | plateau kPa | plateau CV | η | composition | cell |
|---|---|---|---|---|---|---|
| it015_0 | 0.273 | 43.3 | 0.135 | 0.576 | diamond .45 / elytra .22 / honeycomb .18 / re-entrant .16 | 14.6 mm |
| it004_0 | 0.196 | 114.8 | 0.020 | 0.816 | diamond .32 / schwartz .32 / elytra .26 / re-entrant .06 / honeycomb .04 | 20.4 mm |
| it005_0 | 0.195 | 179.0 | 0.012 | 0.874 | schwartz .45 / elytra .44 / diamond .11 | 14.6 mm |

Read the two tables together: **`it018_1`, the SEA champion the loop converged
toward, has η = 0.578** — barely better than a linear spring. It absorbs a lot of
energy by being dense and stiff, which is the opposite of what a liner should do.
The genuinely good cushions are `it005_0` and `it004_0` at ρ ≈ 0.195, with plateau
stress varying only 1–2% across the band. Both are TPMS-dominant (schwartz +
diamond) with a large elytra fraction — a consistent family, not a coincidence of
two samples.

## Caveats — please carry these into any slide

- **n = 27.** The blend weight space is 9-dimensional. The surrogate is
  under-determined and these fronts are directional, not quantitative.
- **Neo-Hooke at 33% validated error.** The calibrated Ogden card does not
  converge in CalculiX 2.22 and Mooney–Rivlin stalls at ~5% crush on a lattice.
  Absolute kPa values inherit that error; the *ordering* is more trustworthy
  than the magnitudes.
- **Quasi-static.** None of this is an impact rate. HIC and BrIC cannot be read
  off these runs — they need physical tests.
- No design here has densified by 20% crush, so the true densification strain
  and the full plateau length are still unmeasured.

## Suggested next step

Re-run the loop with `--energy-metric efficiency --rebuild-hv-reference` so the
proposals chase the decorrelated pair instead of density. Budget ~9 h of solver
time. Worth considering a deeper screening crush (35–40%) on a small subset first,
to measure densification and the real plateau length at least once.
