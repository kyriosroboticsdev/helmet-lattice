"""
make_dummy_data.py
==================
Generate a synthetic master table that matches the Data Handoff Contract
(schema.py) so the ENTIRE pipeline can run end-to-end before real VT data
arrives (spec Section 10, step 2).

The synthetic physics is deliberately constructed so that:
  * relative density has a U-shaped optimum for the LINEAR metric
    (too soft bottoms out; too stiff transmits force) — Gibson-Ashby flavored,
  * shear_anisotropy_ratio is the main lever on the ROTATIONAL metric,
  * the linear and rotational optima pull in DIFFERENT directions
    => a genuine Pareto tradeoff for the multi-objective optimizer to find,
  * each design has correlated replicates (a per-design manufacturing effect)
    => leave-one-group-out CV by design_id actually matters.

None of this is real data. It exists so the code is proven correct on fake
data; swap in the real handoff CSV and the pipeline is unchanged.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import qmc

import schema as S

# Optional: measure sa_to_v from the real TPMS field, exactly like the live pipeline
# (features.derive_geometry). Falls back to the shape-factor proxy if unavailable.
try:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import generate_tiles as _GEN
except Exception:
    _GEN = None

_TPMS = {"gyroid", "schwartz", "diamond"}


def _sa_to_v(topo, cx, cy, cz, rho):
    if _GEN is not None and topo in _TPMS:
        try:
            m = _GEN.measure_tpms_descriptors({topo: 1.0}, (cx, cy, cz), rho)
            if m is not None:
                return float(m["sa_to_v"])
        except Exception:
            pass
    return SA_FACTOR[topo] / ((cx * cy * cz) ** (1.0 / 3.0))

# Nominal nodal connectivity per topology (graph degree of a lattice node).
CONNECTIVITY_Z = {
    "bcc": 8.0, "kelvin": 4.0, "gyroid": 6.0, "octet": 12.0,
    "hybrid": 8.0, "alternating": 7.0, "graded": 6.0,
    "schwartz": 6.0, "diamond": 4.0, "auxetic": 4.0, "voronoi": 5.2, "stochastic": 5.5,
    "re_entrant": 3.0, "honeycomb": 3.0, "elytra": 4.0, "spinodoid": 5.0, "bouligand": 6.0,
}
# Surface-area-to-volume shape factor per topology (multiplies 1/cell_size).
SA_FACTOR = {
    "bcc": 3.2, "kelvin": 3.6, "gyroid": 4.4, "octet": 3.8,
    "hybrid": 3.7, "alternating": 3.9, "graded": 4.2,
    "schwartz": 4.0, "diamond": 4.3, "auxetic": 3.0, "voronoi": 4.6, "stochastic": 4.8,
    "re_entrant": 4.2, "honeycomb": 4.2, "elytra": 4.0, "spinodoid": 4.6, "bouligand": 4.2,
}
# Geometric character per topology (organic lattices are the irregular ones).
IRREGULARITY = {
    "bcc": 0.0, "kelvin": 0.0, "gyroid": 0.0, "octet": 0.0, "schwartz": 0.0,
    "diamond": 0.0, "auxetic": 0.0, "graded": 0.10, "alternating": 0.15, "hybrid": 0.15,
    "voronoi": 0.85, "stochastic": 0.95,
    "re_entrant": 0.0, "honeycomb": 0.0, "elytra": 0.0, "spinodoid": 0.80, "bouligand": 0.10,
}
STRUT_CV = {
    "bcc": 0.05, "kelvin": 0.05, "gyroid": 0.04, "octet": 0.05, "schwartz": 0.04,
    "diamond": 0.04, "auxetic": 0.05, "graded": 0.12, "alternating": 0.12, "hybrid": 0.13,
    "voronoi": 0.35, "stochastic": 0.42,
    "re_entrant": 0.05, "honeycomb": 0.04, "elytra": 0.06, "spinodoid": 0.30, "bouligand": 0.08,
}
# Topology offsets: + worsens that injury axis. Stiff cells (octet) help linear
# but hurt rotational; compliant cells (kelvin/gyroid) do the opposite. Organic
# lattices (voronoi/stochastic) spread load: mild linear penalty, rotational help.
# Second wave: re_entrant tracks auxetic (negative Poisson pulls material inward ->
# strong rotational help); elytra columns are stiff (linear help, rotational cost);
# spinodoid tortuosity dissipates shear; bouligand is the rotational specialist.
TOPO_LIN = {"bcc": 6, "kelvin": 14, "gyroid": 10, "octet": -4,
            "hybrid": 4, "alternating": 5, "graded": 0,
            "schwartz": 9, "diamond": 8, "auxetic": 12, "voronoi": 7, "stochastic": 8,
            "re_entrant": 12, "honeycomb": 6, "elytra": -2, "spinodoid": 4, "bouligand": 5}
TOPO_ROT = {"bcc": 300, "kelvin": -250, "gyroid": -350, "octet": 500,
            "hybrid": 0, "alternating": -100, "graded": -200,
            "schwartz": -300, "diamond": -320, "auxetic": -450,
            "voronoi": -380, "stochastic": -400,
            "re_entrant": -450, "honeycomb": 50, "elytra": 250, "spinodoid": -380,
            "bouligand": -520}


def _strut_thickness(cell_size, rel_density):
    """Beam-lattice scaling rel_density ~ (t/cell)^2 -> t = cell*sqrt(rho)*k."""
    return cell_size * np.sqrt(np.clip(rel_density, 1e-3, None)) * 0.60


def _physics(rng, r, c, g, s, topo, n, irr=0.0, het=0.0, az=1.0, ay=1.0):
    """Vectorized synthetic targets for n replicate rows of one design.

    r=rel_density_measured, c=cell_size_mm, g=gradient_param,
    s=shear_anisotropy_ratio (normal/tangential), topo=topology str,
    irr=irregularity (organic), het=heterogeneity (layered region mix),
    az=cell_aspect_z (z/x), ay=cell_aspect_y (y/x) -> non-cubic cell anisotropy.
    Returns a dict of arrays length n.
    """
    r_opt = 0.27  # density that minimizes the linear metric

    # ---- LINEAR metric: U-shaped in density, mildly worsened by high shear ratio
    peak_lin = (82.0
                + 620.0 * (r - r_opt) ** 2          # Gibson-Ashby-ish optimum
                + 2.0 * (c - 8.0)                    # coarser cells slightly worse
                - 35.0 * g                           # graded soft-entry helps
                + 20.0 * (s - 1.0)                   # shear-compliant => less linear absorption
                - 18.0 * het                         # layered soft-entry helps linear
                + 8.0 * irr                          # organic = less efficient load path
                - 22.0 * (az - 1.0)                  # taller Z cells => more compliant in Z
                + TOPO_LIN[topo])
    peak_lin = peak_lin + rng.normal(0, 3.0, n)
    peak_lin = np.clip(peak_lin, 55, None)

    HIC = 150.0 * (peak_lin / 90.0) ** 2.5 + rng.normal(0, 12.0, n)
    HIC = np.clip(HIC, 50, None)

    # ---- ROTATIONAL metric: improved by high shear ratio and density, hurt by softness
    peak_ang = (5200.0
                - 1500.0 * s                         # compliant shear => less rotational transfer
                + 2600.0 * (0.30 - r)                # very soft liner => more shear motion (worse)
                - 700.0 * g
                - 250.0 * irr                        # organic tortuosity dissipates shear
                + 220.0 * (ay - 1.0)                 # in-plane cell anisotropy alters shear transfer
                + TOPO_ROT[topo])
    peak_ang = np.clip(peak_ang + rng.normal(0, 120.0, n), 800, None)

    peak_ang_vel = peak_ang / 180.0 + rng.normal(0, 0.8, n)
    peak_ang_vel = np.clip(peak_ang_vel, 5, None)
    BrIC = peak_ang_vel / 46.4 + rng.normal(0, 0.02, n)   # BrIC uses critical ang. velocities

    restitution = np.clip(0.22 + 0.35 * r + rng.normal(0, 0.02, n), 0.05, 0.95)

    # ---- Quasi-static metrics
    SEA = 9000.0 * (1 - np.exp(-6.0 * r)) * (0.9 + 0.05 * CONNECTIVITY_Z[topo] / 8) \
          + rng.normal(0, 250.0, n)
    plateau = 1200.0 * r ** 2 * (0.8 + 0.04 * CONNECTIVITY_Z[topo]) + rng.normal(0, 6.0, n)
    densification = np.clip(0.92 - 1.05 * r - 0.10 * (az - 1.0)
                            + rng.normal(0, 0.02, n), 0.2, 0.95)

    return dict(
        peak_lin_accel_g=peak_lin, HIC=HIC,
        peak_ang_accel_rad_s2=peak_ang, peak_ang_vel_rad_s=peak_ang_vel, BrIC=BrIC,
        SEA_J_per_kg=np.clip(SEA, 100, None), plateau_stress_kPa=np.clip(plateau, 5, None),
        densification_strain=densification, restitution=restitution,
    )


def make_dummy(seed: int = 7,
               n_designs: int = 42,
               n_replicates: int = 3,
               frac_impact: float = 0.55) -> pd.DataFrame:
    """Build a synthetic master table.

    n_designs quasi-static screening designs (x n_replicates), of which a
    fraction also receive oblique-impact testing (x n_replicates).
    """
    rng = np.random.default_rng(seed)

    # Space-filling design of experiments over the continuous knobs.
    # cols: rel_density, cell, aspect_y, aspect_z, liner, gradient, shear
    lhs = qmc.LatinHypercube(d=7, seed=seed).random(n_designs)
    lo = np.array([0.18, 5.0, 0.6, 0.6, 22.0, 0.00, 0.6])
    hi = np.array([0.42, 12.0, 1.6, 1.6, 30.0, 0.30, 1.8])
    P = lo + lhs * (hi - lo)

    topo_cycle = (S.TOPOLOGIES * (n_designs // len(S.TOPOLOGIES) + 1))[:n_designs]
    rng.shuffle(topo_cycle)

    impact_designs = set(rng.choice(n_designs, size=int(round(frac_impact * n_designs)),
                                    replace=False).tolist())

    rows = []
    for d in range(n_designs):
        rd_designed, cell, aspect_y, aspect_z, liner, grad, shear = P[d]
        topo = topo_cycle[d]
        design_id = f"D{d:03d}_{topo}"

        # Per-design manufacturing effect shared across that design's replicates
        # (this is what makes grouped CV by design_id necessary).
        dens_bias = rng.normal(0.0, 0.006)            # print over/under-extrusion
        design_effect = dict(lin=rng.normal(0, 4.0), rot=rng.normal(0, 90.0))

        # Per-design geometric character (constant across replicates).
        irr = float(np.clip(IRREGULARITY[topo] + rng.normal(0, 0.02), 0.0, 1.0))
        if topo in ("hybrid", "alternating"):
            region = ('{"layers":[{"topology":"gyroid","frac":0.5},'
                      '{"topology":"schwartz","frac":0.5}]}')
            het = 0.5
        else:
            region = "uniform"
            het = 0.0

        def base_geom(rep):
            rd_meas = float(np.clip(rd_designed + 0.01 + dens_bias
                                    + rng.normal(0, 0.004), 0.1, 0.55))
            cx, cy, cz = cell, cell * aspect_y, cell * aspect_z
            cmean = (cx * cy * cz) ** (1.0 / 3.0)
            strut = float(_strut_thickness(cmean, rd_meas))
            sa_v = float(_sa_to_v(topo, cx, cy, cz, rd_meas) + rng.normal(0, 0.02))
            return dict(
                design_id=design_id, replicate_id=rep,
                topology=topo, region_encoding=region,
                rel_density_designed=round(float(rd_designed), 4),
                rel_density_measured=round(rd_meas, 4),
                cell_size_mm=round(float(cell), 3),
                cell_aspect_y=round(float(aspect_y), 4),
                cell_aspect_z=round(float(aspect_z), 4),
                strut_thickness_mm=round(strut, 4),
                liner_thickness_mm=round(float(liner), 2),
                gradient_param=round(float(grad), 4),
                shear_anisotropy_ratio=round(float(shear), 4),
                sa_to_v=round(sa_v, 4),
                connectivity_z=CONNECTIVITY_Z[topo],
                irregularity=round(irr, 4),
                strut_thickness_cv=round(float(np.clip(
                    STRUT_CV[topo] + rng.normal(0, 0.01), 0.0, None)), 4),
                pore_size_mm=round(float(cmean * max(0.1, 1.0 - np.sqrt(rd_meas))), 4),
                temperature_C=round(float(rng.normal(23.0, 0.4)), 2),
            )

        # --- Quasi-static screening rows (SEA / plateau / densification) ---
        for rep in range(1, n_replicates + 1):
            g = base_geom(rep)
            t = _physics(rng, g["rel_density_measured"], g["cell_size_mm"],
                         g["gradient_param"], g["shear_anisotropy_ratio"], topo, 1,
                         irr=irr, het=het, az=aspect_z, ay=aspect_y)
            row = {**g, "data_source": "quasistatic",
                   "impact_zone": "", "impact_speed_mps": np.nan, "impact_energy_J": np.nan,
                   "SEA_J_per_kg": round(float(t["SEA_J_per_kg"][0]), 1),
                   "plateau_stress_kPa": round(float(t["plateau_stress_kPa"][0]), 2),
                   "densification_strain": round(float(t["densification_strain"][0]), 4),
                   "raw_file_path": f"raw/qs/{design_id}_r{rep}.csv", "notes": ""}
            rows.append(row)

        # --- Oblique-impact rows (linear + rotational headline metrics) ---
        if d in impact_designs:
            for rep in range(1, n_replicates + 1):
                g = base_geom(rep)
                speed = 5.0
                t = _physics(rng, g["rel_density_measured"], g["cell_size_mm"],
                             g["gradient_param"], g["shear_anisotropy_ratio"], topo, 1,
                             irr=irr, het=het, az=aspect_z, ay=aspect_y)
                row = {**g, "data_source": "oblique_impact",
                       "impact_zone": "front", "impact_speed_mps": speed,
                       "impact_energy_J": round(0.5 * 5.0 * speed ** 2, 2),
                       "peak_lin_accel_g": round(float(t["peak_lin_accel_g"][0]) + design_effect["lin"], 2),
                       "HIC": round(float(t["HIC"][0]), 1),
                       "peak_ang_accel_rad_s2": round(float(t["peak_ang_accel_rad_s2"][0]) + design_effect["rot"], 1),
                       "peak_ang_vel_rad_s": round(float(t["peak_ang_vel_rad_s"][0]), 3),
                       "BrIC": round(float(t["BrIC"][0]), 4),
                       "restitution": round(float(t["restitution"][0]), 3),
                       "SEA_J_per_kg": round(float(t["SEA_J_per_kg"][0]), 1),
                       "raw_file_path": f"raw/oblique/{design_id}_r{rep}.csv", "notes": ""}
                rows.append(row)

    df = pd.DataFrame(rows)

    # Inject a couple of CLIPPED sensor rows: implausible peak, flagged in notes.
    # The loader must exclude these from training (spec 2.4).
    clip_idx = df.index[df["data_source"] == "oblique_impact"][:2]
    df.loc[clip_idx, "peak_lin_accel_g"] = 999.0
    df.loc[clip_idx, "notes"] = "clipped sensor signal - exclude"

    # Reindex to the exact contract column order; missing cols become NaN.
    df = df.reindex(columns=S.ALL_COLUMNS)
    return df


def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = make_dummy()
    csv = out_dir / "dummy_master.csv"
    df.to_csv(csv, index=False)
    try:
        df.to_parquet(out_dir / "dummy_master.parquet", index=False)
    except Exception:
        pass
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {csv}")
    print(df.groupby("data_source").size().to_string())
    print(f"Unique designs: {df['design_id'].nunique()} | "
          f"clipped rows flagged: {(df['notes'] != '').sum()}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    main()
