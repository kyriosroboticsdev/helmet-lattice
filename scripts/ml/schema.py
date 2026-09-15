"""
schema.py
=========
THE Data Handoff Contract (spec Section 2) as code — single source of truth.

The testing team and the ML team must keep an identical copy of this contract.
Every other module imports column names, categories, and units from HERE so the
schema is defined in exactly one place. Lock this before any testing starts.

Run `python scripts/ml/schema.py` to write a blank CSV template with all columns.
"""

from __future__ import annotations
import warnings
from pathlib import Path
import pandas as pd

# ── Allowed categorical values ──────────────────────────────────────────────────
DATA_SOURCES = ["fea", "quasistatic", "linear_impact", "oblique_impact", "vt"]
# gyroid/schwartz are TPMS; auxetic = rotating-squares (negative Poisson). The
# `voronoi`/`stochastic` entries are the ORGANIC (irregular) lattices the team is
# adding — they are described mainly by the geometric descriptors below, not by the
# category, so the model can place them without many examples. Confirm when locking.
#
# Second wave (support-free, scripted in generate_tiles.py):
#   re_entrant : re-entrant / bowtie honeycomb — the OTHER standard auxetic family
#                (negative Poisson via inward-folding ribs), distinct from `auxetic`
#                (rotating squares). honeycomb : ordinary hexagonal comb (baseline).
#   elytra     : honeycomb + trans-layer columns (beetle forewing).
#   spinodoid  : spinodal-decomposition / trabecular-bone bicontinuous surface.
#   bouligand  : helicoidal ply stack (mantis-shrimp club) — shear/rotational focus.
# All are geometry-described (see descriptor dicts), so few examples are needed.
TOPOLOGIES   = ["bcc", "kelvin", "gyroid", "octet", "hybrid", "alternating",
                "graded", "schwartz", "diamond", "auxetic", "voronoi", "stochastic",
                "re_entrant", "honeycomb", "elytra", "spinodoid", "bouligand"]
IMPACT_ZONES = ["front", "boss", "side", "back"]

# ── 2.1 Identifier & design (geometry) columns ─────────────────────────────────
ID_COLS = ["design_id", "replicate_id", "data_source"]

GEOMETRY_COLS = [
    "topology",                 # categorical, one of TOPOLOGIES
    "region_encoding",          # string/JSON, for hybrid/alternating
    "rel_density_designed",     # float, as designed in CAD
    "rel_density_measured",     # float, from mass/volume of printed part (REAL feature)
    "cell_size_mm",             # float
    "strut_thickness_mm",       # float, derived from density + cell size
    "liner_thickness_mm",       # float
    "gradient_param",           # float, density slope through thickness (0 = uniform)
    "cell_aspect_y",            # float, cell size ratio y/x (1 = cubic cell)
    "cell_aspect_z",            # float, cell size ratio z/x (non-cubic cells -> anisotropy)
    "shear_anisotropy_ratio",   # float, normal stiffness / tangential stiffness
    "sa_to_v",                  # float, surface-area-to-volume (from CAD)
    "connectivity_z",           # float, nodal connectivity (from CAD)
    # ── Descriptors that let ORGANIC / irregular lattices be modeled by geometry,
    #    not category. Periodic lattices take near-constant values; organic ones vary.
    "irregularity",             # float 0..1, 0 = perfectly periodic, 1 = fully stochastic
    "strut_thickness_cv",       # float, coeff. of variation of strut/wall thickness
    "pore_size_mm",             # float, mean pore/window size (from CAD/mesh)
]

# ── 2.2 Test-condition columns ─────────────────────────────────────────────────
CONDITION_COLS = [
    "impact_zone",        # categorical, one of IMPACT_ZONES (impact tests)
    "impact_speed_mps",   # float
    "impact_energy_J",    # float
    "temperature_C",      # float, should be ~23; log it
]

# ── 2.3 Target (output) columns — units fixed, never mixed ─────────────────────
# Maps column -> (unit, source test). Units are part of the contract.
TARGET_UNITS = {
    "peak_lin_accel_g":       ("g",     "linear/oblique impact"),
    "HIC":                    ("-",     "linear impact"),
    "peak_ang_accel_rad_s2":  ("rad/s2","oblique impact"),
    "peak_ang_vel_rad_s":     ("rad/s", "oblique impact"),
    "BrIC":                   ("-",     "oblique impact"),
    "SEA_J_per_kg":           ("J/kg",  "quasi-static + impact"),
    "plateau_stress_kPa":     ("kPa",   "quasi-static"),
    "densification_strain":   ("-",     "quasi-static"),
    "restitution":            ("-",     "impact"),
}
TARGET_COLS = list(TARGET_UNITS.keys())

# Provenance / free-text columns (kept out of modeling)
META_COLS = ["raw_file_path", "notes"]

# Full ordered column list for the master table.
ALL_COLUMNS = ID_COLS + GEOMETRY_COLS + CONDITION_COLS + TARGET_COLS + META_COLS

# ── Headline optimization objectives (spec 4.1) ────────────────────────────────
# There are two competing injury axes. Both are MINIMIZED.
LINEAR_OBJECTIVE     = "HIC"     # alt: "peak_lin_accel_g"
ROTATIONAL_OBJECTIVE = "BrIC"    # alt: "peak_ang_accel_rad_s2"
OBJECTIVES = [LINEAR_OBJECTIVE, ROTATIONAL_OBJECTIVE]

# Columns the modeling code treats as numeric continuous design inputs.
CONTINUOUS_GEOM = [
    "rel_density_measured", "cell_size_mm", "strut_thickness_mm",
    "liner_thickness_mm", "gradient_param", "shear_anisotropy_ratio",
    "sa_to_v", "connectivity_z",
    "irregularity", "strut_thickness_cv", "pore_size_mm",
    "cell_aspect_y", "cell_aspect_z",
]
CATEGORICAL_INPUTS = ["topology", "impact_zone"]
NUMERIC_CONDITIONS = ["impact_speed_mps"]


# ── SEA depth safety ───────────────────────────────────────────────────────────
# SEA is an INTEGRAL up to some strain, so "SEA" is only meaningful together with
# the depth it was integrated to: SEA-to-20% and SEA-to-densification are different
# quantities and must never be compared or ranked against each other. FEA rows carry
# that depth in `fea_max_strain` (simulations stop wherever they stop); physical
# quasi-static rows carry NaN there and are measured to their own protocol depth.
SEA_DEPTH_COL = "fea_max_strain"


def sea_depths(df: pd.DataFrame, sea_col: str = "SEA_J_per_kg") -> set:
    """Distinct strain depths behind the non-null SEA values in `df`.

    NaN depth means "not an FEA row" (physical test); it is reported as None so a
    caller can see that physical and simulated SEA are being mixed too.
    """
    if sea_col not in df.columns:
        return set()
    have = df[pd.to_numeric(df[sea_col], errors="coerce").notna()]
    if SEA_DEPTH_COL not in have.columns:
        return {None} if len(have) else set()
    d = pd.to_numeric(have[SEA_DEPTH_COL], errors="coerce")
    return {None if pd.isna(v) else round(float(v), 6) for v in d}


def check_sea_depths(df: pd.DataFrame, sea_col: str = "SEA_J_per_kg",
                     strict: bool = True) -> set:
    """Raise (or warn) if `df` mixes SEA values measured to different depths.

    Call this before any ranking, regression or plot that treats an SEA column as a
    single quantity.
    """
    depths = sea_depths(df, sea_col)
    if len(depths) > 1:
        msg = (f"{sea_col} mixes {len(depths)} strain depths: {sorted(depths, key=str)}. "
               f"SEA is an integral to a depth -- comparing across depths is invalid. "
               f"Filter on {SEA_DEPTH_COL}, or recompute every row to one depth from "
               f"its stress-strain curve (see fea_optimize.curve_objectives).")
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
    return depths


def empty_frame() -> pd.DataFrame:
    """An empty DataFrame with every contract column, correctly ordered."""
    return pd.DataFrame(columns=ALL_COLUMNS)


def write_template(path: str | Path) -> Path:
    """Write a header-only CSV template that matches the contract exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    empty_frame().to_csv(path, index=False)
    return path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    out = write_template(root / "data" / "handoff_template.csv")
    print(f"Wrote handoff contract template ({len(ALL_COLUMNS)} columns) -> {out}")
    print("Categorical domains:")
    print(f"  data_source : {DATA_SOURCES}")
    print(f"  topology    : {TOPOLOGIES}")
    print(f"  impact_zone : {IMPACT_ZONES}")
    print(f"Headline objectives (both minimized): {OBJECTIVES}")
