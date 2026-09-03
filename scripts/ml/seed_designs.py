"""
seed_designs.py
===============
Pre-fill the Data Handoff Contract with the five already-printed designs
(D1-D5), so collecting real data is fill-in-the-blanks: the GEOMETRY columns
are populated from each design's parameters; everything that must be MEASURED
(rel_density_measured, every target metric) is left blank for the testing team.

Parameters are taken straight from scripts/generate_tiles.py.
Run:  python scripts/ml/seed_designs.py   ->  data/designs_D1_D5.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

import schema as S
import features as F

# Each design's controllable parameters (from generate_tiles.py).
# strut_thickness_mm / sa_to_v / connectivity_z are DERIVED (Section 6), so we
# compute them with the same function the optimizer uses -> consistent features.
DESIGNS = [
    dict(design_id="D1_gyroid_025_8mm", topology="gyroid",
         rel_density_designed=0.25, cell_size_mm=8.0, gradient_param=0.0,
         region_encoding="uniform"),
    dict(design_id="D2_gyroid_035_6mm", topology="gyroid",
         rel_density_designed=0.35, cell_size_mm=6.0, gradient_param=0.0,
         region_encoding="uniform"),
    dict(design_id="D3_rotating_squares_auxetic", topology="auxetic",
         rel_density_designed=0.29, cell_size_mm=10.0, gradient_param=0.0,
         region_encoding="rotating-squares; s=5mm gap=5mm hinge_r=1.5mm"),
    dict(design_id="D4_schwartz_025_8mm", topology="schwartz",
         rel_density_designed=0.25, cell_size_mm=8.0, gradient_param=0.0,
         region_encoding="uniform"),
    dict(design_id="D5_graded_gyroid_018to035", topology="graded",
         rel_density_designed=0.265, cell_size_mm=8.0, gradient_param=0.17,
         region_encoding='{"base":"gyroid","rho_soft":0.18,"rho_stiff":0.35,"axis":"thickness"}'),
]

# Shared, as-modeled geometry / planned test condition.
LINER_THICKNESS_MM = 225.0          # tile Z-thickness as modeled (generate_tiles.TILE_Z)
N_REPLICATES = 3                    # spec: 3 replicate prints per design
DATA_SOURCE = "oblique_impact"     # headline test -> yields both HIC and BrIC
IMPACT_ZONE = "front"
IMPACT_SPEED_MPS = 5.0             # oblique range is 4.5-6.0 m/s; pick + log one


def build() -> pd.DataFrame:
    rows = []
    for d in DESIGNS:
        # Derive strut / sa_to_v / connectivity_z from the design parameters,
        # using rel_density_designed as the stand-in (measured stays blank).
        geom = F.derive_geometry({
            "topology": d["topology"],
            "rel_density_measured": d["rel_density_designed"],
            "cell_size_mm": d["cell_size_mm"],
            "liner_thickness_mm": LINER_THICKNESS_MM,
            "gradient_param": d["gradient_param"],
            "shear_anisotropy_ratio": float("nan"),  # not shear-tuned; characterize later
        })
        for rep in range(1, N_REPLICATES + 1):
            row = {c: "" for c in S.ALL_COLUMNS}          # blank everything first
            row.update({
                "design_id": d["design_id"],
                "replicate_id": rep,
                "data_source": DATA_SOURCE,
                "topology": d["topology"],
                "region_encoding": d["region_encoding"],
                "rel_density_designed": d["rel_density_designed"],
                # rel_density_measured INTENTIONALLY BLANK -> from coupon mass/volume
                "cell_size_mm": d["cell_size_mm"],
                "strut_thickness_mm": round(geom["strut_thickness_mm"], 4),
                "liner_thickness_mm": LINER_THICKNESS_MM,
                "gradient_param": d["gradient_param"],
                # shear_anisotropy_ratio blank -> characterize from a shear test
                "sa_to_v": round(geom["sa_to_v"], 4),
                "connectivity_z": geom["connectivity_z"],
                "impact_zone": IMPACT_ZONE,
                "impact_speed_mps": IMPACT_SPEED_MPS,
                "temperature_C": 23.0,
                "raw_file_path": f"raw/oblique/{d['design_id']}_r{rep}.bin",
                "notes": "as-designed geometry; measured density + targets pending",
            })
            rows.append(row)
    return pd.DataFrame(rows, columns=S.ALL_COLUMNS)


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "data" / "designs_D1_D5.csv"
    df = build()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows ({df['design_id'].nunique()} designs x {N_REPLICATES} reps) -> {out}")
    blanks = ["rel_density_measured", "shear_anisotropy_ratio"] + S.TARGET_COLS
    print("Blank (to be filled by testing):", ", ".join(blanks))
    print("\nPre-filled geometry per design:")
    cols = ["design_id", "topology", "rel_density_designed", "cell_size_mm",
            "strut_thickness_mm", "gradient_param", "sa_to_v", "connectivity_z"]
    print(df.drop_duplicates("design_id")[cols].to_string(index=False))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    main()
