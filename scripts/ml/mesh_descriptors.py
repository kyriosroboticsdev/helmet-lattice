"""
mesh_descriptors.py
===================
Turn an organic / arbitrary lattice MESH (STL/OBJ/3MF) into the geometric
descriptor columns of the Data Handoff Contract, so teammates' organic-lattice CAD
(Voronoi, stochastic foam, bone-like, ...) can be ingested the same way as the
parametric topologies.

The whole point: the surrogate models lattices by GEOMETRY, not by category — so an
organic lattice it has never seen is still placeable as long as these descriptors
are filled. This script computes the ones that are robust from a mesh and leaves the
rest to be supplied from CAD.

Computed from the mesh:
  * rel_density_measured  = mesh_volume / bounding-box_volume
  * sa_to_v               = surface_area / mesh_volume          (1/mm)
  * strut_thickness_mm    ~ 2 * volume / area   (thin-wall/strut proxy)

Supplied by you (CAD-known or estimated): topology, cell_size_mm, irregularity,
strut_thickness_cv, pore_size_mm, connectivity_z, liner_thickness_mm, ...

Usage
-----
    python scripts/ml/mesh_descriptors.py part.stl --topology voronoi --cell 8
    python scripts/ml/mesh_descriptors.py part.stl --topology stochastic --cell 7 \
        --irregularity 0.9 --append data/raw/master.csv
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema as S          # noqa: E402
import features as F        # noqa: E402


def mesh_descriptors(path: str) -> dict:
    """Robust geometric descriptors computed directly from a mesh file."""
    mesh = trimesh.load(path, force="mesh")
    ext = mesh.bounding_box.extents                  # (dx, dy, dz) mm
    bbox_vol = float(np.prod(ext))
    vol = float(abs(mesh.volume))
    area = float(mesh.area)
    rel_density = vol / bbox_vol if bbox_vol > 0 else float("nan")
    sa_to_v = area / vol if vol > 0 else float("nan")
    strut = 2.0 * vol / area if area > 0 else float("nan")   # thin-feature thickness proxy
    return {
        "rel_density_measured": round(rel_density, 4),
        "sa_to_v": round(sa_to_v, 4),
        "strut_thickness_mm": round(strut, 4),
        "_bbox_mm": tuple(round(float(e), 2) for e in ext),
        "_watertight": bool(mesh.is_watertight),
    }


def build_row(path: str, topology: str, cell: float, **overrides) -> dict:
    """A contract-shaped geometry row for one organic part. Mesh-derived values win
    where computable; topology defaults fill descriptors you don't override."""
    d = mesh_descriptors(path)
    rho = d["rel_density_measured"]
    row = {c: np.nan for c in S.ALL_COLUMNS}
    row.update({
        "topology": topology,
        "region_encoding": "uniform",
        "cell_size_mm": float(cell),
        "rel_density_measured": rho,
        "sa_to_v": d["sa_to_v"],
        "strut_thickness_mm": d["strut_thickness_mm"],
        # Topology-default descriptors (override from CAD when you know better).
        "connectivity_z": F._CONNECTIVITY_Z.get(topology, 5.5),
        "irregularity": F._IRREGULARITY.get(topology, 0.85),
        "strut_thickness_cv": F._STRUT_CV.get(topology, 0.35),
        "pore_size_mm": round(float(F.pore_size(cell, rho)), 4),
    })
    row.update({k: v for k, v in overrides.items() if v is not None})
    return row, d


def main() -> int:
    ap = argparse.ArgumentParser(description="Mesh -> handoff geometric descriptors.")
    ap.add_argument("mesh", help="path to STL/OBJ/3MF")
    ap.add_argument("--topology", required=True, choices=S.TOPOLOGIES)
    ap.add_argument("--cell", type=float, required=True, help="nominal cell/pore pitch (mm)")
    ap.add_argument("--irregularity", type=float, default=None)
    ap.add_argument("--strut-cv", type=float, default=None, dest="strut_thickness_cv")
    ap.add_argument("--pore", type=float, default=None, dest="pore_size_mm")
    ap.add_argument("--connectivity", type=float, default=None, dest="connectivity_z")
    ap.add_argument("--liner", type=float, default=None, dest="liner_thickness_mm")
    ap.add_argument("--append", default=None,
                    help="append this row to an existing master CSV (created if absent)")
    a = ap.parse_args()

    row, d = build_row(
        a.mesh, a.topology, a.cell,
        irregularity=a.irregularity, strut_thickness_cv=a.strut_thickness_cv,
        pore_size_mm=a.pore_size_mm, connectivity_z=a.connectivity_z,
        liner_thickness_mm=a.liner_thickness_mm,
    )

    print(f"mesh: {a.mesh}")
    print(f"  bbox(mm)={d['_bbox_mm']}  watertight={d['_watertight']}")
    if not d["_watertight"]:
        print("  WARNING: mesh not watertight -> volume/rel_density may be off; "
              "repair the mesh or supply rel_density from CAD.")
    print("  descriptor row:")
    for k in ("topology", "cell_size_mm", "rel_density_measured", "sa_to_v",
              "strut_thickness_mm", "pore_size_mm", "connectivity_z",
              "irregularity", "strut_thickness_cv"):
        print(f"    {k:22s} = {row[k]}")

    if a.append:
        path = Path(a.append)
        df_new = pd.DataFrame([row]).reindex(columns=S.ALL_COLUMNS)
        if path.exists():
            df = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            df = df_new
        df.to_csv(path, index=False)
        print(f"  appended -> {path}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
