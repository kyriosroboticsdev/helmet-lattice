"""
features.py
===========
Physics-informed feature engineering (spec Section 3) and the CANONICAL
design -> feature-matrix builder used by BOTH model training and the optimizer.

Using one builder everywhere prevents train/serve skew: the GP is trained on
exactly the columns the Bayesian optimizer later feeds it.

Engineered features (reduce overfitting on small data):
  * rel_density and rel_density**2   (Gibson-Ashby stiffness/plateau scaling)
  * topology x rel_density           (best density depends on topology)
  * sa_to_v, connectivity_z          (already geometric; passed through)
  * inverse cell size, log SA:V      (mild nonlinear basis)
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd

import schema as S

# Geometry that the optimizer is allowed to vary (the controllable "design vector").
DESIGN_PARAMS = [
    "topology", "rel_density_measured", "cell_size_mm",
    "liner_thickness_mm", "gradient_param", "shear_anisotropy_ratio",
]

# Analytic proxies for CAD-derived geometry. In the real pipeline (Section 6) the
# parametric CAD generator computes these and writes them into the handoff table;
# here we reproduce the same relationships so params -> feature_row is self-contained.
_CONNECTIVITY_Z = {
    "bcc": 8.0, "kelvin": 4.0, "gyroid": 6.0, "octet": 12.0,
    "hybrid": 8.0, "alternating": 7.0, "graded": 6.0,
    "schwartz": 6.0, "diamond": 4.0, "auxetic": 4.0, "voronoi": 5.2, "stochastic": 5.5,
    "re_entrant": 3.0, "honeycomb": 3.0, "elytra": 4.0, "spinodoid": 5.0, "bouligand": 6.0,
}
_SA_FACTOR = {
    "bcc": 3.2, "kelvin": 3.6, "gyroid": 4.4, "octet": 3.8,
    "hybrid": 3.7, "alternating": 3.9, "graded": 4.2,
    "schwartz": 4.0, "diamond": 4.3, "auxetic": 3.0, "voronoi": 4.6, "stochastic": 4.8,
    "re_entrant": 4.2, "honeycomb": 4.2, "elytra": 4.0, "spinodoid": 4.6, "bouligand": 4.2,
}
# Geometric character of each topology used to seed PROPOSED designs (real tested
# rows carry measured values from CAD/mesh). Organic lattices are the irregular ones.
_IRREGULARITY = {
    "bcc": 0.0, "kelvin": 0.0, "gyroid": 0.0, "octet": 0.0, "schwartz": 0.0,
    "diamond": 0.0, "auxetic": 0.0, "graded": 0.10, "alternating": 0.15, "hybrid": 0.15,
    "voronoi": 0.85, "stochastic": 0.95,
    # Periodic auxetic/comb ~0; spinodoid is stochastic; bouligand is ordered plies.
    "re_entrant": 0.0, "honeycomb": 0.0, "elytra": 0.0, "spinodoid": 0.80, "bouligand": 0.10,
}
_STRUT_CV = {
    "bcc": 0.05, "kelvin": 0.05, "gyroid": 0.04, "octet": 0.05, "schwartz": 0.04,
    "diamond": 0.04, "auxetic": 0.05, "graded": 0.12, "alternating": 0.12, "hybrid": 0.13,
    "voronoi": 0.35, "stochastic": 0.42,
    "re_entrant": 0.05, "honeycomb": 0.04, "elytra": 0.06, "spinodoid": 0.30, "bouligand": 0.08,
}


def pore_size(cell_size, rel_density):
    """Mean pore/window size proxy: shrinks as the lattice densifies."""
    rho = np.clip(rel_density, 1e-3, None)
    return cell_size * np.clip(1.0 - np.sqrt(rho), 0.1, None)


# Family basis the GENERATIVE optimizer blends over (mirrors
# generate_tiles.BLEND_BASIS -- all nine registered implicit-field families, not just
# the three TPMS ones). A blended design is a point in this simplex; the pure
# topologies are its corners, and the per-family cap keeps proposals off the corners.
BLEND_BASIS = ["gyroid", "schwartz", "diamond", "re_entrant", "honeycomb",
               "elytra", "voronoi", "spinodoid", "bouligand"]

# Back-compat alias: this used to be the 3-member TPMS basis.
TPMS_BLEND_BASIS = BLEND_BASIS

# Mirrors generate_tiles.MAX_FAMILY_WEIGHT -- duplicated here so the feature layer
# can be imported without the meshing stack.
MAX_FAMILY_WEIGHT = 0.45

# Families whose field is z-invariant (vertical walls, zero overhang). A convex blend
# of these is still z-invariant, hence still support-free -- so a mix drawn only from
# this set needs no printability measurement. Mirrors the support_free flags in
# generate_tiles.FIELD_TOPOLOGIES. spinodoid qualifies only in its DEFAULT lamellar
# mode (mode="isotropic" has overhangs); bouligand never does, because a wall only
# lands on the ply below where the rotated patterns cross. This is a cheap pre-check
# -- realize.py measures the real unsupported-area fraction on the actual mesh.
SUPPORT_FREE_FAMILIES = {"gyroid", "schwartz", "diamond", "re_entrant",
                         "honeycomb", "elytra", "voronoi", "spinodoid"}

# Families whose field repeats exactly with the unit cell, so the tile is a true
# repeating pattern instead of a one-off blob. Mirrors generate_tiles.PERIODIC_FAMILIES
# (voronoi needs irregularity=0, spinodoid needs periodic=True; the mesher applies
# both). bouligand is excluded and cannot be fixed by configuration -- each ply is the
# base pattern rotated by an arbitrary angle, which does not map the cell lattice onto
# itself.
PERIODIC_FAMILIES = {"gyroid", "schwartz", "diamond", "re_entrant", "honeycomb",
                     "elytra", "voronoi", "spinodoid"}


def normalize_blend(weights: dict) -> dict:
    """Clamp negatives and L1-normalize blend weights over the family basis."""
    w = {b: max(0.0, float(weights.get(b, 0.0))) for b in BLEND_BASIS}
    s = sum(w.values()) or 1.0
    return {b: v / s for b, v in w.items() if v > 0}


def topology_to_blend(topo: str) -> dict:
    """Topology label -> basis blend weights. A pure family is its own one-hot corner;
    the TPMS-composed labels (graded/alternating/hybrid) sit on the gyroid-schwartz
    edge; anything still outside the basis (octet/kelvin/bcc/stochastic/auxetic) has
    no implicit-field member and stays all zeros, carried by its one-hot instead."""
    w = {b: 0.0 for b in BLEND_BASIS}
    if topo in w:
        w[topo] = 1.0
    elif topo in ("graded", "alternating", "hybrid"):
        w["gyroid"] = w["schwartz"] = 0.5
    return w


def cell_axes(p: dict) -> tuple:
    """Per-axis cell sizes (cx, cy, cz) from cell_size_mm + optional aspect ratios.
    aspect=1 -> isotropic (cubic) cell, the default."""
    c = float(p["cell_size_mm"])
    ay = float(p.get("cell_aspect_y", 1.0) or 1.0)
    az = float(p.get("cell_aspect_z", 1.0) or 1.0)
    return c, c * ay, c * az


# Lazy, optional handle to the geometry generator so sa_to_v can be MEASURED from the
# real (blended/anisotropic) field instead of a per-topology constant. Falls back to
# the proxy if the meshing stack (trimesh/skimage) isn't importable.
_GEN = None


def _gen():
    global _GEN
    if _GEN is None:
        try:
            import os as _os
            import sys as _sys
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
                _os.path.abspath(__file__))))
            import generate_tiles as g
            _GEN = g
        except Exception:
            _GEN = False
    return _GEN


def measured_sa_to_v(weights: dict, cell_xyz: tuple, rho: float):
    """sa_to_v measured from the actual TPMS/blend field, or None if unavailable."""
    g = _gen()
    if not g:
        return None
    try:
        m = g.measure_tpms_descriptors(weights, cell_xyz, rho)
        return None if m is None else float(m["sa_to_v"])
    except Exception:
        return None


def strut_thickness(cell_size, rel_density):
    """Beam-lattice scaling: rel_density ~ (t/cell)^2  =>  t = cell*sqrt(rho)*k."""
    return cell_size * np.sqrt(np.clip(rel_density, 1e-3, None)) * 0.60


def derive_geometry(params: dict) -> dict:
    """Complete a design vector into a full geometry row (Section 6 interface).

    Input: a dict with at least DESIGN_PARAMS. Output: same dict plus the
    CAD-derived fields (strut_thickness_mm, sa_to_v, connectivity_z), so the
    optimizer's proposed `params` becomes a contract-shaped feature row.
    """
    p = dict(params)

    # GENERATIVE path: a blend over the TPMS basis -> a new in-between geometry.
    # CAD-derived descriptors are the weighted blend of the basis values, so the GP
    # interpolates between the pure-topology corners it was trained on.
    cx, cy, cz = cell_axes(p)
    cmean = (cx * cy * cz) ** (1.0 / 3.0)            # geometric-mean cell for proxies
    rho = p["rel_density_measured"]

    blend = p.get("blend")
    if blend:
        bw = {b: max(0.0, float(blend.get(b, 0.0))) for b in TPMS_BLEND_BASIS}
        s = sum(bw.values()) or 1.0
        bw = {b: v / s for b, v in bw.items()}
        for b in TPMS_BLEND_BASIS:
            p[f"blend_{b}"] = bw[b]
        p["topology"] = max(bw, key=bw.get)        # dominant corner (for the one-hot)
        p["strut_thickness_mm"] = float(strut_thickness(cmean, rho))
        # sa_to_v MEASURED from the real blended (anisotropic) field; proxy fallback.
        sav = measured_sa_to_v(bw, (cx, cy, cz), rho)
        p["sa_to_v"] = float(sav) if sav is not None else \
            float(sum(bw[b] * _SA_FACTOR[b] for b in bw) / cmean)
        p["connectivity_z"] = float(sum(bw[b] * _CONNECTIVITY_Z[b] for b in bw))
        p["irregularity"] = float(sum(bw[b] * _IRREGULARITY[b] for b in bw))
        p["strut_thickness_cv"] = float(sum(bw[b] * _STRUT_CV[b] for b in bw))
        p["pore_size_mm"] = float(pore_size(cmean, rho))
        p["region_encoding"] = "uniform"
        return p

    topo = p["topology"]
    p["strut_thickness_mm"] = float(strut_thickness(cmean, rho))
    # MEASURE sa_to_v for the field-based TPMS topologies; proxy for strut/organic.
    sav = measured_sa_to_v({topo: 1.0}, (cx, cy, cz), rho) \
        if topo in TPMS_BLEND_BASIS else None
    p["sa_to_v"] = float(sav) if sav is not None else float(_SA_FACTOR[topo] / cmean)
    p["connectivity_z"] = float(_CONNECTIVITY_Z[topo])
    # Organic / irregular descriptors (so proposals score on geometry, not category).
    p["irregularity"] = float(_IRREGULARITY.get(topo, 0.0))
    p["strut_thickness_cv"] = float(_STRUT_CV.get(topo, 0.05))
    p["pore_size_mm"] = float(pore_size(cmean, rho))
    # Layered topologies need a region encoding so the region features fire; the
    # optimizer only proposes a topology, so default alternating/hybrid to a 50/50
    # gyroid+schwartz stack unless the caller supplied one.
    if not p.get("region_encoding"):
        if topo in ("alternating", "hybrid"):
            p["region_encoding"] = ('{"layers":[{"topology":"gyroid","frac":0.5},'
                                    '{"topology":"schwartz","frac":0.5}]}')
        else:
            p["region_encoding"] = "uniform"
    return p


def _region_stats(val) -> tuple[float, float, float]:
    """Parse one region_encoding cell -> (n_regions, n_interfaces, heterogeneity).

    Accepts "" / "uniform" (single region), a named-region dict like
    {"core":"gyroid","shell":"octet"}, or a layered form
    {"layers":[{"topology":"gyroid","frac":0.5}, ...]}. heterogeneity is the
    fraction NOT in the dominant region (0 = uniform).
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 1.0, 0.0, 0.0
    s = str(val).strip()
    if s == "" or s.lower() == "uniform":
        return 1.0, 0.0, 0.0
    try:
        obj = json.loads(s)
    except Exception:
        return 1.0, 0.0, 0.0
    if isinstance(obj, dict) and isinstance(obj.get("layers"), list) and obj["layers"]:
        layers = obj["layers"]
        n = float(len(layers))
        fracs = [float(l.get("frac", 1.0 / len(layers))) for l in layers]
        het = 1.0 - max(fracs) if fracs else 0.0
        return n, n - 1.0, float(np.clip(het, 0.0, 1.0))
    if isinstance(obj, dict) and obj:
        topos = [v for v in obj.values() if isinstance(v, str)]
        n = float(max(1, len(topos)))
        het = 0.0 if n <= 1 else 1.0 - 1.0 / n     # assume even split when unspecified
        return n, max(0.0, n - 1.0), het
    return 1.0, 0.0, 0.0


def region_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features describing multi-region/layered designs (region_encoding)."""
    enc = (df["region_encoding"] if "region_encoding" in df.columns
           else pd.Series([None] * len(df), index=df.index))
    rows = [_region_stats(v) for v in enc]
    return pd.DataFrame(rows, index=df.index,
                        columns=["n_regions", "n_interfaces", "heterogeneity"]).astype(float)


def _onehot_topology(topo: pd.Series) -> pd.DataFrame:
    """One-hot topology with ALL contract categories present (stable columns)."""
    cat = pd.Categorical(topo, categories=S.TOPOLOGIES)
    oh = pd.get_dummies(cat, prefix="topo")
    oh.index = topo.index
    return oh.astype(float)


def build_features(df: pd.DataFrame, include_conditions: bool = True,
                   drop_sparse_topology: bool = False):
    """Build the numeric design matrix from a contract-shaped DataFrame.

    Returns (X, feature_names). Columns are deterministic and ordered so a
    single proposed design produces a row with identical columns to training.

    drop_sparse_topology=True omits the topology one-hot + topology×rho
    interaction columns (24 mostly-zero columns), keeping only the dense
    geometric/physics features and the continuous TPMS blend weights. Use this
    for the GP: on small data a Gaussian process with an ARD kernel must fit one
    lengthscale per feature, and the sparse categorical expansion drowns the
    real geometric signal (it ~tripled the HIC surrogate's R²). The tree
    ensembles in importance.py keep the full set, where the one-hots are useful.
    """
    df = df.copy()

    cols: dict[str, pd.Series] = {}
    for c in S.CONTINUOUS_GEOM:
        if c in df.columns:
            cols[c] = pd.to_numeric(df[c], errors="coerce")

    # Physics-informed engineered terms
    rho = pd.to_numeric(df["rel_density_measured"], errors="coerce")
    cols["rel_density_sq"] = rho ** 2
    cols["inv_cell_size"] = 1.0 / pd.to_numeric(df["cell_size_mm"], errors="coerce")
    cols["log_sa_to_v"] = np.log(pd.to_numeric(df["sa_to_v"], errors="coerce").clip(lower=1e-3))

    if include_conditions:
        for c in S.NUMERIC_CONDITIONS:
            if c in df.columns:
                cols[c] = pd.to_numeric(df[c], errors="coerce")

    X = pd.DataFrame(cols, index=df.index)

    # Region/layer descriptors (alternating, hybrid, graded) — geometry, not category.
    X = pd.concat([X, region_features(df)], axis=1)

    # TPMS blend weights (continuous topology axis). Use explicit blend_* columns if
    # present (generative proposals); otherwise derive from the topology label so a
    # pure topology sits at its simplex corner.
    basis_cols = [f"blend_{b}" for b in TPMS_BLEND_BASIS]
    if all(c in df.columns for c in basis_cols):
        blendX = df[basis_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        blendX = pd.DataFrame([topology_to_blend(t) for t in df["topology"]],
                              index=df.index)
        blendX.columns = basis_cols
    X = pd.concat([X, blendX.astype(float)], axis=1)

    # Topology one-hot + topology x rel_density interactions. With the geometric
    # descriptors above carrying most of the cross-topology signal, the one-hot is a
    # secondary correction and (critically) is NOT required for an unseen/organic
    # lattice to be placed — its descriptors do that. Skipped for the GP
    # (drop_sparse_topology) because these 24 sparse columns hurt an ARD GP on
    # small data; the dense descriptors + blend weights above still carry topology.
    if not drop_sparse_topology:
        topo_oh = _onehot_topology(df["topology"])
        inter = topo_oh.mul(rho, axis=0)
        inter.columns = [f"{c}_x_rho" for c in topo_oh.columns]
        X = pd.concat([X, topo_oh, inter], axis=1)

    # Drop columns that are entirely NaN in this subset (e.g. impact_speed for QS rows)
    X = X.dropna(axis=1, how="all")
    feature_names = list(X.columns)
    return X, feature_names


def design_to_feature_row(params: dict, conditions: dict | None = None,
                          template_columns: list[str] | None = None) -> pd.DataFrame:
    """Turn one optimizer-proposed design into a single feature row.

    params      : a DESIGN_PARAMS dict (topology + controllable geometry).
    conditions  : fixed test conditions (e.g. {'impact_speed_mps': 5.0}).
    template_columns : if given, the row is reindexed to exactly these columns
                       (the training feature names) so the GP can score it.
    """
    geom = derive_geometry(params)
    if conditions:
        geom.update(conditions)
    row = pd.DataFrame([geom])
    # Fill any contract columns the builder reads but the design dict omits.
    for c in S.CONTINUOUS_GEOM + S.NUMERIC_CONDITIONS:
        if c not in row.columns:
            row[c] = np.nan
    X, _ = build_features(row, include_conditions=bool(conditions))
    if template_columns is not None:
        X = X.reindex(columns=template_columns, fill_value=0.0)
    return X


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from pathlib import Path
    import data as D
    root = Path(__file__).resolve().parents[2]
    df = D.load_master(root / "data" / "raw" / "dummy_master.csv")
    tv = D.training_view(df, data_sources=["oblique_impact"])
    X, names = build_features(tv)
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print("Features:", names)
