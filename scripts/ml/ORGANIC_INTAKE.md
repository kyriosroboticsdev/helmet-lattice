# Adding organic lattice data to the pipeline

For teammates contributing **organic / irregular lattices** (Voronoi, stochastic
foam, bone-like, etc.). The model is *geometry-only*: it scores lattices by their
geometric descriptors, **not** by the topology name. That's deliberate — it's the
only way a lattice the model hasn't seen many times (or at all) can be placed. So
the job when adding organic data is to **fill the descriptors**, not to invent a
category the model has to learn from scratch.

## 1. Schema additions (already in `schema.py`)

New categories: `voronoi`, `stochastic` (in `TOPOLOGIES`).

New descriptor columns (in `GEOMETRY_COLS` and `CONTINUOUS_GEOM`):

| Column | Meaning | Periodic lattices | Organic lattices |
|--------|---------|-------------------|------------------|
| `irregularity` | 0 = perfectly periodic, 1 = fully stochastic | ~0.0 | ~0.85–0.95 |
| `strut_thickness_cv` | coeff. of variation of strut/wall thickness | ~0.05 | ~0.35–0.45 |
| `pore_size_mm` | mean pore/window size | from cell+density | measured from CAD/mesh |

> **Lock the schema with the testing team before they log data.** `schema.py` is the
> single source of truth for both teams; adding columns/categories must be agreed so
> there's no train/serve skew.

## 2. Turning a mesh into a data row

Use `mesh_descriptors.py` to compute the robust descriptors straight from an STL:

```bash
python scripts/ml/mesh_descriptors.py part.stl --topology voronoi --cell 8
# append directly to a master table:
python scripts/ml/mesh_descriptors.py part.stl --topology stochastic --cell 7 \
    --irregularity 0.9 --append data/raw/master.csv
```

Computed from the mesh: `rel_density_measured` (volume / bounding box), `sa_to_v`
(area / volume), `strut_thickness_mm` (≈ 2·V/A). Supply the rest from CAD where you
know better: `cell_size_mm`, `irregularity`, `strut_thickness_cv`, `pore_size_mm`,
`connectivity_z`, `liner_thickness_mm`, plus the **measured test targets**
(`HIC`, `BrIC`, `SEA_J_per_kg`, …) once the part is tested.

If the mesh isn't watertight the volume (hence density) is unreliable — repair it or
enter `rel_density_measured` from CAD.

## 3. What the model does with it

`features.build_features` already consumes every `CONTINUOUS_GEOM` column plus the
`region_encoding` features, so new descriptor columns flow in automatically. The
one-hot topology is kept only as a **secondary correction** — an organic lattice is
positioned by its descriptors, so it doesn't need many same-category examples to be
useful. The optimizer (`optimize.sample_candidates`) enumerates all topologies
including `voronoi`/`stochastic`; organic proposals are scored via descriptors.

## 4. STL generation caveat

`generate_tiles.py` can auto-build TPMS (`gyroid`, `schwartz`, `graded`,
`alternating`) and the perforated `auxetic`. Organic lattices are **not** generated
here — they come from your CAD/Voronoi tooling. The pipeline *consumes* their meshes
(via `mesh_descriptors.py`); it does not synthesize them. Strut lattices
(`bcc/kelvin/octet/...`) likewise come from FreeCAD/nTop.
