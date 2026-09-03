# Helmet Lattice

**ML-optimized 3D-printed TPU lattice liners for football helmets, targeting both linear and rotational concussion metrics.**

This repository holds a generative lattice modeler and a machine-learning design
loop. The modeler builds watertight, printable, single-piece lattice coupons from
implicit fields; the ML loop learns which geometric features drive impact
performance and proposes the next design to print and test. Validation is against
the Virginia Tech STAR protocol.

---

## The one-paragraph version

Nine implicit-field lattice families — three TPMS, two auxetic, four
bio-inspired — are blended on a continuous simplex to produce *new* geometries
rather than picking one family off a list. A Gaussian-process surrogate learns
impact metrics from geometry, and multi-objective Bayesian optimization (qEHVI)
proposes designs on the linear-vs-rotational Pareto front. Every proposal is
meshed and mechanically gated before it is allowed out: it must be a repeating
unit cell, one connected piece, evenly filling, and printable without supports.

---

## Study design — read this first

This is a **geometry-only study**. One TPU, one durometer, one printer, one slicer
profile, held constant across every specimen, so that performance differences are
attributable to *lattice architecture alone*.

That constraint is doing real work, and it is easy to break by accident. Anything
that varies between designs and is not the architecture — bond quality, mounting
hardware, a hole drilled through the lattice, a different infill setting — becomes
a confound that the surrogate will silently attribute to geometry. If you are
about to introduce a per-design difference, raise it before you do.

**Current statistical caveat:** the surrogate has **23 impact designs** behind it.
The blend space adds 9 `blend_*` feature columns on top of the existing geometric
features, so the space is under-determined. Treat surrogate predictions as a
ranking heuristic for what to print next, not as a result.

---

## Quickstart

```bash
git clone <this repo>
cd helmet-lattice
pip install -r requirements.txt

# 1. Generate the baseline tile set (writes ~2.1 GB into stl_output/)
python scripts/generate_tiles.py

# 2. Run the ML pipeline on dummy data
python scripts/run_pipeline.py

# 3. Generative mode — cross-family blended proposals
python scripts/run_pipeline.py --generative --q 4
```

Outputs land in `outputs/`. A committed example run lives in
`docs/example_output/` if you want to see the results without running anything.

`SETUP.md` has the full step-by-step, including slicer settings, print QC, and
what to ask VT for after a test session.

---

## Repository map

```
helmet-lattice/
├── README.md                    ← you are here
├── SETUP.md                     ← step-by-step operating procedure
├── DATA.md                      ← data provenance, what is and isn't committed
├── ML_Computational_Workflow_TEAM.md   ← the workflow the ML team follows
├── requirements.txt
│
├── scripts/
│   ├── generate_tiles.py        ← the generative lattice modeler (the core artifact)
│   ├── run_pipeline.py          ← driver: data → importance → surrogate → optimize → STL
│   ├── make_stl.py              ← single-tile CLI, used by the Fusion add-in
│   ├── make_dogbone.py          ← ASTM D412 Die C tensile specimen generator
│   ├── make_material_specimens.py   ← planar-shear and biaxial specimens
│   ├── make_print_plates.py     ← plate layout for batch printing
│   ├── make_combined_bed.py     ← multi-specimen bed packing
│   ├── analyze_dataset.py       ← analysis of the third-party impact dataset
│   ├── surrogate.py             ← superseded 3-feature prototype, kept for reference
│   └── ml/
│       ├── schema.py            ← ★ the Data Handoff Contract. Lock before testing.
│       ├── data.py              ← load + validate the handoff CSV
│       ├── features.py          ← geometric feature engineering
│       ├── importance.py        ← RF / XGBoost + SHAP, grouped CV by design_id
│       ├── surrogate_gp.py      ← BoTorch GP with a Gibson–Ashby physics mean
│       ├── optimize.py          ← qEHVI multi-objective BO + the blending rules
│       ├── realize.py           ← proposals → meshes, and the validity gates
│       ├── mesh_descriptors.py  ← descriptor measurement from real meshes
│       ├── seed_designs.py      ← initial design-of-experiments
│       └── make_dummy_data.py   ← synthetic data so the pipeline runs pre-testing
│
├── fusion_addin/HelmetLattice/  ← Fusion 360 add-in for batch design review
├── data/                        ← handoff template + as-designed parameters
├── vt_results/                  ← impact results from Virginia Tech
├── docs/                        ← testing guide + committed example output
└── prints_log/                  ← per-print QC checklist
```

---

## The lattice families

`scripts/generate_tiles.py` builds every tile as an implicit field, meshed by
marching cubes, repaired with trimesh, and written as a watertight STL. Nine
families form the blend basis (`FIELD_TOPOLOGIES`):

| Family | Kind | Support-free | Cell-periodic |
|---|---|---|---|
| `gyroid` | TPMS | yes | yes |
| `schwartz` | TPMS | yes | yes |
| `diamond` | TPMS | yes | yes |
| `re_entrant` | auxetic (bowtie honeycomb) | yes | yes |
| `honeycomb` | same field, sign flipped | yes | yes |
| `elytra` | bio — beetle forewing | yes | yes |
| `voronoi` | bio — trabecular bone / open-cell foam | yes | only at `irregularity=0` |
| `spinodoid` | bio — Gaussian random field | yes (`lamellar`) | only with `periodic=True` |
| `bouligand` | bio — mantis shrimp club, targets rotation | **no** | **never** |

Tile geometry is 50.8 × 50.8 × 38.1 mm (2 × 2 × 1.5 in). Quasi-static compression
coupons are separate 40 mm cubes.

### Gotchas worth knowing before you change anything

- **Re-entrant sits at `rho=0.35`, not 0.25.** At 0.25 a wall lattice this open
  cannot have both printable ≥1.2 mm walls *and* enough cells across a 2 in tile
  — it would need a ~20 mm cell, giving only 2 cells. At 0.35 you get 3–4. Density
  is an ML knob, so mixed density across families is allowed and expected.
- **Spinodoid `columnar` fragments below 2D percolation** at `rho≈0.25`, so the
  default mode is `lamellar` (single-body, support-free). `form="sheet"`
  percolates at any density; `form="solid"` only near `rho=0.5`.
- **Density correction is not optional.** For thin walls the quantile point
  fraction is not the meshed volume fraction — `re_entrant` asked for 0.25 and
  came out at 0.133 before correction. `_mesh_to_density()` closes the loop, and
  a blend's density↔isovalue curve is its own, not predictable from its members'.
- **`bouligand` is excluded from generative mode** and cannot be fixed by
  configuration. Each ply is the base pattern rotated by an arbitrary angle about
  the tile centre, which never maps the cell lattice onto itself. Pass
  `--allow-aperiodic` if you deliberately want it.

---

## Generative blending — the rules and why they exist

The goal is designs that *don't look like one another*, built by combining
multiple families. Full detail is in `SETUP.md`; the short version:

- **No family above 45%.** Since weights sum to 1, this ceiling forces at least
  `ceil(1/0.45) = 3` families into every design — "never pure" is a *consequence*
  of the cap, not a separate rule. Weights are projected onto the capped simplex
  by water-filling, redistributing excess proportional to remaining headroom, so
  the cap is exact rather than approximate.
- **3–5 families per design, drawn as a random subset.** Weighting all nine in
  every design makes each tile a blur of all nine, and then they all resemble each
  other — which defeats the entire purpose.
- **Fields are rank-normalised to uniform [0,1] before summing.** This is the
  load-bearing detail. TPMS members are trig sums of magnitude ~1; the wall
  members are distance fields *in mm* whose magnitude scales with cell size. Sum
  them raw and the larger-range field sets the isosurface regardless of its
  weight — a nominal 45/30/25 mix comes out looking ~95% like one family, and the
  cap is enforced on paper while meaning nothing in the mesh.
- **Diversity is enforced in descriptor space, not weight space** (`sa_to_v`,
  `connectivity_z`, shear anisotropy, pore size, irregularity), because two
  different weight vectors can mesh into the same-looking tile.

### Validity gates — a proposal is rejected unless it passes all four

These came from reviewing ten designs in Fusion and rejecting them by hand. Every
rejection reason is now enforced in code.

1. **Repeating unit cell.** A coupon must be a pattern, not a one-off blob.
2. **One connected piece.** `n_bodies > 1` is rejected. `_drop_islands` only
   removes fragments under 1% of volume, so a genuinely split tile survives it —
   and a two-piece coupon is not testable whatever its predicted metrics.
3. **Fills the cube evenly.** `MIN_OCTANT_FILL = 0.55`, emptiest octant against
   the mean. A design can hit its target *average* density while leaving a whole
   region bare, and average density alone cannot see that.
4. **Printable.** `MAX_UNSUPPORTED_FRAC = 0.05`, measured on the mesh. A blend
   drawn only from z-invariant families is support-free *by construction* — a
   weighted sum of z-invariant fields is still z-invariant — and skips the check.

A shortlist of `3 × --q` is ranked and meshed until enough pass, so rejects have
replacements.

---

## Adding a lattice family

The per-topology lookup tables are indexed directly, so a missing key raises
`KeyError` rather than degrading gracefully. All four of these must be updated
together:

1. `scripts/generate_tiles.py` — the field function, plus `FIELD_TOPOLOGIES`, and
   `PERIODIC_FAMILIES` / `PERIODIC_FIELD_KW` if it is cell-periodic.
2. `scripts/ml/schema.py` — add the label to `TOPOLOGIES`.
3. `scripts/ml/features.py` — `_CONNECTIVITY_Z`, `_SA_FACTOR`, `_IRREGULARITY`,
   `_STRUT_CV`.
4. `scripts/ml/make_dummy_data.py` — the same four, plus `TOPO_LIN` and
   `TOPO_ROT`.

`optimize.py` samples proposals straight from `schema.TOPOLOGIES`, so once the
label is registered it becomes proposable automatically.

---

## Fusion 360 add-in

`fusion_addin/HelmetLattice/` adds a **Helmet Lattice Batch** command that runs
the generative pipeline and saves each proposed design as its own Fusion document
with the mesh imported and review facts in the description.

It shells out to a system Python rather than running in-process, and this is not
a shortcut: Fusion's bundled interpreter has no numpy/skimage/trimesh, the ML
stack is worse, and the sandboxed interpreter is replaced on every Fusion update.

**Setup:** copy `local_config.example.py` to `local_config.py` and set `REPO_DIR`
and `PYTHON_EXE` for your machine. `local_config.py` is gitignored. You can use
`HELMET_LATTICE_REPO_DIR` / `HELMET_LATTICE_PYTHON_EXE` environment variables
instead.

> ⚠️ **The add-in is untested.** It was written without a Fusion install
> available. Expect to debug it on first run. `fusion_addin/README.md` has the
> install steps.

---

## Status

- ✅ Generator, blending, validity gates — implemented and verified on a batch of 10
- ✅ ML pipeline — runs end to end on synthetic data
- ⚠️ Surrogate — 23 impact designs, under-determined; a ranking heuristic only
- ⚠️ Fusion add-in — written, never executed
- 🔜 Material characterization — Ogden fit from tension / planar shear / biaxial on an MTS 880
- 🔜 Helmet integration — pad-and-carrier mounting into a production shell

---

## For co-researchers

Start with `SETUP.md`, then `DATA.md`, then `scripts/ml/README.md` for
module-by-module detail. `ML_Computational_Workflow_TEAM.md` describes how the
testing and ML teams hand off to each other.

The single most important shared object is `scripts/ml/schema.py`. **Lock it
before testing starts** — renaming a column after data collection begins
invalidates every row already gathered.
