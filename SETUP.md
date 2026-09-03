# Helmet Lattice Project — Setup & Usage

## Folder Structure

```
helmet-lattice/
├── SETUP.md                        ← start here
├── scripts/
│   ├── generate_tiles.py           ← generates STL files
│   ├── surrogate.py                ← ML model, run after VT testing
│   └── auxetic_instructions.txt   ← manual steps for D3 in FreeCAD
├── stl_output/                     ← STL files appear here
├── vt_results/
│   └── results.csv                 ← fill in after each VT trip
└── prints_log/
    └── print_checklist.md          ← fill out for every print
```

## One-Time Python Setup

Open Command Prompt and run:

```
pip install numpy scikit-image trimesh scipy torch gpytorch botorch pymoo matplotlib pandas
```

## Step 1 — Generate STL Files

```
cd C:\Users\Kyrio\helmet-lattice
python scripts/generate_tiles.py
```

STL files appear in stl_output/. Open in Creality Print to verify before slicing.
D3 (auxetic) requires FreeCAD — see scripts/auxetic_instructions.txt.

## Step 2 — Slice Settings (Creality Print / Cura)

- Nozzle temp:  235°C
- Bed temp:     40°C
- Print speed:  30 mm/s
- Layer height: 0.2 mm
- Walls:        3 perimeters
- Top/bottom:   0 solid layers
- Infill:       0% (lattice is the structure)
- Supports:     Off (gyroid/schwartz). Tree supports for D3 (auxetic) only.

## Step 3 — QC Each Print

Fill out prints_log/print_checklist.md for every tile before going to VT.

## Step 4 — VT Testing

Ask VT for: peak linear G, HIC15, peak angular velocity (rad/s), BrIC.
Enter results into vt_results/results.csv.

## Step 5 — Run the ML pipeline

The full ML workflow (Data Handoff Contract, feature importance, GP surrogate,
multi-objective Bayesian optimization + active-learning loop) lives in
`scripts/ml/` and is driven by:

```
python scripts/run_pipeline.py                 # runs on dummy data (data/raw/dummy_master.csv)
python scripts/run_pipeline.py --data PATH      # point at the real handoff CSV
```

### Generative mode — cross-family blended designs

`--generative` stops picking one topology label and instead searches a continuous
simplex over all nine implicit-field families, so every proposal is a NEW geometry
that mixes them:

```
python scripts/run_pipeline.py --generative --q 4
```

The rules the search obeys:

- **No family above 45%** (`--max-family`). Because weights must sum to 1, this
  ceiling forces at least `ceil(1/0.45) = 3` families into every design — "never
  pure" is a consequence of the cap, not a separate rule. Weights are projected onto
  the capped simplex by water-filling, so the cap is exact, not approximate.
- **3–5 families per design** (`--families MIN MAX`). MIN is raised automatically if
  it is below what the cap allows. Drawing a random SUBSET rather than weighting all
  nine matters: weight on every family in every design makes each tile a blur of all
  nine and makes them all resemble each other.
- **Proposals must look different from each other** (`--min-distance`, default 2.5).
  Enforced as a minimum separation in *descriptor* space (sa_to_v, connectivity_z,
  shear anisotropy, pore size, irregularity), standardised by the candidate pool's
  own spread. Descriptor space rather than weight space because two different weight
  vectors can still mesh into the same-looking tile.
- **Every design is a repeating unit cell.** A coupon has to be a pattern, not a
  one-off blob, so the search is restricted to families whose implicit field repeats
  exactly with the cell. Six do natively; `voronoi` does only at `irregularity=0`
  (jittered seeds are what break it) and `spinodoid` only with `periodic=True`, which
  snaps its wave vectors onto the reciprocal lattice — the mesher applies both
  automatically. **`bouligand` is excluded and cannot be fixed by configuration**:
  each ply is the base pattern rotated by an arbitrary angle about the tile centre,
  which never maps the cell lattice onto itself. A blend is periodic only if *every*
  member is. Pass `--allow-aperiodic` to lift the restriction.
- **Every design is one connected piece** (`realize`, `n_bodies > 1` is rejected).
  `_drop_islands` only removes fragments under 1% of the volume, so a tile that
  genuinely splits in two survives it — and a two-piece coupon is not testable
  whatever its predicted metrics.
- **Every design fills the cube evenly** (`realize.MIN_OCTANT_FILL`, 0.55). The tile
  is split into octants and the emptiest is compared against the mean; a design can
  hit its target *average* density while leaving a whole region bare, and average
  density alone cannot see that.
- **Printability is gated on the mesh** (`realize.MAX_UNSUPPORTED_FRAC`, 5%). The
  optimizer can't know whether a blend will print, since unsupported area is a
  property of the mesh. So a shortlist of `3x --q` is ranked and meshed until enough
  pass. A blend drawn only from z-invariant families is support-free *by
  construction* (a weighted sum of z-invariant fields is still z-invariant) and skips
  the check; mixes containing bouligand are measured.

Blending across families only works because every field is first **rank-normalised**
onto a common [0,1] scale. The TPMS members are trig sums of magnitude ~1 while the
wall members are distance fields in mm — summing them raw would let the larger-range
field set the isosurface regardless of its weight, and a nominal 45/30/25 mix would
come out looking ~95% like one family.

Cost note: measuring sa_to_v from the real blended field runs ~50 ms per candidate,
so a 2000-candidate generative pool adds ~100 s over discrete mode.

Outputs land in `outputs/`:
- `outputs/importance/` — ranked geometric drivers per metric (SHAP/permutation)
- `outputs/pareto_front.png` — linear-vs-rotational Pareto front + next design to print
- `outputs/pipeline_report.md` — summary of every stage

See `scripts/ml/README.md` for module-by-module detail and how each piece maps to
`ML_Computational_Workflow_TEAM.md`. The contract schema is defined once in
`scripts/ml/schema.py` — **lock it with the testing team before testing starts.**

> The older `scripts/surrogate.py` is the original 3-feature prototype, superseded
> by the `scripts/ml/` pipeline. Kept for reference.

## Repeat Steps 1–5 each design iteration.
