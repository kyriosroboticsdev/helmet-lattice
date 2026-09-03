# Helmet Lattice — Fusion 360 add-in

Generates a TPU helmet-liner lattice tile (gyroid / schwartz / graded / auxetic)
and imports it into the active Fusion design as a **mesh body**.

## Why it shells out

Fusion's bundled Python has no `numpy` / `scikit-image` / `trimesh`, and installing
them there breaks on Fusion updates. So the add-in is only the dialog + import; the
actual marching-cubes work runs in your **system Python** via
`scripts/make_stl.py`. You can run that CLI standalone too:

```
python scripts/make_stl.py --topology gyroid --cell 8 --density 0.27 --out tile.stl
```

## Install

1. Open Fusion 360 → **Utilities** tab → **ADD-INS** (Scripts and Add-Ins, or `Shift+S`).
2. **Add-Ins** tab → green **+** → "Add an existing add-in" → select the folder
   `fusion_addin/HelmetLattice` (the folder containing `HelmetLattice.manifest`).
   - Alternatively, copy/symlink that folder into Fusion's add-ins directory:
     `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\`
3. Select **HelmetLattice** → **Run** (tick *Run on Startup* to auto-load).

## Configure (one-time)

Edit the two paths at the top of `HelmetLattice/HelmetLattice.py` if your setup
differs from this machine:

```python
REPO_DIR   = r"C:\Users\Kyrio\helmet-lattice"
PYTHON_EXE = r"C:\Users\Kyrio\AppData\Local\Programs\Python\Python313\python.exe"
```

## Where the buttons are

It's an **add-in**, not a script — in `Shift+S` it lives on the **Add-Ins** tab.

Both commands sit in the **Design** workspace → **SOLID** tab → **CREATE** panel:

- **Helmet Lattice Tile** — one design, from the dialog or the latest proposal
- **Helmet Lattice Batch** — N designs, each saved as its own document

They are promoted, so they appear on the CREATE panel directly. If you don't see
them, check you're in the Design workspace on the SOLID tab (not SURFACE or MESH)
with a document open — that panel doesn't exist elsewhere. After changing add-in
code, restart Fusion: it caches imported modules, so Stop/Run alone may not reload.

## Use

1. Open or create a design.
2. **SOLID** tab → **CREATE** panel → **Helmet Lattice Tile**.
3. Pick a topology and parameters (the dialog shows only the relevant controls):
   - **gyroid / schwartz / diamond** — single TPMS.
   - **graded** — soft→stiff gyroid; uses *Gradient spread*.
   - **auxetic** — perforated sheet (re-entrant square holes).
   - **alternating** — Z-stacked layers; edit the *Layers* field as
     `topo:density:cell,...` (e.g. `gyroid:0.25:8,schwartz:0.30:6`).
   - **blend** — *generative*: mixes the TPMS basis into a new in-between geometry;
     set the **Blend weights** (gyroid/schwartz/diamond, auto-normalized).
4. **Generate** → the STL is built and imported as a mesh body.

### Autonomous (data-driven) generation

Tick **Autonomous: optimize blend from data** and hit **Generate**. The add-in then,
in one step:
1. runs the optimizer (`run_pipeline.py --generative`) — a GP surrogate + Bayesian
   optimization that **chooses the blend weights and parameters from your data**,
2. loads that proposal, and
3. generates + imports the resulting blended geometry.

No sliders — the design is selected by the model, not by hand. Point `DATA_CSV` (top
of `HelmetLattice.py`) at your real handoff table; it defaults to the dummy data.
Expect the optimizer step to take ~10–60 s (GP fit + CV), during which Fusion is busy.

> Scope note: "autonomous" means the optimizer *selects* the design from data — it
> does not invent geometry from scratch (that needs far more data than a small study
> provides). It searches the continuous TPMS blend space the generator can build.

### Load latest proposal

Tick **Load latest proposal** to pull the *most recent* optimizer recommendation from
`outputs/proposed_designs.csv` into the dialog without re-running the optimizer —
including blend weights when the pipeline last ran in `--generative` mode. Useful if
you already ran the pipeline and just want to print its result.

## Batch review — N designs, each its own document

**SOLID → CREATE → Helmet Lattice Batch.** Set a count (default 10) and hit OK. The
add-in runs `run_pipeline.py --generative --q N`, then creates a timestamped folder in
the active project's Data Panel and saves each accepted design as its own document
with the mesh already imported.

The mesh you review **is** the mesh you print — there is no separate preview tier. The
same STL sits in `stl_output/` for the slicer; the document just points at it. One
mesh per document is what makes full resolution affordable; putting all ten in a
single document is what would not be.

Each document's description carries the review facts: blend weights, designed vs
as-built density, cell size, family count, largest family share, predicted HIC/BrIC,
and unsupported-area fraction.

> **"Folder" means a Fusion Teams / Data Panel folder** — cloud storage in the active
> project, not a folder on your C: drive. Ten designs is ~400 MB of mesh across ten
> uploads, so the save step is not instant. If no project is active, the command stops
> and tells you to pick one in the Data Panel.

Timing: roughly a minute per design end-to-end (optimizer once, then meshing each).
The batch asks the pipeline for N designs that clear both the diversity filter and the
printability gate; the pipeline ranks a 3x shortlist internally, so a design rejected
by the gate is replaced rather than shrinking the batch.

## Supportless printing

You can't use supports, so pick accordingly:

| Topology | Supportless | Notes |
|----------|-------------|-------|
| **Auxetic** | Best | All walls vertical, zero overhangs by design. |
| **Gyroid** | Good | Self-supporting TPMS (~45° overhangs). |
| **Graded** | Good | Gyroid topology; soft/stiff halves meet on whole cells. |
| **Schwartz-P** | Risky | Near-horizontal domes droop without supports — avoid or expect rough tops. |

## Whole cells (no half cells)

The requested cell size is **snapped to a whole number of cells per axis** and
resized to `tile_dim / n`, so no cell is sliced at a face. Axes of different length
get slightly different cell sizes (e.g. a 50.8 mm span with a requested 8 mm cell
becomes 6 cells of 8.467 mm; the 38.1 mm height becomes 5 cells of 7.62 mm). For
TPMS, relative density is set by the isovalue, so snapping the cell does **not**
change your target density. The generator prints the final `cells=` and `cell_mm=`.

## Notes

- **Voxel pitch** is the speed/quality knob: 0.5 mm ≈ 1 s; drop to 0.3 mm for finer
  walls (~4–5× slower). Cost scales with the cube of 1/pitch.
- **Units:** the STL is in mm; Fusion imports it at mm scale. If it ever comes in
  10× off, check the document's default units.
- The mesh body is faceted (marching-cubes). To get a B-rep solid, use
  **Mesh → Convert Mesh** (works best on the coarser pitches).
