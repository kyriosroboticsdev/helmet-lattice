# Data in this project

Three different kinds of data live here, with three different provenances and
three different rules. Please read before adding anything.

---

## 1. Committed to git — small, ours, safe to share

| Path | What it is |
|---|---|
| `data/handoff_template.csv` | Empty template for the Data Handoff Contract. The column set is authoritative and defined in `scripts/ml/schema.py`. |
| `data/designs_D1_D5.csv` | As-designed geometry parameters for the first five tiles. |
| `vt_results/results.csv` | Impact results, filled in after each VT trip. |
| `vt_results/baseline_dataset_tidy.csv` | Tidied baseline set used to sanity-check the pipeline. |
| `docs/example_output/` | One committed example pipeline run, so you can see what the outputs look like without running anything. Absolute paths have been scrubbed to `<repo>/`. |

---

## 2. Generated — NOT committed, reproduce locally

| Path | Regenerate with |
|---|---|
| `stl_output/` (~2.1 GB, 81 files) | `python scripts/generate_tiles.py` |
| `outputs/` | `python scripts/run_pipeline.py` |

These are gitignored on purpose. Individual STLs run 17–69 MB, which is at or
over GitHub's 100 MB per-file limit and far past the 1 GB repo soft limit. The
whole point of `generate_tiles.py` is that the geometry is reproducible from a
topology label, a cell size and a target density — so the generator is the
artifact worth versioning, not its output.

If you need a specific tile without running the generator, ask and it can be
shared directly or attached to a release.

---

## 3. Third-party dataset — NOT committed, obtain yourself

`dataverse_files/` (~4.6 GB) is an externally published impact dataset, not ours.
It is excluded from git because we do not have redistribution rights and because
of its size.

**Contents** (per its own `1-Readme.txt`):

- `raw_acceleration_data_labview.rar` (826 MB) — raw 3-axis accelerometer traces
  in LabVIEW TDMS format. Organised as *shape* (cube / wedge / cylinder) →
  *impactor mass* (300 g / 400 g / 500 g) → 16 experiments each.
  - Column C — linear acceleration of the dummy head
  - Column D — force sensor (**not used** in these experiments)
  - Columns E / F / G — accelerometer X / Y / Z
  - Reading TDMS in Excel needs National Instruments' add-in.
- `Dataset_v2.zip` (3.8 GB) — the full dataset.
- `processed_summary.xlsx` — 144 experiments summarised, tabbed by shape, with
  impact velocity, peak head linear acceleration, 3 ms criterion and HIC.

**To use it:** download from the original Dataverse record and unpack into
`dataverse_files/` at the repo root. `scripts/analyze_dataset.py` expects it
there.

> **TODO — @Kyrio:** paste the Dataverse DOI / URL and the required citation
> here. This needs to be recorded before any results derived from this dataset
> go into a writeup, and it is the one thing in this file I could not fill in
> from the local copy.

---

## The Data Handoff Contract

`scripts/ml/schema.py` is the single definition of the columns the testing team
delivers and the ML team consumes. It covers identifiers, as-designed and
as-measured geometry, and the impact metrics.

**Lock it with the testing team before testing starts.** Changing a column name
after data collection begins invalidates every row already gathered. Both teams
should treat `schema.py` as the contract document, not a convenience.
