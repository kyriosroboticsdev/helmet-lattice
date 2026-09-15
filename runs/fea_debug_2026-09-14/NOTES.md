# Seed-11 FEA debugging snapshot (2026-09-14)

Nothing here is committed. The code changes live in scripts/fea, scripts/loop,
scripts/ml/{seed_lhs,surrogate_compare,realize}.py.

## Material cards on the seed-11 lattice (330k C3D4 tets, frictionless symmetry BCs)
| card | result |
|---|---|
| Ogden N=3 (calibrated, 21%) | fails: diverges on a solid cube, residual stalls on the lattice |
| Mooney-Rivlin C10=7.895 C01=-1.519 D1=0.0435 (20.3%) | converges on the cube; stalls at ~5.3% crush on the lattice (residual blow-up, cutbacks below 0.6%) |
| Neo-Hooke C10=5.005 D1=0.0435 (33%) | converged to 12.1% crush (441 kPa), then stalled on ever-smaller cutbacks (below 0.0001 of the step); killed by the 5400 s cap. Stiffness was falling (~39 kPa/% at 0-4% vs ~29 kPa/% at 11-12%), consistent with the onset of wall buckling |

Partial curves: `*/stress_strain.csv`. The laptop slept mid-solve; CalculiX resumed on wake.
The loop's failure path worked end to end: the design was logged as a clean `fea_status=failed` row.
Gaps: that row loses rel_density_measured, and the partial curve is thrown away.

## To finish the Neo-Hooke run (~90 min, 12 threads)
From the repo root (Git Bash):
    export PATH="$PATH:/c/Program Files/FreeCAD 1.1/bin"   # APPEND, never prepend
    export CCX_NPROC_STIFFNESS=$NUMBER_OF_PROCESSORS OMP_NUM_THREADS=$NUMBER_OF_PROCESSORS
    python -u runs/fea_debug_2026-09-14/seed11_e2e.py
(the script writes into the old scratchpad dir; edit `out` if that is gone)

## Still open
- DONE: PAUSE file in the run dir stops the loop between designs; --resume skips finished seeds (tested with a stub).
- A static solve cannot get past ~12% crush -> decide: quasi-static *DYNAMIC, compress to ~10% only, or both.
- Mid-solve *RESTART not tested yet.
- Prep jumped from ~70 s to ~350 s between runs with no code change (CPU-bound). Not profiled yet.
- COMPRESS_FRACTION=0.20 vs the 20-40% plateau band; optimizer is still HIC/BrIC-only.

## 2026-09-14 later: first COMPLETE loop FEA row
- COMPRESS_FRACTION is now 0.10. Plateau and densification stay NaN unless actually reached; new columns fea_max_strain and stress_at_max_strain_kPa.
- Full seed 11 through run_loop.evaluate_design: fea_status=ok, converged to 10%, SEA 51.1 J/kg, 376.4 kPa at 10%,
  rel_density 0.309; prep 176 s + solve 1554 s (26 min). Row + log + ccx files in `full_tile_10pct/`.
- 2x2x2-cell piece (82k tets), `piece_2x2x2/`: static reaches 20% in 189 s with no stall; implicit *DYNAMIC
  (T=0.05 s, HHT -0.3) matches static within 0.05% (peak KE/work 4e-5) in 468 s. The piece is 2-4% stiffer than the
  full tile over 1-12%. Its tangent stiffness falls 40 -> 13 kPa/% from 0-4% to 16-20% (plateau onset), smoothly.
  -> The full-tile stall at 12% is probably mesh-local/numerical, not the lattice's buckling (not proven).
- Open decision: loop FEA = (a) full tile at 10% (~30 min), (b) 2x2x2 piece at 20% (~3 min), or (c) piece to screen + full tile to confirm.

Note: absolute paths in this folder were replaced with <repo>, <scratchpad> and <TEMP> before commit. The test scripts here are a record of what was run; edit those paths before re-running them.
