"""
run_pipeline.py
===============
End-to-end driver for the TPU lattice helmet-liner ML pipeline (spec Section 10).
Runs the WHOLE workflow on the dummy dataset so everything is proven before real
VT data arrives:

  1. load + validate the master handoff table (Data Handoff Contract, Section 2)
  2. RF / gradient-boosting + SHAP feature importance, grouped CV by design_id
  3. Gaussian-process surrogate per headline objective, leave-one-design-out CV
     + uncertainty calibration
  4. multi-objective Bayesian optimization (qEHVI) -> Pareto front of linear vs.
     rotational injury + the active-learning next-design proposal

Usage
-----
    python scripts/run_pipeline.py                 # uses data/raw/dummy_master.csv
    python scripts/run_pipeline.py --data PATH      # point at the real handoff CSV
    python scripts/run_pipeline.py --make-dummy     # regenerate dummy data first

Outputs land in outputs/ (importance/, pareto_front.png, pipeline_report.md).
"""

from __future__ import annotations
import argparse
import math
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml"))

import pandas as pd

import schema as S
import data as D
import features as F          # noqa: F401  (imported so it's part of the package run)
import importance as IMP
import surrogate_gp as G
import optimize as OPT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CONDITIONS = {"impact_speed_mps": 5.0}   # standard impact condition for optimization


def main():
    ap = argparse.ArgumentParser(description="Helmet-lattice ML pipeline (dummy-data scaffold).")
    ap.add_argument("--data", default=str(ROOT / "data" / "raw" / "dummy_master.csv"),
                    help="master handoff table (CSV or Parquet)")
    ap.add_argument("--make-dummy", action="store_true",
                    help="(re)generate the synthetic dataset before running")
    ap.add_argument("--candidates", type=int, default=2000,
                    help="Sobol candidate pool size for the optimizer")
    ap.add_argument("--no-stl", action="store_true",
                    help="skip generating an STL for the proposed design")
    ap.add_argument("--generative", action="store_true",
                    help="search the continuous cross-family blend space (propose NEW "
                         "blended geometries) instead of enumerating discrete topologies")
    ap.add_argument("--blend-floor", type=float, default=OPT.BLEND_MIN_WEIGHT,
                    help="(legacy) min weight per basis; superseded by --max-family")
    ap.add_argument("--max-family", type=float, default=OPT.MAX_FAMILY_WEIGHT,
                    help="per-family ceiling in generative mode (default 0.45): no "
                         "family may exceed this share, which forces >= ceil(1/cap) "
                         "families per design and makes pure designs unreachable")
    ap.add_argument("--families", type=int, nargs=2, metavar=("MIN", "MAX"),
                    default=list(OPT.FAMILIES_PER_DESIGN),
                    help="how many families a design may combine (default 3 5); MIN is "
                         "raised to ceil(1/--max-family) if it is below it")
    ap.add_argument("--allow-aperiodic", action="store_true",
                    help="allow families whose field does NOT repeat with the unit cell "
                         "(bouligand, and the stochastic configurations of voronoi/"
                         "spinodoid). Off by default: coupons need a repeating pattern")
    ap.add_argument("--q", type=int, default=1,
                    help="how many designs to propose this round")
    ap.add_argument("--min-distance", type=float, default=OPT.MIN_DESIGN_DISTANCE,
                    help="minimum descriptor-space separation between proposals, in SDs "
                         "of the candidate pool (0 disables the diversity filter)")
    ap.add_argument("--shortlist", type=int, default=0,
                    help="rank this many candidates so rejects by the printability gate "
                         "can be replaced (default: 3x --q)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # 0. Optionally regenerate dummy data --------------------------------------
    if args.make_dummy:
        import make_dummy_data as MK
        df = MK.make_dummy()
        Path(args.data).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.data, index=False)
        print(f"[pipeline] wrote dummy data -> {args.data}")

    # 1. Load + validate -------------------------------------------------------
    print("\n=== 1. Load + validate (Data Handoff Contract) ===")
    df = D.load_master(args.data)
    problems = D.validate(df)
    print(f"Loaded {len(df)} rows / {df['design_id'].nunique()} designs. "
          f"Contract problems: {len(problems)}")
    n_clip = int(D.is_clipped(df).sum())
    print(f"Clipped/excluded rows: {n_clip}. Sources: "
          + ", ".join(f"{k}={v}" for k, v in df['data_source'].value_counts().items()))

    # 2. Feature importance ----------------------------------------------------
    print("\n=== 2. Feature importance (RF/GBM + SHAP, grouped CV) ===")
    imp_results = IMP.run(df, OUT / "importance")

    # 3. GP surrogate + calibration -------------------------------------------
    print("\n=== 3. GP surrogate (leave-one-design-out CV + calibration) ===")
    gps = G.fit_objectives(df, CONDITIONS)

    # 4. Multi-objective BO + active-learning proposal -------------------------
    print("\n=== 4. Multi-objective BO (qEHVI) + active-learning proposal ===")
    # Rank a shortlist, not just q: the printability gate runs after meshing and can
    # reject, so keep spares ranked and ready rather than ending the round empty.
    shortlist = args.shortlist or max(args.q * 3, args.q + 2)
    proposals, (cands, means, stds) = OPT.propose_next(
        gps, df, CONDITIONS, n_candidates=args.candidates, q=shortlist,
        generative=args.generative, blend_floor=args.blend_floor,
        cap=args.max_family, families=tuple(args.families),
        min_distance=args.min_distance, periodic=not args.allow_aperiodic)
    mask = OPT.predicted_pareto(cands, means)
    OPT.plot_front(df, cands, means, mask, proposals[:args.q], OUT / "pareto_front.png")
    print(f"Pareto front + proposal plotted -> {OUT / 'pareto_front.png'}")
    if args.generative:
        k_min = max(args.families[0], math.ceil(1.0 / args.max_family))
        mode = "aperiodic allowed" if args.allow_aperiodic else \
               "repeating unit cell enforced"
        print(f"(generative mode: NEW cross-family blends, no family above "
              f"{args.max_family:.0%}, {k_min}-{args.families[1]} families per design, "
              f"{mode})")
    for p in proposals[:args.q]:
        what = (f"blend={_fmt_blend(p['blend'])}" if p.get("blend")
                else f"topology={p['topology']}")
        print(f"PROPOSE NEXT: {what}, rho={p['rel_density_measured']:.3f}, "
              f"cell={p['cell_size_mm']:.1f}mm, gradient={p['gradient_param']:.3f}, "
              f"shear={p['shear_anisotropy_ratio']:.2f}  "
              f"=> pred {S.LINEAR_OBJECTIVE}={p['pred_HIC']:.0f}, "
              f"{S.ROTATIONAL_OBJECTIVE}={p['pred_BrIC']:.3f}")

    # 4b. Realize the proposal as a printable STL (Section 6: params -> STL) ----
    if not args.no_stl:
        print("\n=== 4b. Generate STL for the proposed design(s) ===")
        import realize as R
        accepted, stl_infos = R.realize_gated(proposals, ROOT / "stl_output", q=args.q)
        by_design = {id(p): info for p, info in accepted}
        proposals = [p for p, _ in accepted] or proposals[:args.q]
        for p in proposals:
            info = by_design.get(id(p), {})
            p["stl_path"] = info.get("stl_path")
            p["stl_status"] = info.get("status")
            p["unsupported_frac"] = info.get("unsupported_frac")
            p["rel_density_as_built"] = info.get("rel_density_as_built")
            p["n_bodies"] = info.get("n_bodies")
            p["min_octant_fill"] = info.get("min_octant_fill")
        rejected = [i for i in stl_infos if i.get("printable") is False]
        if rejected:
            print(f"[realize] {len(rejected)} candidate(s) rejected "
                  f"(overhang > {R.MAX_UNSUPPORTED_FRAC:.0%}, split into multiple "
                  f"bodies, or uneven fill):")
            for i in rejected:
                print(f"    - {i['design']}: {i['status'].removeprefix('rejected: ')}")
    else:
        proposals = proposals[:args.q]
        print("\n(STL generation skipped: --no-stl)")

    # Always log the proposal(s) so the Fusion add-in's "Load latest proposal" works,
    # whether or not an STL was generated this run.
    pd.DataFrame(proposals).to_csv(OUT / "proposed_designs.csv", index=False)
    print(f"Proposal(s) logged -> {OUT / 'proposed_designs.csv'}")

    # 5. Write a human-readable report ----------------------------------------
    report = write_report(df, imp_results, gps, proposals, OUT / "pipeline_report.md")
    print(f"\n[pipeline] report written -> {report}")
    print("[pipeline] done. All stages ran on the current dataset.")


def _fmt_blend(blend: dict) -> str:
    """Render a blend as a readable, descending-weight mix: 'voronoi 45% + ...'."""
    return " + ".join(f"{k} {v:.0%}" for k, v in
                      sorted(blend.items(), key=lambda kv: -kv[1]))


def write_report(df, imp_results, gps, proposals, path: Path) -> Path:
    lines = ["# Pipeline report", ""]
    lines.append(f"- Rows: **{len(df)}**, designs: **{df['design_id'].nunique()}**, "
                 f"clipped excluded: **{int(D.is_clipped(df).sum())}**")
    lines.append(f"- Sources: " + ", ".join(f"`{k}`={v}" for k, v in
                 df['data_source'].value_counts().items()))
    lines.append("")
    lines.append("## Feature importance (top drivers)")
    for r in imp_results:
        if r.get("skipped"):
            lines.append(f"- `{r['target']}`: skipped (n={r['n']})")
            continue
        cv = r["cv_gbm"]
        lines.append(f"- `{r['target']}` (n={r['n']}, {r['n_designs']} designs): "
                     f"GBM RMSE={cv['rmse']:.3g} "
                     f"[{cv['rmse_ci'][0]:.3g}, {cv['rmse_ci'][1]:.3g}] — "
                     f"top: {', '.join(r['top_features'][:4])}")
    lines.append("")
    lines.append("## GP surrogate (headline objectives)")
    for t, gp in gps.items():
        cv = gp.cv
        lines.append(f"- `{t}`: RMSE={cv['rmse']:.3g}, R²={cv['r2']:.3f}, "
                     f"95% interval coverage={cv['coverage95']:.0%} "
                     f"({'calibrated' if 0.8 <= cv['coverage95'] <= 1.0 else 'review'})")
    lines.append("")
    lines.append("## Active-learning proposal (print & test next)")
    for p in proposals:
        what = (f"blend **{_fmt_blend(p['blend'])}**" if p.get("blend")
                else f"topology=**{p['topology']}**")
        lines.append(f"- {what}, rel_density={p['rel_density_measured']:.3f}, "
                     f"cell={p['cell_size_mm']:.1f} mm, gradient={p['gradient_param']:.3f}, "
                     f"shear_anisotropy={p['shear_anisotropy_ratio']:.2f}")
        if p.get("blend"):
            lines.append(f"  - {p.get('n_families', len(p['blend']))} families, "
                         f"largest share {p.get('max_family_weight', 0):.0%} "
                         f"(cap {OPT.MAX_FAMILY_WEIGHT:.0%})")
        if p.get("unsupported_frac") is not None:
            lines.append(f"  - as-built density {p.get('rel_density_as_built')}, "
                         f"unsupported area {p['unsupported_frac']:.1%}")
        lines.append(f"  - predicted {S.LINEAR_OBJECTIVE}={p['pred_HIC']:.0f}±{p['pred_HIC_sd']:.0f}, "
                     f"{S.ROTATIONAL_OBJECTIVE}={p['pred_BrIC']:.3f}±{p['pred_BrIC_sd']:.3f}")
        if p.get("stl_path"):
            lines.append(f"  - STL: `{p['stl_path']}`")
        elif p.get("stl_status"):
            lines.append(f"  - STL: not generated ({p['stl_status']})")
    lines.append("")
    lines.append("*Generated by `scripts/run_pipeline.py`. On dummy data until the "
                 "real handoff table is supplied via `--data`.*")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
