"""
audit_fea_setup.py  --  Measure what the screening FEA actually builds, per design.
==================================================================================
No solve. For finished piece designs of a loop run it re-meshes the saved STL exactly
as the loop did (fea_evaluator.mesh_for_fea) and reports:

  geometry   bbox vs the intended piece (piece_cells x snapped cell), height H
  density    tet-mesh volume / bbox  vs  STL (surface) volume / bbox  vs  the
             density the proposal asked for; volume lost to dropped slivers
  mass       grams the SEA denominator assumes
  BC sets    share of nodes in the BOT/TOP slabs and on the XSYM/YSYM faces, and
             how thick those slabs are in mm (the 2%-of-extent tolerances)
  material   Neo-Hooke constants -> G, K, E, nu (units N, mm, MPa)

Usage:  python scripts/fea/audit_fea_setup.py runs/loop_2026-09-14 [--n 6]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "ml"))
import fea_evaluator as FE


def material_constants(C10=FE.NEO_HOOKE["C10"], D1=FE.NEO_HOOKE["D1"]):
    G = 2 * C10; K = 2 / D1
    nu = (3 * K - 2 * G) / (2 * (3 * K + G)); E = 2 * G * (1 + nu)
    return dict(C10=C10, D1=D1, G_MPa=G, K_MPa=K, E_MPa=round(E, 2), nu=round(nu, 4))


def audit_one(stl: str, req_rho: float) -> dict:
    import trimesh
    coords, nid, tets, rep = FE.mesh_for_fea(stl)
    ext = np.ptp(coords, axis=0); bbox = float(np.prod(ext))
    vol_tet = FE._mesh_volume(coords, nid, tets)
    surf = trimesh.load(stl)
    z = coords[:, 2]; tolz = ext[2] * 0.02
    stol = max(ext[0], ext[1]) * 0.02
    n = len(nid)
    return dict(
        bbox_x=round(ext[0], 2), bbox_y=round(ext[1], 2), H=round(ext[2], 2),
        stl_bbox=tuple(np.round(surf.extents, 2)),
        rho_requested=req_rho,
        rho_tet=round(vol_tet / bbox, 4),
        rho_stl=round(float(surf.volume) / float(np.prod(surf.extents)), 4),
        watertight=bool(surf.is_watertight),
        dropped_vol_frac=round(rep["dropped_volume_frac"], 5),
        mass_g=round(vol_tet * FE.TPU_DENSITY_G_PER_MM3, 3),
        slab_mm=round(tolz, 3), sym_mm=round(stol, 3),
        bot_frac=round(float((z <= z.min() + tolz).mean()), 4),
        top_frac=round(float((z >= z.max() - tolz).mean()), 4),
        xsym_frac=round(float((coords[:, 0] <= coords[:, 0].min() + stol).mean()), 4),
        ysym_frac=round(float((coords[:, 1] <= coords[:, 1].min() + stol).mean()), 4),
        tets=len(tets), nodes=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    df = pd.read_csv(Path(a.run) / "master_table.csv")
    ok = df[(df.fea_status == "ok") & (df.fea_model == "piece")]
    ok = ok.sort_values("rel_density_measured").iloc[
        np.linspace(0, len(ok) - 1, min(a.n, len(ok))).astype(int)]
    print("material:", material_constants())
    out = []
    for _, r in ok.iterrows():
        req = json.loads(r.proposal_json).get("rel_density_measured")
        d = dict(design=r.design_id, rho_master=r.rel_density_measured,
                 cell_built=r.get("cell_size_built_mm"), **audit_one(r.raw_file_path, req))
        out.append(d); print(d, flush=True)
    res = pd.DataFrame(out)
    res.to_csv(Path(a.run) / "fea_setup_audit.csv", index=False)
    print(res.drop(columns=["stl_bbox"]).to_string(index=False))


if __name__ == "__main__":
    main()
