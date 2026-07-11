"""Synthesis for the multi-asset sweep: apply the pre-registered multiplicity
correction across the whole slices×targets×models grid, test H3 (geometry's edge
scales with manifold diversity), and render the verdict.

Reads the sweep JSON written by multi_asset_experiment.py and answers, honestly:
  * Does any cell show geometry adding over the Absorption Ratio after Holm/BH?
  * Does the full stack (vol+AR+geo) ever beat the volatility baseline after Holm/BH?
  * H3: Spearman(diversity, geometry edge) across slices on the on-thesis target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.evaluation import bh_fdr_correction, holm_bonferroni_correction  # noqa: E402

ON_THESIS_TARGET = "target_fwd_ar_change"


def _grid(results):
    """Flatten to rows: (slice, diversity, target, model, R²s, Δp's)."""
    rows = []
    for r in results:
        for tname, models in r["targets"].items():
            for mname, m in models.items():
                rows.append(
                    {
                        "slice": r["slice"],
                        "diversity": r["diversity"],
                        "target": tname,
                        "model": mname,
                        "r2_vol": m["r2_vol"],
                        "r2_ar": m["r2_ar"],
                        "r2_geo": m["r2_geo"],
                        "r2_ar_geo": m["r2_ar+geo"],
                        "r2_all": m["r2_all"],
                        "p_geo_over_ar": m["delta_r2_geo_over_ar_p"],
                        "d_geo_over_ar": m["delta_r2_geo_over_ar"],
                        "p_all_over_vol": m["delta_r2_all_over_vol_p"],
                        "d_all_over_vol": m["delta_r2_all_over_vol"],
                    }
                )
    return rows


def _correct(rows, pkey):
    ps = [r[pkey] for r in rows]
    valid = [i for i, p in enumerate(ps) if p is not None and np.isfinite(p)]
    pv = [ps[i] for i in valid]
    holm_adj, holm_rej = holm_bonferroni_correction(pv)
    bh_adj, bh_rej = bh_fdr_correction(pv, alpha=0.05)
    for j, i in enumerate(valid):
        rows[i][f"{pkey}_holm"] = float(holm_adj[j])
        rows[i][f"{pkey}_bh"] = float(bh_adj[j])
        rows[i][f"{pkey}_bh_sig"] = bool(bh_rej[j])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sweep", default="experiments/outputs/regime_detection/multi_asset/pred_sweep.json"
    )
    ap.add_argument(
        "--out", default="experiments/outputs/regime_detection/multi_asset/synthesis.json"
    )
    args = ap.parse_args()

    results = json.load(open(args.sweep))
    rows = _grid(results)
    rows = _correct(rows, "p_geo_over_ar")
    rows = _correct(rows, "p_all_over_vol")

    print(f"Grid: {len(rows)} cells (slices×targets×models). Multiplicity: Holm + BH.\n")
    hdr = (
        f"{'slice':16s} {'target':22s} {'mdl':5s} {'vol':>6s} {'ar':>6s} {'geo':>6s} "
        f"{'all':>6s} {'Δgeo|ar':>8s} {'BHp':>6s} {'Δall|vol':>9s} {'BHp':>6s}"
    )
    print(hdr)
    for r in rows:
        print(
            f"{r['slice']:16s} {r['target']:22s} {r['model']:5s} "
            f"{r['r2_vol']:+.2f} {r['r2_ar']:+.2f} {r['r2_geo']:+.2f} {r['r2_all']:+.2f} "
            f"{r['d_geo_over_ar']:+.3f} {r.get('p_geo_over_ar_bh', float('nan')):.3f} "
            f"{r['d_all_over_vol']:+.3f} {r.get('p_all_over_vol_bh', float('nan')):.3f}"
        )

    geo_wins = [r for r in rows if r.get("p_geo_over_ar_bh_sig")]
    vol_wins = [r for r in rows if r.get("p_all_over_vol_bh_sig") and r["d_all_over_vol"] > 0]
    print(f"\nCells where geometry beats Absorption Ratio (BH q<0.05): {len(geo_wins)}")
    for r in geo_wins:
        print(f"   {r['slice']} / {r['target']} / {r['model']}  Δ={r['d_geo_over_ar']:+.3f}")
    print(f"Cells where (vol+AR+geo) beats volatility (BH q<0.05): {len(vol_wins)}")
    for r in vol_wins:
        print(f"   {r['slice']} / {r['target']} / {r['model']}  Δ={r['d_all_over_vol']:+.3f}")

    # H3: edge vs diversity on the on-thesis target (ridge), across slices
    h3 = [
        (r["diversity"], r["d_geo_over_ar"])
        for r in rows
        if r["target"] == ON_THESIS_TARGET
        and r["model"] == "ridge"
        and r["d_geo_over_ar"] is not None
        and np.isfinite(r["d_geo_over_ar"])
    ]
    h3_res = None
    if len(h3) >= 3:
        div, edge = zip(*sorted(h3))
        rho, p = spearmanr(div, edge)
        h3_res = {"spearman_rho": float(rho), "p": float(p), "points": h3}
        print(
            f"\nH3 (edge ∝ diversity, {ON_THESIS_TARGET}, ridge): Spearman ρ={rho:+.3f} (p={p:.3f})"
        )
        for d, e in sorted(h3):
            print(f"   diversity={d:.3f}  Δ(geo|ar)={e:+.3f}")

    payload = {
        "grid": rows,
        "geo_beats_ar_BH": len(geo_wins),
        "stack_beats_vol_BH": len(vol_wins),
        "h3": h3_res,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
