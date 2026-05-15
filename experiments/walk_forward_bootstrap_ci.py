"""Bootstrap 95% CI on the walk-forward nested-HPO median Cohen's d.

Inputs are the 9 per-window OOS d values from Table 4 (tab:wf_hpo) in
paper/qcml_geometric_sde.tex.  Resamples each detector's 9 values with
replacement and reports the percentile bootstrap CI on the median.

Output: experiments/outputs/regime_detection/walk_forward/wf_bootstrap_ci.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

# Per-window OOS Cohen's d from Table 4 (Walk-forward nested HPO).
# Order: Flash 2010, Euro 2011, China 2015, Volmageddon 2018, Q4 2018,
#        Repo 2019, COVID 2020, Rate Hikes 2022, SVB 2023.
WF_HPO_VALUES = {
    "Berry Phase Rate": [0.77, 0.08, 1.18, 1.10, 0.61, 0.53, 0.34, 0.72, 1.54],
    "QFI Determinant":  [0.04, 0.65, 0.13, 1.44, 0.37, 0.28, 0.63, 0.40, 0.34],
    "Multi-Lag Fidelity": [0.52, 1.43, 0.44, 1.99, 0.16, 0.62, 0.88, 0.31, 0.22],
}

N_BOOT = 10000
SEED = 42

OUTPUT_PATH = (
    Path(__file__).parent
    / "outputs"
    / "regime_detection"
    / "walk_forward"
    / "wf_bootstrap_ci.json"
)


def main():
    rng = np.random.default_rng(SEED)
    results = {
        "config": {
            "n_bootstrap": N_BOOT,
            "seed": SEED,
            "source": "Table 4 (tab:wf_hpo) per-window OOS d values",
            "ci_method": "percentile",
        },
        "summary": {},
    }

    for method, d_vals in WF_HPO_VALUES.items():
        arr = np.array(d_vals, dtype=float)
        boot_medians = np.empty(N_BOOT)
        for i in range(N_BOOT):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot_medians[i] = np.median(sample)
        lo, hi = np.quantile(boot_medians, [0.025, 0.975])
        results["summary"][method] = {
            "median_d": float(np.median(arr)),
            "mean_d": float(np.mean(arr)),
            "n_windows": len(arr),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
        }
        print(
            f"{method}: median {np.median(arr):.2f} [95% CI {lo:.2f}, {hi:.2f}]"
        )

    results["timestamp"] = datetime.now().isoformat()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
