"""Is Multi-Lag Fidelity's signal orthogonal to realized volatility?

The surviving detection claim (see research/phase1_gate_preregistration.md) is only
interesting if MLF is not a lossy vol re-encoding like the dead channels (Spectral
Entropy corr +0.67, Reduced Purity +0.63). Computes contemporaneous Pearson and
Spearman correlation between causal MLF scores and the Rolling Vol Z baseline on
the standard SPY+DIA enriched panel. Registered check: |Spearman| < 0.3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.walk_forward_hpo as wf  # noqa: E402
from experiments.baselines import RollingVolatilityDetector  # noqa: E402
from qcml_geometry import MultiLagFidelityDetector  # noqa: E402

# The default MLF config used across the horse races (volatility_forecasting.GEO).
MLF_PARAMS = dict(
    hilbert_dim=4,
    n_pca_components=8,
    rolling_window=20,
    operator_method="pca_inspired",
    seed=42,
    normalization="sphere",
)


def main():
    Xe, dates = wf.prepare_data()
    fit_end = int(0.30 * len(dates))

    mlf = MultiLagFidelityDetector(causal_fit_length=fit_end, **MLF_PARAMS).fit(Xe)
    s_mlf = np.asarray(mlf.compute_regime_scores(Xe), dtype=float)

    vol = RollingVolatilityDetector(vol_window=20).fit(Xe)
    s_vol = np.asarray(vol.compute_regime_scores(Xe), dtype=float)

    m = np.isfinite(s_mlf) & np.isfinite(s_vol)
    m[:fit_end] = False  # score only the out-of-fit region
    r_p, p_p = pearsonr(s_mlf[m], s_vol[m])
    r_s, p_s = spearmanr(s_mlf[m], s_vol[m])
    print(f"n={m.sum()}  (post-fit region, {dates[fit_end].date()}..{dates[-1].date()})")
    print(f"Pearson  r = {r_p:+.3f}  (p={p_p:.2g})")
    print(f"Spearman ρ = {r_s:+.3f}  (p={p_s:.2g})")
    print(f"|Spearman| < 0.3: {'PASS' if abs(r_s) < 0.3 else 'FAIL'}")


if __name__ == "__main__":
    main()
