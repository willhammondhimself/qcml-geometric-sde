"""
Dead Signal Resurrection Smoke Test — Q32–Q35

Tests whether simple architectural fixes can push four low-performing
detectors above the d > 0.3 median threshold.

Questions:
  Q32: Transfer Entropy — different n_bins, window sizes
  Q33: LSTM Autoencoder — larger hidden_dim, deeper network, next-step prediction
  Q34: Kernel PCA — polynomial/sigmoid kernels, different gamma, more components
  Q35: Spectral Flow — different gap_threshold, total spectral velocity, normalized flow

Smoke test protocol:
  - 4 crises: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
  - SPY + DIA data via yfinance
  - Cohen's d per crisis, n_bootstrap=500 (speed; not paper quality)
  - KEEP if median d > 0.3

Usage:
    python research/ideation/dead_signal_fix/test_q32_q35.py
"""

import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from qcml_geometry.observables import BaseRegimeDetector, SpectralFlowDetector

# ---- Config ----
CRISIS_KEYS = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 500   # smoke-test speed
SEED = 42
KEEP_THRESHOLD = 0.30

# ---- Data helpers ----

def load_data():
    """Fetch SPY+DIA data and build feature matrix. Uses cache if available."""
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '1995-01-01', '2024-12-31', use_cache=True)
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    # BaseRegimeDetector enriched features (rolling mean/std/min/max, lookback=20)
    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]   # align after lookback drop
    print(f"Data loaded: {X_enriched.shape[0]} days, {X_enriched.shape[1]} features")
    # dates (raw) aligns with X; dates_enriched aligns with X_enriched
    return X, dates, X_enriched, dates_enriched


def crisis_masks(dates, crisis_info):
    """Boolean masks for crisis and normal periods."""
    start = pd.Timestamp(crisis_info['start'])
    end = pd.Timestamp(crisis_info['end'])
    crisis_mask = (dates >= start) & (dates <= end)
    normal_mask = ~crisis_mask
    return crisis_mask, normal_mask


def causal_fit_end(dates, crisis_info):
    """Index of last date >= 10 days before crisis start (causal fit)."""
    cutoff = pd.Timestamp(crisis_info['start']) - pd.Timedelta(days=10)
    return max(100, int(np.searchsorted(dates, cutoff)))


def cohens_d(scores, c_mask, n_mask):
    """Quick wrapper returning just the point estimate."""
    d, _, _ = compute_cohens_d_with_ci(
        scores[c_mask], scores[n_mask],
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )
    return d if np.isfinite(d) else 0.0


def eval_detector(det, X_fit, X_score, dates, label):
    """Fit detector on X_fit, score on X_score; return per-crisis d dict."""
    results = {}
    for ck in CRISIS_KEYS:
        ci = ALL_CRISES[ck]
        fit_end = causal_fit_end(dates, ci)
        try:
            det_copy = _clone_detector(det)
            det_copy.fit(X_fit[:fit_end])
            scores = det_copy.compute_regime_scores(X_score)
            c_mask, n_mask = crisis_masks(dates, ci)
            d = cohens_d(scores, c_mask, n_mask)
        except Exception as e:
            print(f"    ERROR [{label}] {ck}: {e}")
            d = 0.0
        results[ck] = round(float(d), 3)
    med = float(np.median(list(results.values())))
    return results, med


def _clone_detector(det):
    """Shallow clone via same class + same __dict__."""
    import copy
    return copy.deepcopy(det)


# =============================================================================
# Q32: Transfer Entropy variants
# =============================================================================

def test_q32(X, dates_raw, X_enriched, dates):
    """Test Transfer Entropy with different n_bins and window sizes.

    Current baseline: n_bins=5, te_window=60.
    Variants:
      (a) n_bins in {3, 5, 8}
      (b) te_window in {20, 40, 80}

    TransferEntropyDetector operates on raw X (2 columns = SPY, DIA returns).
    dates_raw must align with X (not with X_enriched).
    """
    from experiments.baselines import TransferEntropyDetector

    print("\n" + "=" * 60)
    print("Q32: Transfer Entropy Variants")
    print("=" * 60)

    # Use raw X — TE needs raw return columns (SPY_ret, DIA_ret).
    # dates_raw is aligned with X (pre-enrichment).
    X_te = X  # shape (T, n_features)

    best_label = None
    best_med = -1.0
    best_results = {}

    variants = []
    # (a) Different n_bins at default window
    for nb in [3, 5, 8]:
        variants.append({
            'label': f'n_bins={nb} window=60',
            'n_bins': nb, 'te_window': 60,
        })
    # (b) Different window sizes at default n_bins
    for w in [20, 40, 80]:
        variants.append({
            'label': f'n_bins=5 window={w}',
            'n_bins': 5, 'te_window': w,
        })

    for v in variants:
        det = TransferEntropyDetector(
            te_window=v['te_window'],
            n_bins=v['n_bins'],
            lag=1,
            min_expanding=60,
        )
        results, med = eval_detector(det, X_te, X_te, dates_raw, v['label'])
        verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
        print(f"  {v['label']:<28} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
              f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
              f"  | median={med:.3f}  {verdict}")
        if med > best_med:
            best_med = med
            best_label = v['label']
            best_results = results

    overall_verdict = "KEEP" if best_med > KEEP_THRESHOLD else "REJECT"
    return {
        'question': 'Q32: Transfer Entropy Fix',
        'best_variant': best_label,
        'crisis_d': best_results,
        'median_d': round(best_med, 3),
        'verdict': overall_verdict,
    }


# =============================================================================
# Q33: LSTM Autoencoder variants
# =============================================================================

def test_q33(X, X_enriched, dates):
    """Test LSTM Autoencoder architectural variants.

    Variants:
      (a) latent_dim in {4, 16, 32}  — baseline=4
      (b) deeper: 2-layer LSTM encoder — local subclass
      (c) next-step prediction (predict X[t+1] from X[:t]) — local subclass

    Keep epochs low (10) and retrain_interval high (500) for smoke-test speed.
    """
    from experiments.baselines import LSTMAutoencoderDetector

    print("\n" + "=" * 60)
    print("Q33: LSTM Autoencoder Variants")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  PyTorch not available — skipping Q33")
        return {
            'question': 'Q33: LSTM Autoencoder Fix',
            'best_variant': 'N/A (no torch)',
            'crisis_d': {k: 0.0 for k in CRISIS_KEYS},
            'median_d': 0.0,
            'verdict': 'REJECT',
            'reason': 'PyTorch not installed',
        }

    best_label = None
    best_med = -1.0
    best_results = {}

    # (a) Larger latent_dim
    for ld in [4, 16, 32]:
        label = f'latent_dim={ld}'
        det = LSTMAutoencoderDetector(
            seq_len=20, latent_dim=ld, n_epochs=10,
            min_expanding=120, retrain_interval=500, seed=SEED,
        )
        results, med = eval_detector(det, X_enriched, X_enriched, dates, label)
        verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
        print(f"  {label:<30} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
              f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
              f"  | median={med:.3f}  {verdict}")
        if med > best_med:
            best_med = med
            best_label = label
            best_results = results

    # (b) Deeper 2-layer LSTM (local subclass)
    class DeepLSTMAutoencoderDetector(LSTMAutoencoderDetector):
        """2-layer LSTM encoder/decoder variant."""

        @property
        def name(self):
            return "Deep LSTM Autoencoder"

        def _build_model(self, n_features):
            import torch
            import torch.nn as nn

            class DeepLSTMAE(nn.Module):
                def __init__(self, n_feat, latent):
                    super().__init__()
                    self.encoder = nn.LSTM(n_feat, latent * 2, num_layers=2,
                                           batch_first=True, dropout=0.1)
                    self.bottleneck = nn.Linear(latent * 2, latent)
                    self.expand = nn.Linear(latent, latent * 2)
                    self.decoder = nn.LSTM(latent * 2, n_feat, num_layers=2,
                                           batch_first=True, dropout=0.1)

                def forward(self, x):
                    _, (h, _) = self.encoder(x)
                    h_top = h[-1]                          # top layer hidden (batch, latent*2)
                    z = self.bottleneck(h_top)             # (batch, latent)
                    z_exp = self.expand(z)                 # (batch, latent*2)
                    z_rep = z_exp.unsqueeze(1).repeat(1, x.size(1), 1)
                    out, _ = self.decoder(z_rep)
                    return out

            torch.manual_seed(self.seed)
            return DeepLSTMAE(n_features, self.latent_dim)

    label = 'deep_2layer_latent16'
    det = DeepLSTMAutoencoderDetector(
        seq_len=20, latent_dim=16, n_epochs=10,
        min_expanding=120, retrain_interval=500, seed=SEED,
    )
    results, med = eval_detector(det, X_enriched, X_enriched, dates, label)
    verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
    print(f"  {label:<30} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
          f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
          f"  | median={med:.3f}  {verdict}")
    if med > best_med:
        best_med = med
        best_label = label
        best_results = results

    # (c) Next-step prediction target (predict X[t] from X[t-seq_len:t-1])
    class NextStepLSTMDetector(LSTMAutoencoderDetector):
        """Predict next window instead of reconstructing same window."""

        @property
        def name(self):
            return "LSTM Next-Step Predictor"

        def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
            if self._scaler is None:
                raise RuntimeError("Call fit() before compute_regime_scores().")
            try:
                import torch
                import torch.nn as nn
            except ImportError:
                return np.full(X.shape[0], np.nan)

            X_scaled = self._scaler.transform(X)
            T, n_feat = X_scaled.shape
            pred_errors = np.full(T, np.nan)

            model = None
            last_train = 0
            max_train = 2000

            for t in range(self.min_expanding, T):
                if model is None or (t - last_train) >= self.retrain_interval:
                    train_data = X_scaled[max(0, t - max_train):t]
                    if len(train_data) < self.seq_len + 10:
                        continue
                    # Build (input, target) pairs: input=X[i:i+seq_len-1], target=X[i+seq_len]
                    inputs, targets = [], []
                    for i in range(len(train_data) - self.seq_len):
                        inputs.append(train_data[i:i + self.seq_len - 1])
                        targets.append(train_data[i + self.seq_len - 1])  # next step
                    inputs = np.array(inputs)
                    targets = np.array(targets)

                    inp_t = torch.FloatTensor(inputs)
                    tgt_t = torch.FloatTensor(targets)

                    model = self._build_model(n_feat)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    loss_fn = nn.MSELoss()
                    batch_size = min(256, len(inp_t))
                    model.train()
                    for _ in range(self.n_epochs):
                        perm = torch.randperm(len(inp_t))
                        for start in range(0, len(inp_t), batch_size):
                            idx = perm[start:start + batch_size]
                            batch_inp = inp_t[idx]
                            batch_tgt = tgt_t[idx]
                            optimizer.zero_grad()
                            # Encode seq, decode last step
                            _, (h, _) = model.encoder(batch_inp)
                            pred = h.squeeze(0)  # (batch, latent) — use as 1-step pred via a linear
                            # Simple: decode last hidden as flat prediction
                            # We repurpose: run decoder for 1 step
                            h_rep = h.squeeze(0).unsqueeze(1)
                            out, _ = model.decoder(h_rep)
                            pred_out = out.squeeze(1)  # (batch, n_feat)
                            loss = loss_fn(pred_out, batch_tgt)
                            loss.backward()
                            optimizer.step()
                    model.eval()
                    last_train = t

                # Score: prediction error at t
                if t >= self.seq_len:
                    window = X_scaled[t - self.seq_len + 1:t]  # seq_len-1 steps
                    if len(window) < self.seq_len - 1:
                        continue
                    window_t = torch.FloatTensor(window).unsqueeze(0)
                    actual = torch.FloatTensor(X_scaled[t]).unsqueeze(0)
                    with torch.no_grad():
                        _, (h, _) = model.encoder(window_t)
                        h_rep = h.squeeze(0).unsqueeze(1)
                        out, _ = model.decoder(h_rep)
                        pred_out = out.squeeze(1)
                    pred_errors[t] = float(torch.mean((actual - pred_out) ** 2).item())

            z_scores = np.full(T, np.nan)
            for t in range(self.min_expanding, T):
                past = pred_errors[self.min_expanding:t]
                past = past[~np.isnan(past)]
                if len(past) < 10:
                    continue
                mu = np.mean(past)
                sigma = np.std(past, ddof=1)
                if sigma > 1e-12 and not np.isnan(pred_errors[t]):
                    z_scores[t] = abs((pred_errors[t] - mu) / sigma)
            return z_scores

    label = 'next_step_prediction'
    det = NextStepLSTMDetector(
        seq_len=20, latent_dim=16, n_epochs=10,
        min_expanding=120, retrain_interval=500, seed=SEED,
    )
    results, med = eval_detector(det, X_enriched, X_enriched, dates, label)
    verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
    print(f"  {label:<30} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
          f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
          f"  | median={med:.3f}  {verdict}")
    if med > best_med:
        best_med = med
        best_label = label
        best_results = results

    overall_verdict = "KEEP" if best_med > KEEP_THRESHOLD else "REJECT"
    return {
        'question': 'Q33: LSTM Autoencoder Fix',
        'best_variant': best_label,
        'crisis_d': best_results,
        'median_d': round(best_med, 3),
        'verdict': overall_verdict,
    }


# =============================================================================
# Q34: Kernel PCA variants
# =============================================================================

def test_q34(X, X_enriched, dates):
    """Test Kernel PCA with different kernels and hyperparameters.

    Variants:
      (a) polynomial kernel (degree 2, 3)
      (b) sigmoid kernel
      (c) RBF with different gamma (0.01, 0.1, 1.0, 'auto')
      (d) more components (8, 16, 32)
    """
    from experiments.baselines import KernelPCABaselineDetector

    print("\n" + "=" * 60)
    print("Q34: Kernel PCA Variants")
    print("=" * 60)

    best_label = None
    best_med = -1.0
    best_results = {}

    # (a) Polynomial kernel
    for degree in [2, 3]:
        label = f'poly_d{degree}'
        from sklearn.decomposition import KernelPCA as _KPCA
        from sklearn.preprocessing import StandardScaler as _SS

        class PolyKPCADetector(KernelPCABaselineDetector):
            def __init__(self, degree):
                super().__init__(n_components=8, rolling_window=20, min_expanding=60, seed=SEED)
                self._degree = degree

            @property
            def name(self):
                return f"Poly-KPCA-d{self._degree}"

            def fit(self, X, **kwargs):
                self._scaler = _SS()
                self._scaler.fit(X)
                X_scaled = self._scaler.transform(X)
                n_comp = min(self.n_components, X.shape[1])
                self._kpca = _KPCA(
                    n_components=n_comp, kernel='poly',
                    degree=self._degree, random_state=self.seed,
                )
                self._kpca.fit(X_scaled)
                return self

        det = PolyKPCADetector(degree=degree)
        results, med = eval_detector(det, X_enriched, X_enriched, dates, label)
        verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
        print(f"  {label:<28} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
              f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
              f"  | median={med:.3f}  {verdict}")
        if med > best_med:
            best_med = med
            best_label = label
            best_results = results

    # (b) Sigmoid kernel
    label = 'sigmoid'
    from sklearn.decomposition import KernelPCA as _KPCA
    from sklearn.preprocessing import StandardScaler as _SS

    class SigmoidKPCADetector(KernelPCABaselineDetector):
        @property
        def name(self):
            return "Sigmoid-KPCA"

        def fit(self, X, **kwargs):
            self._scaler = _SS()
            self._scaler.fit(X)
            X_scaled = self._scaler.transform(X)
            n_comp = min(self.n_components, X.shape[1])
            self._kpca = _KPCA(
                n_components=n_comp, kernel='sigmoid', random_state=self.seed,
            )
            self._kpca.fit(X_scaled)
            return self

    det = SigmoidKPCADetector(n_components=8, rolling_window=20, min_expanding=60, seed=SEED)
    results, med = eval_detector(det, X_enriched, X_enriched, dates, label)
    verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
    print(f"  {label:<28} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
          f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
          f"  | median={med:.3f}  {verdict}")
    if med > best_med:
        best_med = med
        best_label = label
        best_results = results

    # (c) RBF with different gamma
    for gamma in [0.01, 0.1, 1.0]:
        label = f'rbf_gamma={gamma}'
        det = KernelPCABaselineDetector(
            n_components=8, gamma=gamma, rolling_window=20, min_expanding=60, seed=SEED,
        )
        results, med = eval_detector(det, X_enriched, X_enriched, dates, label)
        verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
        print(f"  {label:<28} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
              f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
              f"  | median={med:.3f}  {verdict}")
        if med > best_med:
            best_med = med
            best_label = label
            best_results = results

    # (d) More components (RBF, auto gamma)
    for n_comp in [16, 32]:
        label = f'rbf_n_comp={n_comp}'
        det = KernelPCABaselineDetector(
            n_components=n_comp, gamma=None, rolling_window=20, min_expanding=60, seed=SEED,
        )
        results, med = eval_detector(det, X_enriched, X_enriched, dates, label)
        verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
        print(f"  {label:<28} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
              f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
              f"  | median={med:.3f}  {verdict}")
        if med > best_med:
            best_med = med
            best_label = label
            best_results = results

    overall_verdict = "KEEP" if best_med > KEEP_THRESHOLD else "REJECT"
    return {
        'question': 'Q34: Kernel PCA Fix',
        'best_variant': best_label,
        'crisis_d': best_results,
        'median_d': round(best_med, 3),
        'verdict': overall_verdict,
    }


# =============================================================================
# Q35: Spectral Flow variants
# =============================================================================

def test_q35(X, X_enriched, dates):
    """Test Spectral Flow with different gap thresholds and flow metrics.

    SpectralFlowDetector currently computes:
        flow[t] = ||spectra[t] - spectra[t-1]||_2

    Variants:
      (a) Total spectral velocity: sum of |d lambda_i / dt| (L1 instead of L2)
      (b) Normalized flow: flow / bandwidth (bandwidth = max(eig) - min(eig))
      (c) Spectral gap flow only: d(lambda_1 - lambda_2)/dt
      (d) Use 'random' operator_method instead of default 'pca_inspired'
          (per memory: pca_inspired causes Kramers degeneracy -> zero gap)
    """

    print("\n" + "=" * 60)
    print("Q35: Spectral Flow Variants")
    print("=" * 60)

    best_label = None
    best_med = -1.0
    best_results = {}

    # ---- Inline variant classes ----

    class SpectralFlowL1(SpectralFlowDetector):
        """L1 (sum of |d lambda_i/dt|) instead of L2 norm."""

        @property
        def name(self):
            return "Spectral Flow L1"

        def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
            if self._geometry is None:
                raise RuntimeError("Call fit() first.")
            Xt = _transform_array_sf(X, self)
            T = len(Xt)
            spectra = _compute_spectra(Xt, self)
            flow = np.full(T, np.nan)
            for t in range(1, T):
                flow[t] = np.sum(np.abs(spectra[t] - spectra[t - 1]))  # L1
            return _zscore_rolling(flow, self.rolling_window, self.min_expanding)

    class SpectralFlowNormalized(SpectralFlowDetector):
        """Flow normalized by spectral bandwidth at each step."""

        @property
        def name(self):
            return "Spectral Flow Normalized"

        def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
            if self._geometry is None:
                raise RuntimeError("Call fit() first.")
            Xt = _transform_array_sf(X, self)
            T = len(Xt)
            spectra = _compute_spectra(Xt, self)
            flow = np.full(T, np.nan)
            for t in range(1, T):
                bw = spectra[t].max() - spectra[t].min()
                if bw > 1e-12:
                    flow[t] = np.linalg.norm(spectra[t] - spectra[t - 1]) / bw
                else:
                    flow[t] = 0.0
            return _zscore_rolling(flow, self.rolling_window, self.min_expanding)

    class SpectralGapFlowDetector(SpectralFlowDetector):
        """Rate of change of spectral gap (lambda_1 - lambda_2)."""

        @property
        def name(self):
            return "Spectral Gap Flow"

        def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
            if self._geometry is None:
                raise RuntimeError("Call fit() first.")
            Xt = _transform_array_sf(X, self)
            T = len(Xt)
            spectra = _compute_spectra(Xt, self)
            gaps = np.array([sp[0] - sp[1] if len(sp) >= 2 else np.nan
                             for sp in spectra])
            flow = np.full(T, np.nan)
            for t in range(1, T):
                if np.isfinite(gaps[t]) and np.isfinite(gaps[t - 1]):
                    flow[t] = abs(gaps[t] - gaps[t - 1])
            return _zscore_rolling(flow, self.rolling_window, self.min_expanding)

    # Helper functions for the variant classes
    def _transform_array_sf(X_raw, det):
        """Apply scaler + PCA + normalization using detector's fitted components."""
        from qcml_geometry.observables import _apply_normalization, _transform_array
        return _transform_array(
            X_raw, det._scaler, det._pca,
            normalization=det.normalization,
            train_norms=det._train_norms,
            train_std=det._train_std,
        )

    def _compute_spectra(Xt, det):
        """Compute full spectrum at each time point."""
        T = len(Xt)
        spectra = []
        for t in range(T):
            if det._snapshots is not None:
                geo, xt = det._transform_point_at(X_raw[t], t)
                eigs = geo.full_spectrum(xt)
            else:
                eigs = det._geometry.full_spectrum(Xt[t])
            spectra.append(eigs)
        return np.array(spectra)

    def _zscore_rolling(flow, rolling_window, min_expanding):
        """Apply rolling mean + expanding z-score."""
        T = len(flow)
        rolling_vals = (
            pd.Series(flow)
            .rolling(window=rolling_window, min_periods=1)
            .mean()
            .values
        )
        z_scores = np.full(T, np.nan)
        for t in range(min_expanding, T):
            past = rolling_vals[1:t]
            past_valid = past[~np.isnan(past)]
            if len(past_valid) < 10:
                continue
            mu = np.mean(past_valid)
            sigma = np.std(past_valid, ddof=1)
            if sigma > 1e-12:
                z_scores[t] = (rolling_vals[t] - mu) / sigma
            else:
                z_scores[t] = 0.0
        return z_scores

    # (d) First: baseline with 'random' operator method (fixes pca_inspired degeneracy)
    label = 'random_ops_baseline'
    det_base = SpectralFlowDetector(
        hilbert_dim=8, n_pca_components=8, operator_method='random',
        rolling_window=15, min_expanding=60, seed=SEED, normalization='soft',
        adaptive_epsilon=True,
    )
    results, med = eval_detector(det_base, X_enriched, X_enriched, dates, label)
    verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
    print(f"  {label:<30} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
          f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
          f"  | median={med:.3f}  {verdict}")
    if med > best_med:
        best_med = med
        best_label = label
        best_results = results

    # (a) L1 total spectral velocity with 'random' ops
    label = 'L1_total_velocity'
    det_l1 = SpectralFlowL1(
        hilbert_dim=8, n_pca_components=8, operator_method='random',
        rolling_window=15, min_expanding=60, seed=SEED, normalization='soft',
        adaptive_epsilon=True,
    )
    results, med = eval_detector(det_l1, X_enriched, X_enriched, dates, label)
    verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
    print(f"  {label:<30} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
          f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
          f"  | median={med:.3f}  {verdict}")
    if med > best_med:
        best_med = med
        best_label = label
        best_results = results

    # (b) Normalized flow
    label = 'normalized_flow'
    det_norm = SpectralFlowNormalized(
        hilbert_dim=8, n_pca_components=8, operator_method='random',
        rolling_window=15, min_expanding=60, seed=SEED, normalization='soft',
        adaptive_epsilon=True,
    )
    results, med = eval_detector(det_norm, X_enriched, X_enriched, dates, label)
    verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
    print(f"  {label:<30} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
          f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
          f"  | median={med:.3f}  {verdict}")
    if med > best_med:
        best_med = med
        best_label = label
        best_results = results

    # (c) Gap flow only
    label = 'gap_flow_only'
    det_gap = SpectralGapFlowDetector(
        hilbert_dim=8, n_pca_components=8, operator_method='random',
        rolling_window=15, min_expanding=60, seed=SEED, normalization='soft',
        adaptive_epsilon=True,
    )
    results, med = eval_detector(det_gap, X_enriched, X_enriched, dates, label)
    verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
    print(f"  {label:<30} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
          f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
          f"  | median={med:.3f}  {verdict}")
    if med > best_med:
        best_med = med
        best_label = label
        best_results = results

    # Rolling window sweep with best operator method
    for rw in [5, 30]:
        label = f'random_rw={rw}'
        det = SpectralFlowDetector(
            hilbert_dim=8, n_pca_components=8, operator_method='random',
            rolling_window=rw, min_expanding=60, seed=SEED, normalization='soft',
            adaptive_epsilon=True,
        )
        results, med = eval_detector(det, X_enriched, X_enriched, dates, label)
        verdict = "KEEP" if med > KEEP_THRESHOLD else "REJECT"
        print(f"  {label:<30} | gfc={results['2008_gfc']:.3f}  covid={results['2020_covid']:.3f}"
              f"  rates={results['2022_rates']:.3f}  svb={results['2023_svb']:.3f}"
              f"  | median={med:.3f}  {verdict}")
        if med > best_med:
            best_med = med
            best_label = label
            best_results = results

    overall_verdict = "KEEP" if best_med > KEEP_THRESHOLD else "REJECT"
    return {
        'question': 'Q35: Spectral Flow Fix',
        'best_variant': best_label,
        'crisis_d': best_results,
        'median_d': round(best_med, 3),
        'verdict': overall_verdict,
    }


# =============================================================================
# Main
# =============================================================================

def print_summary(result):
    """Pretty-print a single question result."""
    print(f"\n{result['question']}")
    print(f"  Best variant:  {result['best_variant']}")
    cd = result['crisis_d']
    print(f"  Cohen's d:     gfc={cd.get('2008_gfc', 0):.3f}  "
          f"covid={cd.get('2020_covid', 0):.3f}  "
          f"rates={cd.get('2022_rates', 0):.3f}  "
          f"svb={cd.get('2023_svb', 0):.3f}")
    print(f"  Median d:      {result['median_d']:.3f}")
    print(f"  Verdict:       {result['verdict']}")
    if 'reason' in result:
        print(f"  Reason:        {result['reason']}")


def main():
    print("=" * 60)
    print("Dead Signal Resurrection: Q32–Q35 Smoke Test")
    print(f"Threshold: median Cohen's d > {KEEP_THRESHOLD}")
    print("=" * 60)

    X, dates_raw, X_enriched, dates = load_data()

    r32 = test_q32(X, dates_raw, X_enriched, dates)
    r33 = test_q33(X, X_enriched, dates)
    r34 = test_q34(X, X_enriched, dates)
    r35 = test_q35(X, X_enriched, dates)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for r in [r32, r33, r34, r35]:
        print_summary(r)

    print("\n" + "=" * 60)
    print("Resurrection candidates (median d > 0.30):")
    any_kept = False
    for r in [r32, r33, r34, r35]:
        if r['verdict'] == 'KEEP':
            print(f"  {r['question']}: {r['best_variant']} (d={r['median_d']:.3f})")
            any_kept = True
    if not any_kept:
        print("  None — all signals remain dead.")
    print("=" * 60)


if __name__ == '__main__':
    main()
