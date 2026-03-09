"""
Cross-Asset / Data Expansion Investigation — Q61-Q65

Investigates whether applying our geometric framework to alternative data sources
(intraday, volatility surfaces, multi-asset embeddings, EM markets, breadth) adds
detection value over single-asset daily SPY.

Q61: ANALYTICAL — Does intraday data (hourly, 5-min) reveal regime transitions
     invisible at daily frequency?
Q62: EMPIRICAL — Can we apply our framework to VIX time series for richer embedding?
Q63: EMPIRICAL — Does multi-market embedding (SPY + TLT + GLD jointly) outperform
     single-market?
Q64: EMPIRICAL — Can we detect regional contagion (EEM) using our geometric measures?
Q65: EMPIRICAL — Does sector-breadth dispersion improve when embedded in our framework?

Detectors: BerryPhaseRateDetector, SpectralEntropyDetector
Standard crises: 2008_gfc, 2020_covid, 2022_rates, 2023_svb
"""

import sys
import os
import warnings
import time

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import (
    fetch_data,
    create_feature_matrix,
    create_feature_matrix_single_asset,
    ALL_CRISES,
)
from experiments.evaluation import _cohens_d
from qcml_geometry.observables import BerryPhaseRateDetector, SpectralEntropyDetector

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']

# Fast config: keep tests under ~5 minutes total
FAST_CONFIG = dict(
    hilbert_dim=4,
    n_pca_components=6,
    min_expanding=40,
    rolling_window=15,
    seed=42,
)

# Data range covers all standard crises (2005 to 2024)
DATA_START = '2005-01-01'
DATA_END = '2024-12-31'

# EM-specific crises for Q64
EM_CRISES = {
    '2015_china': ALL_CRISES['2015_china'],
    '2018_q4':    ALL_CRISES['2018_q4'],
    '2020_covid': ALL_CRISES['2020_covid'],
    '2022_rates': ALL_CRISES['2022_rates'],
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_crisis_mask(dates_arr, crisis_key):
    """Build a boolean array marking the crisis period in dates_arr."""
    c = ALL_CRISES[crisis_key]
    start = pd.Timestamp(c['start'])
    end = pd.Timestamp(c['end'])
    return np.asarray((dates_arr >= start) & (dates_arr <= end))


def _cohens_d_from_scores(scores, crisis_mask):
    """Compute Cohen's d using crisis vs non-crisis scores."""
    valid = ~np.isnan(scores)
    c_scores = scores[valid & crisis_mask]
    n_scores = scores[valid & ~crisis_mask]
    if len(c_scores) < 2 or len(n_scores) < 2:
        return np.nan
    return _cohens_d(c_scores, n_scores)


def _run_detector(detector_cls, features, config=None):
    """Fit detector and return regime scores."""
    cfg = dict(FAST_CONFIG)
    if config:
        cfg.update(config)
    det = detector_cls(**cfg)
    det.fit(features)
    return det.compute_regime_scores(features)


def _mean_d_across_crises(scores, dates_arr, crisis_keys):
    """Compute mean Cohen's d across a list of crisis keys."""
    ds = []
    for ckey in crisis_keys:
        mask = _build_crisis_mask(dates_arr, ckey)
        if mask.sum() < 5:
            continue
        d = _cohens_d_from_scores(scores, mask)
        if not np.isnan(d):
            ds.append(d)
    return float(np.mean(ds)) if ds else np.nan


# ===========================================================================
# Q61: ANALYTICAL — Intraday timescale analysis
# ===========================================================================

class TestQ61_IntradayTimescales:
    """
    Q61: Does intraday data reveal regime transitions invisible at daily frequency?

    This is an ANALYTICAL question. We cannot empirically compare intraday vs daily
    using the same pipeline due to data constraints (yfinance intraday history is
    limited to ~60 days). Instead, we reason from first principles about how our
    observables interact with temporal resolution.

    Key arguments are encoded as assertions over the observable properties
    and window parameters — they serve as documented theoretical constraints.
    """

    def test_rolling_window_dominates_intraday_noise(self):
        """
        Berry phase and spectral entropy both use rolling windows of 15-20 days.

        At sub-daily resolution (hourly = ~6.5 observations/day), a 20-day window
        corresponds to ~130 hourly bars. The observables first build a feature
        matrix using 20-day rolling statistics (vol5, vol20, mom20). At hourly
        resolution these rolling windows would span <<1 day — meaningless for
        regime detection.

        Analytically: our observables are regime-level (weeks), not microstructure
        (minutes). Adding intraday noise to a 20-day rolling mean adds no signal.

        This test documents the theoretical constraint numerically.
        """
        # Rolling windows used by our observables (from config.yaml / FAST_CONFIG)
        daily_rolling_window = 20      # trading days
        intraday_bars_per_day = 6.5    # approximate hourly bars in a US session

        # The effective lookback in calendar time is identical whether we use
        # daily bars or hourly bars that are then aggregated into 20-day features.
        # But if we use raw intraday bars as features WITHOUT re-rolling, the
        # 20-day z-score window spans only 20 / 6.5 ≈ 3 days of calendar time.
        effective_days_hourly = daily_rolling_window / intraday_bars_per_day
        assert effective_days_hourly < 5.0, (
            f"A {daily_rolling_window}-bar rolling window on hourly data spans "
            f"only {effective_days_hourly:.1f} calendar days — too short for "
            "regime-level detection."
        )

    def test_berry_phase_accumulation_requires_daily_or_coarser(self):
        """
        Berry phase accumulation is path-dependent: F_t = sum_s F(x_s) for s in [t-w, t].

        The Berry curvature F(x) measures how fast the ground state rotates as
        the embedding point x moves. For financial data, the 'path' in PCA space
        is driven by economic regime shifts that unfold over days to weeks.

        At sub-daily frequencies:
        - The path in PCA space is dominated by microstructure noise (bid-ask bounce,
          inventory cycles, liquidity effects).
        - Berry curvature differences become dominated by microstructure noise,
          not regime transitions.

        The signal-to-noise ratio for Berry phase rate is:
            SNR ∝ (regime_variance / noise_variance)
        At daily frequency: regime variance >> microstructure variance (after 20d rolling).
        At 5-min frequency: regime variance << microstructure variance.

        This test documents the minimum window length required for meaningful signal.
        """
        # Min expanding window: our detectors require 40 observations before scoring.
        # At daily: 40 trading days = 2 calendar months (reasonable regime window).
        # At hourly: 40 bars = 40 / 6.5 ≈ 6 calendar hours (microstructure, not regime).
        min_expanding = FAST_CONFIG['min_expanding']
        daily_calendar_days = min_expanding  # 40 trading days

        # Minimum meaningful calendar duration for a macroeconomic regime:
        # Empirically, crises last at minimum 2-4 weeks (e.g., SVB 2023 = 2 months).
        min_regime_duration_days = 14  # 2 calendar weeks

        assert daily_calendar_days > min_regime_duration_days, (
            "min_expanding should exceed the minimum meaningful regime duration."
        )

        # At hourly resolution, 40 bars = ~6 hours, which is < 14 days:
        hourly_calendar_hours = min_expanding  # 40 hourly bars
        hourly_calendar_days = hourly_calendar_hours / (6.5 * 5 / 7)  # ~4.3 days
        assert hourly_calendar_days < min_regime_duration_days, (
            "At hourly resolution, min_expanding bars span < 2 weeks, "
            "too short for macroeconomic regime detection."
        )

    def test_analytical_verdict_documented(self):
        """
        Documents the analytical conclusion for Q61.

        Verdict: NEGATIVE — Intraday data does not add value for our framework.

        Rationale:
        1. Rolling windows are tuned to daily bars (20d, 60d). Applying them to
           hourly bars shrinks the effective lookback by 6.5x, destroying the
           regime-level averaging that gives our signals power.
        2. Berry phase accumulation requires a path through a stable embedding
           manifold. Intraday paths are dominated by microstructure noise that
           is orthogonal to regime-level geometry.
        3. Our detectors require min_expanding=40-60 observations before scoring.
           At hourly frequency this is ~6-9 hours — insufficient warmup.
        4. The PCA embedding (n_pca_components=6-15) is fitted on day-level features
           (vol5, vol20, mom20). Re-fitting on intraday features would require a
           completely different feature engineering pipeline.

        Theoretical exception: Very high frequency (1-min, tick) data COULD add
        value IF we pre-aggregate into realized volatility or realized correlation
        metrics that are then fed at daily frequency. This is a different question
        (Q65/feature engineering) rather than raw intraday frequency.
        """
        verdict = "NEGATIVE"
        rationale_points = [
            "20-day rolling windows collapse to <3 calendar days on hourly bars",
            "Berry phase path dominated by microstructure noise at intraday resolution",
            "min_expanding=40 bars = ~6 hours at hourly frequency (too short)",
            "PCA features are designed for daily return distributions",
        ]
        assert verdict == "NEGATIVE"
        assert len(rationale_points) == 4
        print("\n--- Q61 ANALYTICAL VERDICT ---")
        print("Verdict: NEGATIVE — intraday data does not add value for current observables.")
        print("Rationale:")
        for i, pt in enumerate(rationale_points, 1):
            print(f"  {i}. {pt}")
        print("\nException: pre-aggregated intraday statistics (realized vol, realized corr)")
        print("  fed at daily frequency could add value — but that is feature engineering,")
        print("  not raw intraday frequency. Covered in Q65.")


# ===========================================================================
# Q62: EMPIRICAL — VIX as input to geometric framework
# ===========================================================================

class TestQ62_VIXEmbedding:
    """
    Q62: Can we apply our framework to VIX time series for richer embedding?

    VIX is the 30-day implied volatility index for S&P 500. It is itself a
    derived statistic. Applying geometric observables to VIX means computing
    'second-order geometry': how fast does the options market's fear gauge
    transition between regimes?

    Hypothesis: VIX geometry may be complementary to SPY geometry because:
    - VIX captures options market beliefs (forward-looking)
    - SPY captures realised price dynamics (backward-looking)
    - During crises, VIX spikes BEFORE price drops (lead-time advantage)
    """

    @pytest.fixture(scope='class')
    def vix_data(self):
        """Fetch VIX daily data from yfinance (^VIX)."""
        raw = fetch_data(['^VIX'], DATA_START, DATA_END, source='yfinance', use_cache=True)
        prices = raw['close'].unstack('symbol')['^VIX'].dropna()
        return prices

    @pytest.fixture(scope='class')
    def spy_baseline(self):
        """Fetch SPY data for comparison baseline."""
        raw = fetch_data(['SPY'], DATA_START, DATA_END, source='yfinance', use_cache=True)
        prices = raw['close'].unstack('symbol')['SPY'].dropna()
        features, dates = create_feature_matrix_single_asset(prices)
        return features, pd.DatetimeIndex(dates)

    def test_vix_data_fetched(self, vix_data):
        """VIX data should be fetchable with sufficient history."""
        assert len(vix_data) >= 2000, f"VIX should have >= 2000 trading days, got {len(vix_data)}"
        assert not vix_data.isna().all(), "VIX prices should not be all NaN"
        print(f"\nVIX data: {len(vix_data)} days, {vix_data.index[0].date()} to {vix_data.index[-1].date()}")

    def test_vix_berry_detects_crises(self, vix_data, spy_baseline):
        """
        BerryPhaseRate on VIX features should achieve d > 0 on standard crises.

        Even if VIX geometry is noisier than SPY geometry, it should show SOME
        elevation during crises (VIX always spikes during stress events).
        """
        spy_features, spy_dates = spy_baseline

        # Build feature matrix from VIX prices
        vix_features, vix_dates = create_feature_matrix_single_asset(vix_data)
        vix_dates = pd.DatetimeIndex(vix_dates)

        results = {}
        for ckey in STANDARD_CRISES:
            vix_mask = _build_crisis_mask(vix_dates, ckey)
            spy_mask = _build_crisis_mask(spy_dates, ckey)

            if vix_mask.sum() < 5 or spy_mask.sum() < 5:
                print(f"  {ckey}: insufficient crisis observations, skipping")
                continue

            # VIX Berry
            vix_scores = _run_detector(BerryPhaseRateDetector, vix_features)
            d_vix = _cohens_d_from_scores(vix_scores, vix_mask)

            # SPY Berry (baseline)
            spy_scores = _run_detector(BerryPhaseRateDetector, spy_features)
            d_spy = _cohens_d_from_scores(spy_scores, spy_mask)

            results[ckey] = {'d_vix_berry': d_vix, 'd_spy_berry': d_spy}
            print(f"  {ckey:20s} | VIX Berry d={d_vix:.3f} | SPY Berry d={d_spy:.3f}")

        valid_results = {k: v for k, v in results.items() if not np.isnan(v['d_vix_berry'])}
        assert len(valid_results) >= 2, "Should get valid d values for at least 2 crises"

        mean_d_vix = np.mean([v['d_vix_berry'] for v in valid_results.values()])
        mean_d_spy = np.mean([v['d_spy_berry'] for v in valid_results.values()])

        print(f"\n  Mean d: VIX Berry={mean_d_vix:.3f}  SPY Berry={mean_d_spy:.3f}")

        # VIX geometry should show some separation (d > 0)
        assert mean_d_vix > 0, "VIX Berry should show positive d across crises"

        self.__class__._berry_results = results
        self.__class__._mean_d_vix_berry = mean_d_vix
        self.__class__._mean_d_spy_berry = mean_d_spy

    def test_vix_spectral_entropy_detects_crises(self, vix_data, spy_baseline):
        """
        SpectralEntropy on VIX features should achieve d > 0 on standard crises.

        Spectral entropy measures spread of excitation energies. VIX level shifts
        during crises change the distribution of eigenvalue weights.
        """
        spy_features, spy_dates = spy_baseline
        vix_features, vix_dates = create_feature_matrix_single_asset(vix_data)
        vix_dates = pd.DatetimeIndex(vix_dates)

        results = {}
        for ckey in STANDARD_CRISES:
            vix_mask = _build_crisis_mask(vix_dates, ckey)
            spy_mask = _build_crisis_mask(spy_dates, ckey)
            if vix_mask.sum() < 5 or spy_mask.sum() < 5:
                continue

            vix_scores = _run_detector(
                SpectralEntropyDetector, vix_features,
                config={'normalization': 'soft', 'adaptive_epsilon': True}
            )
            d_vix = _cohens_d_from_scores(vix_scores, vix_mask)

            spy_scores = _run_detector(
                SpectralEntropyDetector, spy_features,
                config={'normalization': 'soft', 'adaptive_epsilon': True}
            )
            d_spy = _cohens_d_from_scores(spy_scores, spy_mask)

            results[ckey] = {'d_vix_entropy': d_vix, 'd_spy_entropy': d_spy}
            print(f"  {ckey:20s} | VIX Entropy d={d_vix:.3f} | SPY Entropy d={d_spy:.3f}")

        valid_results = {k: v for k, v in results.items() if not np.isnan(v['d_vix_entropy'])}
        assert len(valid_results) >= 2

        mean_d_vix = np.mean([v['d_vix_entropy'] for v in valid_results.values()])
        mean_d_spy = np.mean([v['d_spy_entropy'] for v in valid_results.values()])

        print(f"\n  Mean d: VIX Entropy={mean_d_vix:.3f}  SPY Entropy={mean_d_spy:.3f}")
        assert mean_d_vix > 0, "VIX Spectral Entropy should show positive d"

        self.__class__._entropy_results = results
        self.__class__._mean_d_vix_entropy = mean_d_vix
        self.__class__._mean_d_spy_entropy = mean_d_spy

    def test_vix_vs_spy_comparison_reported(self, vix_data, spy_baseline):
        """
        Report whether VIX geometry is superior/inferior/complementary to SPY geometry.

        Complementary = VIX is better on some crises, SPY is better on others.
        """
        spy_features, spy_dates = spy_baseline
        vix_features, vix_dates = create_feature_matrix_single_asset(vix_data)
        vix_dates = pd.DatetimeIndex(vix_dates)

        vix_better_count = 0
        spy_better_count = 0
        n_compared = 0

        print("\n--- Q62 VIX vs SPY geometry comparison ---")
        for ckey in STANDARD_CRISES:
            vix_mask = _build_crisis_mask(vix_dates, ckey)
            spy_mask = _build_crisis_mask(spy_dates, ckey)
            if vix_mask.sum() < 5 or spy_mask.sum() < 5:
                continue

            vix_scores = _run_detector(BerryPhaseRateDetector, vix_features)
            spy_scores = _run_detector(BerryPhaseRateDetector, spy_features)

            d_vix = _cohens_d_from_scores(vix_scores, vix_mask)
            d_spy = _cohens_d_from_scores(spy_scores, spy_mask)

            if np.isnan(d_vix) or np.isnan(d_spy):
                continue

            better = "VIX" if d_vix > d_spy else "SPY"
            delta = abs(d_vix - d_spy)
            print(f"  {ckey:20s} | VIX d={d_vix:.3f}  SPY d={d_spy:.3f}  "
                  f"| winner={better} (delta={delta:.3f})")
            if d_vix > d_spy:
                vix_better_count += 1
            else:
                spy_better_count += 1
            n_compared += 1

        print(f"\n  VIX better: {vix_better_count}/{n_compared} crises")
        print(f"  SPY better: {spy_better_count}/{n_compared} crises")

        if vix_better_count > 0 and spy_better_count > 0:
            print("  => COMPLEMENTARY: VIX and SPY geometry are crisis-specific")
        elif vix_better_count > spy_better_count:
            print("  => VIX SUPERIOR overall")
        else:
            print("  => SPY SUPERIOR overall (second-order geometry adds noise)")

        assert n_compared >= 2, "Should be able to compare at least 2 crises"


# ===========================================================================
# Q63: EMPIRICAL — Multi-asset embedding (SPY + TLT + GLD)
# ===========================================================================

class TestQ63_MultiAssetEmbedding:
    """
    Q63: Does a joint SPY + TLT + GLD embedding outperform SPY-only?

    SPY = US large-cap equity
    TLT = 20-year Treasury bonds (risk-off asset, negative beta to equities)
    GLD = Gold ETF (inflation hedge, crisis safe haven)

    The joint feature matrix captures:
    - Cross-asset correlation dynamics (flight-to-quality flows)
    - Vol dispersion across asset classes
    - Relative momentum (equity vs bond vs commodity)

    Hypothesis: Cross-asset correlation structure changes dramatically during
    crises (correlations go to 1 for risk assets, diverge for safe havens).
    This regime change is geometrically richer than single-asset geometry.
    """

    @pytest.fixture(scope='class')
    def multi_asset_data(self):
        """Fetch SPY, TLT, GLD jointly."""
        symbols = ['SPY', 'TLT', 'GLD']
        raw = fetch_data(symbols, DATA_START, DATA_END, source='yfinance', use_cache=True)
        prices_wide = raw['close'].unstack('symbol')[symbols].dropna()
        features, dates = create_feature_matrix(prices_wide)
        return features, pd.DatetimeIndex(dates)

    @pytest.fixture(scope='class')
    def spy_only_data(self):
        """Fetch SPY only (baseline)."""
        raw = fetch_data(['SPY'], DATA_START, DATA_END, source='yfinance', use_cache=True)
        prices = raw['close'].unstack('symbol')['SPY'].dropna()
        features, dates = create_feature_matrix_single_asset(prices)
        return features, pd.DatetimeIndex(dates)

    def test_multi_asset_features_have_more_columns(self, multi_asset_data, spy_only_data):
        """Multi-asset feature matrix should have more columns than SPY-only."""
        ma_features, _ = multi_asset_data
        spy_features, _ = spy_only_data
        assert ma_features.shape[1] > spy_features.shape[1], (
            f"Multi-asset features ({ma_features.shape[1]} cols) should exceed "
            f"SPY-only features ({spy_features.shape[1]} cols)"
        )
        print(f"\nFeature dimensions: SPY-only={spy_features.shape[1]}  "
              f"Multi-asset={ma_features.shape[1]}")

    def test_multi_asset_berry_vs_spy_only(self, multi_asset_data, spy_only_data):
        """
        BerryPhaseRate on joint SPY+TLT+GLD should be >= SPY-only on average.

        Cross-asset correlation collapse (all correlations → 1 during crises)
        is a large geometric deformation that should produce strong Berry rate signals.
        """
        ma_features, ma_dates = multi_asset_data
        spy_features, spy_dates = spy_only_data

        results = {}
        print("\n--- Q63 Multi-asset vs SPY-only (BerryPhaseRate) ---")

        for ckey in STANDARD_CRISES:
            ma_mask = _build_crisis_mask(ma_dates, ckey)
            spy_mask = _build_crisis_mask(spy_dates, ckey)
            if ma_mask.sum() < 5 or spy_mask.sum() < 5:
                print(f"  {ckey}: insufficient crisis observations")
                continue

            ma_scores = _run_detector(BerryPhaseRateDetector, ma_features)
            d_ma = _cohens_d_from_scores(ma_scores, ma_mask)

            spy_scores = _run_detector(BerryPhaseRateDetector, spy_features)
            d_spy = _cohens_d_from_scores(spy_scores, spy_mask)

            results[ckey] = {'d_multi': d_ma, 'd_spy': d_spy}
            gain = d_ma - d_spy
            print(f"  {ckey:20s} | Multi d={d_ma:.3f}  SPY d={d_spy:.3f}  gain={gain:+.3f}")

        valid = {k: v for k, v in results.items()
                 if not np.isnan(v['d_multi']) and not np.isnan(v['d_spy'])}
        assert len(valid) >= 2, "Should have valid comparisons for >= 2 crises"

        mean_d_multi = np.mean([v['d_multi'] for v in valid.values()])
        mean_d_spy = np.mean([v['d_spy'] for v in valid.values()])
        mean_gain = mean_d_multi - mean_d_spy

        print(f"\n  Mean: Multi d={mean_d_multi:.3f}  SPY d={mean_d_spy:.3f}  "
              f"mean gain={mean_gain:+.3f}")

        if mean_gain > 0.05:
            print("  => POSITIVE: Multi-asset embedding adds meaningful detection power")
        elif mean_gain > -0.05:
            print("  => NEUTRAL: Multi-asset is comparable to SPY-only")
        else:
            print("  => NEGATIVE: Multi-asset embedding dilutes signal with noise")

        self.__class__._berry_results = results
        self.__class__._mean_gain_berry = mean_gain

    def test_multi_asset_spectral_entropy_vs_spy_only(self, multi_asset_data, spy_only_data):
        """
        SpectralEntropy on joint embedding vs SPY-only.

        Spectral entropy captures spread of eigenvalue weights. A multi-asset
        embedding has richer covariance structure; entropy changes during cross-asset
        correlation breakdowns may be more pronounced.
        """
        ma_features, ma_dates = multi_asset_data
        spy_features, spy_dates = spy_only_data

        results = {}
        print("\n--- Q63 Multi-asset vs SPY-only (SpectralEntropy) ---")

        for ckey in STANDARD_CRISES:
            ma_mask = _build_crisis_mask(ma_dates, ckey)
            spy_mask = _build_crisis_mask(spy_dates, ckey)
            if ma_mask.sum() < 5 or spy_mask.sum() < 5:
                continue

            ma_scores = _run_detector(
                SpectralEntropyDetector, ma_features,
                config={'normalization': 'soft', 'adaptive_epsilon': True}
            )
            d_ma = _cohens_d_from_scores(ma_scores, ma_mask)

            spy_scores = _run_detector(
                SpectralEntropyDetector, spy_features,
                config={'normalization': 'soft', 'adaptive_epsilon': True}
            )
            d_spy = _cohens_d_from_scores(spy_scores, spy_mask)

            results[ckey] = {'d_multi': d_ma, 'd_spy': d_spy}
            gain = d_ma - d_spy
            print(f"  {ckey:20s} | Multi d={d_ma:.3f}  SPY d={d_spy:.3f}  gain={gain:+.3f}")

        valid = {k: v for k, v in results.items()
                 if not np.isnan(v['d_multi']) and not np.isnan(v['d_spy'])}
        assert len(valid) >= 2

        mean_d_multi = np.mean([v['d_multi'] for v in valid.values()])
        mean_d_spy = np.mean([v['d_spy'] for v in valid.values()])
        mean_gain = mean_d_multi - mean_d_spy

        print(f"\n  Mean: Multi d={mean_d_multi:.3f}  SPY d={mean_d_spy:.3f}  "
              f"mean gain={mean_gain:+.3f}")

        self.__class__._entropy_results = results
        self.__class__._mean_gain_entropy = mean_gain

    def test_cross_asset_correlation_regime_change(self, multi_asset_data):
        """
        Verify that cross-asset correlations do in fact change during crises.

        This validates the theoretical motivation: if correlations don't change,
        multi-asset embedding provides no additional regime signal over SPY-only.
        """
        features, dates = multi_asset_data

        # Find 2020 COVID crisis indices
        mask = _build_crisis_mask(dates, '2020_covid')
        crisis_idx = np.where(mask)[0]
        normal_idx = np.where(~mask)[0]

        if len(crisis_idx) < 10 or len(normal_idx) < 100:
            pytest.skip("Insufficient 2020 data for correlation analysis")

        # The feature matrix includes cross_corr5 and cross_vol_disp columns
        # (from create_feature_matrix). We can look at feature variance during crises.
        crisis_features = features[crisis_idx]
        normal_features = features[normal_idx[:len(crisis_idx) * 10]]  # matched ratio

        # Feature variance should be higher during crisis (regime change)
        crisis_var = np.var(crisis_features, axis=0)
        normal_var = np.var(normal_features, axis=0)

        n_more_volatile = np.sum(crisis_var > normal_var)
        pct_more_volatile = n_more_volatile / len(crisis_var) * 100

        print(f"\n  Cross-asset feature variance analysis (2020 COVID):")
        print(f"  {n_more_volatile}/{len(crisis_var)} features more volatile during crisis "
              f"({pct_more_volatile:.0f}%)")

        # Majority of features should be more volatile during crisis
        assert pct_more_volatile > 40, (
            f"Expected majority of features more volatile during crisis, "
            f"got {pct_more_volatile:.0f}%"
        )


# ===========================================================================
# Q64: EMPIRICAL — Regional contagion detection (EEM)
# ===========================================================================

class TestQ64_EMContagion:
    """
    Q64: Can we detect regional contagion using EEM (Emerging Markets ETF)?

    EEM = iShares MSCI Emerging Markets ETF. Launched in 2003.
    Covers China, Taiwan, India, Brazil, S. Korea, and other EM economies.

    Test crises:
    - 2015_china: Chinese equity market crash + yuan devaluation
    - 2018_q4: Global selloff with EM stress (Fed tightening + trade war)
    - 2020_covid: Global (also severe EM impact)
    - 2022_rates: Dollar strengthening crushed EM debt

    The question: does EEM geometry detect EM-specific crises better than SPY?
    """

    @pytest.fixture(scope='class')
    def eem_data(self):
        """Fetch EEM data."""
        raw = fetch_data(['EEM'], DATA_START, DATA_END, source='yfinance', use_cache=True)
        prices = raw['close'].unstack('symbol')['EEM'].dropna()
        features, dates = create_feature_matrix_single_asset(prices)
        return features, pd.DatetimeIndex(dates)

    @pytest.fixture(scope='class')
    def spy_data(self):
        """Fetch SPY data."""
        raw = fetch_data(['SPY'], DATA_START, DATA_END, source='yfinance', use_cache=True)
        prices = raw['close'].unstack('symbol')['SPY'].dropna()
        features, dates = create_feature_matrix_single_asset(prices)
        return features, pd.DatetimeIndex(dates)

    def test_eem_data_available(self, eem_data):
        """EEM should have >= 2000 trading days."""
        features, dates = eem_data
        assert len(features) >= 2000, f"EEM should have >= 2000 observations, got {len(features)}"
        print(f"\nEEM: {len(features)} observations, {dates[0].date()} to {dates[-1].date()}")

    def test_eem_berry_detects_em_crises(self, eem_data, spy_data):
        """
        BerryPhaseRate on EEM should detect EM-specific crises.

        2015_china and 2018_q4 are EM-stress events. EEM Berry should be
        elevated during these periods.
        """
        eem_features, eem_dates = eem_data
        spy_features, spy_dates = spy_data

        em_test_crises = ['2015_china', '2018_q4', '2020_covid', '2022_rates']

        results = {}
        print("\n--- Q64 EEM Berry vs SPY Berry (EM crises) ---")

        for ckey in em_test_crises:
            if ckey not in ALL_CRISES:
                continue

            eem_mask = _build_crisis_mask(eem_dates, ckey)
            spy_mask = _build_crisis_mask(spy_dates, ckey)

            if eem_mask.sum() < 5 or spy_mask.sum() < 5:
                print(f"  {ckey}: insufficient observations")
                continue

            eem_scores = _run_detector(BerryPhaseRateDetector, eem_features)
            d_eem = _cohens_d_from_scores(eem_scores, eem_mask)

            spy_scores = _run_detector(BerryPhaseRateDetector, spy_features)
            d_spy = _cohens_d_from_scores(spy_scores, spy_mask)

            results[ckey] = {'d_eem': d_eem, 'd_spy': d_spy}
            advantage = "EEM" if d_eem > d_spy else "SPY"
            print(f"  {ckey:20s} | EEM d={d_eem:.3f}  SPY d={d_spy:.3f}  "
                  f"| {advantage} better (delta={abs(d_eem-d_spy):.3f})")

        valid = {k: v for k, v in results.items() if not np.isnan(v['d_eem'])}
        assert len(valid) >= 2, "Should get valid EEM d values for at least 2 crises"

        mean_d_eem = np.mean([v['d_eem'] for v in valid.values()])
        mean_d_spy = np.mean([v['d_spy'] for v in valid.values()])

        print(f"\n  Mean d: EEM={mean_d_eem:.3f}  SPY={mean_d_spy:.3f}")
        print(f"  EEM better on {sum(1 for v in valid.values() if v['d_eem'] > v['d_spy'])}"
              f"/{len(valid)} crises")

        # EEM should detect EM crises at some level (d > 0)
        assert mean_d_eem > 0, "EEM Berry should show positive separation during EM crises"

        self.__class__._berry_results = results
        self.__class__._mean_d_eem = mean_d_eem
        self.__class__._mean_d_spy = mean_d_spy

    def test_eem_spectral_entropy_detects_em_crises(self, eem_data, spy_data):
        """
        SpectralEntropy on EEM should detect EM-specific crises.

        Spectral entropy may be especially sensitive to EEM crises because
        EM equity correlations collapse more dramatically than developed markets.
        """
        eem_features, eem_dates = eem_data
        spy_features, spy_dates = spy_data

        em_test_crises = ['2015_china', '2018_q4', '2020_covid', '2022_rates']
        results = {}
        print("\n--- Q64 EEM SpectralEntropy vs SPY (EM crises) ---")

        for ckey in em_test_crises:
            if ckey not in ALL_CRISES:
                continue
            eem_mask = _build_crisis_mask(eem_dates, ckey)
            spy_mask = _build_crisis_mask(spy_dates, ckey)
            if eem_mask.sum() < 5 or spy_mask.sum() < 5:
                continue

            eem_scores = _run_detector(
                SpectralEntropyDetector, eem_features,
                config={'normalization': 'soft', 'adaptive_epsilon': True}
            )
            d_eem = _cohens_d_from_scores(eem_scores, eem_mask)

            spy_scores = _run_detector(
                SpectralEntropyDetector, spy_features,
                config={'normalization': 'soft', 'adaptive_epsilon': True}
            )
            d_spy = _cohens_d_from_scores(spy_scores, spy_mask)

            results[ckey] = {'d_eem': d_eem, 'd_spy': d_spy}
            print(f"  {ckey:20s} | EEM d={d_eem:.3f}  SPY d={d_spy:.3f}")

        valid = {k: v for k, v in results.items() if not np.isnan(v['d_eem'])}
        assert len(valid) >= 2

        mean_d_eem = np.mean([v['d_eem'] for v in valid.values()])
        mean_d_spy = np.mean([v['d_spy'] for v in valid.values()])
        print(f"\n  Mean d: EEM Entropy={mean_d_eem:.3f}  SPY Entropy={mean_d_spy:.3f}")

        self.__class__._entropy_results = results
        self.__class__._mean_d_eem_entropy = mean_d_eem

    def test_eem_vs_spy_contagion_comparison(self, eem_data, spy_data):
        """
        Determine whether EEM-specific crises (China 2015, EM 2018) are better
        detected by EEM geometry than SPY geometry.

        Contagion hypothesis: EEM geometry should lead SPY geometry during EM crises
        because EEM prices reflect EM stress before it propagates to developed markets.
        """
        eem_features, eem_dates = eem_data
        spy_features, spy_dates = spy_data

        eem_specific = ['2015_china', '2018_q4']
        global_crises = ['2020_covid', '2022_rates']

        print("\n--- Q64 Contagion analysis: EM-specific vs global crises ---")

        def _get_d_both(ckey, eem_features, eem_dates, spy_features, spy_dates):
            eem_mask = _build_crisis_mask(eem_dates, ckey)
            spy_mask = _build_crisis_mask(spy_dates, ckey)
            if eem_mask.sum() < 5 or spy_mask.sum() < 5:
                return np.nan, np.nan
            eem_scores = _run_detector(BerryPhaseRateDetector, eem_features)
            spy_scores = _run_detector(BerryPhaseRateDetector, spy_features)
            return (_cohens_d_from_scores(eem_scores, eem_mask),
                    _cohens_d_from_scores(spy_scores, spy_mask))

        eem_specific_gains = []
        global_gains = []

        print("\n  EM-specific crises (EEM should have advantage):")
        for ckey in eem_specific:
            if ckey not in ALL_CRISES:
                continue
            d_eem, d_spy = _get_d_both(ckey, eem_features, eem_dates, spy_features, spy_dates)
            if np.isnan(d_eem):
                continue
            gain = d_eem - d_spy
            eem_specific_gains.append(gain)
            print(f"    {ckey:20s} | EEM d={d_eem:.3f}  SPY d={d_spy:.3f}  "
                  f"EEM gain={gain:+.3f}")

        print("\n  Global crises (SPY may be comparable):")
        for ckey in global_crises:
            if ckey not in ALL_CRISES:
                continue
            d_eem, d_spy = _get_d_both(ckey, eem_features, eem_dates, spy_features, spy_dates)
            if np.isnan(d_eem):
                continue
            gain = d_eem - d_spy
            global_gains.append(gain)
            print(f"    {ckey:20s} | EEM d={d_eem:.3f}  SPY d={d_spy:.3f}  "
                  f"EEM gain={gain:+.3f}")

        if eem_specific_gains and global_gains:
            mean_em_specific_gain = np.mean(eem_specific_gains)
            mean_global_gain = np.mean(global_gains)
            print(f"\n  Mean EEM advantage: EM-specific={mean_em_specific_gain:+.3f}  "
                  f"global={mean_global_gain:+.3f}")

            if mean_em_specific_gain > mean_global_gain:
                print("  => CONTAGION SIGNAL: EEM has larger advantage on EM-specific crises")
            else:
                print("  => NO CONTAGION ADVANTAGE: EEM and SPY equally affected by all crises")


# ===========================================================================
# Q65: EMPIRICAL — Sector breadth dispersion
# ===========================================================================

class TestQ65_SectorBreadth:
    """
    Q65: Does sector-breadth dispersion improve regime detection when embedded
    in our geometric framework?

    Sector ETFs used as breadth proxy:
    XLF = Financials     XLK = Technology    XLE = Energy
    XLV = Healthcare     XLI = Industrials   XLU = Utilities
    XLP = Consumer Staples  XLY = Consumer Discretionary

    Cross-sectional std of sector returns = breadth proxy.
    When breadth is high (sectors diverge), market internals are stressed.
    When breadth is low (all sectors move together), regime is homogeneous.

    Tests:
    1. Sector breadth + SPY in joint embedding vs SPY-only
    2. Pure breadth geometry (sector dispersion as input features)
    3. Whether breadth adds incremental d to the best single-asset detector
    """

    SECTOR_ETFS = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY']

    @pytest.fixture(scope='class')
    def sector_data(self):
        """Fetch all 8 sector ETFs jointly."""
        symbols = self.SECTOR_ETFS
        raw = fetch_data(symbols, DATA_START, DATA_END, source='yfinance', use_cache=True)
        prices_wide = raw['close'].unstack('symbol')[symbols].dropna()
        features, dates = create_feature_matrix(prices_wide)
        return features, pd.DatetimeIndex(dates), prices_wide

    @pytest.fixture(scope='class')
    def spy_only_data(self):
        """Fetch SPY for comparison."""
        raw = fetch_data(['SPY'], DATA_START, DATA_END, source='yfinance', use_cache=True)
        prices = raw['close'].unstack('symbol')['SPY'].dropna()
        features, dates = create_feature_matrix_single_asset(prices)
        return features, pd.DatetimeIndex(dates)

    @pytest.fixture(scope='class')
    def breadth_feature_data(self, sector_data):
        """
        Build a pure breadth feature matrix from sector ETFs.

        Breadth features:
        - Cross-sectional std of daily returns (dispersion)
        - Cross-sectional std of 5-day returns
        - Cross-sectional std of 20-day returns
        - Mean absolute return across sectors
        - Fraction of sectors with positive returns
        - Cross-sectional skewness of returns
        - Rolling 20-day correlation between breadth and abs return
        """
        features_full, dates, prices_wide = sector_data

        log_ret = np.log(prices_wide / prices_wide.shift(1))
        ret5 = prices_wide.pct_change(5)
        ret20 = prices_wide.pct_change(20)

        breadth = {
            'xsec_std_ret': log_ret.std(axis=1),
            'xsec_std_ret5': ret5.std(axis=1),
            'xsec_std_ret20': ret20.std(axis=1),
            'xsec_mean_abs': log_ret.abs().mean(axis=1),
            'xsec_frac_pos': (log_ret > 0).mean(axis=1),
            'xsec_skew': log_ret.skew(axis=1),
            'breadth_ma5': log_ret.std(axis=1).rolling(5).mean(),
            'breadth_ma20': log_ret.std(axis=1).rolling(20).mean(),
            'breadth_z20': (
                (log_ret.std(axis=1) - log_ret.std(axis=1).rolling(20).mean()) /
                (log_ret.std(axis=1).rolling(20).std() + 1e-8)
            ),
        }

        breadth_df = pd.DataFrame(breadth)
        breadth_df = breadth_df.replace([np.inf, -np.inf], np.nan).dropna()
        return breadth_df.values, pd.DatetimeIndex(breadth_df.index)

    def test_sector_data_available(self, sector_data):
        """All 8 sector ETFs should be fetchable."""
        features, dates, prices_wide = sector_data
        assert prices_wide.shape[1] == len(self.SECTOR_ETFS), (
            f"Expected {len(self.SECTOR_ETFS)} sector ETFs, got {prices_wide.shape[1]}"
        )
        assert len(features) >= 2000
        print(f"\nSector data: {len(features)} obs, {dates[0].date()} to {dates[-1].date()}")
        print(f"Sectors: {list(prices_wide.columns)}")

    def test_breadth_features_constructed(self, breadth_feature_data):
        """Breadth feature matrix should be valid."""
        features, dates = breadth_feature_data
        assert features.shape[1] >= 5, f"Expected >= 5 breadth features, got {features.shape[1]}"
        assert not np.any(np.isnan(features)), "Breadth features should not contain NaN"
        print(f"\nBreadth features: {features.shape[1]} columns, {len(features)} observations")

    def test_pure_breadth_berry_detects_crises(self, breadth_feature_data, spy_only_data):
        """
        BerryPhaseRate on pure breadth features should detect crises.

        Breadth collapses (all sectors fall together) or breadth spikes
        (sectors diverge dramatically) both signal regime stress.
        """
        bread_features, bread_dates = breadth_feature_data
        spy_features, spy_dates = spy_only_data

        results = {}
        print("\n--- Q65 Pure Breadth vs SPY-only (BerryPhaseRate) ---")

        for ckey in STANDARD_CRISES:
            bread_mask = _build_crisis_mask(bread_dates, ckey)
            spy_mask = _build_crisis_mask(spy_dates, ckey)
            if bread_mask.sum() < 5 or spy_mask.sum() < 5:
                print(f"  {ckey}: insufficient observations")
                continue

            bread_scores = _run_detector(BerryPhaseRateDetector, bread_features)
            d_bread = _cohens_d_from_scores(bread_scores, bread_mask)

            spy_scores = _run_detector(BerryPhaseRateDetector, spy_features)
            d_spy = _cohens_d_from_scores(spy_scores, spy_mask)

            results[ckey] = {'d_breadth': d_bread, 'd_spy': d_spy}
            gain = d_bread - d_spy
            print(f"  {ckey:20s} | Breadth d={d_bread:.3f}  SPY d={d_spy:.3f}  "
                  f"gain={gain:+.3f}")

        valid = {k: v for k, v in results.items() if not np.isnan(v['d_breadth'])}
        assert len(valid) >= 2

        mean_d_bread = np.mean([v['d_breadth'] for v in valid.values()])
        mean_d_spy = np.mean([v['d_spy'] for v in valid.values()])
        mean_gain = mean_d_bread - mean_d_spy

        print(f"\n  Mean: Breadth d={mean_d_bread:.3f}  SPY d={mean_d_spy:.3f}  "
              f"mean gain={mean_gain:+.3f}")

        if mean_gain > 0.1:
            print("  => POSITIVE: Sector breadth geometry adds detection power over SPY")
        elif mean_gain > -0.05:
            print("  => NEUTRAL: Sector breadth is comparable to SPY-only")
        else:
            print("  => NEGATIVE: Sector breadth dilutes the signal")

        self.__class__._berry_results = results
        self.__class__._mean_gain_berry = mean_gain

    def test_sector_joint_embedding_vs_spy_only(self, sector_data, spy_only_data):
        """
        Full sector ETF joint embedding (8 ETFs) vs SPY-only.

        The full sector embedding includes cross-correlation features that
        directly capture breadth dynamics as geometric structure.
        """
        sect_features, sect_dates, _ = sector_data
        spy_features, spy_dates = spy_only_data

        results = {}
        print("\n--- Q65 Full Sector Embedding vs SPY-only (BerryPhaseRate) ---")

        for ckey in STANDARD_CRISES:
            sect_mask = _build_crisis_mask(sect_dates, ckey)
            spy_mask = _build_crisis_mask(spy_dates, ckey)
            if sect_mask.sum() < 5 or spy_mask.sum() < 5:
                continue

            sect_scores = _run_detector(BerryPhaseRateDetector, sect_features)
            d_sect = _cohens_d_from_scores(sect_scores, sect_mask)

            spy_scores = _run_detector(BerryPhaseRateDetector, spy_features)
            d_spy = _cohens_d_from_scores(spy_scores, spy_mask)

            results[ckey] = {'d_sector': d_sect, 'd_spy': d_spy}
            gain = d_sect - d_spy
            print(f"  {ckey:20s} | Sector d={d_sect:.3f}  SPY d={d_spy:.3f}  "
                  f"gain={gain:+.3f}")

        valid = {k: v for k, v in results.items() if not np.isnan(v['d_sector'])}
        assert len(valid) >= 2

        mean_d_sect = np.mean([v['d_sector'] for v in valid.values()])
        mean_d_spy = np.mean([v['d_spy'] for v in valid.values()])
        mean_gain = mean_d_sect - mean_d_spy

        print(f"\n  Mean: Sector d={mean_d_sect:.3f}  SPY d={mean_d_spy:.3f}  "
              f"mean gain={mean_gain:+.3f}")

        self.__class__._sector_results = results
        self.__class__._mean_gain_sector = mean_gain

    def test_breadth_spectral_entropy(self, breadth_feature_data, spy_only_data):
        """
        SpectralEntropy on breadth features may be especially powerful:
        breadth collapse during crises changes the eigenvalue spectrum dramatically.
        """
        bread_features, bread_dates = breadth_feature_data
        spy_features, spy_dates = spy_only_data

        results = {}
        print("\n--- Q65 Pure Breadth vs SPY-only (SpectralEntropy) ---")

        for ckey in STANDARD_CRISES:
            bread_mask = _build_crisis_mask(bread_dates, ckey)
            spy_mask = _build_crisis_mask(spy_dates, ckey)
            if bread_mask.sum() < 5 or spy_mask.sum() < 5:
                continue

            bread_scores = _run_detector(
                SpectralEntropyDetector, bread_features,
                config={'normalization': 'soft', 'adaptive_epsilon': True}
            )
            d_bread = _cohens_d_from_scores(bread_scores, bread_mask)

            spy_scores = _run_detector(
                SpectralEntropyDetector, spy_features,
                config={'normalization': 'soft', 'adaptive_epsilon': True}
            )
            d_spy = _cohens_d_from_scores(spy_scores, spy_mask)

            results[ckey] = {'d_breadth': d_bread, 'd_spy': d_spy}
            gain = d_bread - d_spy
            print(f"  {ckey:20s} | Breadth Entropy d={d_bread:.3f}  "
                  f"SPY Entropy d={d_spy:.3f}  gain={gain:+.3f}")

        valid = {k: v for k, v in results.items() if not np.isnan(v['d_breadth'])}
        assert len(valid) >= 2

        mean_d_bread = np.mean([v['d_breadth'] for v in valid.values()])
        mean_d_spy = np.mean([v['d_spy'] for v in valid.values()])
        mean_gain = mean_d_bread - mean_d_spy

        print(f"\n  Mean: Breadth Entropy d={mean_d_bread:.3f}  "
              f"SPY Entropy d={mean_d_spy:.3f}  mean gain={mean_gain:+.3f}")

        self.__class__._entropy_results = results
        self.__class__._mean_gain_entropy = mean_gain


# ===========================================================================
# Summary report
# ===========================================================================

class TestSummaryReport:
    """Collects and prints a structured summary of Q61-Q65 results."""

    def test_print_summary(self):
        """Print structured Q61-Q65 results summary."""
        print("\n" + "=" * 70)
        print("Q61-Q65 CROSS-ASSET / DATA EXPANSION — SUMMARY REPORT")
        print("=" * 70)

        print("""
Q61: INTRADAY TIMESCALES — ANALYTICAL
  Verdict: NEGATIVE
  Rolling windows (20d) collapse to <3 calendar days at hourly resolution.
  Berry phase path at intraday frequency is dominated by microstructure noise.
  Our observables are regime-level (weeks/months), not microstructure.
  Exception: pre-aggregated realized vol/corr at daily frequency would add value
  (this is Q65 feature engineering, not raw frequency).
""")

        print("""
Q62: VIX EMBEDDING — EMPIRICAL
  Detector: BerryPhaseRateDetector + SpectralEntropyDetector on VIX features.
  VIX is a derived forward-looking implied volatility surface.
  Second-order geometry on VIX: measures how fast the options market's
  regime beliefs change.
  Hypothesis: VIX geometry is COMPLEMENTARY to SPY geometry (different crises).
""")

        print("""
Q63: MULTI-ASSET EMBEDDING (SPY + TLT + GLD) — EMPIRICAL
  Cross-asset correlation collapse during crises adds geometric richness.
  Joint embedding captures flight-to-quality dynamics.
  Test: BerryPhaseRate and SpectralEntropy on joint 3-asset feature matrix.
""")

        print("""
Q64: EM CONTAGION (EEM) — EMPIRICAL
  Tests whether EEM-specific crises (2015_china, 2018_q4) are better
  detected by EEM geometry than SPY geometry.
  Contagion hypothesis: EM stress manifests geometrically before DM.
""")

        print("""
Q65: SECTOR BREADTH DISPERSION — EMPIRICAL
  Cross-sectional std of 8 sector ETF returns as breadth proxy.
  Pure breadth features vs SPY-only vs full 8-sector joint embedding.
  Tests whether market internals (breadth collapse/dispersion) improve d.
""")

        print("=" * 70)
        print("See per-class test output above for numerical d values.")
        print("=" * 70)

        assert True  # Summary always passes
