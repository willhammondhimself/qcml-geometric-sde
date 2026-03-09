"""
Cross-Asset Generalization Investigation — Q56-Q60

Tests whether our geometric observables (BerryPhaseRateDetector,
SpectralEntropyDetector) generalize beyond US equities to other asset classes.

Q56: Bond markets (TLT) — rate regime detection
Q57: Crypto (BTC-USD) — fast-shifting regimes
Q58: Commodities (GLD) — gold crash + COVID oil crash
Q59: Multi-asset sector ETFs [SPY, XLF, XLK, XLE] vs SPY-only
Q60: FX markets (EURUSD=X) — currency regime detection

Data: yfinance, real data only.
Detectors: BerryPhaseRateDetector, SpectralEntropyDetector (top-2 by d).
Metric: Cohen's d (crisis vs non-crisis scores), pooled-std formulation.
"""

import sys
import os
import warnings
import time

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
sys.path.insert(0, PROJECT_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix, create_feature_matrix_single_asset
from experiments.evaluation import _cohens_d
from qcml_geometry.observables import BerryPhaseRateDetector, SpectralEntropyDetector


# ---------------------------------------------------------------------------
# Detector configuration — fast but non-trivial
# ---------------------------------------------------------------------------

FAST_CONFIG = dict(
    hilbert_dim=4,
    n_pca_components=6,
    min_expanding=40,
    rolling_window=15,
    seed=42,
)

DETECTOR_CLASSES = [BerryPhaseRateDetector, SpectralEntropyDetector]
DETECTOR_NAMES = ['BerryPhaseRate', 'SpectralEntropy']


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cohens_d_from_scores(scores: np.ndarray, crisis_mask: np.ndarray) -> float:
    """Compute Cohen's d between crisis-period and non-crisis-period scores.

    Args:
        scores: 1-D regime score time series.
        crisis_mask: Boolean array, True = in crisis window.

    Returns:
        Cohen's d (float). Returns np.nan when either group has < 2 observations.
    """
    valid = ~np.isnan(scores)
    crisis_scores = scores[valid & crisis_mask]
    normal_scores = scores[valid & ~crisis_mask]
    if len(crisis_scores) < 2 or len(normal_scores) < 2:
        return np.nan
    return _cohens_d(crisis_scores, normal_scores)


def _make_crisis_mask(dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    """Build a boolean numpy mask for dates in [start, end].

    Args:
        dates: DatetimeIndex aligned to the feature matrix rows.
        start: Crisis start date string 'YYYY-MM-DD'.
        end: Crisis end date string 'YYYY-MM-DD'.

    Returns:
        Boolean numpy array, True = crisis period.
    """
    dates_arr = pd.DatetimeIndex(dates)
    return np.asarray(
        (dates_arr >= pd.Timestamp(start)) & (dates_arr <= pd.Timestamp(end))
    )


def _run_detectors(features: np.ndarray) -> dict:
    """Fit all detectors on features and return score arrays.

    Args:
        features: Feature matrix (T, d).

    Returns:
        Dict mapping detector name -> score array (length T).
    """
    scores = {}
    for cls, name in zip(DETECTOR_CLASSES, DETECTOR_NAMES):
        det = cls(**FAST_CONFIG)
        det.fit(features)
        scores[name] = det.compute_regime_scores(features)
    return scores


def _fetch_prices(symbol: str, start: str, end: str) -> pd.Series:
    """Fetch adjusted close prices for a single symbol.

    Args:
        symbol: Ticker string (yfinance format).
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD'.

    Returns:
        pd.Series with DatetimeIndex and daily close prices.
    """
    raw = fetch_data([symbol], start, end, source='yfinance', use_cache=True)
    prices = raw['close'].unstack('symbol')[symbol].dropna()
    return prices


def _fetch_multi_prices(symbols: list, start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices for multiple symbols, aligned on common dates.

    Args:
        symbols: List of ticker strings.
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD'.

    Returns:
        DataFrame (dates x symbols) with NaNs dropped.
    """
    raw = fetch_data(symbols, start, end, source='yfinance', use_cache=True)
    prices = raw['close'].unstack('symbol').dropna()
    return prices


def _print_crisis_results(
    asset: str,
    crisis_name: str,
    scores_dict: dict,
    crisis_mask: np.ndarray,
    n_crisis_days: int,
) -> dict:
    """Print and return per-detector Cohen's d for a single crisis.

    Args:
        asset: Asset label for display.
        crisis_name: Human-readable crisis label.
        scores_dict: Dict mapping detector name -> score array.
        crisis_mask: Boolean crisis period mask.
        n_crisis_days: Number of days in the crisis window (in the data).

    Returns:
        Dict mapping detector name -> Cohen's d.
    """
    results = {}
    print(f"\n  {asset} | {crisis_name} | crisis_days={n_crisis_days}")
    for name, scores in scores_dict.items():
        d = _cohens_d_from_scores(scores, crisis_mask)
        results[name] = d
        label = 'large' if d >= 0.8 else ('medium' if d >= 0.5 else ('small' if d >= 0.2 else 'negligible'))
        print(f"    {name:20s}: d={d:.3f}  [{label}]")
    return results


# ===========================================================================
# Q56: Bond markets (TLT) — rate regime detection
# ===========================================================================

class TestQ56_BondMarkets:
    """
    Q56: Do our observables work on bond markets for rate regime detection?

    Asset: TLT (iShares 20+ Year Treasury Bond ETF)
    Crisis periods:
      - 2013 Taper Tantrum: 2013-05-20 to 2013-08-31
      - 2022 Rate Hike Cycle: 2022-01-01 to 2022-12-31
    """

    SYMBOL = 'TLT'
    HISTORY_START = '2005-01-01'
    HISTORY_END = '2024-12-31'

    CRISES = {
        'Taper Tantrum 2013': ('2013-05-20', '2013-08-31'),
        'Rate Hike Cycle 2022': ('2022-01-01', '2022-12-31'),
    }

    @pytest.fixture(scope='class')
    def tlt_data(self):
        """Fetch TLT prices and build feature matrix."""
        prices = _fetch_prices(self.SYMBOL, self.HISTORY_START, self.HISTORY_END)
        features, dates = create_feature_matrix_single_asset(prices)
        scores = _run_detectors(features)
        return prices, features, pd.DatetimeIndex(dates), scores

    def test_taper_tantrum_2013(self, tlt_data):
        """TLT Taper Tantrum 2013: Bond selloff should register as regime change."""
        prices, features, dates, scores = tlt_data
        start, end = self.CRISES['Taper Tantrum 2013']
        mask = _make_crisis_mask(dates, start, end)
        n_crisis = mask.sum()

        print("\n--- Q56: TLT Bond Markets ---")
        results = _print_crisis_results('TLT', 'Taper Tantrum 2013', scores, mask, n_crisis)

        # Store results on class for summary
        if not hasattr(self.__class__, '_results'):
            self.__class__._results = {}
        self.__class__._results['taper_tantrum_2013'] = results

        # Basic sanity: mask must have some crisis days and some data
        assert n_crisis >= 5, f"Expected >= 5 crisis days in mask, got {n_crisis}"
        # At least one detector should give a non-NaN d
        valid_ds = [d for d in results.values() if not np.isnan(d)]
        assert len(valid_ds) >= 1, "Expected at least one valid Cohen's d"

    def test_rate_hike_cycle_2022(self, tlt_data):
        """TLT Rate Hike Cycle 2022: TLT fell ~30% — should be highly detectable."""
        prices, features, dates, scores = tlt_data
        start, end = self.CRISES['Rate Hike Cycle 2022']
        mask = _make_crisis_mask(dates, start, end)
        n_crisis = mask.sum()

        results = _print_crisis_results('TLT', 'Rate Hike Cycle 2022', scores, mask, n_crisis)
        self.__class__._results['rate_hike_2022'] = results

        assert n_crisis >= 20, f"Expected >= 20 crisis days in mask, got {n_crisis}"
        valid_ds = [d for d in results.values() if not np.isnan(d)]
        assert len(valid_ds) >= 1, "Expected at least one valid Cohen's d"

    def test_summary_q56(self, tlt_data):
        """Print Q56 summary and assess cross-asset generalization."""
        if not hasattr(self.__class__, '_results'):
            pytest.skip("Run earlier tests first")

        r = self.__class__._results
        print("\n--- Q56 SUMMARY: TLT Bond Markets ---")
        for crisis_key, dmap in r.items():
            for det_name, d in dmap.items():
                print(f"  {crisis_key:30s} | {det_name:20s} | d={d:.3f}")

        mean_ds = {
            name: np.nanmean([r.get(c, {}).get(name, np.nan) for c in r])
            for name in DETECTOR_NAMES
        }
        print("\n  Mean d across crises:")
        for name, d in mean_ds.items():
            print(f"    {name:20s}: mean_d={d:.3f}")

        generalizes = any(d >= 0.3 for d in mean_ds.values() if not np.isnan(d))
        print(f"\n  Generalization to bonds: {'YES (d >= 0.3)' if generalizes else 'WEAK (all d < 0.3)'}")

        assert True  # Summary always passes


# ===========================================================================
# Q57: Crypto markets (BTC-USD) — fast-shifting regimes
# ===========================================================================

class TestQ57_CryptoMarkets:
    """
    Q57: Can we detect crypto regime changes where regimes shift faster?

    Asset: BTC-USD
    Crisis periods:
      - Crypto Winter 2018: 2018-01-01 to 2018-12-31
      - FTX/Luna Collapse 2022: 2022-05-01 to 2022-12-31
    Note: BTC is ~5-10x more volatile than SPY; normalization may behave differently.
    """

    SYMBOL = 'BTC-USD'
    HISTORY_START = '2015-01-01'
    HISTORY_END = '2024-12-31'

    CRISES = {
        'Crypto Winter 2018': ('2018-01-01', '2018-12-31'),
        'FTX/Luna Collapse 2022': ('2022-05-01', '2022-12-31'),
    }

    @pytest.fixture(scope='class')
    def btc_data(self):
        """Fetch BTC-USD prices and build feature matrix."""
        prices = _fetch_prices(self.SYMBOL, self.HISTORY_START, self.HISTORY_END)
        features, dates = create_feature_matrix_single_asset(prices)
        scores = _run_detectors(features)
        return prices, features, pd.DatetimeIndex(dates), scores

    def test_crypto_winter_2018(self, btc_data):
        """BTC Crypto Winter 2018: BTC fell ~85% — extreme regime shift."""
        prices, features, dates, scores = btc_data
        start, end = self.CRISES['Crypto Winter 2018']
        mask = _make_crisis_mask(dates, start, end)
        n_crisis = mask.sum()

        print("\n--- Q57: BTC-USD Crypto Markets ---")
        results = _print_crisis_results('BTC-USD', 'Crypto Winter 2018', scores, mask, n_crisis)

        if not hasattr(self.__class__, '_results'):
            self.__class__._results = {}
        self.__class__._results['crypto_winter_2018'] = results

        assert n_crisis >= 20, f"Expected >= 20 crisis days, got {n_crisis}"
        valid_ds = [d for d in results.values() if not np.isnan(d)]
        assert len(valid_ds) >= 1, "Expected at least one valid Cohen's d"

    def test_ftx_luna_2022(self, btc_data):
        """BTC FTX/Luna 2022: Multi-month crypto crash cascade."""
        prices, features, dates, scores = btc_data
        start, end = self.CRISES['FTX/Luna Collapse 2022']
        mask = _make_crisis_mask(dates, start, end)
        n_crisis = mask.sum()

        results = _print_crisis_results('BTC-USD', 'FTX/Luna Collapse 2022', scores, mask, n_crisis)
        self.__class__._results['ftx_luna_2022'] = results

        assert n_crisis >= 20, f"Expected >= 20 crisis days, got {n_crisis}"
        valid_ds = [d for d in results.values() if not np.isnan(d)]
        assert len(valid_ds) >= 1, "Expected at least one valid Cohen's d"

    def test_volatility_note(self, btc_data):
        """Report BTC vs SPY volatility ratio to contextualize d values."""
        prices, features, dates, scores = btc_data
        log_ret = np.log(prices / prices.shift(1)).dropna()
        daily_vol_btc = float(log_ret.std())

        print("\n--- Q57 Volatility Context ---")
        print(f"  BTC-USD daily vol: {daily_vol_btc:.4f} ({daily_vol_btc * 100:.2f}%)")
        print(f"  SPY baseline ~0.01 (1.0%)")
        print(f"  BTC/SPY vol ratio: ~{daily_vol_btc / 0.01:.1f}x")
        print("  Note: Higher vol means z-score normalization faces noisier baseline.")

        assert daily_vol_btc > 0.005, "BTC should have non-trivial daily vol"

    def test_summary_q57(self, btc_data):
        """Print Q57 summary."""
        if not hasattr(self.__class__, '_results'):
            pytest.skip("Run earlier tests first")

        r = self.__class__._results
        print("\n--- Q57 SUMMARY: BTC-USD Crypto Markets ---")
        for crisis_key, dmap in r.items():
            for det_name, d in dmap.items():
                print(f"  {crisis_key:30s} | {det_name:20s} | d={d:.3f}")

        mean_ds = {
            name: np.nanmean([r.get(c, {}).get(name, np.nan) for c in r])
            for name in DETECTOR_NAMES
        }
        print("\n  Mean d across crises:")
        for name, d in mean_ds.items():
            print(f"    {name:20s}: mean_d={d:.3f}")

        generalizes = any(d >= 0.3 for d in mean_ds.values() if not np.isnan(d))
        print(f"\n  Generalization to crypto: {'YES (d >= 0.3)' if generalizes else 'WEAK (all d < 0.3)'}")

        assert True


# ===========================================================================
# Q58: Commodity markets (GLD, USO) — gold crash + COVID oil
# ===========================================================================

class TestQ58_CommodityMarkets:
    """
    Q58: Do our observables work on commodity markets?

    Assets:
      - GLD (SPDR Gold ETF): 2013 Gold Crash (Apr-Jul 2013)
      - USO (US Oil ETF): 2020 COVID Oil Crash (Mar-Apr 2020)
    """

    HISTORY_START = '2006-01-01'
    HISTORY_END = '2024-12-31'

    GOLD_CRISIS = ('2013-04-01', '2013-07-31')
    OIL_CRISIS = ('2020-02-20', '2020-04-30')

    @pytest.fixture(scope='class')
    def gld_data(self):
        """Fetch GLD prices and build feature matrix."""
        prices = _fetch_prices('GLD', self.HISTORY_START, self.HISTORY_END)
        features, dates = create_feature_matrix_single_asset(prices)
        scores = _run_detectors(features)
        return prices, features, pd.DatetimeIndex(dates), scores

    @pytest.fixture(scope='class')
    def uso_data(self):
        """Fetch USO prices and build feature matrix."""
        prices = _fetch_prices('USO', self.HISTORY_START, self.HISTORY_END)
        features, dates = create_feature_matrix_single_asset(prices)
        scores = _run_detectors(features)
        return prices, features, pd.DatetimeIndex(dates), scores

    def test_gold_crash_2013(self, gld_data):
        """GLD Gold Crash 2013: Gold fell ~25% in 3 months — commodity regime shift."""
        prices, features, dates, scores = gld_data
        start, end = self.GOLD_CRISIS
        mask = _make_crisis_mask(dates, start, end)
        n_crisis = mask.sum()

        print("\n--- Q58: Commodity Markets ---")
        results = _print_crisis_results('GLD', 'Gold Crash 2013', scores, mask, n_crisis)

        if not hasattr(self.__class__, '_results'):
            self.__class__._results = {}
        self.__class__._results['gold_crash_2013'] = results

        assert n_crisis >= 5, f"Expected >= 5 crisis days, got {n_crisis}"
        valid_ds = [d for d in results.values() if not np.isnan(d)]
        assert len(valid_ds) >= 1, "Expected at least one valid Cohen's d"

    def test_covid_oil_crash_2020(self, uso_data):
        """USO COVID Oil Crash 2020: Oil went negative — extreme commodity stress."""
        prices, features, dates, scores = uso_data
        start, end = self.OIL_CRISIS
        mask = _make_crisis_mask(dates, start, end)
        n_crisis = mask.sum()

        results = _print_crisis_results('USO', 'COVID Oil Crash 2020', scores, mask, n_crisis)
        self.__class__._results['oil_crash_2020'] = results

        assert n_crisis >= 5, f"Expected >= 5 crisis days, got {n_crisis}"
        valid_ds = [d for d in results.values() if not np.isnan(d)]
        assert len(valid_ds) >= 1, "Expected at least one valid Cohen's d"

    def test_summary_q58(self, gld_data, uso_data):
        """Print Q58 summary."""
        if not hasattr(self.__class__, '_results'):
            pytest.skip("Run earlier tests first")

        r = self.__class__._results
        print("\n--- Q58 SUMMARY: Commodity Markets ---")
        for crisis_key, dmap in r.items():
            for det_name, d in dmap.items():
                print(f"  {crisis_key:30s} | {det_name:20s} | d={d:.3f}")

        mean_ds = {
            name: np.nanmean([r.get(c, {}).get(name, np.nan) for c in r])
            for name in DETECTOR_NAMES
        }
        print("\n  Mean d across crises:")
        for name, d in mean_ds.items():
            print(f"    {name:20s}: mean_d={d:.3f}")

        generalizes = any(d >= 0.3 for d in mean_ds.values() if not np.isnan(d))
        print(f"\n  Generalization to commodities: {'YES (d >= 0.3)' if generalizes else 'WEAK (all d < 0.3)'}")

        assert True


# ===========================================================================
# Q59: Multi-asset sector ETFs vs SPY-only
# ===========================================================================

class TestQ59_MultiAssetInput:
    """
    Q59: Does adding sector ETFs [SPY, XLF, XLK, XLE] improve detection
    via richer geometry compared to SPY-only?

    Hypothesis: multi-asset cross-sectional geometry captures sector rotation
    signals not visible in SPY alone.

    Standard crises used: 2008_gfc, 2020_covid, 2022_rates, 2015_china.
    """

    SYMBOLS_MULTI = ['SPY', 'XLF', 'XLK', 'XLE']
    SYMBOL_SINGLE = 'SPY'
    HISTORY_START = '2005-01-01'
    HISTORY_END = '2024-12-31'

    # Standard 4 crises (same as Paper 1 standard set)
    CRISES = {
        'GFC 2008': ('2008-09-01', '2009-03-31'),
        'COVID 2020': ('2020-02-20', '2020-04-30'),
        'Rate Hikes 2022': ('2022-01-01', '2022-10-31'),
        'China Crash 2015': ('2015-07-01', '2015-09-30'),
    }

    @pytest.fixture(scope='class')
    def single_asset_data(self):
        """Fetch SPY-only data and build single-asset feature matrix."""
        prices = _fetch_prices(self.SYMBOL_SINGLE, self.HISTORY_START, self.HISTORY_END)
        features, dates = create_feature_matrix_single_asset(prices)
        scores = _run_detectors(features)
        return features, pd.DatetimeIndex(dates), scores

    @pytest.fixture(scope='class')
    def multi_asset_data(self):
        """Fetch [SPY, XLF, XLK, XLE] and build multi-asset feature matrix."""
        prices_df = _fetch_multi_prices(self.SYMBOLS_MULTI, self.HISTORY_START, self.HISTORY_END)
        features, dates = create_feature_matrix(prices_df)
        scores = _run_detectors(features)
        return features, pd.DatetimeIndex(dates), scores

    def _compute_mean_d(self, scores_dict: dict, dates: pd.DatetimeIndex) -> dict:
        """Compute mean Cohen's d across all 4 standard crises for each detector."""
        mean_ds = {}
        for name, scores in scores_dict.items():
            per_crisis_ds = []
            for crisis_label, (start, end) in self.CRISES.items():
                mask = _make_crisis_mask(dates, start, end)
                d = _cohens_d_from_scores(scores, mask)
                per_crisis_ds.append(d)
            mean_ds[name] = np.nanmean(per_crisis_ds)
        return mean_ds

    def test_gfc_2008_comparison(self, single_asset_data, multi_asset_data):
        """GFC 2008: Compare single-asset vs multi-asset d."""
        single_features, single_dates, single_scores = single_asset_data
        multi_features, multi_dates, multi_scores = multi_asset_data

        start, end = self.CRISES['GFC 2008']

        mask_s = _make_crisis_mask(single_dates, start, end)
        mask_m = _make_crisis_mask(multi_dates, start, end)

        print("\n--- Q59: Multi-Asset vs Single-Asset Input ---")
        print("  Crisis: GFC 2008")
        for name in DETECTOR_NAMES:
            d_single = _cohens_d_from_scores(single_scores[name], mask_s)
            d_multi = _cohens_d_from_scores(multi_scores[name], mask_m)
            delta = d_multi - d_single
            print(f"    {name:20s}: single_d={d_single:.3f}  multi_d={d_multi:.3f}  "
                  f"delta={delta:+.3f}  {'IMPROVED' if delta > 0 else 'DEGRADED'}")

        assert True

    def test_covid_2020_comparison(self, single_asset_data, multi_asset_data):
        """COVID 2020: Compare single-asset vs multi-asset d."""
        single_features, single_dates, single_scores = single_asset_data
        multi_features, multi_dates, multi_scores = multi_asset_data

        start, end = self.CRISES['COVID 2020']
        mask_s = _make_crisis_mask(single_dates, start, end)
        mask_m = _make_crisis_mask(multi_dates, start, end)

        print("\n  Crisis: COVID 2020")
        for name in DETECTOR_NAMES:
            d_single = _cohens_d_from_scores(single_scores[name], mask_s)
            d_multi = _cohens_d_from_scores(multi_scores[name], mask_m)
            delta = d_multi - d_single
            print(f"    {name:20s}: single_d={d_single:.3f}  multi_d={d_multi:.3f}  "
                  f"delta={delta:+.3f}  {'IMPROVED' if delta > 0 else 'DEGRADED'}")

        assert True

    def test_mean_d_comparison_all_crises(self, single_asset_data, multi_asset_data):
        """Compare mean d across all 4 standard crises: single vs multi."""
        single_features, single_dates, single_scores = single_asset_data
        multi_features, multi_dates, multi_scores = multi_asset_data

        mean_d_single = self._compute_mean_d(single_scores, single_dates)
        mean_d_multi = self._compute_mean_d(multi_scores, multi_dates)

        print("\n--- Q59 SUMMARY: Mean d Across 4 Standard Crises ---")
        print(f"  {'Detector':20s}  {'Single(SPY)':>12}  {'Multi(4-ETF)':>12}  {'Delta':>8}")
        print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*8}")

        if not hasattr(self.__class__, '_results'):
            self.__class__._results = {}

        for name in DETECTOR_NAMES:
            d_s = mean_d_single.get(name, np.nan)
            d_m = mean_d_multi.get(name, np.nan)
            delta = d_m - d_s if not (np.isnan(d_s) or np.isnan(d_m)) else np.nan
            verdict = 'IMPROVED' if (not np.isnan(delta) and delta > 0) else 'DEGRADED'
            print(f"  {name:20s}  {d_s:12.3f}  {d_m:12.3f}  {delta:+8.3f}  {verdict}")

            self.__class__._results[name] = {
                'single_d': d_s,
                'multi_d': d_m,
                'delta': delta,
            }

        # At least one detector should have computable mean d in both modes
        valid_single = sum(1 for d in mean_d_single.values() if not np.isnan(d))
        valid_multi = sum(1 for d in mean_d_multi.values() if not np.isnan(d))
        assert valid_single >= 1, "Single-asset: expected at least 1 valid mean d"
        assert valid_multi >= 1, "Multi-asset: expected at least 1 valid mean d"

        n_improved = sum(
            1 for name in DETECTOR_NAMES
            if not np.isnan(self.__class__._results[name]['delta'])
            and self.__class__._results[name]['delta'] > 0
        )
        print(f"\n  Detectors improved by multi-asset input: {n_improved}/{len(DETECTOR_NAMES)}")
        generalizes = n_improved > 0
        print(f"  Conclusion: Multi-asset richer geometry {'HELPS' if generalizes else 'does NOT help'}")


# ===========================================================================
# Q60: FX markets (EURUSD=X) — currency regime detection
# ===========================================================================

class TestQ60_FXMarkets:
    """
    Q60: Can we detect regimes in FX markets?

    Asset: EURUSD=X (EUR/USD exchange rate via yfinance)
    Crisis periods:
      - EUR/USD crash 2014-2015: 2014-09-01 to 2015-03-31
      - USD surge 2022: 2022-01-01 to 2022-10-31
    """

    SYMBOL = 'EURUSD=X'
    HISTORY_START = '2005-01-01'
    HISTORY_END = '2024-12-31'

    CRISES = {
        'EUR/USD Crash 2014-2015': ('2014-09-01', '2015-03-31'),
        'USD Surge 2022': ('2022-01-01', '2022-10-31'),
    }

    @pytest.fixture(scope='class')
    def fx_data(self):
        """Fetch EURUSD=X prices and build feature matrix."""
        prices = _fetch_prices(self.SYMBOL, self.HISTORY_START, self.HISTORY_END)
        features, dates = create_feature_matrix_single_asset(prices)
        scores = _run_detectors(features)
        return prices, features, pd.DatetimeIndex(dates), scores

    def test_eurusd_crash_2014_2015(self, fx_data):
        """EUR/USD Crash 2014-2015: EUR fell ~25% vs USD over 6 months."""
        prices, features, dates, scores = fx_data
        start, end = self.CRISES['EUR/USD Crash 2014-2015']
        mask = _make_crisis_mask(dates, start, end)
        n_crisis = mask.sum()

        print("\n--- Q60: FX Markets (EURUSD=X) ---")
        results = _print_crisis_results('EURUSD=X', 'EUR/USD Crash 2014-2015', scores, mask, n_crisis)

        if not hasattr(self.__class__, '_results'):
            self.__class__._results = {}
        self.__class__._results['eurusd_crash_2014'] = results

        assert n_crisis >= 10, f"Expected >= 10 crisis days, got {n_crisis}"
        valid_ds = [d for d in results.values() if not np.isnan(d)]
        assert len(valid_ds) >= 1, "Expected at least one valid Cohen's d"

    def test_usd_surge_2022(self, fx_data):
        """USD Surge 2022: DXY hit 20-year high; EUR/USD broke parity."""
        prices, features, dates, scores = fx_data
        start, end = self.CRISES['USD Surge 2022']
        mask = _make_crisis_mask(dates, start, end)
        n_crisis = mask.sum()

        results = _print_crisis_results('EURUSD=X', 'USD Surge 2022', scores, mask, n_crisis)
        self.__class__._results['usd_surge_2022'] = results

        assert n_crisis >= 20, f"Expected >= 20 crisis days, got {n_crisis}"
        valid_ds = [d for d in results.values() if not np.isnan(d)]
        assert len(valid_ds) >= 1, "Expected at least one valid Cohen's d"

    def test_fx_properties(self, fx_data):
        """Report FX-specific properties: mean-reversion, lower vol than equities."""
        prices, features, dates, scores = fx_data
        log_ret = np.log(prices / prices.shift(1)).dropna()
        daily_vol_fx = float(log_ret.std())
        mean_ret = float(log_ret.mean())

        print("\n--- Q60 FX Properties ---")
        print(f"  EURUSD=X daily vol: {daily_vol_fx:.4f} ({daily_vol_fx * 100:.3f}%)")
        print(f"  EURUSD=X mean daily ret: {mean_ret:.6f}")
        print(f"  FX vol is ~{daily_vol_fx / 0.01:.1f}x SPY vol (SPY baseline ~1.0%/day)")
        print(f"  FX is lower vol than equities → z-score thresholds may differ")

        assert daily_vol_fx > 0.001, "EURUSD should have non-trivial daily vol"

    def test_summary_q60(self, fx_data):
        """Print Q60 summary and assess FX generalization."""
        if not hasattr(self.__class__, '_results'):
            pytest.skip("Run earlier tests first")

        r = self.__class__._results
        print("\n--- Q60 SUMMARY: FX Markets (EURUSD=X) ---")
        for crisis_key, dmap in r.items():
            for det_name, d in dmap.items():
                print(f"  {crisis_key:30s} | {det_name:20s} | d={d:.3f}")

        mean_ds = {
            name: np.nanmean([r.get(c, {}).get(name, np.nan) for c in r])
            for name in DETECTOR_NAMES
        }
        print("\n  Mean d across crises:")
        for name, d in mean_ds.items():
            print(f"    {name:20s}: mean_d={d:.3f}")

        generalizes = any(d >= 0.3 for d in mean_ds.values() if not np.isnan(d))
        print(f"\n  Generalization to FX: {'YES (d >= 0.3)' if generalizes else 'WEAK (all d < 0.3)'}")

        assert True


# ===========================================================================
# Cross-Asset Summary Reporter
# ===========================================================================

class TestCrossAssetSummary:
    """
    Print a consolidated cross-asset summary table after all Q56-Q60 tests.

    Compares Cohen's d across asset classes and assesses generalization.
    Threshold: d >= 0.3 = 'generalizes'; d >= 0.8 = 'large effect'.
    """

    def test_print_cross_asset_table(self):
        """Print structured cross-asset results summary."""
        print("\n" + "=" * 78)
        print("CROSS-ASSET GENERALIZATION SUMMARY: Q56-Q60")
        print("=" * 78)
        print(f"{'Asset Class':20s}  {'Crisis':30s}  {'BerryPhaseRate':14s}  {'SpectralEntropy':14s}")
        print(f"{'-'*20}  {'-'*30}  {'-'*14}  {'-'*14}")

        rows = []

        # Q56 bonds
        if hasattr(TestQ56_BondMarkets, '_results'):
            r = TestQ56_BondMarkets._results
            for crisis_key, dmap in r.items():
                bpr = dmap.get('BerryPhaseRate', np.nan)
                se = dmap.get('SpectralEntropy', np.nan)
                print(f"{'TLT (Bonds)':20s}  {crisis_key:30s}  {bpr:14.3f}  {se:14.3f}")
                rows.append({'asset': 'TLT', 'crisis': crisis_key, 'BerryPhaseRate': bpr, 'SpectralEntropy': se})

        # Q57 crypto
        if hasattr(TestQ57_CryptoMarkets, '_results'):
            r = TestQ57_CryptoMarkets._results
            for crisis_key, dmap in r.items():
                bpr = dmap.get('BerryPhaseRate', np.nan)
                se = dmap.get('SpectralEntropy', np.nan)
                print(f"{'BTC-USD (Crypto)':20s}  {crisis_key:30s}  {bpr:14.3f}  {se:14.3f}")
                rows.append({'asset': 'BTC-USD', 'crisis': crisis_key, 'BerryPhaseRate': bpr, 'SpectralEntropy': se})

        # Q58 commodities
        if hasattr(TestQ58_CommodityMarkets, '_results'):
            r = TestQ58_CommodityMarkets._results
            for crisis_key, dmap in r.items():
                bpr = dmap.get('BerryPhaseRate', np.nan)
                se = dmap.get('SpectralEntropy', np.nan)
                asset = 'GLD' if '2013' in crisis_key else 'USO'
                print(f"{f'{asset} (Commodity)':20s}  {crisis_key:30s}  {bpr:14.3f}  {se:14.3f}")
                rows.append({'asset': asset, 'crisis': crisis_key, 'BerryPhaseRate': bpr, 'SpectralEntropy': se})

        # Q59 multi-asset
        if hasattr(TestQ59_MultiAssetInput, '_results'):
            r = TestQ59_MultiAssetInput._results
            print(f"\n{'Q59 Multi-Asset SPY+XLF+XLK+XLE vs SPY-only':}")
            print(f"  {'Detector':20s}  {'Single-d':10s}  {'Multi-d':10s}  {'Delta':8s}")
            for name in DETECTOR_NAMES:
                if name in r:
                    rd = r[name]
                    print(f"  {name:20s}  {rd['single_d']:10.3f}  {rd['multi_d']:10.3f}  "
                          f"{rd['delta']:+8.3f}")

        # Q60 FX
        if hasattr(TestQ60_FXMarkets, '_results'):
            r = TestQ60_FXMarkets._results
            for crisis_key, dmap in r.items():
                bpr = dmap.get('BerryPhaseRate', np.nan)
                se = dmap.get('SpectralEntropy', np.nan)
                print(f"{'EURUSD=X (FX)':20s}  {crisis_key:30s}  {bpr:14.3f}  {se:14.3f}")
                rows.append({'asset': 'EURUSD=X', 'crisis': crisis_key, 'BerryPhaseRate': bpr, 'SpectralEntropy': se})

        # Overall generalization score
        if rows:
            all_bpr = [r['BerryPhaseRate'] for r in rows if not np.isnan(r.get('BerryPhaseRate', np.nan))]
            all_se = [r['SpectralEntropy'] for r in rows if not np.isnan(r.get('SpectralEntropy', np.nan))]

            print(f"\n{'='*78}")
            print(f"  Overall mean d (BerryPhaseRate): {np.nanmean(all_bpr):.3f}" if all_bpr else "")
            print(f"  Overall mean d (SpectralEntropy): {np.nanmean(all_se):.3f}" if all_se else "")
            n_strong_bpr = sum(1 for d in all_bpr if d >= 0.5)
            n_strong_se = sum(1 for d in all_se if d >= 0.5)
            print(f"  BerryPhaseRate: {n_strong_bpr}/{len(all_bpr)} crises with d >= 0.5")
            print(f"  SpectralEntropy: {n_strong_se}/{len(all_se)} crises with d >= 0.5")

        print("=" * 78)
        assert True  # Summary always passes
