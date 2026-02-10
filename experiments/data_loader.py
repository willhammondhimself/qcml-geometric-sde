"""
Shared data loading and crisis definitions for all experiments.

Provides:
- fetch_polygon_data(): Fetch daily OHLCV from Polygon API
- create_feature_matrix(): Build feature matrix from close prices
- ALL_CRISES: 12 crisis definitions with start/end dates
- CRISIS_CATEGORIES: Pre-defined novel vs conventional classification
- PolygonDataSource / MinimalFeatureEngine: Legacy-compatible wrappers

Usage:
    from experiments.data_loader import fetch_polygon_data, create_feature_matrix, ALL_CRISES
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Crisis Definitions
# =============================================================================

ALL_CRISES = {
    '2007_quant': {
        'start': '2007-08-01', 'end': '2007-09-30',
        'label': 'Quant Crisis 2007',
    },
    '2008_gfc': {
        'start': '2008-09-01', 'end': '2009-03-31',
        'label': 'GFC 2008',
    },
    '2010_flash': {
        'start': '2010-05-01', 'end': '2010-06-30',
        'label': 'Flash Crash 2010',
    },
    '2011_euro': {
        'start': '2011-07-01', 'end': '2011-10-31',
        'label': 'Euro Crisis 2011',
    },
    '2015_china': {
        'start': '2015-07-01', 'end': '2015-09-30',
        'label': 'China Crash 2015',
    },
    '2018_volmageddon': {
        'start': '2018-01-26', 'end': '2018-04-30',
        'label': 'Volmageddon 2018',
    },
    '2018_q4': {
        'start': '2018-10-01', 'end': '2018-12-31',
        'label': 'Q4 Selloff 2018',
    },
    '2019_repo': {
        'start': '2019-09-01', 'end': '2019-10-31',
        'label': 'Repo Crisis 2019',
    },
    '2020_covid': {
        'start': '2020-02-20', 'end': '2020-04-30',
        'label': 'COVID 2020',
    },
    '2022_rates': {
        'start': '2022-01-01', 'end': '2022-10-31',
        'label': 'Rate Hikes 2022',
    },
    '2023_svb': {
        'start': '2023-03-01', 'end': '2023-04-30',
        'label': 'SVB 2023',
    },
    '2024_carry': {
        'start': '2024-07-15', 'end': '2024-08-31',
        'label': 'Carry Unwind 2024',
    },
}

# Pre-defined BEFORE seeing results (Review Item 8).
# Novel = new market mechanisms without historical precedent.
# Conventional = crises with recognizable historical parallels.
CRISIS_CATEGORIES = {
    'novel': [
        '2018_volmageddon',  # Unprecedented short-vol blowup
        '2018_q4',           # Algorithmic-driven selloff
        '2019_repo',         # Plumbing crisis, no equity analog
        '2022_rates',        # Fastest rate cycle in 40 years
        '2023_svb',          # Social-media-driven bank run
        '2024_carry',        # Yen carry unwind + AI rotation
    ],
    'conventional': [
        '2007_quant',   # Factor crowding (resembles LTCM)
        '2008_gfc',     # Credit crisis (resembles 1929, 1987)
        '2010_flash',   # Liquidity shock (resembles 1987)
        '2011_euro',    # Sovereign debt (resembles EM crises)
        '2015_china',   # Emerging market contagion (resembles 1997)
        '2020_covid',   # Exogenous shock (resembles 1918, 2003)
    ],
}

# Narrative figures use wider context windows
NARRATIVE_CRISES = {
    '2008_gfc': {
        'label': '2008 Global Financial Crisis',
        'short': 'GFC 2008',
        'start': '2008-09-01', 'end': '2009-03-31',
        'context_start': '2007-06-01', 'context_end': '2009-12-31',
    },
    '2020_covid': {
        'label': '2020 COVID-19 Pandemic Crash',
        'short': 'COVID 2020',
        'start': '2020-02-20', 'end': '2020-04-30',
        'context_start': '2019-06-01', 'context_end': '2020-12-31',
    },
    '2022_rates': {
        'label': '2022 Federal Reserve Rate Hikes',
        'short': 'Rate Hikes 2022',
        'start': '2022-01-01', 'end': '2022-10-31',
        'context_start': '2021-01-01', 'context_end': '2023-06-30',
    },
}


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_polygon_data(symbols, start_date, end_date):
    """Fetch daily OHLCV from Polygon API.

    Args:
        symbols: List of ticker symbols, e.g. ['SPY', 'DIA'].
        start_date: Start date string 'YYYY-MM-DD'.
        end_date: End date string 'YYYY-MM-DD'.

    Returns:
        DataFrame with MultiIndex (symbol, timestamp) and columns
        [open, high, low, close, volume].
    """
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')

    from polygon import RESTClient

    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not found in environment")

    client = RESTClient(api_key)
    all_data = []

    for symbol in symbols:
        logger.info(f"  Fetching {symbol} {start_date} to {end_date}...")
        aggs = list(client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan='day',
            from_=start_date,
            to=end_date,
            limit=50000,
        ))

        for a in aggs:
            all_data.append({
                'symbol': symbol,
                'timestamp': pd.Timestamp(a.timestamp, unit='ms'),
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume,
            })

    df = pd.DataFrame(all_data)
    df = df.set_index(['symbol', 'timestamp']).sort_index()
    return df


def create_feature_matrix(prices_df):
    """Create a minimal feature matrix from a close-price DataFrame.

    Needs >= 2 symbols for cross-sectional features (cross_corr5,
    cross_vol_disp, avg_ret). Single-symbol gives 0 rows after warmup.

    Args:
        prices_df: DataFrame with columns = symbols, index = dates,
            values = close prices.

    Returns:
        features: np.ndarray (T', d) after warmup.
        dates: DatetimeIndex aligned with features.
    """
    log_ret = np.log(prices_df / prices_df.shift(1))

    features_dict = {}
    for col in prices_df.columns:
        features_dict[f'{col}_ret'] = log_ret[col]
        features_dict[f'{col}_vol5'] = log_ret[col].rolling(5).std()
        features_dict[f'{col}_vol20'] = log_ret[col].rolling(20).std()
        features_dict[f'{col}_mom5'] = prices_df[col].pct_change(5)
        features_dict[f'{col}_mom20'] = prices_df[col].pct_change(20)

    if len(prices_df.columns) > 1:
        features_dict['cross_corr5'] = (
            log_ret.rolling(5).corr().groupby(level=0).mean().mean(axis=1)
        )
        features_dict['cross_vol_disp'] = log_ret.rolling(20).std().std(axis=1)
        features_dict['avg_ret'] = log_ret.mean(axis=1)

    feat_df = pd.DataFrame(features_dict)
    feat_df = feat_df.dropna()

    return feat_df.values, feat_df.index


def load_default_data(symbols=None, start='2005-01-01', end='2024-12-31'):
    """Convenience: fetch data and build feature matrix in one call.

    Args:
        symbols: List of tickers (default ['SPY', 'DIA']).
        start: Start date string.
        end: End date string.

    Returns:
        X: Feature matrix (T, d).
        dates: DatetimeIndex.
        prices_df: Close prices DataFrame.
    """
    if symbols is None:
        symbols = ['SPY', 'DIA']

    raw = fetch_polygon_data(symbols, start, end)
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    return X, dates, prices_df


# =============================================================================
# Legacy-compatible wrappers (for tests that import experiments.data)
# =============================================================================

class PolygonDataSource:
    """Wrapper matching the old qcml.data.PolygonDataSource interface."""

    def fetch_equities(self, symbols, start_date, end_date):
        return fetch_polygon_data(symbols, start_date, end_date)


class MinimalFeatureEngine:
    """Wrapper matching the old qcml.data.MinimalFeatureEngine interface."""

    def __init__(self, window=20):
        self.window = window

    def create_feature_matrix(self, prices_df):
        """Return a DataFrame (not tuple) for backward compatibility."""
        arr, dates = create_feature_matrix(prices_df)
        return pd.DataFrame(arr, index=dates)
