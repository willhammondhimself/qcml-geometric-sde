"""
Shared data loading and crisis definitions for all experiments.

Provides:
- fetch_data(): Unified data fetcher (yfinance by default, WRDS optional)
- fetch_polygon_data(): Fetch daily OHLCV from Polygon API (legacy)
- create_feature_matrix(): Build feature matrix from close prices
- ALL_CRISES: 19 crisis definitions with start/end dates (15 post-2005 + 4 pre-2005)
- CRISIS_CATEGORIES: Pre-defined novel vs conventional classification
- PolygonDataSource / MinimalFeatureEngine: Legacy-compatible wrappers

Usage:
    from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
"""

import hashlib
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = ROOT / 'data'


# =============================================================================
# Crisis Definitions
# =============================================================================

ALL_CRISES = {
    # --- Pre-2005 crises (SPY launched 1993-01-29) ---
    '1997_asia': {
        'start': '1997-10-20', 'end': '1997-11-30',
        'label': 'Asian Crisis 1997',
    },
    '1998_ltcm': {
        'start': '1998-08-01', 'end': '1998-10-15',
        'label': 'LTCM/Russia 1998',
    },
    '2000_dotcom': {
        'start': '2000-03-10', 'end': '2000-10-09',
        'label': 'Dot-Com Crash 2000',
    },
    '2001_911': {
        'start': '2001-09-10', 'end': '2001-10-15',
        'label': 'September 11 2001',
    },
    # --- Post-2005 crises (original 12) ---
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
    '2013_taper': {
        'start': '2013-05-20', 'end': '2013-07-15',
        'label': 'Taper Tantrum 2013',
    },
    '2015_china': {
        'start': '2015-07-01', 'end': '2015-09-30',
        'label': 'China Crash 2015',
    },
    '2016_brexit': {
        'start': '2016-06-20', 'end': '2016-07-31',
        'label': 'Brexit 2016',
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
    '2021_meme': {
        'start': '2021-01-25', 'end': '2021-04-15',
        'label': 'Meme/Archegos 2021',
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

# Pre-SPY crises that require CRSP index data (not available via yfinance ETFs).
# Kept separate; requires WRDS access.
CRSP_ONLY_CRISES = {
    '1987_crash': {
        'start': '1987-10-14', 'end': '1987-11-30',
        'label': 'Black Monday 1987',
    },
}

# SPY-era crises (post-1993): accessible via yfinance (free) or WRDS CRSP (institutional)
SPY_ERA_CRISES = {k: v for k, v in ALL_CRISES.items()}

# All crises including CRSP-only (for WRDS-based analysis)
ALL_CRISES_EXTENDED = {**CRSP_ONLY_CRISES, **ALL_CRISES}

# Pre-defined BEFORE seeing results (Review Item 8).
# Novel = new market mechanisms without historical precedent.
# Conventional = crises with recognizable historical parallels.
CRISIS_CATEGORIES = {
    'novel': [
        '2016_brexit',       # Geopolitical shock, no prior analog
        '2018_volmageddon',  # Unprecedented short-vol blowup
        '2018_q4',           # Algorithmic-driven selloff
        '2019_repo',         # Plumbing crisis, no equity analog
        '2021_meme',         # Hidden stress (SPY calm, internals extreme)
        '2022_rates',        # Fastest rate cycle in 40 years
        '2023_svb',          # Social-media-driven bank run
        '2024_carry',        # Yen carry unwind + AI rotation
    ],
    'conventional': [
        '1997_asia',    # Asian contagion (EM crisis archetype)
        '1998_ltcm',    # Leverage/liquidity crisis (resembles 2007 quant)
        '2000_dotcom',  # Valuation bubble burst (resembles 1929)
        '2001_911',     # Exogenous shock (resembles 2020 COVID)
        '2007_quant',   # Factor crowding (resembles LTCM)
        '2008_gfc',     # Credit crisis (resembles 1929, 1987)
        '2010_flash',   # Liquidity shock (resembles 1987)
        '2011_euro',    # Sovereign debt (resembles EM crises)
        '2013_taper',   # Rate shock (resembles 2022, but milder)
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
# Data Caching (DVC-compatible)
# =============================================================================

def _cache_key(symbols, start_date, end_date, source='yfinance'):
    """Generate a deterministic cache key for a data fetch request."""
    key_str = f"{sorted(symbols)}_{start_date}_{end_date}_{source}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def cache_data(df, symbols, start_date, end_date, source='yfinance'):
    """Save fetched data as a parquet file in data/ for DVC tracking.

    Args:
        df: DataFrame with MultiIndex (symbol, timestamp).
        symbols: List of ticker symbols.
        start_date: Start date string.
        end_date: End date string.
        source: Data source name.

    Returns:
        Path to the saved parquet file.
    """
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(symbols, start_date, end_date, source)
    path = DATA_CACHE_DIR / f'market_data_{key}.parquet'
    df.to_parquet(path)
    logger.info(f"Cached data to {path}")
    return path


def load_cached_data(symbols, start_date, end_date, source='yfinance'):
    """Load cached data if available.

    Args:
        symbols: List of ticker symbols.
        start_date: Start date string.
        end_date: End date string.
        source: Data source name.

    Returns:
        DataFrame if cache hit, None if cache miss.
    """
    key = _cache_key(symbols, start_date, end_date, source)
    path = DATA_CACHE_DIR / f'market_data_{key}.parquet'
    if path.exists():
        logger.info(f"Cache hit: {path}")
        return pd.read_parquet(path)
    return None


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


def fetch_yfinance_data(symbols, start_date, end_date):
    """Fetch daily OHLCV from Yahoo Finance via yfinance.

    Free, no API key required. 30+ years of history for liquid ETFs.
    Returns split/dividend-adjusted prices (same adjustment as Polygon default).

    Args:
        symbols: List of ticker symbols, e.g. ['SPY', 'DIA'].
        start_date: Start date string 'YYYY-MM-DD'.
        end_date: End date string 'YYYY-MM-DD'.

    Returns:
        DataFrame with MultiIndex (symbol, timestamp) and columns
        [open, high, low, close, volume].
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    raw = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {symbols} {start_date}:{end_date}")

    # yfinance 0.2.x returns MultiIndex columns (field, ticker) for any input
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns.names = ['field', 'symbol']
        df = raw.stack(level='symbol', future_stack=True)
        df.columns = [c.lower() for c in df.columns]
        df.index = df.index.rename(['timestamp', 'symbol'])
        df = df.swaplevel().sort_index()
    else:
        # Flat columns — single ticker, older yfinance behaviour
        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        df.index.name = 'timestamp'
        df.index = pd.MultiIndex.from_tuples(
            [(symbols[0], t) for t in df.index],
            names=['symbol', 'timestamp'],
        )

    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_data(symbols, start_date, end_date, source='yfinance', use_cache=True):
    """Unified data fetcher. Uses yfinance by default (free, reproducible).

    Args:
        symbols: List of ticker symbols, e.g. ['SPY', 'DIA'].
        start_date: Start date string 'YYYY-MM-DD'.
        end_date: End date string 'YYYY-MM-DD'.
        source: 'yfinance' (default), 'polygon', 'wrds', or 'auto'.
            'auto' tries WRDS if configured, otherwise uses yfinance.
            WARNING: WRDS returns split-adjusted but NOT dividend-adjusted
            prices, which differ from yfinance's fully-adjusted prices.
        use_cache: If True, check data/ for cached parquet files before fetching.
            Cached files are DVC-tracked for reproducibility.

    Returns:
        DataFrame with MultiIndex (symbol, timestamp) and columns
        [open, high, low, close, volume].
    """
    if use_cache:
        cached = load_cached_data(symbols, start_date, end_date, source)
        if cached is not None:
            return cached

    if source == 'yfinance':
        logger.info("Data source: yfinance")
        df = fetch_yfinance_data(symbols, start_date, end_date)
    elif source == 'polygon':
        logger.info("Data source: polygon")
        df = fetch_polygon_data(symbols, start_date, end_date)
    else:
        # source == 'auto' or 'wrds': try WRDS first, fall back to yfinance
        try:
            from experiments.wrds_data_loader import (
                fetch_wrds_equities, wrds_prices_to_polygon_format,
            )
            from dotenv import load_dotenv
            load_dotenv(ROOT / '.env')
            if not os.environ.get('WRDS_USERNAME'):
                raise RuntimeError("WRDS_USERNAME not set")
            prices_wide = fetch_wrds_equities(symbols, start_date, end_date)
            logger.info("Data source: wrds")
            df = wrds_prices_to_polygon_format(prices_wide)
        except Exception as e:
            logger.warning(f"WRDS fetch failed ({e}), falling back to yfinance")
            logger.info("Data source: yfinance (fallback from auto)")
            df = fetch_yfinance_data(symbols, start_date, end_date)

    if use_cache:
        cache_data(df, symbols, start_date, end_date, source)

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
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()

    return feat_df.values, feat_df.index


def create_feature_matrix_single_asset(prices_series, extra_lags=True):
    """Create feature matrix from a single price series.

    Unlike create_feature_matrix() which needs >= 2 symbols for cross-sectional
    features, this works with a single asset by using only per-asset features
    plus additional lag/momentum features to reach sufficient dimensionality.

    Args:
        prices_series: Series with DatetimeIndex and close prices, or
            single-column DataFrame.
        extra_lags: If True, add extra rolling windows (10, 40, 60) to
            increase dimensionality for PCA.

    Returns:
        features: np.ndarray (T', d) after warmup. d >= 15 if extra_lags=True.
        dates: DatetimeIndex aligned with features.
    """
    if isinstance(prices_series, pd.DataFrame):
        if prices_series.shape[1] == 1:
            prices_series = prices_series.iloc[:, 0]
        else:
            # If multi-column, use create_feature_matrix instead
            return create_feature_matrix(prices_series)

    log_ret = np.log(prices_series / prices_series.shift(1))

    features_dict = {
        'ret': log_ret,
        'vol5': log_ret.rolling(5).std(),
        'vol20': log_ret.rolling(20).std(),
        'mom5': prices_series.pct_change(5),
        'mom20': prices_series.pct_change(20),
    }

    if extra_lags:
        features_dict.update({
            'vol10': log_ret.rolling(10).std(),
            'vol40': log_ret.rolling(40).std(),
            'vol60': log_ret.rolling(60).std(),
            'mom10': prices_series.pct_change(10),
            'mom40': prices_series.pct_change(40),
            'mom60': prices_series.pct_change(60),
            'ret_sq': log_ret ** 2,
            'range_20': (
                prices_series.rolling(20).max() / prices_series.rolling(20).min() - 1
            ),
            'skew_20': log_ret.rolling(20).skew(),
            'kurt_20': log_ret.rolling(20).kurt(),
        })

    feat_df = pd.DataFrame(features_dict)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()

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

    raw = fetch_data(symbols, start, end)
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)
    return X, dates, prices_df


# =============================================================================
# Legacy-compatible wrappers (for tests that import experiments.data)
# =============================================================================

class PolygonDataSource:
    """Wrapper matching the old qcml.data.PolygonDataSource interface.

    Now uses yfinance by default via fetch_data().
    """

    def fetch_equities(self, symbols, start_date, end_date):
        return fetch_data(symbols, start_date, end_date)


class MinimalFeatureEngine:
    """Wrapper matching the old qcml.data.MinimalFeatureEngine interface."""

    def __init__(self, window=20):
        self.window = window

    def create_feature_matrix(self, prices_df):
        """Return a DataFrame (not tuple) for backward compatibility."""
        arr, dates = create_feature_matrix(prices_df)
        return pd.DataFrame(arr, index=dates)
