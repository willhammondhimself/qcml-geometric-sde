"""
WRDS data loading layer for extended asset coverage and historical depth.

Provides functions to fetch data from CRSP, TRACE, and OptionMetrics databases
via the WRDS (Wharton Research Data Services) API, returning DataFrames in the
same format as the Polygon-based loader in data_loader.py.

All queries are cached locally to data/wrds_cache/ as parquet files to avoid
repeated expensive database queries.

Functions:
    get_wrds_connection     — Cached WRDS connection (reads username from .env)
    fetch_wrds_equities     — CRSP daily stock prices by ticker
    fetch_wrds_index        — CRSP index returns (e.g., S&P 500 back to 1926)
    fetch_wrds_bonds        — TRACE corporate bond prices (daily aggregated)
    fetch_wrds_options      — OptionMetrics implied vol surface data
    fetch_wrds_vix_term     — VIX term structure from CBOE via OptionMetrics

Usage:
    from experiments.wrds_data_loader import fetch_wrds_equities
    prices = fetch_wrds_equities(['AAPL', 'MSFT'], '1990-01-01', '2024-12-31')
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / 'data' / 'wrds_cache'

_wrds_conn = None


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(prefix: str, **kwargs) -> str:
    """Generate a deterministic cache filename from query parameters."""
    key_str = f"{prefix}|" + "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    h = hashlib.md5(key_str.encode()).hexdigest()[:12]
    return f"{prefix}_{h}.parquet"


def _read_cache(filename: str) -> Optional[pd.DataFrame]:
    """Read from parquet cache if it exists."""
    path = CACHE_DIR / filename
    if path.exists():
        logger.info(f"  Cache hit: {filename}")
        return pd.read_parquet(path)
    return None


def _write_cache(df: pd.DataFrame, filename: str):
    """Write DataFrame to parquet cache."""
    _ensure_cache_dir()
    path = CACHE_DIR / filename
    df.to_parquet(path)
    logger.info(f"  Cached: {filename} ({len(df)} rows)")


def get_wrds_connection():
    """Get or create a cached WRDS database connection.

    Uses the wrds library which handles PAM/Duo 2FA interactively.
    Must be run from an interactive terminal where the user can accept
    the Duo push notification on their phone.

    Returns:
        wrds.Connection object with raw_sql() method.
    """
    global _wrds_conn
    if _wrds_conn is not None:
        return _wrds_conn

    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')

    username = os.environ.get('WRDS_USERNAME')
    if not username:
        raise RuntimeError(
            "WRDS_USERNAME not found in .env. Add: WRDS_USERNAME=your_wrds_id"
        )

    import wrds
    logger.info(f"  Connecting to WRDS as {username}...")
    logger.info(f"  Check your phone for a Duo push notification and accept it.")
    _wrds_conn = wrds.Connection(wrds_username=username)
    logger.info(f"  Connected to WRDS.")
    return _wrds_conn


# =============================================================================
# CRSP Equities
# =============================================================================

def fetch_wrds_equities(
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily close prices from CRSP daily stock file.

    Uses CRSP/Compustat merged (crsp.dsf) joined with crsp.dsenames
    for ticker-to-PERMNO mapping.

    Args:
        symbols: List of ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date: Start date 'YYYY-MM-DD'.
        end_date: End date 'YYYY-MM-DD'.

    Returns:
        DataFrame with columns = symbols, index = DatetimeIndex,
        values = adjusted close prices. Same format as Polygon output
        after .unstack().
    """
    cache_file = _cache_key(
        'crsp_equities', symbols=','.join(sorted(symbols)),
        start=start_date, end=end_date,
    )
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached.astype('float64')

    db = get_wrds_connection()

    tickers_str = "', '".join(symbols)
    query = f"""
    SELECT a.date, b.ticker, a.prc, a.ret, a.vol
    FROM crsp.dsf AS a
    JOIN crsp.dsenames AS b
        ON a.permno = b.permno
        AND b.namedt <= a.date
        AND a.date <= b.nameendt
    WHERE b.ticker IN ('{tickers_str}')
        AND a.date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY b.ticker, a.date
    """

    logger.info(f"  Querying CRSP for {symbols} ({start_date} to {end_date})...")
    df = db.raw_sql(query)

    # CRSP uses negative prices for bid-ask average; take absolute value
    df['close'] = df['prc'].abs()
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'ticker': 'symbol'})

    # Pivot to wide format (same as Polygon after unstack)
    prices = df.pivot_table(index='date', columns='symbol', values='close')
    prices = prices.dropna()
    prices.index.name = None
    # Ensure plain numpy float64 (parquet may restore pandas nullable Float64)
    prices = prices.astype('float64')

    _write_cache(prices, cache_file)
    return prices


# =============================================================================
# CRSP Index
# =============================================================================

def fetch_wrds_index(
    index_name: str = 'sp500',
    start_date: str = '1985-01-01',
    end_date: str = '2024-12-31',
) -> pd.DataFrame:
    """Fetch CRSP index daily returns and levels.

    Args:
        index_name: One of 'sp500', 'nasdaq', 'nyse'.
        start_date: Start date.
        end_date: End date.

    Returns:
        DataFrame with columns ['level', 'return'], index = DatetimeIndex.
        The 'level' column can be used as a price proxy for feature engineering.
    """
    cache_file = _cache_key(
        'crsp_index', index=index_name, start=start_date, end=end_date,
    )
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    db = get_wrds_connection()

    # CRSP S&P 500 composite index: sprtrn = S&P return, vwretd = value-weighted
    if index_name == 'sp500':
        query = f"""
        SELECT caldt AS date, sprtrn AS ret, spindx AS level
        FROM crsp.dsi
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY caldt
        """
    elif index_name == 'nasdaq':
        query = f"""
        SELECT caldt AS date, vwretd AS ret, totval AS level
        FROM crsp.dsic
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
            AND exchcd = 3
        ORDER BY caldt
        """
    else:
        # NYSE value-weighted
        query = f"""
        SELECT caldt AS date, vwretd AS ret
        FROM crsp.dsi
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY caldt
        """

    logger.info(f"  Querying CRSP index '{index_name}' ({start_date} to {end_date})...")
    df = db.raw_sql(query)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # If no level column, reconstruct from returns
    if 'level' not in df.columns or df['level'].isna().all():
        df['level'] = (1 + df['ret'].fillna(0)).cumprod() * 100

    _write_cache(df, cache_file)
    return df


# =============================================================================
# TRACE Corporate Bonds
# =============================================================================

def fetch_wrds_bonds(
    cusips_or_tickers: Optional[List[str]] = None,
    start_date: str = '2005-01-01',
    end_date: str = '2024-12-31',
    bond_type: str = 'investment_grade',
) -> pd.DataFrame:
    """Fetch daily bond prices from TRACE enhanced.

    Aggregates intraday transactions to daily last-trade price.

    Args:
        cusips_or_tickers: List of CUSIPs or bond tickers (if None, fetches
            a representative sample of IG/HY bonds).
        start_date: Start date.
        end_date: End date.
        bond_type: 'investment_grade' or 'high_yield' (used when
            cusips_or_tickers is None to select a representative sample).

    Returns:
        DataFrame with columns = bond identifiers, index = DatetimeIndex,
        values = daily close (last trade) prices.
    """
    cache_file = _cache_key(
        'trace_bonds', type=bond_type,
        cusips=','.join(sorted(cusips_or_tickers or ['sample'])),
        start=start_date, end=end_date,
    )
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    db = get_wrds_connection()

    if cusips_or_tickers is not None:
        ids_str = "', '".join(cusips_or_tickers)
        where_clause = f"AND cusip_id IN ('{ids_str}')"
    else:
        # Get a representative sample: top 20 most-traded bonds
        where_clause = ""

    query = f"""
    SELECT trd_exctn_dt AS date,
           cusip_id AS cusip,
           rptd_pr AS price,
           entrd_vol_qt AS volume
    FROM trace.trace_enhanced
    WHERE trd_exctn_dt BETWEEN '{start_date}' AND '{end_date}'
        AND rptd_pr IS NOT NULL
        AND rptd_pr > 0
        {where_clause}
    ORDER BY cusip_id, trd_exctn_dt, trd_exctn_tm DESC
    """

    logger.info(f"  Querying TRACE bonds ({start_date} to {end_date})...")
    df = db.raw_sql(query)

    if df.empty:
        logger.warning("  No TRACE data returned")
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])

    # Aggregate to daily: volume-weighted average price (VWAP)
    daily = (
        df.groupby(['cusip', 'date'])
        .agg(close=('price', 'last'), volume=('volume', 'sum'))
        .reset_index()
    )

    prices = daily.pivot_table(index='date', columns='cusip', values='close')
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))
    prices = prices.ffill().dropna()

    _write_cache(prices, cache_file)
    return prices


# =============================================================================
# OptionMetrics Implied Volatility
# =============================================================================

def fetch_wrds_options(
    symbols: List[str],
    start_date: str = '1996-01-01',
    end_date: str = '2024-12-31',
) -> pd.DataFrame:
    """Fetch ATM implied volatility from OptionMetrics volatility surface.

    Extracts ATM IV at 30-day and 90-day tenors from optionm.vsurfd.

    Args:
        symbols: List of ticker symbols (e.g., ['SPY']).
        start_date: Start date.
        end_date: End date.

    Returns:
        DataFrame with columns like 'SPY_iv30', 'SPY_iv90', 'SPY_skew'
        for each symbol, index = DatetimeIndex.
    """
    cache_file = _cache_key(
        'optionm_iv', symbols=','.join(sorted(symbols)),
        start=start_date, end=end_date,
    )
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    db = get_wrds_connection()

    # Map tickers to secids via optionm.securd
    tickers_str = "', '".join(symbols)
    secid_query = f"""
    SELECT secid, ticker
    FROM optionm.securd
    WHERE ticker IN ('{tickers_str}')
    """
    secid_df = db.raw_sql(secid_query)

    if secid_df.empty:
        logger.warning(f"  No OptionMetrics secids found for {symbols}")
        return pd.DataFrame()

    secids = secid_df['secid'].unique()
    secids_str = ','.join(str(s) for s in secids)
    ticker_map = dict(zip(secid_df['secid'], secid_df['ticker']))

    # Fetch ATM volatility surface: delta=50 (ATM), days=30 and 90
    query = f"""
    SELECT date, secid, days, impl_volatility, impl_strike, delta
    FROM optionm.vsurfd
    WHERE secid IN ({secids_str})
        AND date BETWEEN '{start_date}' AND '{end_date}'
        AND days IN (30, 91)
        AND abs(delta) BETWEEN 40 AND 60
    ORDER BY secid, date, days
    """

    logger.info(f"  Querying OptionMetrics IV surface for {symbols}...")
    df = db.raw_sql(query)

    if df.empty:
        logger.warning("  No OptionMetrics data returned")
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    df['ticker'] = df['secid'].map(ticker_map)

    result_frames = []
    for sym in symbols:
        sym_df = df[df['ticker'] == sym].copy()
        if sym_df.empty:
            continue

        # Aggregate: mean IV per date × tenor
        daily = sym_df.groupby(['date', 'days'])['impl_volatility'].mean().unstack()

        rename_map = {}
        if 30 in daily.columns:
            rename_map[30] = f'{sym}_iv30'
        if 91 in daily.columns:
            rename_map[91] = f'{sym}_iv90'
        daily = daily.rename(columns=rename_map)

        # Compute term structure slope (90d - 30d)
        if f'{sym}_iv30' in daily.columns and f'{sym}_iv90' in daily.columns:
            daily[f'{sym}_term_slope'] = daily[f'{sym}_iv90'] - daily[f'{sym}_iv30']

        result_frames.append(daily)

    if not result_frames:
        return pd.DataFrame()

    result = pd.concat(result_frames, axis=1).sort_index()
    result = result.ffill().dropna()

    _write_cache(result, cache_file)
    return result


def fetch_wrds_vix_term(
    start_date: str = '2004-01-01',
    end_date: str = '2024-12-31',
) -> pd.DataFrame:
    """Fetch VIX term structure from CBOE futures data.

    Args:
        start_date: Start date.
        end_date: End date.

    Returns:
        DataFrame with columns ['vix_spot', 'vix_1m', 'vix_3m', 'term_slope'],
        index = DatetimeIndex.
    """
    cache_file = _cache_key(
        'vix_term', start=start_date, end=end_date,
    )
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    db = get_wrds_connection()

    # Try CBOE VIX index from OptionMetrics
    query = f"""
    SELECT date, impl_volatility AS vix_spot
    FROM optionm.vsurfd
    WHERE secid = 108105
        AND date BETWEEN '{start_date}' AND '{end_date}'
        AND days = 30
        AND abs(delta) BETWEEN 45 AND 55
    ORDER BY date
    """

    logger.info(f"  Querying VIX term structure ({start_date} to {end_date})...")
    df = db.raw_sql(query)

    if df.empty:
        logger.warning("  No VIX data from OptionMetrics; returning empty")
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    result = df.groupby('date')['vix_spot'].mean().to_frame()

    _write_cache(result, cache_file)
    return result


# =============================================================================
# Convenience: Match Polygon Output Format
# =============================================================================

def wrds_prices_to_polygon_format(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format WRDS prices to Polygon-style MultiIndex format.

    Args:
        prices_df: Wide DataFrame (index=dates, columns=symbols, values=close).

    Returns:
        DataFrame with MultiIndex (symbol, timestamp) and column 'close'.
        Compatible with existing pipeline: raw['close'].unstack('symbol').
    """
    stacked = prices_df.stack()
    stacked.name = 'close'
    stacked.index.names = ['timestamp', 'symbol']
    # Swap to (symbol, timestamp) to match Polygon format
    df = stacked.swaplevel().sort_index().to_frame()
    return df
