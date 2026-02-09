"""
Data Acquisition Module

Fetches market data from Polygon.io and Alpaca APIs with fallback support,
rate limiting, and comprehensive error handling.

Classes:
    PolygonDataSource: Primary API client for Polygon.io
    AlpacaDataSource: Fallback API client for Alpaca
    UniverseManager: Trading universe and symbol list management
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

try:
    from polygon import RESTClient as PolygonClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    logging.warning("polygon package not installed. PolygonDataSource will not be available.")

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("alpaca-py package not installed. AlpacaDataSource will not be available.")

logger = logging.getLogger(__name__)


@dataclass
class DataFetchConfig:
    """Configuration for data fetching"""
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.2
    timeout: int = 30


class PolygonDataSource:
    """
    Polygon.io API client for market data

    Handles fetching OHLCV data for equities, options chains, and real-time snapshots
    with automatic rate limiting and retry logic.

    Args:
        api_key: Polygon.io API key (defaults to POLYGON_API_KEY env var)
        config: Optional DataFetchConfig for retry and rate limiting settings

    Example:
        >>> source = PolygonDataSource()
        >>> df = source.fetch_equities(["AAPL", "MSFT"], "2024-01-01", "2024-01-31")
        >>> print(df.head())
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[DataFetchConfig] = None
    ):
        if not POLYGON_AVAILABLE:
            raise ImportError("polygon package required. Install with: pip install polygon-api-client")

        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set POLYGON_API_KEY environment variable or pass api_key parameter")

        self.client = PolygonClient(self.api_key)
        self.config = config or DataFetchConfig()
        logger.info("PolygonDataSource initialized")

    def fetch_equities(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for equities

        Args:
            symbols: List of ticker symbols (e.g., ["AAPL", "MSFT"])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Time resolution - "1d" (daily), "1h" (hourly), "1m" (minute)

        Returns:
            DataFrame with columns: [symbol, timestamp, open, high, low, close, volume]
            MultiIndex: (symbol, timestamp)

        Example:
            >>> df = source.fetch_equities(["AAPL"], "2024-01-01", "2024-01-31", "1d")
            >>> print(df.loc["AAPL"].head())
        """
        logger.info(f"Fetching {len(symbols)} symbols from {start_date} to {end_date} ({timeframe})")

        all_data = []

        for symbol in symbols:
            try:
                data = self._fetch_single_equity(symbol, start_date, end_date, timeframe)
                if data is not None and not data.empty:
                    data['symbol'] = symbol
                    all_data.append(data)

                # Rate limiting
                time.sleep(self.config.rate_limit_delay)

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue

        if not all_data:
            logger.warning("No data fetched for any symbols")
            return pd.DataFrame()

        # Combine all symbols
        df = pd.concat(all_data, ignore_index=True)

        # Set MultiIndex
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index(['symbol', 'timestamp']).sort_index()

        logger.info(f"Fetched {len(df)} total bars across {len(symbols)} symbols")
        return df

    def _fetch_single_equity(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single equity with retry logic"""

        for attempt in range(self.config.max_retries):
            try:
                # Convert timeframe to Polygon format
                multiplier, span = self._parse_timeframe(timeframe)

                # Fetch bars
                bars = self.client.get_aggs(
                    ticker=symbol,
                    multiplier=multiplier,
                    timespan=span,
                    from_=start_date,
                    to=end_date,
                    limit=50000
                )

                if not bars:
                    logger.warning(f"No data returned for {symbol}")
                    return None

                # Convert to DataFrame
                data = []
                for bar in bars:
                    data.append({
                        'timestamp': pd.Timestamp(bar.timestamp, unit='ms'),
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    })

                return pd.DataFrame(data)

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.config.max_retries} failed for {symbol}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All retries exhausted for {symbol}")
                    return None

    def _parse_timeframe(self, timeframe: str) -> Tuple[int, str]:
        """Parse timeframe string into Polygon API format"""
        timeframe = timeframe.lower()

        if timeframe.endswith('d'):
            return int(timeframe[:-1]) if timeframe[:-1] else 1, 'day'
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) if timeframe[:-1] else 1, 'hour'
        elif timeframe.endswith('m'):
            return int(timeframe[:-1]) if timeframe[:-1] else 1, 'minute'
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}. Use format like '1d', '1h', '5m'")

    def fetch_options_chain(
        self,
        underlying: str,
        date: str
    ) -> pd.DataFrame:
        """
        Fetch options chain data for a given underlying and date

        Args:
            underlying: Underlying ticker (e.g., "SPY")
            date: Date in YYYY-MM-DD format

        Returns:
            DataFrame with options contract data

        Note:
            Requires Polygon.io options data subscription
        """
        logger.info(f"Fetching options chain for {underlying} on {date}")

        try:
            # This is a placeholder - actual implementation depends on Polygon options API
            # which requires a higher tier subscription
            logger.warning("Options chain fetching not yet implemented")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch options chain: {e}")
            return pd.DataFrame()

    def fetch_market_snapshot(
        self,
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Fetch real-time market snapshot for given symbols

        Args:
            symbols: List of ticker symbols

        Returns:
            DataFrame with current market data (last price, bid/ask, volume, etc.)
        """
        logger.info(f"Fetching market snapshot for {len(symbols)} symbols")

        try:
            # Placeholder for snapshot API
            logger.warning("Market snapshot fetching not yet implemented")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch market snapshot: {e}")
            return pd.DataFrame()


class AlpacaDataSource:
    """
    Alpaca API fallback client

    Provides fallback data source when Polygon.io is unavailable or for
    assets not covered by Polygon.

    Args:
        api_key: Alpaca API key (defaults to ALPACA_API_KEY env var)
        secret_key: Alpaca secret key (defaults to ALPACA_SECRET_KEY env var)
        config: Optional DataFetchConfig

    Example:
        >>> source = AlpacaDataSource()
        >>> df = source.fetch_bars(["AAPL"], "2024-01-01", "2024-01-31")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        config: Optional[DataFetchConfig] = None
    ):
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py package required. Install with: pip install alpaca-py")

        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError("API credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")

        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.config = config or DataFetchConfig()
        logger.info("AlpacaDataSource initialized")

    def fetch_bars(
        self,
        symbols: List[str],
        start: str,
        end: str,
        timeframe: str = "1Day"
    ) -> pd.DataFrame:
        """
        Fetch bar data from Alpaca

        Args:
            symbols: List of ticker symbols
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            timeframe: Alpaca TimeFrame (e.g., "1Day", "1Hour", "1Min")

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {len(symbols)} symbols from Alpaca ({timeframe})")

        try:
            # Parse timeframe
            tf = self._parse_alpaca_timeframe(timeframe)

            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end)
            )

            # Fetch data
            bars = self.client.get_stock_bars(request)

            # Convert to DataFrame
            df = bars.df

            if df.empty:
                logger.warning("No data returned from Alpaca")
                return df

            # Rename columns to match Polygon format
            df = df.rename(columns={'timestamp': 'timestamp'})
            df = df.reset_index()

            logger.info(f"Fetched {len(df)} bars from Alpaca")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch from Alpaca: {e}")
            return pd.DataFrame()

    def _parse_alpaca_timeframe(self, timeframe: str):
        """Parse timeframe string into Alpaca TimeFrame object"""
        timeframe = timeframe.lower()

        if 'day' in timeframe:
            amount = int(timeframe.replace('day', '')) if timeframe.replace('day', '') else 1
            return TimeFrame(amount, TimeFrameUnit.Day)
        elif 'hour' in timeframe:
            amount = int(timeframe.replace('hour', '')) if timeframe.replace('hour', '') else 1
            return TimeFrame(amount, TimeFrameUnit.Hour)
        elif 'min' in timeframe:
            amount = int(timeframe.replace('min', '')) if timeframe.replace('min', '') else 1
            return TimeFrame(amount, TimeFrameUnit.Minute)
        else:
            raise ValueError(f"Invalid Alpaca timeframe: {timeframe}")


class UniverseManager:
    """
    Manage trading universe and symbol lists

    Provides utilities for getting predefined universes like S&P 500,
    sector ETFs, and filtered liquid assets.

    Example:
        >>> manager = UniverseManager()
        >>> sp500 = manager.get_sp500_constituents()
        >>> liquid = manager.get_liquid_universe(min_volume=5e6)
    """

    def __init__(self):
        logger.info("UniverseManager initialized")

    def get_sp500_constituents(self) -> List[str]:
        """
        Get current S&P 500 constituents

        Returns:
            List of S&P 500 ticker symbols

        Note:
            This is a static list. For production, fetch from a live source
            like Wikipedia or Index provider APIs.
        """
        # Top 50 S&P 500 by market cap (as of 2024)
        # In production, this should be fetched dynamically
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'UNH', 'JNJ', 'V', 'XOM', 'WMT', 'LLY', 'JPM', 'MA', 'PG', 'AVGO',
            'HD', 'CVX', 'MRK', 'ABBV', 'KO', 'PEP', 'COST', 'ADBE', 'BAC',
            'CRM', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT', 'NFLX', 'AMD', 'NKE',
            'TXN', 'DHR', 'PM', 'DIS', 'VZ', 'WFC', 'ORCL', 'INTC', 'CMCSA',
            'QCOM', 'UPS', 'HON', 'IBM', 'AMGN'
        ]

    def get_sector_etfs(self) -> Dict[str, List[str]]:
        """
        Get sector ETF tickers mapped by sector

        Returns:
            Dictionary mapping sector names to ETF tickers

        Example:
            >>> sectors = manager.get_sector_etfs()
            >>> print(sectors['Technology'])
            ['XLK', 'VGT', 'IYW']
        """
        return {
            'Technology': ['XLK', 'VGT', 'IYW'],
            'Financials': ['XLF', 'VFH', 'IYF'],
            'Healthcare': ['XLV', 'VHT', 'IYH'],
            'Consumer Discretionary': ['XLY', 'VCR', 'IYC'],
            'Consumer Staples': ['XLP', 'VDC', 'IYK'],
            'Energy': ['XLE', 'VDE', 'IYE'],
            'Industrials': ['XLI', 'VIS', 'IYJ'],
            'Materials': ['XLB', 'VAW', 'IYM'],
            'Real Estate': ['XLRE', 'VNQ', 'IYR'],
            'Utilities': ['XLU', 'VPU', 'IDU'],
            'Communication Services': ['XLC', 'VOX', 'IYZ'],
            'Broad Market': ['SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI']
        }

    def get_liquid_universe(
        self,
        min_volume: float = 1e6,
        min_price: float = 5.0
    ) -> List[str]:
        """
        Filter for liquid assets

        Args:
            min_volume: Minimum average daily volume
            min_price: Minimum price per share

        Returns:
            List of liquid ticker symbols

        Note:
            This is a placeholder. In production, fetch real-time volume
            and price data to filter dynamically.
        """
        logger.info(f"Getting liquid universe (min_volume={min_volume}, min_price={min_price})")

        # For now, return S&P 500 as they're all liquid
        # In production, fetch actual volume/price data and filter
        return self.get_sp500_constituents()

    def get_crisis_universe(self, crisis_name: str) -> List[str]:
        """
        Get universe appropriate for historical crisis analysis

        Args:
            crisis_name: "2008_financial", "2020_covid", "2022_rates", etc.

        Returns:
            List of ticker symbols relevant for that crisis
        """
        if crisis_name == "2008_financial":
            # Focus on financials and broad market
            return ['SPY', 'XLF', 'BAC', 'JPM', 'C', 'GS', 'MS', 'WFC', 'USB', 'PNC']

        elif crisis_name == "2020_covid":
            # Broad market + sectors most impacted
            return ['SPY', 'QQQ', 'XLF', 'XLE', 'XLK', 'XLV', 'XLY', 'IWM']

        elif crisis_name == "2022_rates":
            # Rate-sensitive sectors
            return ['SPY', 'XLF', 'XLRE', 'XLU', 'TLT', 'IEF', 'SHY']

        else:
            logger.warning(f"Unknown crisis: {crisis_name}. Returning default universe")
            return self.get_sp500_constituents()[:20]
