"""
Full Feature Engineering Module for QCML Regime Detection

Production-grade feature engineering with 30+ technical indicators across 5 categories:
- Volatility: rolling_vol, ATR, Parkinson, Garman-Klass
- Momentum: EMA, RSI, MACD, ROC
- Mean Reversion: Bollinger Bands, Z-score, Stochastic
- Volume: OBV, Volume Ratio, MFI
- Cross-Sectional: Beta, Correlation, Relative Strength, Dispersion

Design Decisions:
- Separate class from MinimalFeatureEngine for clarity (different responsibility)
- Implemented from scratch to control lookahead bias (no pandas-ta dependency)
- Daily timeframe focus (validated hypothesis uses daily data)
- Single class with category system for easier maintenance

Author: QCML Research
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Feature categories for selective feature engineering."""
    VOLATILITY = auto()
    MOMENTUM = auto()
    MEAN_REVERSION = auto()
    VOLUME = auto()
    CROSS_SECTIONAL = auto()
    ALL = auto()


@dataclass
class FeatureConfig:
    """
    Configuration for feature engineering.

    Args:
        categories: Set of FeatureCategory to compute (default: ALL)
        volatility_windows: Windows for rolling volatility [5, 10, 20, 60]
        momentum_windows: Windows for momentum indicators
        ema_spans: Spans for EMA calculation [12, 26]
        rsi_period: Period for RSI calculation (default: 14)
        macd_fast: Fast period for MACD (default: 12)
        macd_slow: Slow period for MACD (default: 26)
        macd_signal: Signal period for MACD (default: 9)
        bollinger_window: Window for Bollinger Bands (default: 20)
        bollinger_std: Standard deviations for Bollinger Bands (default: 2.0)
        stochastic_k: %K period for Stochastic (default: 14)
        stochastic_d: %D smoothing period (default: 3)
        benchmark_col: Column name of benchmark (default: 'SPY')
        beta_window: Window for beta calculation (default: 60)
        correlation_window: Window for correlation (default: 20)
        min_periods_ratio: Minimum periods as ratio of window (default: 0.5)

    Example:
        >>> config = FeatureConfig(
        ...     categories={FeatureCategory.VOLATILITY, FeatureCategory.MOMENTUM},
        ...     volatility_windows=[10, 20],
        ...     benchmark_col='SPY'
        ... )
    """
    categories: Set[FeatureCategory] = field(
        default_factory=lambda: {FeatureCategory.ALL}
    )
    # Volatility settings
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    atr_period: int = 14

    # Momentum settings
    ema_spans: List[int] = field(default_factory=lambda: [12, 26])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    roc_period: int = 10

    # Mean Reversion settings
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    stochastic_k: int = 14
    stochastic_d: int = 3
    zscore_window: int = 20

    # Volume settings
    mfi_period: int = 14
    volume_ratio_window: int = 20

    # Cross-sectional settings
    benchmark_col: str = "SPY"
    beta_window: int = 60
    correlation_window: int = 20

    # General settings
    min_periods_ratio: float = 0.5

    def should_compute(self, category: FeatureCategory) -> bool:
        """Check if a category should be computed."""
        return FeatureCategory.ALL in self.categories or category in self.categories


class FeatureEngine:
    """
    Full-featured technical indicator engine for QCML regime detection.

    Computes 30+ technical indicators across 5 categories for rich feature
    representation suitable for topological regime detection.

    Args:
        config: FeatureConfig with all settings (default: FeatureConfig())

    Example:
        >>> engine = FeatureEngine()
        >>> # With just prices (close only)
        >>> features = engine.create_feature_matrix(prices_df, benchmark_col='SPY')
        >>>
        >>> # With OHLCV data for full features
        >>> features = engine.create_feature_matrix(
        ...     prices_df, ohlcv_df=ohlcv_data, benchmark_col='SPY'
        ... )
        >>> print(f"Features: {features.shape}")
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        logger.info(f"FeatureEngine initialized with categories: {self.config.categories}")

    # =========================================================================
    # VOLATILITY INDICATORS (7 features)
    # =========================================================================

    def compute_returns(self, prices: pd.Series) -> pd.Series:
        """
        Compute log returns from price series.

        Args:
            prices: Price series

        Returns:
            Log returns series (first value is NaN)
        """
        return np.log(prices / prices.shift(1))

    def compute_rolling_volatility(
        self,
        returns: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Compute rolling volatility (standard deviation of returns).

        Args:
            returns: Returns series
            window: Rolling window size

        Returns:
            Rolling volatility series (annualized by sqrt(252))
        """
        min_periods = max(1, int(window * self.config.min_periods_ratio))
        return returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(252)

    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """
        Compute Average True Range (ATR).

        True Range = max(high - low, |high - prev_close|, |low - prev_close|)

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period (default: config.atr_period)

        Returns:
            ATR series
        """
        period = period or self.config.atr_period
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        min_periods = max(1, int(period * self.config.min_periods_ratio))
        return true_range.rolling(window=period, min_periods=min_periods).mean()

    def compute_parkinson_volatility(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Compute Parkinson volatility estimator.

        More efficient than close-to-close volatility as it uses high-low range.
        Parkinson = sqrt(1/(4*n*ln(2)) * sum(ln(high/low)^2))

        Args:
            high: High prices
            low: Low prices
            window: Rolling window

        Returns:
            Parkinson volatility (annualized)
        """
        log_hl = np.log(high / low) ** 2
        factor = 1 / (4 * np.log(2))

        min_periods = max(1, int(window * self.config.min_periods_ratio))
        parkinson = np.sqrt(
            factor * log_hl.rolling(window=window, min_periods=min_periods).mean()
        )

        return parkinson * np.sqrt(252)

    def compute_garman_klass_volatility(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Compute Garman-Klass volatility estimator.

        Uses OHLC data for more efficient volatility estimation.
        GK = sqrt(0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2)

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window

        Returns:
            Garman-Klass volatility (annualized)
        """
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2

        gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

        min_periods = max(1, int(window * self.config.min_periods_ratio))
        gk_vol = np.sqrt(
            gk_var.rolling(window=window, min_periods=min_periods).mean().clip(lower=0)
        )

        return gk_vol * np.sqrt(252)

    # =========================================================================
    # MOMENTUM INDICATORS (7 features)
    # =========================================================================

    def compute_ema(
        self,
        prices: pd.Series,
        span: int
    ) -> pd.Series:
        """
        Compute Exponential Moving Average.

        Args:
            prices: Price series
            span: EMA span

        Returns:
            EMA series
        """
        min_periods = max(1, int(span * self.config.min_periods_ratio))
        return prices.ewm(span=span, min_periods=min_periods, adjust=False).mean()

    def compute_rsi(
        self,
        prices: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss

        Args:
            prices: Price series
            period: RSI period (default: config.rsi_period)

        Returns:
            RSI series (0-100 scale)
        """
        period = period or self.config.rsi_period
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        # Use EMA for smoother RSI (Wilder's method)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def compute_macd(
        self,
        prices: pd.Series,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Compute MACD (Moving Average Convergence Divergence).

        Args:
            prices: Price series
            fast: Fast EMA period (default: config.macd_fast)
            slow: Slow EMA period (default: config.macd_slow)
            signal: Signal EMA period (default: config.macd_signal)

        Returns:
            Dictionary with 'macd_line', 'signal_line', 'histogram'
        """
        fast = fast or self.config.macd_fast
        slow = slow or self.config.macd_slow
        signal = signal or self.config.macd_signal

        ema_fast = self.compute_ema(prices, fast)
        ema_slow = self.compute_ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = self.compute_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }

    def compute_roc(
        self,
        prices: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """
        Compute Rate of Change (ROC).

        ROC = (price - price_n_periods_ago) / price_n_periods_ago * 100

        Args:
            prices: Price series
            period: ROC period (default: config.roc_period)

        Returns:
            ROC series (percentage)
        """
        period = period or self.config.roc_period
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100

    # =========================================================================
    # MEAN REVERSION INDICATORS (7 features)
    # =========================================================================

    def compute_bollinger_bands(
        self,
        prices: pd.Series,
        window: Optional[int] = None,
        num_std: Optional[float] = None
    ) -> Dict[str, pd.Series]:
        """
        Compute Bollinger Bands.

        Args:
            prices: Price series
            window: Moving average window (default: config.bollinger_window)
            num_std: Number of standard deviations (default: config.bollinger_std)

        Returns:
            Dictionary with 'upper', 'middle', 'lower', 'pct_b'
            pct_b = (price - lower) / (upper - lower)
        """
        window = window or self.config.bollinger_window
        num_std = num_std or self.config.bollinger_std

        min_periods = max(1, int(window * self.config.min_periods_ratio))

        middle = prices.rolling(window=window, min_periods=min_periods).mean()
        std = prices.rolling(window=window, min_periods=min_periods).std()

        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        # Percent B: where price is relative to bands
        band_width = upper - lower
        pct_b = (prices - lower) / band_width.replace(0, np.nan)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'pct_b': pct_b
        }

    def compute_zscore(
        self,
        prices: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Compute rolling Z-score.

        Z-score = (price - rolling_mean) / rolling_std

        Args:
            prices: Price series
            window: Rolling window (default: config.zscore_window)

        Returns:
            Z-score series
        """
        window = window or self.config.zscore_window
        min_periods = max(1, int(window * self.config.min_periods_ratio))

        rolling_mean = prices.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = prices.rolling(window=window, min_periods=min_periods).std()

        return (prices - rolling_mean) / rolling_std.replace(0, np.nan)

    def compute_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: Optional[int] = None,
        d_period: Optional[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Compute Stochastic Oscillator (%K and %D).

        %K = (close - lowest_low) / (highest_high - lowest_low) * 100
        %D = SMA(%K, d_period)

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period (default: config.stochastic_k)
            d_period: %D smoothing (default: config.stochastic_d)

        Returns:
            Dictionary with 'k' and 'd'
        """
        k_period = k_period or self.config.stochastic_k
        d_period = d_period or self.config.stochastic_d

        min_periods = max(1, int(k_period * self.config.min_periods_ratio))

        lowest_low = low.rolling(window=k_period, min_periods=min_periods).min()
        highest_high = high.rolling(window=k_period, min_periods=min_periods).max()

        range_ = highest_high - lowest_low
        k = ((close - lowest_low) / range_.replace(0, np.nan)) * 100

        d = k.rolling(window=d_period, min_periods=1).mean()

        return {'k': k, 'd': d}

    # =========================================================================
    # VOLUME INDICATORS (3 features)
    # =========================================================================

    def compute_obv(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Compute On-Balance Volume (OBV).

        OBV adds volume on up days and subtracts on down days.

        Args:
            close: Close prices
            volume: Volume series

        Returns:
            OBV series
        """
        direction = np.sign(close.diff())
        direction.iloc[0] = 0  # First day has no direction

        obv = (direction * volume).cumsum()

        return obv

    def compute_volume_ratio(
        self,
        volume: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Compute volume ratio (current volume / average volume).

        Args:
            volume: Volume series
            window: Averaging window (default: config.volume_ratio_window)

        Returns:
            Volume ratio series
        """
        window = window or self.config.volume_ratio_window
        min_periods = max(1, int(window * self.config.min_periods_ratio))

        avg_volume = volume.rolling(window=window, min_periods=min_periods).mean()

        return volume / avg_volume.replace(0, np.nan)

    def compute_mfi(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """
        Compute Money Flow Index (MFI).

        MFI is RSI weighted by volume - shows buying/selling pressure.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume series
            period: MFI period (default: config.mfi_period)

        Returns:
            MFI series (0-100 scale)
        """
        period = period or self.config.mfi_period

        # Typical price
        typical_price = (high + low + close) / 3

        # Raw money flow
        raw_money_flow = typical_price * volume

        # Positive and negative money flow
        price_change = typical_price.diff()
        positive_flow = raw_money_flow.where(price_change > 0, 0)
        negative_flow = raw_money_flow.where(price_change < 0, 0)

        min_periods = max(1, int(period * self.config.min_periods_ratio))
        positive_mf = positive_flow.rolling(window=period, min_periods=min_periods).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=min_periods).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.nan)))

        return mfi

    # =========================================================================
    # CROSS-SECTIONAL INDICATORS (4 features)
    # =========================================================================

    def compute_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Compute rolling beta to benchmark.

        Beta = Cov(asset, benchmark) / Var(benchmark)

        Args:
            returns: Asset returns
            benchmark_returns: Benchmark returns
            window: Rolling window (default: config.beta_window)

        Returns:
            Rolling beta series
        """
        window = window or self.config.beta_window
        min_periods = max(1, int(window * self.config.min_periods_ratio))

        # Covariance and variance
        cov = returns.rolling(window=window, min_periods=min_periods).cov(benchmark_returns)
        var = benchmark_returns.rolling(window=window, min_periods=min_periods).var()

        return cov / var.replace(0, np.nan)

    def compute_correlation(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Compute rolling correlation to benchmark.

        Args:
            returns: Asset returns
            benchmark_returns: Benchmark returns
            window: Rolling window (default: config.correlation_window)

        Returns:
            Rolling correlation series
        """
        window = window or self.config.correlation_window
        min_periods = max(1, int(window * self.config.min_periods_ratio))

        return returns.rolling(window=window, min_periods=min_periods).corr(benchmark_returns)

    def compute_relative_strength(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> pd.Series:
        """
        Compute relative strength (excess returns vs benchmark).

        Args:
            returns: Asset returns
            benchmark_returns: Benchmark returns

        Returns:
            Relative strength series (asset return - benchmark return)
        """
        return returns - benchmark_returns

    def compute_dispersion(
        self,
        returns_df: pd.DataFrame,
        window: int
    ) -> pd.Series:
        """
        Compute cross-sectional dispersion (std of returns across assets).

        High dispersion indicates differentiated returns across assets.

        Args:
            returns_df: DataFrame of returns (columns are assets)
            window: Rolling window for averaging dispersion

        Returns:
            Dispersion series
        """
        # Cross-sectional std at each point in time
        cross_std = returns_df.std(axis=1)

        min_periods = max(1, int(window * self.config.min_periods_ratio))
        return cross_std.rolling(window=window, min_periods=min_periods).mean()

    # =========================================================================
    # FEATURE MATRIX CREATION
    # =========================================================================

    def create_feature_matrix(
        self,
        prices_df: pd.DataFrame,
        ohlcv_df: Optional[pd.DataFrame] = None,
        categories: Optional[Set[FeatureCategory]] = None,
        benchmark_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create feature matrix from price/OHLCV DataFrame.

        Computes all enabled features for each symbol and combines into a
        single feature matrix suitable for QCML processing.

        Args:
            prices_df: DataFrame with close prices (columns=symbols, index=dates)
            ohlcv_df: Optional DataFrame with OHLCV data (MultiIndex: symbol, timestamp)
                      Required for ATR, Parkinson, Garman-Klass, Stochastic, MFI
            categories: Categories to compute (default: config.categories)
            benchmark_col: Benchmark column (default: config.benchmark_col)

        Returns:
            DataFrame with features for each symbol, aligned by date.
            NaN values from warmup period are dropped.

        Example:
            >>> engine = FeatureEngine()
            >>> features = engine.create_feature_matrix(
            ...     prices_df=close_prices,
            ...     ohlcv_df=full_ohlcv_data,
            ...     benchmark_col='SPY'
            ... )
        """
        categories = categories or self.config.categories
        benchmark_col = benchmark_col or self.config.benchmark_col

        # Temporarily update config categories if provided
        original_categories = self.config.categories
        self.config.categories = categories

        logger.info(
            f"Creating feature matrix for {len(prices_df.columns)} symbols, "
            f"categories: {categories}, benchmark={benchmark_col}"
        )

        if benchmark_col not in prices_df.columns:
            raise ValueError(f"Benchmark column '{benchmark_col}' not in prices DataFrame")

        # Compute all returns first
        returns_df = prices_df.apply(self.compute_returns)
        benchmark_returns = returns_df[benchmark_col]

        # Compute dispersion (cross-sectional)
        dispersion = None
        if self.config.should_compute(FeatureCategory.CROSS_SECTIONAL):
            dispersion = self.compute_dispersion(returns_df, self.config.correlation_window)

        feature_dfs = []

        for col in prices_df.columns:
            prices = prices_df[col]
            returns = returns_df[col]

            # Get OHLCV data for this symbol if available
            ohlcv = None
            if ohlcv_df is not None:
                try:
                    if isinstance(ohlcv_df.index, pd.MultiIndex):
                        ohlcv = ohlcv_df.loc[col].copy()
                    else:
                        ohlcv = ohlcv_df.copy()
                except KeyError:
                    logger.warning(f"OHLCV data not found for {col}")

            # Compute features for this symbol
            symbol_features = self._compute_symbol_features(
                col, prices, returns, benchmark_returns, ohlcv, dispersion
            )

            feature_dfs.append(symbol_features)

        # Combine all features
        result = pd.concat(feature_dfs, axis=1)

        # Restore original categories
        self.config.categories = original_categories

        # Drop warmup period (rows with any NaN)
        n_before = len(result)
        result = result.dropna()
        n_after = len(result)

        logger.info(
            f"Feature matrix created: {result.shape[0]} samples, "
            f"{result.shape[1]} features (dropped {n_before - n_after} warmup rows)"
        )

        return result

    def _compute_symbol_features(
        self,
        symbol: str,
        prices: pd.Series,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        ohlcv: Optional[pd.DataFrame],
        dispersion: Optional[pd.Series]
    ) -> pd.DataFrame:
        """Compute all features for a single symbol."""
        features = {}

        # =====================================================================
        # VOLATILITY FEATURES
        # =====================================================================
        if self.config.should_compute(FeatureCategory.VOLATILITY):
            # Rolling volatility at multiple windows
            for window in self.config.volatility_windows:
                features[f"{symbol}_vol_{window}d"] = self.compute_rolling_volatility(
                    returns, window
                )

            # OHLCV-based volatility if available
            if ohlcv is not None and all(c in ohlcv.columns for c in ['open', 'high', 'low', 'close']):
                features[f"{symbol}_atr"] = self.compute_atr(
                    ohlcv['high'], ohlcv['low'], ohlcv['close']
                )
                features[f"{symbol}_parkinson_vol"] = self.compute_parkinson_volatility(
                    ohlcv['high'], ohlcv['low'], 20
                )
                features[f"{symbol}_gk_vol"] = self.compute_garman_klass_volatility(
                    ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], 20
                )

        # =====================================================================
        # MOMENTUM FEATURES
        # =====================================================================
        if self.config.should_compute(FeatureCategory.MOMENTUM):
            # EMA at multiple spans
            for span in self.config.ema_spans:
                ema = self.compute_ema(prices, span)
                # Use EMA ratio (price / EMA) for scale-independence
                features[f"{symbol}_ema_{span}_ratio"] = prices / ema

            # RSI
            features[f"{symbol}_rsi"] = self.compute_rsi(prices)

            # MACD components
            macd = self.compute_macd(prices)
            features[f"{symbol}_macd_line"] = macd['macd_line']
            features[f"{symbol}_macd_signal"] = macd['signal_line']
            features[f"{symbol}_macd_hist"] = macd['histogram']

            # Rate of Change
            features[f"{symbol}_roc"] = self.compute_roc(prices)

        # =====================================================================
        # MEAN REVERSION FEATURES
        # =====================================================================
        if self.config.should_compute(FeatureCategory.MEAN_REVERSION):
            # Bollinger Bands
            bb = self.compute_bollinger_bands(prices)
            features[f"{symbol}_bb_upper"] = bb['upper']
            features[f"{symbol}_bb_lower"] = bb['lower']
            features[f"{symbol}_bb_middle"] = bb['middle']
            features[f"{symbol}_bb_pct_b"] = bb['pct_b']

            # Z-score
            features[f"{symbol}_zscore"] = self.compute_zscore(prices)

            # Stochastic (requires high/low/close)
            if ohlcv is not None and all(c in ohlcv.columns for c in ['high', 'low', 'close']):
                stoch = self.compute_stochastic(
                    ohlcv['high'], ohlcv['low'], ohlcv['close']
                )
                features[f"{symbol}_stoch_k"] = stoch['k']
                features[f"{symbol}_stoch_d"] = stoch['d']

        # =====================================================================
        # VOLUME FEATURES
        # =====================================================================
        if self.config.should_compute(FeatureCategory.VOLUME):
            if ohlcv is not None and 'volume' in ohlcv.columns:
                # OBV
                close_col = ohlcv['close'] if 'close' in ohlcv.columns else prices
                features[f"{symbol}_obv"] = self.compute_obv(close_col, ohlcv['volume'])

                # Volume ratio
                features[f"{symbol}_volume_ratio"] = self.compute_volume_ratio(
                    ohlcv['volume']
                )

                # MFI (requires OHLCV)
                if all(c in ohlcv.columns for c in ['high', 'low', 'close']):
                    features[f"{symbol}_mfi"] = self.compute_mfi(
                        ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']
                    )

        # =====================================================================
        # CROSS-SECTIONAL FEATURES
        # =====================================================================
        if self.config.should_compute(FeatureCategory.CROSS_SECTIONAL):
            # Beta
            features[f"{symbol}_beta"] = self.compute_beta(returns, benchmark_returns)

            # Correlation
            features[f"{symbol}_corr_benchmark"] = self.compute_correlation(
                returns, benchmark_returns
            )

            # Relative strength
            features[f"{symbol}_rel_strength"] = self.compute_relative_strength(
                returns, benchmark_returns
            )

            # Dispersion (same for all symbols - cross-sectional measure)
            if dispersion is not None:
                features[f"{symbol}_dispersion"] = dispersion

        return pd.DataFrame(features)

    def validate_features(self, features: pd.DataFrame) -> Dict:
        """
        Validate feature matrix for common issues.

        Args:
            features: Feature DataFrame

        Returns:
            Dictionary with validation results including:
            - is_valid: Boolean indicating if features pass all checks
            - n_samples: Number of samples
            - n_features: Number of features
            - issues: List of detected issues
            - feature_stats: Summary statistics for each feature

        Example:
            >>> validation = engine.validate_features(features)
            >>> if validation['is_valid']:
            ...     print("Features ready for QCML")
            >>> else:
            ...     print(f"Issues: {validation['issues']}")
        """
        issues = []

        # Check for NaN values
        nan_count = features.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"Contains {nan_count} NaN values")

        # Check for infinite values
        inf_count = np.isinf(features.values).sum()
        if inf_count > 0:
            issues.append(f"Contains {inf_count} infinite values")

        # Check for constant columns
        constant_cols = features.columns[features.std() == 0].tolist()
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols}")

        # Check for reasonable value ranges by feature type
        for col in features.columns:
            col_lower = col.lower()

            # RSI should be 0-100
            if 'rsi' in col_lower:
                if (features[col] < 0).any() or (features[col] > 100).any():
                    issues.append(f"{col} has values outside 0-100 range")

            # MFI should be 0-100
            elif 'mfi' in col_lower:
                if (features[col] < 0).any() or (features[col] > 100).any():
                    issues.append(f"{col} has values outside 0-100 range")

            # Stochastic should be 0-100
            elif 'stoch' in col_lower:
                if (features[col] < 0).any() or (features[col] > 100).any():
                    issues.append(f"{col} has values outside 0-100 range")

            # Correlation should be in [-1, 1]
            elif 'corr' in col_lower:
                if (features[col].abs() > 1.001).any():
                    issues.append(f"{col} has correlations outside [-1, 1]")

            # Volatility should be non-negative
            elif 'vol' in col_lower and 'volume' not in col_lower:
                if (features[col] < 0).any():
                    issues.append(f"{col} has negative volatility values")

            # BB %B typically 0-1 but can exceed
            elif 'pct_b' in col_lower:
                extreme_count = ((features[col] < -0.5) | (features[col] > 1.5)).sum()
                if extreme_count > len(features) * 0.1:  # More than 10% extreme
                    issues.append(f"{col} has many extreme values (>10% outside [-0.5, 1.5])")

        return {
            "is_valid": len(issues) == 0,
            "n_samples": len(features),
            "n_features": len(features.columns),
            "issues": issues,
            "feature_stats": features.describe().to_dict(),
        }

    def get_feature_names(
        self,
        symbols: List[str],
        categories: Optional[Set[FeatureCategory]] = None,
        has_ohlcv: bool = True
    ) -> List[str]:
        """
        Get list of feature names that will be generated.

        Useful for understanding feature matrix structure before creation.

        Args:
            symbols: List of symbol names
            categories: Categories to include (default: all)
            has_ohlcv: Whether OHLCV data is available

        Returns:
            List of feature column names
        """
        categories = categories or self.config.categories
        features = []

        for symbol in symbols:
            if FeatureCategory.ALL in categories or FeatureCategory.VOLATILITY in categories:
                for window in self.config.volatility_windows:
                    features.append(f"{symbol}_vol_{window}d")
                if has_ohlcv:
                    features.extend([
                        f"{symbol}_atr",
                        f"{symbol}_parkinson_vol",
                        f"{symbol}_gk_vol"
                    ])

            if FeatureCategory.ALL in categories or FeatureCategory.MOMENTUM in categories:
                for span in self.config.ema_spans:
                    features.append(f"{symbol}_ema_{span}_ratio")
                features.extend([
                    f"{symbol}_rsi",
                    f"{symbol}_macd_line",
                    f"{symbol}_macd_signal",
                    f"{symbol}_macd_hist",
                    f"{symbol}_roc"
                ])

            if FeatureCategory.ALL in categories or FeatureCategory.MEAN_REVERSION in categories:
                features.extend([
                    f"{symbol}_bb_upper",
                    f"{symbol}_bb_lower",
                    f"{symbol}_bb_middle",
                    f"{symbol}_bb_pct_b",
                    f"{symbol}_zscore"
                ])
                if has_ohlcv:
                    features.extend([
                        f"{symbol}_stoch_k",
                        f"{symbol}_stoch_d"
                    ])

            if FeatureCategory.ALL in categories or FeatureCategory.VOLUME in categories:
                if has_ohlcv:
                    features.extend([
                        f"{symbol}_obv",
                        f"{symbol}_volume_ratio",
                        f"{symbol}_mfi"
                    ])

            if FeatureCategory.ALL in categories or FeatureCategory.CROSS_SECTIONAL in categories:
                features.extend([
                    f"{symbol}_beta",
                    f"{symbol}_corr_benchmark",
                    f"{symbol}_rel_strength",
                    f"{symbol}_dispersion"
                ])

        return features
