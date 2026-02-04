"""
Volatility Data Pipeline for QCML Research

Fetches VIX and SPY data via yfinance and computes features for
quantum volatility forecasting experiments.

Features computed:
- vix: VIX daily close (IV proxy)
- rv_5d: 5-day realized volatility from SPY
- rv_20d: 20-day realized volatility from SPY
- vix_change: Daily VIX change
- vix_rv_ratio: VIX / RV_20d (variance risk premium proxy)

Target:
- rv_5d_ahead: 5-day RV shifted forward (prediction target)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VolatilityDataset:
    """
    Container for volatility forecasting data.

    Attributes:
        features: DataFrame with columns [vix, rv_5d, rv_20d, vix_change, vix_rv_ratio]
        target: Series with rv_5d_ahead (5-day realized vol shifted forward)
        dates: DatetimeIndex aligned with features and target
        spy_prices: SPY close prices for reference
        vix_prices: VIX close prices for reference
    """
    features: pd.DataFrame
    target: pd.Series
    dates: pd.DatetimeIndex
    spy_prices: pd.Series
    vix_prices: pd.Series

    @property
    def X(self) -> np.ndarray:
        """Return features as numpy array."""
        return self.features.values

    @property
    def y(self) -> np.ndarray:
        """Return target as numpy array."""
        return self.target.values

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.features)

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.features.shape[1]

    @property
    def feature_names(self) -> list:
        """List of feature names."""
        return list(self.features.columns)

    def train_test_split(
        self,
        test_ratio: float = 0.2,
        by_date: bool = True
    ) -> Tuple['VolatilityDataset', 'VolatilityDataset']:
        """
        Split dataset into train and test sets.

        Args:
            test_ratio: Fraction of data for test set
            by_date: If True, split chronologically (recommended for time series)

        Returns:
            train_dataset, test_dataset
        """
        n = len(self.features)
        split_idx = int(n * (1 - test_ratio))

        train = VolatilityDataset(
            features=self.features.iloc[:split_idx].copy(),
            target=self.target.iloc[:split_idx].copy(),
            dates=self.dates[:split_idx],
            spy_prices=self.spy_prices.iloc[:split_idx].copy(),
            vix_prices=self.vix_prices.iloc[:split_idx].copy(),
        )

        test = VolatilityDataset(
            features=self.features.iloc[split_idx:].copy(),
            target=self.target.iloc[split_idx:].copy(),
            dates=self.dates[split_idx:],
            spy_prices=self.spy_prices.iloc[split_idx:].copy(),
            vix_prices=self.vix_prices.iloc[split_idx:].copy(),
        )

        return train, test

    def walk_forward_splits(
        self,
        train_months: int = 12,
        test_months: int = 1,
        step_months: int = 1
    ) -> list:
        """
        Generate walk-forward validation splits.

        Args:
            train_months: Number of months for training window
            test_months: Number of months for test window
            step_months: Number of months to step forward each iteration

        Returns:
            List of (train_dataset, test_dataset) tuples
        """
        splits = []

        # Get unique year-months
        dates = pd.Series(self.dates)
        start_date = dates.min()
        end_date = dates.max()

        current_train_start = start_date

        while True:
            train_end = current_train_start + pd.DateOffset(months=train_months)
            test_end = train_end + pd.DateOffset(months=test_months)

            if test_end > end_date:
                break

            # Create masks
            train_mask = (dates >= current_train_start) & (dates < train_end)
            test_mask = (dates >= train_end) & (dates < test_end)

            train_idx = train_mask.values
            test_idx = test_mask.values

            if train_idx.sum() > 20 and test_idx.sum() > 5:  # Minimum samples
                train_ds = VolatilityDataset(
                    features=self.features.loc[train_idx].copy(),
                    target=self.target.loc[train_idx].copy(),
                    dates=self.dates[train_idx],
                    spy_prices=self.spy_prices.loc[train_idx].copy(),
                    vix_prices=self.vix_prices.loc[train_idx].copy(),
                )

                test_ds = VolatilityDataset(
                    features=self.features.loc[test_idx].copy(),
                    target=self.target.loc[test_idx].copy(),
                    dates=self.dates[test_idx],
                    spy_prices=self.spy_prices.loc[test_idx].copy(),
                    vix_prices=self.vix_prices.loc[test_idx].copy(),
                )

                splits.append((train_ds, test_ds))

            current_train_start = current_train_start + pd.DateOffset(months=step_months)

        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits


class VolatilityDataPipeline:
    """
    Data pipeline for fetching and processing volatility data.

    Uses yfinance to fetch VIX (implied volatility proxy) and SPY
    (for realized volatility computation).

    Example:
        >>> pipeline = VolatilityDataPipeline()
        >>> dataset = pipeline.fetch_and_process('2010-01-01', '2024-01-01')
        >>> print(f"Dataset: {dataset.n_samples} samples, {dataset.n_features} features")
    """

    def __init__(self, annualize: bool = True):
        """
        Initialize pipeline.

        Args:
            annualize: If True, annualize volatility measures (multiply by sqrt(252))
        """
        self.annualize = annualize
        self._annualization_factor = np.sqrt(252) if annualize else 1.0

    def fetch_and_process(
        self,
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None,
        rv_windows: Tuple[int, int] = (5, 20),
        forecast_horizon: int = 5
    ) -> VolatilityDataset:
        """
        Fetch VIX and SPY data, compute features, and return dataset.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date (default: today)
            rv_windows: Tuple of (short_window, long_window) for RV computation
            forecast_horizon: Days ahead to forecast (default: 5)

        Returns:
            VolatilityDataset with features and target
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

        logger.info(f"Fetching data from {start_date} to {end_date or 'today'}")

        # Fetch data
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)

        if vix.empty or spy.empty:
            raise ValueError("Failed to fetch data from yfinance")

        # Align dates
        common_dates = vix.index.intersection(spy.index)
        vix = vix.loc[common_dates]
        spy = spy.loc[common_dates]

        logger.info(f"Fetched {len(common_dates)} trading days")

        # Extract close prices
        vix_close = vix['Close'].squeeze() if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
        spy_close = spy['Close'].squeeze() if isinstance(spy['Close'], pd.DataFrame) else spy['Close']

        # Compute features
        features = self._compute_features(
            vix_close=vix_close,
            spy_close=spy_close,
            rv_windows=rv_windows
        )

        # Compute target (future realized volatility)
        target = self._compute_target(spy_close, forecast_horizon)

        # Align features and target, drop NaN
        combined = pd.concat([features, target.rename('target')], axis=1)
        combined = combined.dropna()

        logger.info(f"After NaN removal: {len(combined)} samples")

        # Create dataset
        dataset = VolatilityDataset(
            features=combined.drop(columns=['target']),
            target=combined['target'],
            dates=combined.index,
            spy_prices=spy_close.loc[combined.index],
            vix_prices=vix_close.loc[combined.index],
        )

        return dataset

    def _compute_features(
        self,
        vix_close: pd.Series,
        spy_close: pd.Series,
        rv_windows: Tuple[int, int] = (5, 20)
    ) -> pd.DataFrame:
        """
        Compute volatility features.

        Features:
        - vix: VIX close (already in annualized % terms)
        - rv_5d: 5-day realized volatility
        - rv_20d: 20-day realized volatility
        - vix_change: Daily VIX change (points)
        - vix_rv_ratio: VIX / RV_20d (variance risk premium proxy)
        """
        features = pd.DataFrame(index=vix_close.index)

        # VIX as implied volatility proxy (divide by 100 to get decimal)
        features['vix'] = vix_close / 100.0

        # Compute log returns for SPY
        spy_returns = np.log(spy_close / spy_close.shift(1))

        # Realized volatility at different windows
        rv_short, rv_long = rv_windows
        features['rv_5d'] = self._compute_realized_vol(spy_returns, rv_short)
        features['rv_20d'] = self._compute_realized_vol(spy_returns, rv_long)

        # VIX change (daily difference in VIX points)
        features['vix_change'] = vix_close.diff() / 100.0

        # Variance risk premium proxy: VIX / RV_20d
        # High ratio = IV > RV = expensive options
        features['vix_rv_ratio'] = features['vix'] / features['rv_20d'].replace(0, np.nan)

        return features

    def _compute_realized_vol(self, returns: pd.Series, window: int) -> pd.Series:
        """
        Compute realized volatility as rolling std of returns.

        Args:
            returns: Log returns series
            window: Rolling window size

        Returns:
            Realized volatility series (annualized if self.annualize=True)
        """
        rv = returns.rolling(window=window, min_periods=window).std()
        return rv * self._annualization_factor

    def _compute_target(self, spy_close: pd.Series, horizon: int = 5) -> pd.Series:
        """
        Compute target: forward realized volatility.

        This is the RV over the next `horizon` days, which we want to predict.

        Args:
            spy_close: SPY close prices
            horizon: Forecast horizon in days

        Returns:
            Forward realized volatility series
        """
        spy_returns = np.log(spy_close / spy_close.shift(1))

        # Forward-looking RV (shifted back to align with current date)
        # At time t, target is the RV from t+1 to t+horizon
        forward_rv = spy_returns.rolling(window=horizon, min_periods=horizon).std()
        forward_rv = forward_rv.shift(-horizon)  # Shift to align with current date

        return forward_rv * self._annualization_factor


def load_volatility_data(
    start_date: str = '2010-01-01',
    end_date: Optional[str] = None,
    cache_path: Optional[str] = None
) -> VolatilityDataset:
    """
    Convenience function to load volatility data.

    Args:
        start_date: Start date
        end_date: End date (default: today)
        cache_path: Optional path to cache data (not implemented yet)

    Returns:
        VolatilityDataset
    """
    pipeline = VolatilityDataPipeline()
    return pipeline.fetch_and_process(start_date, end_date)


if __name__ == "__main__":
    # Test the pipeline
    logging.basicConfig(level=logging.INFO)

    print("Testing Volatility Data Pipeline...")

    pipeline = VolatilityDataPipeline()
    dataset = pipeline.fetch_and_process(
        start_date='2020-01-01',
        end_date='2024-01-01'
    )

    print(f"\nDataset Summary:")
    print(f"  Samples: {dataset.n_samples}")
    print(f"  Features: {dataset.n_features}")
    print(f"  Feature names: {dataset.feature_names}")
    print(f"  Date range: {dataset.dates[0]} to {dataset.dates[-1]}")

    print(f"\nFeature Statistics:")
    print(dataset.features.describe())

    print(f"\nTarget Statistics:")
    print(dataset.target.describe())

    # Test walk-forward splits
    splits = dataset.walk_forward_splits(train_months=12, test_months=1)
    print(f"\nWalk-forward splits: {len(splits)}")
    if splits:
        train, test = splits[0]
        print(f"  First split - Train: {train.n_samples}, Test: {test.n_samples}")

    print("\nVolatility Data Pipeline tests passed!")
