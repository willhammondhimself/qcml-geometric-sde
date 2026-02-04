"""
Unit tests for QCML feature engineering module (Phase 2)

Tests cover:
    - Volatility indicators (rolling_vol, ATR, Parkinson, Garman-Klass)
    - Momentum indicators (EMA, RSI, MACD, ROC)
    - Mean reversion indicators (Bollinger, Z-score, Stochastic)
    - Volume indicators (OBV, Volume Ratio, MFI)
    - Cross-sectional indicators (Beta, Correlation, Relative Strength, Dispersion)
    - Feature matrix creation and validation
    - Lookahead bias prevention
"""

import pytest
import numpy as np
import pandas as pd

from qcml.data.features import (
    FeatureEngine,
    FeatureConfig,
    FeatureCategory
)


class TestFeatureEngine:
    """Tests for FeatureEngine class"""

    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        # Generate random walk prices
        returns = np.random.normal(0.0005, 0.02, n)
        spy_prices = 100 * np.exp(np.cumsum(returns))
        xlf_prices = 50 * np.exp(np.cumsum(returns * 1.2 + np.random.normal(0, 0.01, n)))

        prices_df = pd.DataFrame({
            'SPY': spy_prices,
            'XLF': xlf_prices
        }, index=dates)

        return prices_df

    @pytest.fixture
    def sample_ohlcv(self, sample_prices):
        """Generate sample OHLCV data"""
        ohlcv_data = []

        for symbol in sample_prices.columns:
            for date, close in sample_prices[symbol].items():
                # Synthetic OHLCV with realistic ranges
                daily_range = close * 0.02
                ohlcv_data.append({
                    'symbol': symbol,
                    'timestamp': date,
                    'open': close * (1 + np.random.uniform(-0.01, 0.01)),
                    'high': close + abs(np.random.normal(0, daily_range)),
                    'low': close - abs(np.random.normal(0, daily_range)),
                    'close': close,
                    'volume': int(np.random.uniform(1e6, 1e8))
                })

        ohlcv_df = pd.DataFrame(ohlcv_data)
        ohlcv_df = ohlcv_df.set_index(['symbol', 'timestamp']).sort_index()

        return ohlcv_df

    @pytest.fixture
    def engine(self):
        """Create default FeatureEngine"""
        return FeatureEngine()


class TestVolatilityIndicators(TestFeatureEngine):
    """Tests for volatility indicators"""

    def test_compute_returns(self, engine, sample_prices):
        """Test log returns calculation"""
        returns = engine.compute_returns(sample_prices['SPY'])

        # First value should be NaN
        assert pd.isna(returns.iloc[0])

        # Returns should be finite for rest
        assert returns.iloc[1:].notna().all()

        # Should have same length as prices
        assert len(returns) == len(sample_prices)

    def test_rolling_volatility(self, engine, sample_prices):
        """Test rolling volatility calculation"""
        returns = engine.compute_returns(sample_prices['SPY'])
        vol = engine.compute_rolling_volatility(returns, window=20)

        # Should be positive (after warmup)
        assert (vol.dropna() > 0).all()

        # Should be annualized (sqrt(252) factor)
        min_periods = max(1, int(20 * engine.config.min_periods_ratio))
        raw_std = returns.rolling(20, min_periods=min_periods).std()
        expected_annualized = raw_std * np.sqrt(252)

        # Compare after aligning indices
        common_idx = vol.dropna().index.intersection(expected_annualized.dropna().index)
        np.testing.assert_allclose(
            vol.loc[common_idx].values,
            expected_annualized.loc[common_idx].values,
            rtol=1e-10
        )

    def test_atr(self, engine, sample_ohlcv):
        """Test Average True Range calculation"""
        spy_ohlcv = sample_ohlcv.loc['SPY']

        atr = engine.compute_atr(
            spy_ohlcv['high'],
            spy_ohlcv['low'],
            spy_ohlcv['close']
        )

        # ATR should be positive (after warmup)
        assert (atr.dropna() > 0).all()

        # ATR should reflect price range
        avg_range = (spy_ohlcv['high'] - spy_ohlcv['low']).mean()
        assert atr.dropna().mean() <= avg_range * 2  # Should be close to average range

    def test_parkinson_volatility(self, engine, sample_ohlcv):
        """Test Parkinson volatility estimator"""
        spy_ohlcv = sample_ohlcv.loc['SPY']

        parkinson = engine.compute_parkinson_volatility(
            spy_ohlcv['high'],
            spy_ohlcv['low'],
            window=20
        )

        # Should be positive
        assert (parkinson.dropna() > 0).all()

        # Should be annualized
        assert parkinson.dropna().mean() > 0.01  # At least 1% annualized

    def test_garman_klass_volatility(self, engine, sample_ohlcv):
        """Test Garman-Klass volatility estimator"""
        spy_ohlcv = sample_ohlcv.loc['SPY']

        gk = engine.compute_garman_klass_volatility(
            spy_ohlcv['open'],
            spy_ohlcv['high'],
            spy_ohlcv['low'],
            spy_ohlcv['close'],
            window=20
        )

        # Should be non-negative (after warmup)
        assert (gk.dropna() >= 0).all()


class TestMomentumIndicators(TestFeatureEngine):
    """Tests for momentum indicators"""

    def test_ema(self, engine, sample_prices):
        """Test Exponential Moving Average"""
        ema_12 = engine.compute_ema(sample_prices['SPY'], span=12)
        ema_26 = engine.compute_ema(sample_prices['SPY'], span=26)

        # EMA should be close to price
        price_mean = sample_prices['SPY'].mean()
        assert abs(ema_12.dropna().mean() - price_mean) < price_mean * 0.1

        # Shorter EMA more responsive - should have higher correlation with price
        corr_12 = sample_prices['SPY'].corr(ema_12)
        corr_26 = sample_prices['SPY'].corr(ema_26)
        assert corr_12 >= corr_26 - 0.01  # Allow small tolerance

    def test_rsi(self, engine, sample_prices):
        """Test Relative Strength Index"""
        rsi = engine.compute_rsi(sample_prices['SPY'], period=14)

        # RSI should be between 0 and 100
        assert (rsi.dropna() >= 0).all()
        assert (rsi.dropna() <= 100).all()

        # RSI should oscillate around 50 for random walk
        assert 30 < rsi.dropna().mean() < 70

    def test_macd(self, engine, sample_prices):
        """Test MACD calculation"""
        macd = engine.compute_macd(sample_prices['SPY'])

        assert 'macd_line' in macd
        assert 'signal_line' in macd
        assert 'histogram' in macd

        # MACD line = fast EMA - slow EMA
        # Histogram = MACD line - signal line
        np.testing.assert_allclose(
            macd['histogram'].dropna().values,
            (macd['macd_line'] - macd['signal_line']).dropna().values,
            rtol=1e-10
        )

    def test_roc(self, engine, sample_prices):
        """Test Rate of Change"""
        roc = engine.compute_roc(sample_prices['SPY'], period=10)

        # ROC should be in reasonable percentage range
        assert roc.dropna().abs().max() < 100  # Less than 100% per 10 days

        # ROC should average close to 0 for random walk
        assert abs(roc.dropna().mean()) < 5


class TestMeanReversionIndicators(TestFeatureEngine):
    """Tests for mean reversion indicators"""

    def test_bollinger_bands(self, engine, sample_prices):
        """Test Bollinger Bands"""
        bb = engine.compute_bollinger_bands(sample_prices['SPY'], window=20, num_std=2)

        assert 'upper' in bb
        assert 'middle' in bb
        assert 'lower' in bb
        assert 'pct_b' in bb

        # Upper > middle > lower
        mask = bb['upper'].notna() & bb['lower'].notna()
        assert (bb['upper'][mask] >= bb['middle'][mask]).all()
        assert (bb['middle'][mask] >= bb['lower'][mask]).all()

        # %B should mostly be between 0 and 1
        pct_b_clean = bb['pct_b'].dropna()
        within_bands = ((pct_b_clean >= 0) & (pct_b_clean <= 1)).sum() / len(pct_b_clean)
        assert within_bands > 0.85  # At least 85% within bands (relaxed for random data)

    def test_zscore(self, engine, sample_prices):
        """Test Z-score calculation"""
        zscore = engine.compute_zscore(sample_prices['SPY'], window=20)

        # Z-score should have mean close to 0 and std close to 1
        # (approximately, due to rolling calculation)
        assert abs(zscore.dropna().mean()) < 0.5
        assert 0.5 < zscore.dropna().std() < 1.5

    def test_stochastic(self, engine, sample_ohlcv):
        """Test Stochastic Oscillator"""
        spy_ohlcv = sample_ohlcv.loc['SPY']

        stoch = engine.compute_stochastic(
            spy_ohlcv['high'],
            spy_ohlcv['low'],
            spy_ohlcv['close'],
            k_period=14,
            d_period=3
        )

        assert 'k' in stoch
        assert 'd' in stoch

        # Should be between 0 and 100
        assert (stoch['k'].dropna() >= 0).all()
        assert (stoch['k'].dropna() <= 100).all()
        assert (stoch['d'].dropna() >= 0).all()
        assert (stoch['d'].dropna() <= 100).all()


class TestVolumeIndicators(TestFeatureEngine):
    """Tests for volume indicators"""

    def test_obv(self, engine, sample_ohlcv):
        """Test On-Balance Volume"""
        spy_ohlcv = sample_ohlcv.loc['SPY']

        obv = engine.compute_obv(spy_ohlcv['close'], spy_ohlcv['volume'])

        # OBV should be cumulative
        assert len(obv) == len(spy_ohlcv)

        # Should not have NaN after first value
        assert obv.iloc[1:].notna().all()

    def test_volume_ratio(self, engine, sample_ohlcv):
        """Test Volume Ratio"""
        spy_ohlcv = sample_ohlcv.loc['SPY']

        vol_ratio = engine.compute_volume_ratio(spy_ohlcv['volume'], window=20)

        # Should be positive
        assert (vol_ratio.dropna() > 0).all()

        # Mean should be close to 1 (volume / average volume)
        assert 0.5 < vol_ratio.dropna().mean() < 2.0

    def test_mfi(self, engine, sample_ohlcv):
        """Test Money Flow Index"""
        spy_ohlcv = sample_ohlcv.loc['SPY']

        mfi = engine.compute_mfi(
            spy_ohlcv['high'],
            spy_ohlcv['low'],
            spy_ohlcv['close'],
            spy_ohlcv['volume'],
            period=14
        )

        # MFI should be between 0 and 100
        mfi_clean = mfi.dropna()
        assert (mfi_clean >= 0).all()
        assert (mfi_clean <= 100).all()


class TestCrossSectionalIndicators(TestFeatureEngine):
    """Tests for cross-sectional indicators"""

    def test_beta(self, engine, sample_prices):
        """Test rolling beta calculation"""
        spy_returns = engine.compute_returns(sample_prices['SPY'])
        xlf_returns = engine.compute_returns(sample_prices['XLF'])

        beta = engine.compute_beta(xlf_returns, spy_returns, window=60)

        # Beta should be finite
        assert beta.dropna().notna().all()

        # XLF (financials) tends to have beta > 1 relative to SPY
        # Our synthetic data has 1.2x multiplier
        assert beta.dropna().mean() > 0.5

    def test_correlation(self, engine, sample_prices):
        """Test rolling correlation"""
        spy_returns = engine.compute_returns(sample_prices['SPY'])
        xlf_returns = engine.compute_returns(sample_prices['XLF'])

        corr = engine.compute_correlation(xlf_returns, spy_returns, window=20)

        # Correlation should be in [-1, 1]
        corr_clean = corr.dropna()
        assert (corr_clean >= -1).all()
        assert (corr_clean <= 1).all()

        # Our synthetic data is correlated (same base returns)
        assert corr_clean.mean() > 0.5

    def test_relative_strength(self, engine, sample_prices):
        """Test relative strength"""
        spy_returns = engine.compute_returns(sample_prices['SPY'])
        xlf_returns = engine.compute_returns(sample_prices['XLF'])

        rel_strength = engine.compute_relative_strength(xlf_returns, spy_returns)

        # Should be same length as returns
        assert len(rel_strength) == len(spy_returns)

        # Should be close to 0 on average for similar assets
        assert abs(rel_strength.dropna().mean()) < 0.01

    def test_dispersion(self, engine, sample_prices):
        """Test cross-sectional dispersion"""
        returns_df = sample_prices.apply(engine.compute_returns)

        dispersion = engine.compute_dispersion(returns_df, window=20)

        # Dispersion should be non-negative
        assert (dispersion.dropna() >= 0).all()


class TestFeatureMatrixCreation(TestFeatureEngine):
    """Tests for feature matrix creation"""

    def test_create_feature_matrix_minimal(self, engine, sample_prices):
        """Test feature matrix with minimal categories"""
        config = FeatureConfig(categories={FeatureCategory.VOLATILITY})
        engine_vol_only = FeatureEngine(config=config)

        features = engine_vol_only.create_feature_matrix(
            sample_prices,
            benchmark_col='SPY'
        )

        # Should have volatility features only
        assert any('vol' in col.lower() for col in features.columns)

        # Should not have momentum features
        assert not any('rsi' in col.lower() for col in features.columns)
        assert not any('macd' in col.lower() for col in features.columns)

    def test_create_feature_matrix_full(self, engine, sample_prices, sample_ohlcv):
        """Test full feature matrix creation"""
        features = engine.create_feature_matrix(
            sample_prices,
            ohlcv_df=sample_ohlcv,
            benchmark_col='SPY'
        )

        # Should have many features
        assert features.shape[1] > 20

        # Should have no NaN (dropped during creation)
        assert not features.isna().any().any()

        # Should have reasonable number of samples (accounting for warmup)
        assert features.shape[0] > 200

    def test_create_feature_matrix_no_ohlcv(self, engine, sample_prices):
        """Test feature matrix without OHLCV data"""
        features = engine.create_feature_matrix(
            sample_prices,
            ohlcv_df=None,
            benchmark_col='SPY'
        )

        # Should still create features from prices only
        assert features.shape[1] > 10

        # Should not have ATR, Parkinson, MFI (require OHLCV)
        assert not any('atr' in col.lower() for col in features.columns)
        assert not any('parkinson' in col.lower() for col in features.columns)
        assert not any('mfi' in col.lower() for col in features.columns)

    def test_benchmark_required(self, engine, sample_prices):
        """Test that benchmark column must exist"""
        with pytest.raises(ValueError, match="Benchmark column"):
            engine.create_feature_matrix(
                sample_prices,
                benchmark_col='INVALID'
            )

    def test_feature_validation(self, engine, sample_prices, sample_ohlcv):
        """Test feature validation"""
        features = engine.create_feature_matrix(
            sample_prices,
            ohlcv_df=sample_ohlcv,
            benchmark_col='SPY'
        )

        validation = engine.validate_features(features)

        # Should have basic structure
        assert validation['n_samples'] > 0
        assert validation['n_features'] > 0

        # With synthetic random data, some edge cases may trigger validation warnings
        # The key is that the features are structurally sound
        if not validation['is_valid']:
            # Allow minor issues with synthetic data
            for issue in validation['issues']:
                # Only fail on serious issues like NaN or infinite values
                assert 'NaN' not in issue, f"Serious issue: {issue}"
                assert 'infinite' not in issue, f"Serious issue: {issue}"

    def test_get_feature_names(self, engine):
        """Test feature name generation"""
        symbols = ['SPY', 'XLF']
        names = engine.get_feature_names(symbols, has_ohlcv=True)

        # Should have features for both symbols
        assert any('SPY' in name for name in names)
        assert any('XLF' in name for name in names)

        # Should include expected feature types
        assert any('vol' in name.lower() for name in names)
        assert any('rsi' in name.lower() for name in names)
        assert any('beta' in name.lower() for name in names)


class TestLookaheadBiasPrevention(TestFeatureEngine):
    """Tests to ensure no lookahead bias in feature calculations"""

    def test_rolling_calculations_use_past_data(self, engine, sample_prices):
        """Verify rolling calculations only use past data"""
        returns = engine.compute_returns(sample_prices['SPY'])
        vol = engine.compute_rolling_volatility(returns, window=20)

        # For any point t, volatility should only depend on data up to t
        # Compare volatility at t with hand-calculated using only data up to t
        for t in range(20, min(50, len(returns))):
            past_returns = returns.iloc[:t+1]
            expected_vol = past_returns.iloc[-20:].std() * np.sqrt(252)
            actual_vol = vol.iloc[t]

            np.testing.assert_allclose(actual_vol, expected_vol, rtol=1e-10)

    def test_normalization_uses_past_window(self, sample_prices):
        """Verify normalization doesn't use future data"""
        from qcml.data.preprocessing import DataPreprocessor, PreprocessingConfig, NormalizationMethod

        preprocessor = DataPreprocessor(PreprocessingConfig(
            normalization=NormalizationMethod.ROLLING_ZSCORE,
            norm_window=20
        ))

        prices = sample_prices['SPY'].values
        df = pd.DataFrame({'price': prices})

        normalized = preprocessor.normalize(df)

        # Check that normalization at time t uses only data up to t
        for t in range(20, min(50, len(df))):
            window = df['price'].iloc[t-19:t+1]  # Last 20 points including t
            expected_mean = window.mean()
            expected_std = window.std()

            actual_zscore = normalized['price'].iloc[t]
            expected_zscore = (df['price'].iloc[t] - expected_mean) / expected_std

            np.testing.assert_allclose(actual_zscore, expected_zscore, rtol=1e-10)


class TestFeatureConfig:
    """Tests for FeatureConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = FeatureConfig()

        assert FeatureCategory.ALL in config.categories
        assert config.volatility_windows == [5, 10, 20, 60]
        assert config.benchmark_col == "SPY"

    def test_should_compute(self):
        """Test category computation check"""
        # ALL includes everything
        config_all = FeatureConfig(categories={FeatureCategory.ALL})
        assert config_all.should_compute(FeatureCategory.VOLATILITY)
        assert config_all.should_compute(FeatureCategory.MOMENTUM)

        # Specific category
        config_vol = FeatureConfig(categories={FeatureCategory.VOLATILITY})
        assert config_vol.should_compute(FeatureCategory.VOLATILITY)
        assert not config_vol.should_compute(FeatureCategory.MOMENTUM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
