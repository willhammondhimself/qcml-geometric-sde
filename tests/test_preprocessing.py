"""
Unit tests for QCML preprocessing module (Phase 2)

Tests cover:
    - Missing data handling (forward fill, interpolation, drop)
    - Outlier detection and treatment (rolling z-score, IQR, clipping)
    - Normalization methods (rolling z-score, cross-sectional, MinMax)
    - Walk-forward validation with embargo period
    - Lookahead bias prevention
"""

import pytest
import numpy as np
import pandas as pd

from qcml.data.preprocessing import (
    DataPreprocessor,
    PreprocessingConfig,
    NormalizationMethod,
    OutlierMethod,
    MissingDataMethod,
    WalkForwardFold,
    create_preprocessor
)


class TestDataPreprocessor:
    """Tests for DataPreprocessor class"""

    @pytest.fixture
    def sample_features(self):
        """Generate sample feature data"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        features = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n) * 2,
            'feature_3': np.random.randn(n) + 5,
        }, index=dates)

        return features

    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        returns = np.random.normal(0.0005, 0.02, n)
        prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)

        return prices

    @pytest.fixture
    def preprocessor(self):
        """Create default DataPreprocessor"""
        return DataPreprocessor()


class TestMissingDataHandling(TestDataPreprocessor):
    """Tests for missing data handling"""

    def test_forward_fill(self, preprocessor, sample_features):
        """Test forward fill method"""
        # Add missing values
        df = sample_features.copy()
        df.iloc[10:13, 0] = np.nan  # 3 consecutive NaN

        result = preprocessor.handle_missing_data(df, method=MissingDataMethod.FORWARD_FILL)

        # Should fill NaN values
        assert result.iloc[10:13, 0].notna().all()

        # Forward fill uses previous value
        assert result.iloc[10, 0] == df.iloc[9, 0]
        assert result.iloc[11, 0] == df.iloc[9, 0]

    def test_forward_fill_limit(self, sample_features):
        """Test forward fill respects limit"""
        preprocessor = DataPreprocessor(PreprocessingConfig(ffill_limit=2))

        df = sample_features.copy()
        df.iloc[10:15, 0] = np.nan  # 5 consecutive NaN

        result = preprocessor.handle_missing_data(df, method=MissingDataMethod.FORWARD_FILL)

        # Should only fill first 2 NaN (limit=2)
        assert result.iloc[10, 0] == df.iloc[9, 0]
        assert result.iloc[11, 0] == df.iloc[9, 0]
        assert pd.isna(result.iloc[12, 0])

    def test_interpolate(self, preprocessor, sample_features):
        """Test linear interpolation"""
        df = sample_features.copy()
        df.iloc[10, 0] = np.nan  # Single NaN

        result = preprocessor.handle_missing_data(df, method=MissingDataMethod.INTERPOLATE)

        # Should interpolate between neighbors
        assert not pd.isna(result.iloc[10, 0])
        expected = (df.iloc[9, 0] + df.iloc[11, 0]) / 2
        np.testing.assert_allclose(result.iloc[10, 0], expected, rtol=0.01)

    def test_drop_missing(self, preprocessor, sample_features):
        """Test dropping rows with missing data"""
        df = sample_features.copy()
        df.iloc[10, 0] = np.nan
        df.iloc[20, 1] = np.nan

        result = preprocessor.handle_missing_data(df, method=MissingDataMethod.DROP)

        # Should have fewer rows
        assert len(result) == len(df) - 2
        assert result.notna().all().all()


class TestOutlierDetection(TestDataPreprocessor):
    """Tests for outlier detection and treatment"""

    def test_detect_outliers_rolling_zscore(self, preprocessor, sample_features):
        """Test rolling z-score outlier detection"""
        df = sample_features.copy()
        # Add obvious outlier
        df.iloc[100, 0] = df['feature_1'].mean() + 10 * df['feature_1'].std()

        outliers = preprocessor.detect_outliers(df, method=OutlierMethod.ROLLING_ZSCORE)

        # Should detect the outlier
        # Note: May not detect if rolling window hasn't seen enough data yet
        assert outliers.iloc[100:120, 0].any()  # Some detection near outlier

    def test_treat_outliers_clipping(self, sample_features):
        """Test outlier clipping"""
        config = PreprocessingConfig(
            outlier_method=OutlierMethod.ROLLING_ZSCORE,
            outlier_threshold=3.0
        )
        preprocessor = DataPreprocessor(config)

        df = sample_features.copy()
        # Add extreme outlier
        original_outlier = df['feature_1'].mean() + 20 * df['feature_1'].std()
        df.iloc[200, 0] = original_outlier

        result = preprocessor.treat_outliers(df)

        # Outlier should be clipped
        assert result.iloc[200, 0] < original_outlier

    def test_no_outlier_treatment(self, sample_features):
        """Test disabled outlier treatment"""
        config = PreprocessingConfig(outlier_method=OutlierMethod.NONE)
        preprocessor = DataPreprocessor(config)

        df = sample_features.copy()
        df.iloc[100, 0] = 1000  # Extreme value

        result = preprocessor.treat_outliers(df)

        # Should not change outlier
        assert result.iloc[100, 0] == 1000


class TestNormalization(TestDataPreprocessor):
    """Tests for normalization methods"""

    def test_rolling_zscore_normalization(self, sample_features):
        """Test rolling z-score normalization"""
        config = PreprocessingConfig(
            normalization=NormalizationMethod.ROLLING_ZSCORE,
            norm_window=50
        )
        preprocessor = DataPreprocessor(config)

        result = preprocessor.normalize(sample_features)

        # After warmup, normalized values should have ~0 mean and ~1 std
        # (approximately, for rolling window)
        clean_result = result.dropna()
        assert abs(clean_result.mean().mean()) < 0.5
        assert 0.5 < clean_result.std().mean() < 1.5

    def test_cross_sectional_normalization(self, sample_features):
        """Test cross-sectional normalization"""
        config = PreprocessingConfig(normalization=NormalizationMethod.CROSS_SECTIONAL)
        preprocessor = DataPreprocessor(config)

        result = preprocessor.normalize(sample_features)

        # Each row should have mean ~0 and std ~1 across columns
        row_means = result.mean(axis=1)
        row_stds = result.std(axis=1)

        np.testing.assert_allclose(row_means.dropna().values, 0, atol=1e-10)
        np.testing.assert_allclose(row_stds.dropna().values, 1, atol=1e-10)

    def test_minmax_normalization(self, sample_features):
        """Test MinMax normalization"""
        config = PreprocessingConfig(
            normalization=NormalizationMethod.MINMAX,
            norm_window=50
        )
        preprocessor = DataPreprocessor(config)

        result = preprocessor.normalize(sample_features)

        # After warmup, values should mostly be in [0, 1]
        clean_result = result.dropna()
        assert (clean_result >= -0.1).all().all()  # Allow small undershoot
        assert (clean_result <= 1.1).all().all()  # Allow small overshoot

    def test_no_normalization(self, sample_features):
        """Test disabled normalization"""
        config = PreprocessingConfig(normalization=NormalizationMethod.NONE)
        preprocessor = DataPreprocessor(config)

        result = preprocessor.normalize(sample_features)

        # Should be unchanged
        pd.testing.assert_frame_equal(result, sample_features)


class TestFullPipeline(TestDataPreprocessor):
    """Tests for full preprocessing pipeline"""

    def test_fit_transform(self, preprocessor, sample_features):
        """Test full fit_transform pipeline"""
        # Add some issues to the data
        df = sample_features.copy()
        df.iloc[10, 0] = np.nan
        df.iloc[200, 1] = df['feature_2'].mean() + 20 * df['feature_2'].std()

        result = preprocessor.fit_transform(df)

        # Should have no NaN
        assert result.notna().all().all()

        # Should have fewer samples (warmup dropped)
        assert len(result) <= len(df)

    def test_transform_consistency(self, sample_features):
        """Test transform uses fitted parameters"""
        config = PreprocessingConfig(
            normalization=NormalizationMethod.ROLLING_ZSCORE,
            norm_window=50
        )
        preprocessor = DataPreprocessor(config)

        # Fit and transform
        train_result = preprocessor.fit_transform(sample_features)

        # Transform again (simulating test data)
        test_result = preprocessor.transform(sample_features)

        # Results should be similar (same data)
        # Note: Won't be identical due to rolling nature
        assert train_result.shape == test_result.shape


class TestWalkForwardValidation(TestDataPreprocessor):
    """Tests for walk-forward validation"""

    def test_basic_walk_forward_split(self, sample_features, sample_prices):
        """Test basic walk-forward splitting"""
        # Use config with smaller min_train_samples for test data
        config = PreprocessingConfig(
            train_window=100,
            test_window=21,
            embargo_days=5,
            min_train_samples=50  # Lower threshold for testing
        )
        preprocessor = DataPreprocessor(config)

        folds = list(preprocessor.walk_forward_split(
            sample_features,
            sample_prices,
            train_window=100,
            test_window=21,
            embargo_days=5
        ))

        assert len(folds) > 0

        # Check first fold structure
        fold = folds[0]
        assert isinstance(fold, WalkForwardFold)
        assert len(fold.train_features) == 100
        assert len(fold.test_features) == 21
        assert fold.train_end + 5 <= fold.test_start  # Embargo respected

    def test_embargo_prevents_leakage(self, preprocessor, sample_features, sample_prices):
        """Test that embargo period creates gap between train and test"""
        embargo_days = 10

        folds = list(preprocessor.walk_forward_split(
            sample_features,
            sample_prices,
            train_window=100,
            test_window=21,
            embargo_days=embargo_days
        ))

        for fold in folds:
            # Test start should be at least embargo_days after train end
            assert fold.test_start >= fold.train_end + embargo_days

    def test_no_data_overlap(self, preprocessor, sample_features, sample_prices):
        """Test that train and test periods don't overlap"""
        folds = list(preprocessor.walk_forward_split(
            sample_features,
            sample_prices,
            train_window=100,
            test_window=21,
            embargo_days=5
        ))

        for fold in folds:
            train_dates = set(fold.train_features.index)
            test_dates = set(fold.test_features.index)

            # No overlap
            assert len(train_dates & test_dates) == 0

    def test_expanding_window(self, sample_features, sample_prices):
        """Test expanding training window"""
        config = PreprocessingConfig(
            train_window=100,
            test_window=21,
            embargo_days=5,
            expanding_train=True,
            min_train_samples=50
        )
        preprocessor = DataPreprocessor(config)

        folds = list(preprocessor.walk_forward_split(
            sample_features,
            sample_prices,
            expanding=True
        ))

        # Each successive fold should have more training data
        train_sizes = [len(fold.train_features) for fold in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1]

    def test_rolling_window(self, sample_features, sample_prices):
        """Test rolling (fixed) training window"""
        config = PreprocessingConfig(
            train_window=100,
            test_window=21,
            embargo_days=5,
            expanding_train=False
        )
        preprocessor = DataPreprocessor(config)

        folds = list(preprocessor.walk_forward_split(
            sample_features,
            sample_prices,
            expanding=False
        ))

        # All folds should have same training size
        train_sizes = [len(fold.train_features) for fold in folds]
        assert all(size == 100 for size in train_sizes)

    def test_insufficient_data_error(self, preprocessor):
        """Test error when insufficient data for walk-forward"""
        # Create small dataset
        small_features = pd.DataFrame(
            np.random.randn(50, 3),
            index=pd.date_range("2020-01-01", periods=50, freq="D")
        )
        small_prices = pd.Series(
            np.random.randn(50) + 100,
            index=pd.date_range("2020-01-01", periods=50, freq="D")
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            list(preprocessor.walk_forward_split(
                small_features,
                small_prices,
                train_window=100,  # More than available data
                test_window=21,
                embargo_days=5
            ))

    def test_walk_forward_info(self, preprocessor):
        """Test walk-forward info without generating data"""
        info = preprocessor.get_walk_forward_info(
            n_samples=500,
            train_window=100,
            test_window=21,
            embargo_days=5
        )

        assert info['sufficient_data']
        assert info['n_folds'] > 0
        assert info['train_window'] == 100
        assert info['test_window'] == 21

    def test_walk_forward_info_insufficient(self, preprocessor):
        """Test walk-forward info with insufficient data"""
        info = preprocessor.get_walk_forward_info(
            n_samples=50,  # Too small
            train_window=100,
            test_window=21,
            embargo_days=5
        )

        assert not info['sufficient_data']
        assert info['n_folds'] == 0


class TestValidation(TestDataPreprocessor):
    """Tests for preprocessing validation"""

    def test_validate_preprocessing(self, preprocessor, sample_features):
        """Test validation of preprocessed data"""
        result = preprocessor.fit_transform(sample_features)
        validation = preprocessor.validate_preprocessing(result)

        assert validation['is_valid']
        assert validation['n_samples'] > 0
        assert len(validation['issues']) == 0

    def test_validate_detects_nan(self, preprocessor):
        """Test validation detects NaN values"""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [1, 2, 3, 4]
        })

        validation = preprocessor.validate_preprocessing(df)

        assert not validation['is_valid']
        assert any('NaN' in issue for issue in validation['issues'])


class TestFactoryFunction:
    """Tests for create_preprocessor factory function"""

    def test_create_preprocessor_default(self):
        """Test default preprocessor creation"""
        preprocessor = create_preprocessor()

        assert preprocessor.config.normalization == NormalizationMethod.ROLLING_ZSCORE
        assert preprocessor.config.train_window == 252

    def test_create_preprocessor_custom(self):
        """Test custom preprocessor creation"""
        preprocessor = create_preprocessor(
            normalization="cross_sectional",
            train_window=500,
            test_window=42
        )

        assert preprocessor.config.normalization == NormalizationMethod.CROSS_SECTIONAL
        assert preprocessor.config.train_window == 500
        assert preprocessor.config.test_window == 42


class TestLookaheadBiasPrevention(TestDataPreprocessor):
    """Tests to ensure no lookahead bias in preprocessing"""

    def test_walk_forward_no_future_data(self, preprocessor, sample_features, sample_prices):
        """Verify walk-forward splits don't leak future data"""
        folds = list(preprocessor.walk_forward_split(
            sample_features,
            sample_prices,
            train_window=100,
            test_window=21,
            embargo_days=5
        ))

        for fold in folds:
            # Training end date must be before test start date
            train_end_date = fold.train_dates[1]
            test_start_date = fold.test_dates[0]

            # Embargo ensures gap
            assert train_end_date < test_start_date

            # Verify no data from test period appears in training
            train_indices = set(fold.train_features.index)
            test_indices = set(fold.test_features.index)

            assert len(train_indices & test_indices) == 0

    def test_rolling_normalization_no_future(self, sample_features):
        """Verify rolling normalization only uses past data"""
        config = PreprocessingConfig(
            normalization=NormalizationMethod.ROLLING_ZSCORE,
            norm_window=20
        )
        preprocessor = DataPreprocessor(config)

        result = preprocessor.normalize(sample_features)

        # For any point t, normalization should only use data up to t
        # Verify by manual calculation
        for t in range(20, min(50, len(sample_features))):
            for col in sample_features.columns:
                window = sample_features[col].iloc[t-19:t+1]  # Last 20 points including t
                expected_mean = window.mean()
                expected_std = window.std()

                actual_zscore = result[col].iloc[t]
                expected_zscore = (sample_features[col].iloc[t] - expected_mean) / expected_std

                np.testing.assert_allclose(actual_zscore, expected_zscore, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
