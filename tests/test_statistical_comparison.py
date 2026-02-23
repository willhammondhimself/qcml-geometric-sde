"""
Unit tests for the statistical comparison framework.

Tests cover: Model Confidence Set, Bayesian signed-rank test, bootstrap rank CIs,
win-rate analysis, and oracle/complementarity analysis.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Shared RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def d_matrix_with_dominant(rng):
    """D-value matrix where method 0 clearly dominates (highest d everywhere).

    Shape: (12, 4) — 12 crises, 4 methods.
    Method 0: d ~ 2.0, Methods 1-3: d ~ 0.5.
    """
    n_crises, n_methods = 12, 4
    d = rng.uniform(0.3, 0.7, size=(n_crises, n_methods))
    d[:, 0] = rng.uniform(1.8, 2.2, size=n_crises)  # Method 0 dominates
    return d


@pytest.fixture
def d_matrix_identical():
    """D-value matrix where all methods are identical.

    Shape: (12, 3) — 12 crises, 3 methods, all with d=1.0.
    """
    return np.ones((12, 3))


@pytest.fixture
def d_matrix_realistic(rng):
    """Realistic d-value matrix with overlapping performance.

    Shape: (12, 5) — 12 crises, 5 methods.
    Methods 0-1 slightly better on average, but no clear winner.
    """
    base = rng.uniform(0.3, 1.2, size=(12, 5))
    base[:, 0] += 0.2  # Slight edge
    base[:, 1] += 0.15
    return base


@pytest.fixture
def method_names_4():
    """Method names for 4-method matrix."""
    return ["Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity", "Random Forest"]


@pytest.fixture
def method_names_5():
    """Method names for 5-method matrix."""
    return [
        "Berry Phase Rate",
        "QFI Determinant",
        "Multi-Lag Fidelity",
        "Random Forest",
        "CUSUM",
    ]


# ---------------------------------------------------------------------------
# TestModelConfidenceSet
# ---------------------------------------------------------------------------


class TestModelConfidenceSet:
    """Test the Hansen-Lunde-Nason Model Confidence Set implementation."""

    def test_dominant_method_always_in_mcs(self, d_matrix_with_dominant):
        """When one method clearly dominates, it must be in the MCS."""
        from experiments.statistical_comparison import compute_model_confidence_set

        mcs = compute_model_confidence_set(
            d_matrix_with_dominant, alpha=0.10, n_bootstrap=5000, seed=42
        )
        assert 0 in mcs, "Dominant method (index 0) must be in the MCS"

    def test_identical_methods_all_in_mcs(self, d_matrix_identical):
        """When all methods are identical, all should be in the MCS."""
        from experiments.statistical_comparison import compute_model_confidence_set

        mcs = compute_model_confidence_set(
            d_matrix_identical, alpha=0.10, n_bootstrap=5000, seed=42
        )
        assert set(mcs) == {0, 1, 2}, "All identical methods should be in the MCS"

    def test_mcs_nonempty(self, d_matrix_realistic):
        """MCS should never be empty."""
        from experiments.statistical_comparison import compute_model_confidence_set

        mcs = compute_model_confidence_set(
            d_matrix_realistic, alpha=0.10, n_bootstrap=5000, seed=42
        )
        assert len(mcs) > 0, "MCS must contain at least one method"

    def test_mcs_returns_list_of_ints(self, d_matrix_realistic):
        """MCS should return a list of integer indices."""
        from experiments.statistical_comparison import compute_model_confidence_set

        mcs = compute_model_confidence_set(
            d_matrix_realistic, alpha=0.10, n_bootstrap=2000, seed=42
        )
        assert isinstance(mcs, list)
        for idx in mcs:
            assert isinstance(idx, (int, np.integer))

    def test_mcs_subset_of_all_methods(self, d_matrix_with_dominant):
        """MCS indices must be valid column indices."""
        from experiments.statistical_comparison import compute_model_confidence_set

        n_methods = d_matrix_with_dominant.shape[1]
        mcs = compute_model_confidence_set(
            d_matrix_with_dominant, alpha=0.10, n_bootstrap=2000, seed=42
        )
        for idx in mcs:
            assert 0 <= idx < n_methods


# ---------------------------------------------------------------------------
# TestBayesianSignedRank
# ---------------------------------------------------------------------------


class TestBayesianSignedRank:
    """Test the Bayesian signed-rank test wrapper."""

    def test_identical_arrays_high_p_equivalent(self):
        """Identical arrays should yield high P(equivalent)."""
        from experiments.statistical_comparison import bayesian_signed_rank

        d_a = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.55, 0.65, 0.75, 0.85])
        d_b = d_a.copy()
        result = bayesian_signed_rank(d_a, d_b, rope=0.1)
        assert result["p_equivalent"] > 0.5, (
            f"P(equivalent) should be > 0.5 for identical arrays, got {result['p_equivalent']}"
        )

    def test_clearly_different_arrays(self, rng):
        """When A clearly dominates B, P(A > B) should be high."""
        from experiments.statistical_comparison import bayesian_signed_rank

        d_a = np.array([2.0, 2.1, 1.9, 2.2, 1.8, 2.0, 2.3, 1.95, 2.05, 2.15])
        d_b = np.array([0.3, 0.4, 0.2, 0.5, 0.1, 0.35, 0.25, 0.45, 0.15, 0.3])
        result = bayesian_signed_rank(d_a, d_b, rope=0.1)
        assert result["p_a_better"] > 0.8, (
            f"P(A > B) should be > 0.8 for clearly different arrays, got {result['p_a_better']}"
        )

    def test_returns_three_probabilities(self):
        """Result dict must have p_a_better, p_equivalent, p_b_better."""
        from experiments.statistical_comparison import bayesian_signed_rank

        d_a = np.array([1.0, 1.1, 0.9, 1.0, 1.05])
        d_b = np.array([0.8, 0.9, 0.7, 1.0, 0.85])
        result = bayesian_signed_rank(d_a, d_b, rope=0.1)
        assert "p_a_better" in result
        assert "p_equivalent" in result
        assert "p_b_better" in result

    def test_probabilities_sum_to_one(self):
        """The three probabilities should sum to approximately 1.0."""
        from experiments.statistical_comparison import bayesian_signed_rank

        d_a = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.2, 0.85])
        d_b = np.array([0.8, 0.9, 0.7, 1.0, 0.85, 0.75, 1.1, 0.65])
        result = bayesian_signed_rank(d_a, d_b, rope=0.1)
        total = result["p_a_better"] + result["p_equivalent"] + result["p_b_better"]
        assert abs(total - 1.0) < 0.05, f"Probabilities should sum to ~1.0, got {total}"


# ---------------------------------------------------------------------------
# TestBootstrapRankCIs
# ---------------------------------------------------------------------------


class TestBootstrapRankCIs:
    """Test bootstrap confidence intervals for mean ranks."""

    def test_ci_contains_actual_mean_rank(self, d_matrix_realistic):
        """The CI should contain the actual (non-bootstrapped) mean rank."""
        from experiments.statistical_comparison import bootstrap_rank_cis
        from scipy.stats import rankdata

        result = bootstrap_rank_cis(d_matrix_realistic, n_bootstrap=5000, seed=42)

        # Compute actual mean ranks
        n_crises, n_methods = d_matrix_realistic.shape
        actual_ranks = np.zeros((n_crises, n_methods))
        for i in range(n_crises):
            actual_ranks[i] = rankdata(-d_matrix_realistic[i])
        actual_mean_ranks = actual_ranks.mean(axis=0)

        for j in range(n_methods):
            ci_lo = result[j]["ci_lo"]
            ci_hi = result[j]["ci_hi"]
            actual = actual_mean_ranks[j]
            assert ci_lo <= actual <= ci_hi, (
                f"Method {j}: actual mean rank {actual:.2f} not in "
                f"CI [{ci_lo:.2f}, {ci_hi:.2f}]"
            )

    def test_returns_all_methods(self, d_matrix_realistic):
        """Should return results for all methods in the matrix."""
        from experiments.statistical_comparison import bootstrap_rank_cis

        result = bootstrap_rank_cis(d_matrix_realistic, n_bootstrap=2000, seed=42)
        n_methods = d_matrix_realistic.shape[1]
        assert len(result) == n_methods

    def test_ci_has_required_keys(self, d_matrix_realistic):
        """Each method's result must have mean_rank, ci_lo, ci_hi."""
        from experiments.statistical_comparison import bootstrap_rank_cis

        result = bootstrap_rank_cis(d_matrix_realistic, n_bootstrap=2000, seed=42)
        for j in result:
            assert "mean_rank" in result[j]
            assert "ci_lo" in result[j]
            assert "ci_hi" in result[j]

    def test_ci_lo_le_ci_hi(self, d_matrix_realistic):
        """Lower CI bound must be <= upper CI bound."""
        from experiments.statistical_comparison import bootstrap_rank_cis

        result = bootstrap_rank_cis(d_matrix_realistic, n_bootstrap=2000, seed=42)
        for j in result:
            assert result[j]["ci_lo"] <= result[j]["ci_hi"]


# ---------------------------------------------------------------------------
# TestWinRateAnalysis
# ---------------------------------------------------------------------------


class TestWinRateAnalysis:
    """Test pairwise win-rate computation."""

    def test_dominant_method_wins_all(self, d_matrix_with_dominant, method_names_4):
        """A dominant method should have 100% win rate vs all others."""
        from experiments.statistical_comparison import compute_win_rates

        result = compute_win_rates(d_matrix_with_dominant, method_names_4)
        dominant = method_names_4[0]
        for other in method_names_4[1:]:
            wins = result[dominant][other]
            total = d_matrix_with_dominant.shape[0]
            assert wins == total, (
                f"{dominant} should win all {total} crises vs {other}, got {wins}"
            )

    def test_win_rates_symmetric(self, d_matrix_realistic, method_names_5):
        """Win(A,B) + Win(B,A) + Ties = n_crises."""
        from experiments.statistical_comparison import compute_win_rates

        n_crises = d_matrix_realistic.shape[0]
        result = compute_win_rates(d_matrix_realistic, method_names_5)
        for i, name_a in enumerate(method_names_5):
            for j, name_b in enumerate(method_names_5):
                if i == j:
                    continue
                wins_ab = result[name_a][name_b]
                wins_ba = result[name_b][name_a]
                # wins_ab + wins_ba + ties = n_crises
                assert wins_ab + wins_ba <= n_crises

    def test_returns_nested_dict(self, d_matrix_realistic, method_names_5):
        """Result should be a nested dict with all method pairs."""
        from experiments.statistical_comparison import compute_win_rates

        result = compute_win_rates(d_matrix_realistic, method_names_5)
        for name in method_names_5:
            assert name in result
            for other in method_names_5:
                if other != name:
                    assert other in result[name]


# ---------------------------------------------------------------------------
# TestOracleAndComplementarity
# ---------------------------------------------------------------------------


class TestOracleAndComplementarity:
    """Test oracle and complementarity analysis."""

    def test_oracle_at_least_as_good_as_best_single(self, d_matrix_realistic, method_names_5):
        """Oracle median d should be >= best single method's median d."""
        from experiments.statistical_comparison import compute_oracle_and_complementarity

        result = compute_oracle_and_complementarity(d_matrix_realistic, method_names_5)
        assert result["oracle_median_d"] >= result["best_single_median_d"]

    def test_complementarity_score_in_range(self, d_matrix_realistic, method_names_5):
        """Complementarity score should be in [0, 1]."""
        from experiments.statistical_comparison import compute_oracle_and_complementarity

        result = compute_oracle_and_complementarity(d_matrix_realistic, method_names_5)
        assert 0.0 <= result["complementarity_score"] <= 1.0

    def test_oracle_improvement_nonneg(self, d_matrix_realistic, method_names_5):
        """Oracle improvement should be >= 0."""
        from experiments.statistical_comparison import compute_oracle_and_complementarity

        result = compute_oracle_and_complementarity(d_matrix_realistic, method_names_5)
        assert result["oracle_improvement"] >= 0.0

    def test_has_correlation_matrix(self, d_matrix_realistic, method_names_5):
        """Result should include a correlation matrix of correct shape."""
        from experiments.statistical_comparison import compute_oracle_and_complementarity

        result = compute_oracle_and_complementarity(d_matrix_realistic, method_names_5)
        corr = result["correlation_matrix"]
        n = len(method_names_5)
        assert corr.shape == (n, n)


# ---------------------------------------------------------------------------
# TestCDDiagram (integration — requires autorank)
# ---------------------------------------------------------------------------


class TestCDDiagram:
    """Test critical difference diagram generation (requires autorank)."""

    def test_generate_cd_diagram_returns_result(self, d_matrix_realistic, method_names_5):
        """generate_cd_diagram should return an autorank result object."""
        from experiments.statistical_comparison import generate_cd_diagram

        result = generate_cd_diagram(d_matrix_realistic, method_names_5, output_path=None)
        # autorank result should have rankdf attribute
        assert hasattr(result, "rankdf")

    def test_generate_cd_diagram_with_output(self, d_matrix_realistic, method_names_5, tmp_path):
        """generate_cd_diagram should save a PDF when output_path is given."""
        from experiments.statistical_comparison import generate_cd_diagram

        out = tmp_path / "cd_test.pdf"
        result = generate_cd_diagram(
            d_matrix_realistic, method_names_5, output_path=str(out)
        )
        assert result is not None
        assert out.exists()


# ---------------------------------------------------------------------------
# TestBuildDMatrix
# ---------------------------------------------------------------------------


class TestBuildDMatrix:
    """Test d-value matrix construction from JSON results dict."""

    def test_build_d_matrix_basic(self):
        """Build a d-value matrix from a results dict."""
        from experiments.statistical_comparison import build_d_matrix

        results = {
            "Method A": {
                "2008_gfc": {"d": 1.0},
                "2020_covid": {"d": 0.8},
            },
            "Method B": {
                "2008_gfc": {"d": 0.5},
                "2020_covid": {"d": 0.6},
            },
        }
        d_matrix, method_names, crisis_names = build_d_matrix(results)
        assert d_matrix.shape == (2, 2)
        assert len(method_names) == 2
        assert len(crisis_names) == 2

    def test_build_d_matrix_nan_rows_removed(self):
        """Rows with NaN should be removed from the matrix."""
        from experiments.statistical_comparison import build_d_matrix

        results = {
            "Method A": {
                "2008_gfc": {"d": 1.0},
                "2020_covid": {"d": 0.8},
            },
            "Method B": {
                "2008_gfc": {"d": 0.5},
                # 2020_covid missing => NaN
            },
        }
        d_matrix, method_names, crisis_names = build_d_matrix(results)
        # Should only keep the row where both methods have d-values
        assert d_matrix.shape[0] == 1
        assert not np.any(np.isnan(d_matrix))
