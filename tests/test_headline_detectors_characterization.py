"""Characterization + behavioral tests for the headline geometric detectors.

These detectors back the Paper-1 results but had no direct coverage: the core
geometry math is tested elsewhere, but the detector wrappers (PCA pipeline,
rolling smoothing, expanding z-score, diff-and-prepend alignment, signed vs.
absolute conventions) were not. This module pins them so a refactor of the
shared ``fit`` / ``_expanding_zscore`` machinery cannot silently change outputs.

The golden snapshot (``fixtures/headline_detector_golden.npz``) is generated
from the detectors as configured below. Regenerate intentionally with::

    python tests/test_headline_detectors_characterization.py

Each detector is deterministic given its seed, so the snapshot is an exact
behavioral lock (``np.allclose`` with NaN-awareness).
"""

import os

import numpy as np
import pytest

from qcml_geometry.observables import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    GeometricEnsembleDetector,
    MetricConditionDetector,
    MultiLagFidelityDetector,
    QFIDeterminantDetector,
    SpectralGapDetector,
)

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "headline_detector_golden.npz")

# Detectors keyed by display name, each with representative HPO-style params
# exercising the distinguishing code paths (berry_aggregation, qfi_mode,
# adaptive_epsilon, sphere vs soft normalization, signed vs abs z-score).
CONFIGS = {
    "Berry Phase Rate": (
        BerryPhaseRateDetector,
        dict(
            hilbert_dim=6,
            n_pca_components=8,
            rolling_window=15,
            operator_method="random",
            seed=42,
            normalization="sphere",
            berry_aggregation="f01",
        ),
    ),
    "QFI Determinant": (
        QFIDeterminantDetector,
        dict(
            hilbert_dim=8,
            n_pca_components=12,
            rolling_window=20,
            operator_method="pca_inspired",
            seed=42,
            normalization="soft",
            qfi_mode="logdet",
            adaptive_epsilon=True,
        ),
    ),
    "Multi-Lag Fidelity": (
        MultiLagFidelityDetector,
        dict(
            hilbert_dim=4,
            n_pca_components=8,
            rolling_window=20,
            operator_method="pca_inspired",
            seed=42,
            normalization="sphere",
        ),
    ),
    "Spectral Gap": (
        SpectralGapDetector,
        dict(
            hilbert_dim=8,
            n_pca_components=12,
            rolling_window=20,
            operator_method="random",
            seed=42,
            normalization="soft",
            adaptive_epsilon=True,
        ),
    ),
    "Metric Condition": (
        MetricConditionDetector,
        dict(
            hilbert_dim=8,
            n_pca_components=12,
            rolling_window=20,
            operator_method="random",
            seed=42,
            normalization="soft",
            adaptive_epsilon=True,
        ),
    ),
    "Geometric Ensemble": (
        GeometricEnsembleDetector,
        dict(
            hilbert_dim=8,
            n_pca_components=12,
            rolling_window=20,
            operator_method="random",
            seed=42,
            normalization="sphere",
        ),
    ),
}


def make_data(T=160, d=5, seed=0):
    """Deterministic synthetic returns with an injected mid-series regime shift."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, (T, d))
    rets[T // 2 : T // 2 + 15] *= 6.0
    return rets


def _enriched():
    return BaseRegimeDetector.build_enriched_features(make_data(), lookback=20)


def _scores(cls, params, Xe, fit_end):
    det = cls(causal_fit_length=fit_end, **params)
    det.fit(Xe)
    return np.asarray(det.compute_regime_scores(Xe), dtype=float)


def _all_scores():
    Xe = _enriched()
    fit_end = Xe.shape[0] // 2
    return {name: _scores(cls, params, Xe, fit_end) for name, (cls, params) in CONFIGS.items()}


@pytest.fixture(scope="module")
def computed():
    return _all_scores()


@pytest.mark.parametrize("name", list(CONFIGS))
def test_output_shape_and_warmup(name, computed):
    Xe = _enriched()
    s = computed[name]
    assert s.shape == (Xe.shape[0],), "score series must align to the input length"
    assert np.isnan(s[:3]).all(), "expanding-window warmup must be NaN at the start"
    assert np.isfinite(s).sum() > 20, "must produce a non-trivial number of finite scores"


@pytest.mark.parametrize("name", list(CONFIGS))
def test_deterministic(name, computed):
    Xe = _enriched()
    fit_end = Xe.shape[0] // 2
    cls, params = CONFIGS[name]
    again = _scores(cls, params, Xe, fit_end)
    np.testing.assert_allclose(again, computed[name], equal_nan=True)


@pytest.mark.parametrize("name", ["Berry Phase Rate", "Spectral Gap"])
def test_scoring_is_causal(name):
    """Perturbing data strictly after index k must not change scores at/before k.

    Fit is frozen on the pre-cutoff prefix, so a change in the post-cutoff tail
    can only affect scores from the perturbation point onward.
    """
    Xe = _enriched()
    fit_end = Xe.shape[0] // 2
    cls, params = CONFIGS[name]
    base = _scores(cls, params, Xe, fit_end)

    k = fit_end + 30
    Xe2 = Xe.copy()
    Xe2[k:] += 5.0  # large perturbation, entirely after the cutoff and after k
    perturbed = _scores(cls, params, Xe2, fit_end)

    np.testing.assert_allclose(base[:k], perturbed[:k], equal_nan=True)


def test_golden_snapshot(computed):
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip("golden snapshot missing; regenerate with `python tests/<thisfile>.py`")
    golden = np.load(GOLDEN_PATH)
    for name in CONFIGS:
        key = name.replace(" ", "_")
        assert key in golden.files, f"golden missing channel {name!r}"
        np.testing.assert_allclose(
            computed[name],
            golden[key],
            equal_nan=True,
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"{name} output drifted from golden snapshot",
        )


def _regenerate_golden():
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    scores = {name.replace(" ", "_"): s for name, s in _all_scores().items()}
    np.savez(GOLDEN_PATH, **scores)
    print(f"wrote golden snapshot: {GOLDEN_PATH} ({len(scores)} channels)")


if __name__ == "__main__":
    _regenerate_golden()
