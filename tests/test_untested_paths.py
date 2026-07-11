"""Behavioral coverage for previously-untested modules.

The headline detectors are pinned by ``test_headline_detectors_characterization``;
this module covers the other zero-coverage, high-bug-surface paths:

  * ``adaptive_threshold`` — score → boolean-alarm conversion + persistence.
  * ``topology``           — rolling Chern, transition detection, event classify.
  * ``online_detection``   — strictly-causal streaming feature/P(crisis) path.
  * Paper-2 ``fusion``     — hierarchical / regime-adaptive / SPRT accumulators.

These are smoke + key-invariant checks (boolean alarms, P(crisis) in [0,1],
finite outputs, causal thresholds), not golden snapshots.
"""

import numpy as np

from qcml_geometry.adaptive_threshold import (
    CombinedAdaptiveThreshold,
    RollingQuantileThreshold,
    _persistence_filter,
)
from qcml_geometry.core import QCMLGeometry
from qcml_geometry.fusion import (
    ACTIVE_CHANNELS,
    BayesianEvidenceAccumulator,
    HierarchicalFusionDetector,
    RegimeAdaptiveFusionDetector,
)
from qcml_geometry.observables import BaseRegimeDetector
from qcml_geometry.online_detection import (
    ExpandingPercentileDetector,
    OnlineEnsembleDetector,
    OnlineGeometricFeatureComputer,
)
from qcml_geometry.topology import TopologicalRegimeDetector


def _scores_with_spike(T=400, spike=(200, 230), seed=0):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 1, T)
    s[spike[0] : spike[1]] += 8.0
    return s


# --------------------------------------------------------------------------- #
# adaptive_threshold
# --------------------------------------------------------------------------- #


def test_persistence_filter_removes_short_runs():
    mask = np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 0], dtype=bool)
    out = _persistence_filter(mask, min_persistence=3)
    # Only the length-3 run survives.
    assert out.tolist() == [False, False, False, True, True, True, False, False, False, False]


def test_persistence_filter_identity_when_one():
    mask = np.array([0, 1, 0, 1], dtype=bool)
    assert np.array_equal(_persistence_filter(mask, 1), mask)


def test_rolling_quantile_flags_spike_and_is_boolean():
    s = _scores_with_spike()
    det = RollingQuantileThreshold(lookback=120, quantile=0.95, persistence=3, min_history=60)
    alarm, thr = det.detect(s)
    assert alarm.dtype == bool and alarm.shape == s.shape
    assert thr.shape == s.shape
    assert alarm[200:230].sum() >= 5, "the injected spike should raise alarms"
    assert not alarm[:60].any(), "no alarms during warmup"


def test_quantile_threshold_is_causal():
    """Thresholds use only trailing data (with a gap), so perturbing the
    future must not change an earlier threshold."""
    s = _scores_with_spike()
    det = RollingQuantileThreshold(lookback=120, quantile=0.95, min_history=60, gap=5)
    base = det.compute_thresholds(s)
    s2 = s.copy()
    s2[300:] += 50.0
    pert = det.compute_thresholds(s2)
    k = 300 - det.gap  # threshold at t uses scores up to t - gap
    np.testing.assert_allclose(base[:k], pert[:k], equal_nan=True)


def test_combined_threshold_returns_boolean_and_details():
    s = _scores_with_spike()
    alarm, details = CombinedAdaptiveThreshold().detect(s)
    assert alarm.dtype == bool and alarm.shape == s.shape
    assert {"quantile_alarm", "velocity_alarm"} <= set(details)


# --------------------------------------------------------------------------- #
# topology
# --------------------------------------------------------------------------- #


def _fitted_geometry(n_features=3, hilbert_dim=4, seed=0):
    rng = np.random.default_rng(seed)
    Xfit = rng.normal(0, 1, (80, n_features))
    Xfit /= np.linalg.norm(Xfit, axis=1, keepdims=True) + 1e-9
    g = QCMLGeometry(n_features=n_features, hilbert_dim=hilbert_dim)
    g.fit_operators(Xfit, method="random")
    return g


def _toy_path(n_features=3, T=60, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.3, (T, n_features))
    X[T // 2 :] += 2.0  # regime shift
    return X


def test_topology_rolling_chern_is_finite():
    det = TopologicalRegimeDetector(_fitted_geometry(), window_size=20, smoothing_window=3)
    C = det.rolling_chern_number(_toy_path())
    assert np.all(np.isfinite(C))


def test_topology_detect_transitions_returns_wellformed_list():
    det = TopologicalRegimeDetector(_fitted_geometry(), window_size=20, chern_threshold=0.1)
    transitions = det.detect_transitions(_toy_path())
    assert isinstance(transitions, list)
    for t in transitions:
        assert t.start_idx <= t.end_idx
        assert np.isfinite(t.delta_chern)


def test_topology_classify_event_returns_valid_label():
    det = TopologicalRegimeDetector(_fitted_geometry(), window_size=10)
    X = _toy_path(T=60)
    res = det.classify_event(X[:20], X[20:40], X[40:])
    assert res["event_type"] in {"REGIME_CHANGE", "EXTREME_EVENT", "NORMAL_FLUCTUATION"}


# --------------------------------------------------------------------------- #
# online_detection
# --------------------------------------------------------------------------- #


def _enriched_stream(T=80, d=4, seed=2):
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 0.01, (T, d))
    raw[T // 2 : T // 2 + 10] *= 6.0
    return BaseRegimeDetector.build_enriched_features(raw, lookback=20)


def _feature_computer():
    return OnlineGeometricFeatureComputer(
        hilbert_dim=4, n_pca_components=6, min_history=30, refit_interval=15
    )


def test_online_feature_computer_warmup_then_dict():
    Xe = _enriched_stream()
    fc = _feature_computer()
    out = [fc.update(Xe[t]) for t in range(len(Xe))]
    assert out[0] is None, "warmup must yield None"
    later = [o for o in out if o is not None]
    assert later, "must emit features after warmup"
    assert all(isinstance(o, dict) for o in later)


def test_online_percentile_p_in_unit_interval():
    Xe = _enriched_stream()
    fc = _feature_computer()
    det = ExpandingPercentileDetector(min_history=10)
    ps = [det.update(fc.update(Xe[t])) for t in range(len(Xe))]
    finite = [p for p in ps if np.isfinite(p)]
    assert finite, "should emit at least one finite P(crisis)"
    assert all(0.0 <= p <= 1.0 for p in finite)


def test_online_ensemble_combines_detectors():
    Xe = _enriched_stream()
    fc = _feature_computer()
    ens = OnlineEnsembleDetector([ExpandingPercentileDetector(10), ExpandingPercentileDetector(10)])
    ps = [ens.update(fc.update(Xe[t])) for t in range(len(Xe))]
    finite = [p for p in ps if np.isfinite(p)]
    assert all(0.0 <= p <= 1.0 for p in finite)


# --------------------------------------------------------------------------- #
# Paper-2 fusion (precomputed-score path)
# --------------------------------------------------------------------------- #


def _score_matrix(T, names, seed=3):
    rng = np.random.default_rng(seed)
    M = rng.normal(0, 1, (T, len(names)))
    M[T // 2 : T // 2 + 20] += 5.0
    return M


def test_hierarchical_fusion_smoke():
    names = list(ACTIVE_CHANNELS)
    M = _score_matrix(200, names)
    det = HierarchicalFusionDetector(channel_names=names).set_precomputed_scores(M)
    out = det.compute_regime_scores(M)
    assert out.shape == (200,)
    assert np.isfinite(out).sum() > 20


def test_regime_adaptive_fusion_smoke():
    names = list(ACTIVE_CHANNELS)
    M = _score_matrix(160, names)
    det = RegimeAdaptiveFusionDetector(
        channel_names=names, min_train_obs=60, retrain_interval=30
    ).set_precomputed_scores(M)
    out = det.compute_regime_scores(M)
    assert out.shape == (160,)


def test_bayesian_evidence_smoke():
    names = list(ACTIVE_CHANNELS)
    M = _score_matrix(160, names)
    det = BayesianEvidenceAccumulator(channel_names=names).set_precomputed_scores(M)
    out = det.compute_regime_scores(M)
    assert out.shape == (160,)
