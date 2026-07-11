"""Correctness of the (detector, params, crisis) Cohen's d memoization cache.

The cache must be a transparent pass-through: a hit returns exactly what a
recompute would, and the key must capture everything that changes the value
(params, crisis, window, library, data) so stale results can't leak.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")  # experiments stack; core-only installs skip

from experiments.hpo_cache import CohensDCache, data_fingerprint  # noqa: E402


class _Dummy:
    __name__ = "DummyDetector"


def _cache(tmp_path, compute_fn):
    c = CohensDCache(cache_dir=tmp_path, compute_fn=compute_fn)
    c.attach_data(np.zeros((10, 3)))
    return c


def test_key_is_deterministic_and_sensitive(tmp_path):
    c = _cache(tmp_path, lambda *a: 1.0)
    k = c.key(_Dummy, {"a": 1}, "2020_covid", 10)
    assert k == c.key(_Dummy, {"a": 1}, "2020_covid", 10)  # deterministic
    assert k != c.key(_Dummy, {"a": 2}, "2020_covid", 10)  # params matter
    assert k != c.key(_Dummy, {"a": 1}, "2008_gfc", 10)  # crisis matters
    assert k != c.key(_Dummy, {"a": 1}, "2020_covid", 20)  # window matters


def test_data_fingerprint_changes_with_data():
    assert data_fingerprint(np.zeros((5, 2))) != data_fingerprint(np.ones((5, 2)))


def test_hit_equals_recompute_and_computes_once(tmp_path):
    calls = {"n": 0}

    def compute(detector_class, params, X, dates, crisis_key, window_size):
        calls["n"] += 1
        return 0.4242

    c = _cache(tmp_path, compute)
    v1 = c.get_or_compute(_Dummy, {"a": 1}, None, None, "2020_covid")
    v2 = c.get_or_compute(_Dummy, {"a": 1}, None, None, "2020_covid")
    assert v1 == v2 == 0.4242
    assert calls["n"] == 1, "second call must hit the cache, not recompute"
    assert c.hits == 1 and c.misses == 1


def test_different_params_recompute(tmp_path):
    calls = {"n": 0}

    def compute(*a):
        calls["n"] += 1
        return float(calls["n"])

    c = _cache(tmp_path, compute)
    c.get_or_compute(_Dummy, {"a": 1}, None, None, "2020_covid")
    c.get_or_compute(_Dummy, {"a": 2}, None, None, "2020_covid")
    assert calls["n"] == 2


def test_persists_across_instances(tmp_path):
    c1 = _cache(tmp_path, lambda *a: 0.99)
    c1.get_or_compute(_Dummy, {"a": 1}, None, None, "2020_covid")

    # A fresh instance (same dir, same data) must read the pickle, not recompute.
    def boom(*a):
        raise AssertionError("should have been a cache hit")

    c2 = _cache(tmp_path, boom)
    assert c2.get_or_compute(_Dummy, {"a": 1}, None, None, "2020_covid") == 0.99
    assert c2.hits == 1 and c2.misses == 0


def test_install_uninstall_rebinds_hook(tmp_path):
    import experiments.walk_forward_hpo as wf

    original = wf._D_FN
    try:
        c = _cache(tmp_path, lambda *a: 1.0)
        c.install()
        assert wf._D_FN == c.get_or_compute
        CohensDCache.uninstall()
        assert wf._D_FN is wf._compute_d
    finally:
        wf._D_FN = original
