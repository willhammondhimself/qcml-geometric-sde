"""Leak-free invariants for the nested walk-forward HPO protocol.

These verify the core honesty guarantee without touching the network or running
detectors: hyperparameter selection for a test crisis may only use crises that
*ended strictly before* that crisis *started*.
"""

import pandas as pd
import pytest

pytest.importorskip("optuna")  # experiments extra; core-only installs skip this module

from experiments.data_loader import ALL_CRISES  # noqa: E402
from experiments.walk_forward_hpo import (  # noqa: E402
    DEFAULT_OOS_KEYS,
    chronological_keys,
    selection_pool,
)


def test_chronological_keys_sorted_by_start():
    keys = chronological_keys(ALL_CRISES.keys())
    starts = [pd.Timestamp(ALL_CRISES[k]["start"]) for k in keys]
    assert starts == sorted(starts)


@pytest.mark.parametrize("test_key", DEFAULT_OOS_KEYS)
def test_selection_pool_has_no_lookahead(test_key):
    test_start = pd.Timestamp(ALL_CRISES[test_key]["start"])
    pool = selection_pool(test_key)
    assert test_key not in pool, "test crisis must not tune itself"
    for k in pool:
        assert (
            pd.Timestamp(ALL_CRISES[k]["end"]) < test_start
        ), f"{k} ends at/after {test_key} starts — that is look-ahead"


def test_selection_pool_is_expanding_over_time():
    """Later OOS windows see a superset of earlier windows' selection crises."""
    oos = chronological_keys(DEFAULT_OOS_KEYS)
    pools = [set(selection_pool(k)) for k in oos]
    for earlier, later in zip(pools, pools[1:]):
        assert earlier <= later, "expanding-window pools must be monotone"


def test_first_window_seeds_from_prior_real_crises():
    """The earliest reported OOS window (2010 flash) tunes on pre-2010 crises
    such as the GFC, never on later ones."""
    pool = selection_pool("2010_flash")
    assert "2008_gfc" in pool
    assert "2011_euro" not in pool
    assert "2020_covid" not in pool
