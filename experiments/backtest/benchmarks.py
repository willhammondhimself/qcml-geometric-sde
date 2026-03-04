"""
Benchmark strategies for comparison.

The key benchmark is ConstantVolSPY — vol-targeted SPY WITHOUT regime signal.
This isolates the alpha from the regime signal itself.
"""

import numpy as np
from .portfolio import vol_target_weight


def buy_and_hold_spy(spy_returns):
    """Buy and hold SPY (weight = 1.0 always).

    Returns:
        weights: Array (T,) of 1.0.
    """
    return np.ones(len(spy_returns))


def buy_and_hold_equal_weight(asset_returns):
    """Equal-weight buy-and-hold across assets.

    Args:
        asset_returns: (T, n_assets).

    Returns:
        weights: (T, n_assets).
    """
    T, n = asset_returns.shape
    return np.full((T, n), 1.0 / n)


def sixty_forty(spy_returns):
    """60/40 SPY/cash allocation.

    Returns:
        weights: Array (T,) of 0.6.
    """
    return np.full(len(spy_returns), 0.6)


def constant_vol_spy(spy_returns, target_vol=0.10):
    """Vol-targeted SPY WITHOUT any regime signal.

    THE KEY BENCHMARK: same vol targeting as geometric strategies,
    but no regime adjustment. If geometric strategies beat this,
    the regime signal itself adds alpha.

    Args:
        spy_returns: Daily SPY returns (T,).
        target_vol: Annualized target vol.

    Returns:
        weights: Array (T,) of vol-targeted weights, shifted by 1 day.
    """
    T = len(spy_returns)
    base_w = vol_target_weight(spy_returns, target_vol=target_vol)

    # Shift by 1 day (causal)
    weights = np.zeros(T)
    weights[1:] = base_w[:-1]

    return weights


def constant_vol_multi_asset(asset_returns, target_vol=0.10):
    """Vol-targeted equal-weight without regime signal.

    Args:
        asset_returns: (T, n_assets).
        target_vol: Annualized target vol.

    Returns:
        weights: (T, n_assets).
    """
    T, n = asset_returns.shape
    ew_returns = np.mean(asset_returns, axis=1)
    base_w = vol_target_weight(ew_returns, target_vol=target_vol)

    # Shift by 1 day, distribute equally
    weights = np.zeros((T, n))
    for i in range(n):
        weights[1:, i] = base_w[:-1] / n

    return weights
