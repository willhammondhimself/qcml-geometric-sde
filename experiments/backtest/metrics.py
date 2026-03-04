"""
Backtest performance metrics with statistical significance tests.
"""

import numpy as np
from scipy import stats


def compute_backtest_metrics(returns, rf_rate=0.0):
    """Compute standard backtest performance metrics.

    Args:
        returns: Daily return series (T,).
        rf_rate: Annualized risk-free rate (default: 0).

    Returns:
        Dict of metrics.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return _empty_metrics()

    T = len(returns)
    daily_rf = rf_rate / 252

    # Basic stats
    total_return = float(np.prod(1 + returns) - 1)
    annual_return = float((1 + total_return) ** (252 / T) - 1)
    annual_vol = float(np.std(returns, ddof=1) * np.sqrt(252))

    # Sharpe
    excess = returns - daily_rf
    sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252)) if np.std(excess) > 1e-12 else 0.0

    # Sortino
    downside = excess[excess < 0]
    downside_vol = float(np.std(downside, ddof=1) * np.sqrt(252)) if len(downside) > 1 else 1e-6
    sortino = float(np.mean(excess) * 252 / downside_vol) if downside_vol > 1e-12 else 0.0

    # Drawdown
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    drawdowns = (peak - cum) / peak
    max_dd = float(np.max(drawdowns))
    calmar = float(annual_return / max_dd) if max_dd > 0.01 else 0.0

    # Drawdown duration
    dd_durations = _compute_drawdown_durations(drawdowns)
    max_dd_duration = int(np.max(dd_durations)) if len(dd_durations) > 0 else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'max_dd_duration_days': max_dd_duration,
        'n_days': T,
        'daily_mean': float(np.mean(returns)),
        'daily_std': float(np.std(returns, ddof=1)),
        'skewness': float(stats.skew(returns)),
        'kurtosis': float(stats.kurtosis(returns)),
        'pct_positive_days': float(np.mean(returns > 0)),
    }


def _compute_drawdown_durations(drawdowns):
    """Compute duration of each drawdown episode."""
    durations = []
    current_duration = 0

    for dd in drawdowns:
        if dd > 0.001:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        durations.append(current_duration)

    return durations


def _empty_metrics():
    return {k: 0.0 for k in [
        'total_return', 'annual_return', 'annual_vol', 'sharpe', 'sortino',
        'max_drawdown', 'calmar', 'max_dd_duration_days', 'n_days',
        'daily_mean', 'daily_std', 'skewness', 'kurtosis', 'pct_positive_days',
    ]}


def bootstrap_sharpe_ci(returns, n_bootstrap=10000, seed=42):
    """Bootstrap confidence interval for annualized Sharpe ratio.

    Args:
        returns: Daily returns (T,).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        (sharpe, ci_lo, ci_hi): Point estimate and 95% CI.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    T = len(returns)

    if T < 30:
        return 0.0, np.nan, np.nan

    sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252))

    rng = np.random.default_rng(seed)
    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(T, size=T, replace=True)
        sample = returns[idx]
        s = np.std(sample, ddof=1)
        boot_sharpes[i] = np.mean(sample) / s * np.sqrt(252) if s > 1e-12 else 0.0

    ci_lo, ci_hi = np.percentile(boot_sharpes, [2.5, 97.5])
    return sharpe, float(ci_lo), float(ci_hi)


def ledoit_wolf_sharpe_test(returns_1, returns_2):
    """Ledoit-Wolf (2008) test for equality of Sharpe ratios.

    Uses HAC standard errors to handle autocorrelation.

    Args:
        returns_1: Daily returns of strategy 1 (T,).
        returns_2: Daily returns of strategy 2 (T,).

    Returns:
        (delta_sharpe, p_value): Difference in Sharpe and p-value.
    """
    r1 = np.asarray(returns_1, dtype=float)
    r2 = np.asarray(returns_2, dtype=float)

    # Align lengths
    T = min(len(r1), len(r2))
    r1, r2 = r1[:T], r2[:T]

    # Remove NaN
    valid = ~np.isnan(r1) & ~np.isnan(r2)
    r1, r2 = r1[valid], r2[valid]
    T = len(r1)

    if T < 30:
        return 0.0, 1.0

    mu1, mu2 = np.mean(r1), np.mean(r2)
    sig1, sig2 = np.std(r1, ddof=1), np.std(r2, ddof=1)

    if sig1 < 1e-12 or sig2 < 1e-12:
        return 0.0, 1.0

    sr1 = mu1 / sig1 * np.sqrt(252)
    sr2 = mu2 / sig2 * np.sqrt(252)
    delta_sr = sr1 - sr2

    # HAC variance of the difference (Newey-West with Bartlett kernel)
    # Following Ledoit-Wolf (2008) Appendix
    d = r1 - r2
    gamma_hat = _newey_west_var(d, T)

    se = np.sqrt(gamma_hat / T) * np.sqrt(252)

    if se < 1e-12:
        return float(delta_sr), 1.0

    z_stat = delta_sr / se
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    return float(delta_sr), p_value


def _newey_west_var(x, T, max_lag=None):
    """Newey-West HAC variance estimator."""
    if max_lag is None:
        max_lag = int(np.floor(4 * (T / 100) ** (2/9)))

    x_demeaned = x - np.mean(x)
    gamma_0 = np.mean(x_demeaned ** 2)

    gamma_sum = 0.0
    for j in range(1, max_lag + 1):
        weight = 1 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = np.mean(x_demeaned[j:] * x_demeaned[:-j])
        gamma_sum += 2 * weight * gamma_j

    return gamma_0 + gamma_sum


def crisis_period_returns(strategy_returns, dates, crisis_dict):
    """Compute cumulative return during each crisis period.

    Args:
        strategy_returns: Daily returns (T,).
        dates: DatetimeIndex.
        crisis_dict: Dict of crisis definitions.

    Returns:
        Dict mapping crisis_key to cumulative return.
    """
    import pandas as pd

    results = {}
    for ck, ci in crisis_dict.items():
        cs = pd.Timestamp(ci['start'])
        ce = pd.Timestamp(ci['end'])
        mask = (dates >= cs) & (dates <= ce)
        crisis_rets = strategy_returns[mask]

        if len(crisis_rets) > 0:
            cum_ret = float(np.prod(1 + crisis_rets) - 1)
            results[ck] = cum_ret
        else:
            results[ck] = np.nan

    return results
