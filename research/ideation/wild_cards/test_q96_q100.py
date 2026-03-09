"""
Wild Card Questions Q96-Q100: Empirical and Analytical Investigation

Q96: Near-miss crisis detection (2011 debt ceiling, 2015 China, 2019 repo)
Q97: Periodic patterns in quantum state (seasonal, FOMC, election cycles)
Q98: Crisis severity classification via z-score magnitude
Q99: Flash crash precursor detection (2010-05-06, 2015-08-24)
Q100: Universal critical exponent across crises (power law fit)

All tests use real SPY data via yfinance.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO_ROOT)

from experiments.data_loader import fetch_data, create_feature_matrix_single_asset
from qcml_geometry.observables import BerryPhaseRateDetector, QFIDeterminantDetector


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_spy_series(start='2005-01-01', end='2025-01-01'):
    """Fetch SPY close prices as a clean Series."""
    raw = fetch_data(['SPY'], start, end, use_cache=True)
    prices = raw['close'].xs('SPY', level='symbol').sort_index()
    return prices


def build_z_scores(prices: pd.Series, min_expanding: int = 60):
    """Build BerryPhaseRate and SpectralEntropy (QFI entropy) z-score series."""
    features, feat_dates = create_feature_matrix_single_asset(prices, extra_lags=True)

    # Berry Phase Rate
    berry_det = BerryPhaseRateDetector(
        hilbert_dim=8, n_pca_components=15, rolling_window=20,
        min_expanding=min_expanding, seed=42,
    )
    berry_det.fit(features)
    berry_scores = berry_det.compute_regime_scores(features)

    # Spectral Entropy (QFIDeterminant with entropy mode)
    entropy_det = QFIDeterminantDetector(
        hilbert_dim=8, n_pca_components=15, rolling_window=20,
        min_expanding=min_expanding, seed=42, qfi_mode='entropy',
    )
    entropy_det.fit(features)
    entropy_scores = entropy_det.compute_regime_scores(features)

    berry_series = pd.Series(berry_scores, index=feat_dates, name='berry')
    entropy_series = pd.Series(entropy_scores, index=feat_dates, name='entropy')

    return berry_series, entropy_series, prices.reindex(feat_dates)


# ---------------------------------------------------------------------------
# Q96: Near-miss crisis detection
# ---------------------------------------------------------------------------

NEAR_MISS_WINDOWS = {
    '2011_debt_ceiling': ('2011-07-01', '2011-08-15'),
    '2015_china_fears':  ('2015-07-10', '2015-09-30'),
    '2019_repo_crisis':  ('2019-09-01', '2019-10-31'),
}

CONFIRMED_CRISIS_WINDOWS = {
    '2008_gfc':    ('2008-09-01', '2009-03-31'),
    '2020_covid':  ('2020-02-20', '2020-04-30'),
    '2022_rates':  ('2022-01-01', '2022-10-31'),
    '2023_svb':    ('2023-03-01', '2023-04-30'),
}


def compute_window_mean_z(series: pd.Series, start: str, end: str, label: str = '') -> dict:
    """Compute mean z-score (ignoring NaN) within a date window."""
    mask = (series.index >= start) & (series.index <= end)
    window = series[mask].dropna()
    if len(window) == 0:
        return {'label': label, 'n_days': 0, 'mean_z': np.nan, 'max_z': np.nan, 'pct_above_2': np.nan}
    return {
        'label': label,
        'n_days': len(window),
        'mean_z': float(window.mean()),
        'max_z': float(window.max()),
        'pct_above_2': float((window > 2.0).mean() * 100),
    }


def run_q96(berry: pd.Series, entropy: pd.Series) -> dict:
    """Q96: Do observables fire for near-miss crises?"""
    results = {'question': 'Q96', 'near_misses': {}, 'confirmed_crises': {}, 'baseline': {}}

    # Baseline: full series median z-score
    results['baseline']['berry_median'] = float(berry.dropna().median())
    results['baseline']['entropy_median'] = float(entropy.dropna().median())

    for name, (s, e) in NEAR_MISS_WINDOWS.items():
        results['near_misses'][name] = {
            'berry': compute_window_mean_z(berry, s, e, name),
            'entropy': compute_window_mean_z(entropy, s, e, name),
        }

    for name, (s, e) in CONFIRMED_CRISIS_WINDOWS.items():
        results['confirmed_crises'][name] = {
            'berry': compute_window_mean_z(berry, s, e, name),
            'entropy': compute_window_mean_z(entropy, s, e, name),
        }

    # Summarise: mean z of near-misses vs confirmed crises
    nm_berry = [v['berry']['mean_z'] for v in results['near_misses'].values()]
    cc_berry = [v['berry']['mean_z'] for v in results['confirmed_crises'].values()]
    nm_ent = [v['entropy']['mean_z'] for v in results['near_misses'].values()]
    cc_ent = [v['entropy']['mean_z'] for v in results['confirmed_crises'].values()]

    results['summary'] = {
        'near_miss_mean_berry': float(np.nanmean(nm_berry)),
        'confirmed_mean_berry': float(np.nanmean(cc_berry)),
        'near_miss_mean_entropy': float(np.nanmean(nm_ent)),
        'confirmed_mean_entropy': float(np.nanmean(cc_ent)),
        'ratio_berry': float(np.nanmean(nm_berry) / np.nanmean(cc_berry)) if np.nanmean(cc_berry) > 0 else np.nan,
        'ratio_entropy': float(np.nanmean(nm_ent) / np.nanmean(cc_ent)) if np.nanmean(cc_ent) > 0 else np.nan,
    }

    return results


# ---------------------------------------------------------------------------
# Q97: Periodic patterns (autocorrelation + FOMC)
# ---------------------------------------------------------------------------

# Approximate FOMC meeting dates (8 per year, scheduled)
FOMC_DATES_APPROX = [
    # 2010
    '2010-01-27', '2010-03-16', '2010-04-28', '2010-06-23',
    '2010-08-10', '2010-09-21', '2010-11-03', '2010-12-14',
    # 2015
    '2015-01-28', '2015-03-18', '2015-04-29', '2015-06-17',
    '2015-07-29', '2015-09-17', '2015-10-28', '2015-12-16',
    # 2019
    '2019-01-30', '2019-03-20', '2019-05-01', '2019-06-19',
    '2019-07-31', '2019-09-18', '2019-10-30', '2019-12-11',
    # 2020
    '2020-01-29', '2020-03-03', '2020-03-15', '2020-04-29',
    '2020-06-10', '2020-07-29', '2020-09-16', '2020-11-05', '2020-12-16',
    # 2022
    '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15',
    '2022-07-27', '2022-09-21', '2022-11-02', '2022-12-14',
    # 2023
    '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14',
    '2023-07-26', '2023-09-20', '2023-11-01', '2023-12-13',
]


def autocorr_at_lag(series: pd.Series, lag: int) -> float:
    """Pearson autocorrelation at a specific lag (ignoring NaN pairs)."""
    clean = series.dropna()
    if len(clean) <= lag:
        return np.nan
    return float(clean.iloc[:-lag].corr(clean.iloc[lag:].reset_index(drop=True)
                                         .set_axis(clean.iloc[:-lag].index)))


def fomc_effect(series: pd.Series, fomc_dates: list, window_days: int = 5) -> dict:
    """Compute mean z-score in ±window_days around FOMC meetings vs rest."""
    fomc_ts = pd.to_datetime(fomc_dates)
    is_fomc = pd.Series(False, index=series.index)

    for d in fomc_ts:
        mask = (series.index >= d - pd.Timedelta(days=window_days)) & \
               (series.index <= d + pd.Timedelta(days=window_days))
        is_fomc |= mask

    fomc_window = series[is_fomc].dropna()
    non_fomc = series[~is_fomc].dropna()

    if len(fomc_window) == 0 or len(non_fomc) == 0:
        return {'fomc_mean': np.nan, 'non_fomc_mean': np.nan, 't_stat': np.nan, 'p_value': np.nan}

    t_stat, p_value = stats.ttest_ind(fomc_window, non_fomc)
    return {
        'fomc_mean': float(fomc_window.mean()),
        'non_fomc_mean': float(non_fomc.mean()),
        'fomc_n': len(fomc_window),
        'non_fomc_n': len(non_fomc),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
    }


def run_q97(berry: pd.Series, entropy: pd.Series) -> dict:
    """Q97: Autocorrelation periodicities and FOMC effects."""
    target_lags = {
        '21d_monthly': 21,
        '63d_quarterly': 63,
        '252d_annual': 252,
    }

    results = {'question': 'Q97', 'autocorrelation': {}, 'fomc_effect': {}}

    for period, lag in target_lags.items():
        results['autocorrelation'][period] = {
            'berry': autocorr_at_lag(berry, lag),
            'entropy': autocorr_at_lag(entropy, lag),
        }

    results['fomc_effect']['berry'] = fomc_effect(berry, FOMC_DATES_APPROX)
    results['fomc_effect']['entropy'] = fomc_effect(entropy, FOMC_DATES_APPROX)

    # Additional: look for dominant periodicity via FFT on non-NaN segment
    for name, series in [('berry', berry), ('entropy', entropy)]:
        clean = series.dropna()
        if len(clean) > 252:
            fft_vals = np.abs(np.fft.rfft(clean.values - clean.mean()))
            freqs = np.fft.rfftfreq(len(clean), d=1.0)  # cycles per day
            # Skip DC (freq=0)
            dominant_idx = np.argmax(fft_vals[1:]) + 1
            dominant_period = 1.0 / freqs[dominant_idx] if freqs[dominant_idx] > 0 else np.nan
            results['autocorrelation'][f'dominant_period_days_{name}'] = float(dominant_period)

    return results


# ---------------------------------------------------------------------------
# Q98: Crisis severity correlation
# ---------------------------------------------------------------------------

CRISIS_WINDOWS_WITH_BUFFER = {
    '2008_gfc':   ('2008-09-01', '2009-03-31'),
    '2020_covid': ('2020-02-20', '2020-04-30'),
    '2022_rates': ('2022-01-01', '2022-10-31'),
    '2023_svb':   ('2023-03-01', '2023-04-30'),
}


def max_drawdown(prices: pd.Series, start: str, end: str) -> float:
    """Compute max peak-to-trough drawdown (negative fraction) in window."""
    mask = (prices.index >= start) & (prices.index <= end)
    window = prices[mask].dropna()
    if len(window) < 2:
        return np.nan
    roll_max = window.cummax()
    drawdown = (window - roll_max) / roll_max
    return float(drawdown.min())  # most negative value


def run_q98(berry: pd.Series, entropy: pd.Series, spy_prices: pd.Series) -> dict:
    """Q98: Does z-score magnitude correlate with crisis severity?"""
    records = []

    for crisis, (s, e) in CRISIS_WINDOWS_WITH_BUFFER.items():
        dd = max_drawdown(spy_prices, s, e)
        peak_berry = compute_window_mean_z(berry, s, e)['max_z']
        peak_entropy = compute_window_mean_z(entropy, s, e)['max_z']
        records.append({
            'crisis': crisis,
            'max_drawdown': dd,
            'peak_berry_z': peak_berry,
            'peak_entropy_z': peak_entropy,
        })

    df = pd.DataFrame(records).sort_values('max_drawdown')

    # Severity rank (most severe drawdown = rank 1)
    df['severity_rank'] = df['max_drawdown'].rank(ascending=True)  # most negative = rank 1
    df['berry_rank'] = df['peak_berry_z'].rank(ascending=False)
    df['entropy_rank'] = df['peak_entropy_z'].rank(ascending=False)

    # Spearman correlation: severity (drawdown magnitude, negative) vs peak z-score
    drawdowns = df['max_drawdown'].values
    berry_peaks = df['peak_berry_z'].values
    entropy_peaks = df['peak_entropy_z'].values

    valid_b = ~(np.isnan(drawdowns) | np.isnan(berry_peaks))
    valid_e = ~(np.isnan(drawdowns) | np.isnan(entropy_peaks))

    spearman_berry = stats.spearmanr(drawdowns[valid_b], berry_peaks[valid_b]) if valid_b.sum() >= 3 else (np.nan, np.nan)
    spearman_entropy = stats.spearmanr(drawdowns[valid_e], entropy_peaks[valid_e]) if valid_e.sum() >= 3 else (np.nan, np.nan)

    results = {
        'question': 'Q98',
        'crisis_details': df.to_dict(orient='records'),
        'spearman_berry': {
            'rho': float(spearman_berry[0]),
            'p_value': float(spearman_berry[1]),
            'n': int(valid_b.sum()),
        },
        'spearman_entropy': {
            'rho': float(spearman_entropy[0]),
            'p_value': float(spearman_entropy[1]),
            'n': int(valid_e.sum()),
        },
        'interpretation': (
            'Negative rho means: larger drawdown -> higher peak z-score (correct direction). '
            'Positive rho means observables mismatch severity ordering.'
        ),
    }

    return results


# ---------------------------------------------------------------------------
# Q99: Flash crash precursor detection
# ---------------------------------------------------------------------------

FLASH_CRASH_EVENTS = {
    '2010_flash_crash': ('2010-05-06', '2010-03-01', '2010-05-05'),  # event, window_start, window_end
    '2015_aug_selloff': ('2015-08-24', '2015-06-01', '2015-08-21'),
}


def z_score_trajectory(series: pd.Series, start: str, end: str, event_date: str) -> dict:
    """Compute z-score stats in lead-up window before event_date."""
    mask = (series.index >= start) & (series.index <= end)
    lead_up = series[mask].dropna()

    event_ts = pd.Timestamp(event_date)

    # Day-by-day approach in last 20 trading days before event
    last_20 = lead_up.tail(20)

    return {
        'window_start': start,
        'event_date': event_date,
        'n_days_before': len(lead_up),
        'mean_z_leadup': float(lead_up.mean()) if len(lead_up) else np.nan,
        'max_z_leadup': float(lead_up.max()) if len(lead_up) else np.nan,
        'last_10d_mean': float(lead_up.tail(10).mean()) if len(lead_up) >= 10 else np.nan,
        'last_5d_mean': float(lead_up.tail(5).mean()) if len(lead_up) >= 5 else np.nan,
        'trend_slope': float(np.polyfit(np.arange(len(last_20)), last_20.values, 1)[0])
                       if len(last_20) >= 5 else np.nan,
        'pct_above_2sigma': float((lead_up > 2.0).mean() * 100) if len(lead_up) else np.nan,
    }


def run_q99(berry: pd.Series, entropy: pd.Series) -> dict:
    """Q99: Flash crash precursor detection.

    Analytical note: 20-day rolling window means the first detectable signal
    appears ~20 trading days before the event, not on the day of. This is
    a fundamental temporal resolution constraint.
    """
    results = {'question': 'Q99', 'events': {}, 'analytical_notes': []}

    results['analytical_notes'] = [
        "20-day rolling window = ~4 week minimum detection latency.",
        "Flash crashes (< 1 day) are sub-resolution for 20d-window observables.",
        "Precursors, if present, reflect multi-week buildup of geometric stress.",
        "True intraday flash crash detection requires tick/minute data.",
    ]

    for event_name, (event_date, win_start, win_end) in FLASH_CRASH_EVENTS.items():
        results['events'][event_name] = {
            'berry': z_score_trajectory(berry, win_start, win_end, event_date),
            'entropy': z_score_trajectory(entropy, win_start, win_end, event_date),
        }

        # Also look at the day-of and day-after z-scores (if in the series)
        event_ts = pd.Timestamp(event_date)
        days_around = []
        for offset in [-5, -4, -3, -2, -1, 0, 1, 2]:
            t = event_ts + pd.Timedelta(days=offset)
            # find nearest trading day
            b_val = berry.iloc[berry.index.searchsorted(t)] if t >= berry.index[0] else np.nan
            e_val = entropy.iloc[entropy.index.searchsorted(t)] if t >= entropy.index[0] else np.nan
            days_around.append({'offset': offset, 'date': str(t.date()), 'berry_z': float(b_val) if not np.isnan(float(b_val)) else None, 'entropy_z': float(e_val) if not np.isnan(float(e_val)) else None})
        results['events'][event_name]['days_around_event'] = days_around

    return results


# ---------------------------------------------------------------------------
# Q100: Universal critical exponent
# ---------------------------------------------------------------------------

CRISIS_POWER_LAW_WINDOWS = {
    '2008_gfc':   ('2008-01-01', '2008-09-15'),   # 8-month lead-up
    '2020_covid': ('2019-11-01', '2020-02-28'),   # 4-month lead-up
    '2022_rates': ('2021-06-01', '2022-01-15'),   # 7-month lead-up
    '2023_svb':   ('2022-12-01', '2023-03-10'),   # 3-month lead-up
}

CRISIS_PEAK_DATES = {
    '2008_gfc':   '2008-09-15',
    '2020_covid': '2020-02-20',
    '2022_rates': '2022-01-03',
    '2023_svb':   '2023-03-10',
}


def fit_power_law(series: pd.Series, t_c_str: str, lookback_days: int = 120) -> dict:
    """Fit z(t) ~ (t_c - t)^{-alpha} in log-log space.

    Only uses data where z > 0 and t < t_c.

    Returns alpha, r_squared, and fit quality info.
    """
    t_c = pd.Timestamp(t_c_str)
    t_start = t_c - pd.Timedelta(days=lookback_days)

    mask = (series.index >= t_start) & (series.index < t_c)
    window = series[mask].dropna()
    window = window[window > 0.05]  # avoid log(0)

    if len(window) < 10:
        return {'alpha': np.nan, 'r_squared': np.nan, 'n_points': len(window), 'status': 'insufficient_data'}

    # Time-to-crisis in trading days
    time_to_crisis = np.array([(t_c - t).days for t in window.index], dtype=float)
    valid = time_to_crisis > 0
    time_to_crisis = time_to_crisis[valid]
    z_vals = window.values[valid]

    if len(time_to_crisis) < 5:
        return {'alpha': np.nan, 'r_squared': np.nan, 'n_points': len(time_to_crisis), 'status': 'insufficient_valid'}

    log_tau = np.log(time_to_crisis)
    log_z = np.log(z_vals)

    # Linear fit in log-log: log(z) = -alpha * log(tau) + const
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tau, log_z)
    alpha = -slope  # power law: z ~ tau^{-alpha}

    return {
        'alpha': float(alpha),
        'intercept': float(intercept),
        'r_squared': float(r_value ** 2),
        'p_value': float(p_value),
        'std_err': float(std_err),
        'n_points': int(len(time_to_crisis)),
        'status': 'ok',
    }


def run_q100(berry: pd.Series, entropy: pd.Series) -> dict:
    """Q100: Is there a universal critical exponent across crises?"""
    results = {'question': 'Q100', 'crisis_fits': {}, 'universality_test': {}}

    alphas_berry = []
    alphas_entropy = []

    for crisis, (win_start, win_end) in CRISIS_POWER_LAW_WINDOWS.items():
        t_c = CRISIS_PEAK_DATES[crisis]
        # Use up to 180 calendar days before onset
        berry_fit = fit_power_law(berry, t_c, lookback_days=180)
        entropy_fit = fit_power_law(entropy, t_c, lookback_days=180)

        results['crisis_fits'][crisis] = {
            'berry': berry_fit,
            'entropy': entropy_fit,
            't_c': t_c,
        }

        if berry_fit['status'] == 'ok' and not np.isnan(berry_fit['alpha']):
            alphas_berry.append(berry_fit['alpha'])
        if entropy_fit['status'] == 'ok' and not np.isnan(entropy_fit['alpha']):
            alphas_entropy.append(entropy_fit['alpha'])

    # Universality: mean, std, and coefficient of variation of alpha
    def universality_stats(alphas, label):
        if len(alphas) < 2:
            return {'label': label, 'n': len(alphas), 'mean': np.nan, 'std': np.nan, 'cv': np.nan, 'is_universal': False}
        mu = float(np.mean(alphas))
        sigma = float(np.std(alphas, ddof=1))
        cv = sigma / abs(mu) if abs(mu) > 1e-8 else np.nan
        # "Universal" if CV < 0.3 and all alphas > 0 (divergent approach to crisis)
        is_universal = (cv < 0.3) and all(a > 0 for a in alphas) and len(alphas) >= 3
        return {
            'label': label,
            'n': len(alphas),
            'mean_alpha': mu,
            'std_alpha': sigma,
            'cv': float(cv) if not np.isnan(cv) else np.nan,
            'all_alphas': [float(a) for a in alphas],
            'is_universal': bool(is_universal),
            'interpretation': (
                f"CV={cv:.3f}: {'consistent (universal candidate)' if cv < 0.3 else 'inconsistent (not universal)'}"
                if not np.isnan(cv) else 'insufficient data'
            ),
        }

    results['universality_test']['berry'] = universality_stats(alphas_berry, 'BerryPhaseRate')
    results['universality_test']['entropy'] = universality_stats(alphas_entropy, 'SpectralEntropy')

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_q96(r: dict):
    print_separator("Q96: Near-Miss Crisis Detection")
    print(f"\nBaseline (full series median): Berry={r['baseline']['berry_median']:.3f}, Entropy={r['baseline']['entropy_median']:.3f}")

    print("\n--- Near-Miss Windows ---")
    for nm, vals in r['near_misses'].items():
        b = vals['berry']
        e = vals['entropy']
        print(f"  {nm}: Berry mean_z={b['mean_z']:.3f} max_z={b['max_z']:.3f} | "
              f"Entropy mean_z={e['mean_z']:.3f} max_z={e['max_z']:.3f}")

    print("\n--- Confirmed Crisis Windows ---")
    for cc, vals in r['confirmed_crises'].items():
        b = vals['berry']
        e = vals['entropy']
        print(f"  {cc}: Berry mean_z={b['mean_z']:.3f} max_z={b['max_z']:.3f} | "
              f"Entropy mean_z={e['mean_z']:.3f} max_z={e['max_z']:.3f}")

    s = r['summary']
    print(f"\n--- Summary ---")
    print(f"  Near-miss avg Berry: {s['near_miss_mean_berry']:.3f} | Confirmed avg Berry: {s['confirmed_mean_berry']:.3f} | Ratio: {s['ratio_berry']:.2f}x")
    print(f"  Near-miss avg Entropy: {s['near_miss_mean_entropy']:.3f} | Confirmed avg Entropy: {s['confirmed_mean_entropy']:.3f} | Ratio: {s['ratio_entropy']:.2f}x")
    if s['ratio_berry'] > 0.5:
        print("  FINDING: Observables fire at >50% intensity for near-misses vs confirmed crises.")
    else:
        print("  FINDING: Observables do NOT strongly differentiate near-misses from non-crisis periods.")


def print_q97(r: dict):
    print_separator("Q97: Periodic Patterns in Quantum State")

    print("\n--- Autocorrelation at Target Lags ---")
    for period, vals in r['autocorrelation'].items():
        if 'dominant' in period:
            print(f"  {period}: {vals:.1f} days")
        else:
            b = vals.get('berry', np.nan)
            e = vals.get('entropy', np.nan)
            print(f"  {period}: Berry r={b:.4f}, Entropy r={e:.4f}")

    print("\n--- FOMC Effect ---")
    for obs, vals in r['fomc_effect'].items():
        sig = '*' if vals.get('significant') else ''
        print(f"  {obs}: FOMC-window mean={vals['fomc_mean']:.3f}, non-FOMC mean={vals['non_fomc_mean']:.3f}, "
              f"t={vals['t_stat']:.3f}, p={vals['p_value']:.4f}{sig}")


def print_q98(r: dict):
    print_separator("Q98: Crisis Severity Classification")

    print("\n--- Crisis Ranking by Severity ---")
    for record in sorted(r['crisis_details'], key=lambda x: x['max_drawdown']):
        print(f"  {record['crisis']}: drawdown={record['max_drawdown']*100:.1f}%, "
              f"peak_berry_z={record['peak_berry_z']:.3f}, peak_entropy_z={record['peak_entropy_z']:.3f}")

    sb = r['spearman_berry']
    se = r['spearman_entropy']
    print(f"\n--- Spearman Correlation (drawdown vs peak z-score) ---")
    print(f"  Berry:   rho={sb['rho']:.3f}, p={sb['p_value']:.4f}, n={sb['n']}")
    print(f"  Entropy: rho={se['rho']:.3f}, p={se['p_value']:.4f}, n={se['n']}")
    print(f"\n  {r['interpretation']}")

    if sb['rho'] < -0.5:
        print("  FINDING: Berry z-score magnitude tracks crisis severity (larger crash -> higher peak z).")
    elif sb['rho'] > 0.5:
        print("  FINDING: Counter-intuitive — higher z for smaller crashes. May reflect duration vs depth.")
    else:
        print("  FINDING: Weak or no correlation between z-score peak and drawdown magnitude.")


def print_q99(r: dict):
    print_separator("Q99: Flash Crash Precursor Detection")

    for note in r['analytical_notes']:
        print(f"  NOTE: {note}")

    for event_name, vals in r['events'].items():
        print(f"\n--- {event_name} ---")
        for obs in ['berry', 'entropy']:
            v = vals[obs]
            print(f"  {obs}: lead_up_mean={v['mean_z_leadup']:.3f}, max={v['max_z_leadup']:.3f}, "
                  f"last_5d_mean={v['last_5d_mean']:.3f}, trend_slope={v['trend_slope']:.5f}")

        print("  Days around event:")
        for day in vals.get('days_around_event', []):
            b = f"{day['berry_z']:.3f}" if day['berry_z'] is not None else "NaN"
            e = f"{day['entropy_z']:.3f}" if day['entropy_z'] is not None else "NaN"
            print(f"    offset={day['offset']:+2d} ({day['date']}): Berry={b}, Entropy={e}")


def print_q100(r: dict):
    print_separator("Q100: Universal Critical Exponent")

    print("\n--- Per-Crisis Power Law Fits ---")
    for crisis, vals in r['crisis_fits'].items():
        bf = vals['berry']
        ef = vals['entropy']
        b_str = f"alpha={bf['alpha']:.3f} (R^2={bf['r_squared']:.3f})" if bf['status'] == 'ok' else f"FAILED: {bf['status']}"
        e_str = f"alpha={ef['alpha']:.3f} (R^2={ef['r_squared']:.3f})" if ef['status'] == 'ok' else f"FAILED: {ef['status']}"
        print(f"  {crisis}: Berry {b_str} | Entropy {e_str}")

    print("\n--- Universality Assessment ---")
    for obs, u in r['universality_test'].items():
        print(f"  {obs}: mean_alpha={u['mean_alpha']:.3f}, std={u['std_alpha']:.3f}, "
              f"CV={u.get('cv', np.nan):.3f}, n={u['n']}")
        print(f"    -> {u['interpretation']}")
        if u.get('all_alphas'):
            print(f"    Individual alphas: {[f'{a:.3f}' for a in u['all_alphas']]}")


def main():
    print("Loading SPY data and computing geometric observables...")
    spy = load_spy_series(start='2005-01-01', end='2025-01-01')
    print(f"SPY data: {spy.index[0].date()} to {spy.index[-1].date()}, N={len(spy)}")

    print("Fitting BerryPhaseRate and SpectralEntropy detectors...")
    berry, entropy, spy_aligned = build_z_scores(spy)
    print(f"Z-scores computed. Non-NaN: Berry={berry.notna().sum()}, Entropy={entropy.notna().sum()}")

    # Run all 5 questions
    r96 = run_q96(berry, entropy)
    print_q96(r96)

    r97 = run_q97(berry, entropy)
    print_q97(r97)

    r98 = run_q98(berry, entropy, spy)
    print_q98(r98)

    r99 = run_q99(berry, entropy)
    print_q99(r99)

    r100 = run_q100(berry, entropy)
    print_q100(r100)

    print("\n" + "="*70)
    print("  ALL QUESTIONS COMPLETE")
    print("="*70)

    return {
        'Q96': r96,
        'Q97': r97,
        'Q98': r98,
        'Q99': r99,
        'Q100': r100,
    }


if __name__ == '__main__':
    results = main()
