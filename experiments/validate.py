"""
Automated post-run validation for experiment results.

Catches bugs before they propagate to the paper:
- Bounds check: d-values not NaN, not negative, not suspiciously large (>5.0)
- Completeness check: all expected (method, crisis) cells have results
- Statistical significance: Friedman test warning if p > 0.05
- Causal ordering: verify preprocessing cutoff dates precede crisis starts

Runs automatically after ExperimentRunner.run_comparison().

Usage:
    from experiments.validate import validate_results
    issues = validate_results(results, cfg)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_results(output: dict, cfg: Optional[dict] = None) -> list[dict]:
    """Run all validation checks on experiment results.

    Args:
        output: Results dict from ExperimentRunner.run_comparison().
        cfg: Config dict (for completeness checks). Optional.

    Returns:
        List of issue dicts with keys: severity (CRITICAL/WARNING/INFO),
        check, message, details.
    """
    issues = []

    results = output.get('results', {})
    summary = output.get('summary', {})
    config = output.get('config', {})

    # 1. Bounds check
    issues.extend(_check_bounds(results))

    # 2. Completeness check
    if cfg:
        issues.extend(_check_completeness(results, output, cfg))

    # 3. Statistical significance
    issues.extend(_check_significance(summary))

    # 4. NaN/None density
    issues.extend(_check_nan_density(results))

    # 5. Suspiciously identical values
    issues.extend(_check_identical_values(results))

    # 6. Timing anomalies
    issues.extend(_check_timing(results))

    # Log summary
    n_critical = sum(1 for i in issues if i['severity'] == 'CRITICAL')
    n_warning = sum(1 for i in issues if i['severity'] == 'WARNING')
    n_info = sum(1 for i in issues if i['severity'] == 'INFO')

    if n_critical > 0:
        logger.error(f"Validation: {n_critical} CRITICAL, {n_warning} WARNING, {n_info} INFO")
    elif n_warning > 0:
        logger.warning(f"Validation: {n_warning} WARNING, {n_info} INFO")
    else:
        logger.info(f"Validation passed: {n_info} informational notes")

    return issues


def _check_bounds(results: dict) -> list[dict]:
    """Check d-values are within reasonable bounds."""
    issues = []
    for method, crises in results.items():
        for crisis, cell in crises.items():
            if not isinstance(cell, dict):
                continue
            d = cell.get('d')
            if d is None:
                continue

            if d < 0:
                issues.append({
                    'severity': 'CRITICAL',
                    'check': 'bounds',
                    'message': f'{method} x {crisis}: negative d-value ({d:.3f})',
                    'details': cell,
                })
            elif d > 5.0:
                issues.append({
                    'severity': 'WARNING',
                    'check': 'bounds',
                    'message': f'{method} x {crisis}: suspiciously large d ({d:.3f})',
                    'details': cell,
                })

            ci_lo = cell.get('ci_lo')
            ci_hi = cell.get('ci_hi')
            if ci_lo is not None and ci_hi is not None:
                if ci_lo > ci_hi:
                    issues.append({
                        'severity': 'CRITICAL',
                        'check': 'bounds',
                        'message': f'{method} x {crisis}: CI inverted ({ci_lo:.3f} > {ci_hi:.3f})',
                        'details': cell,
                    })
    return issues


def _check_completeness(results: dict, output: dict, cfg: dict) -> list[dict]:
    """Check all expected (method, crisis) cells have results."""
    issues = []

    exp_name = output.get('experiment', 'default')
    exp_cfg = cfg.get('experiments', {}).get(exp_name, {})
    expected_subset = exp_cfg.get('crisis_subset', 'post_2005')
    expected_crises = cfg.get('crisis_subsets', {}).get(expected_subset, [])

    if not expected_crises:
        return issues

    for method, crises in results.items():
        missing = [c for c in expected_crises if c not in crises]
        if missing:
            issues.append({
                'severity': 'WARNING',
                'check': 'completeness',
                'message': f'{method}: missing {len(missing)} crises: {missing}',
                'details': {'expected': expected_crises, 'missing': missing},
            })

        # Check for skipped cells
        skipped = [c for c, cell in crises.items()
                   if isinstance(cell, dict) and cell.get('skipped')]
        if skipped:
            issues.append({
                'severity': 'INFO',
                'check': 'completeness',
                'message': f'{method}: {len(skipped)} crises skipped ({skipped})',
                'details': {c: crises[c].get('reason') for c in skipped},
            })

    return issues


def _check_significance(summary: dict) -> list[dict]:
    """Check Friedman test significance."""
    issues = []
    p = summary.get('friedman_p')
    if p is not None and p > 0.05:
        issues.append({
            'severity': 'WARNING',
            'check': 'significance',
            'message': f'Friedman test not significant (p={p:.4f} > 0.05)',
            'details': {
                'chi_sq': summary.get('friedman_chi_sq'),
                'p_value': p,
            },
        })
    elif p is None:
        issues.append({
            'severity': 'INFO',
            'check': 'significance',
            'message': 'Friedman test not computed (insufficient data)',
            'details': {},
        })
    return issues


def _check_nan_density(results: dict) -> list[dict]:
    """Warn if too many cells have NaN/None d-values."""
    issues = []
    for method, crises in results.items():
        total = len(crises)
        n_none = sum(1 for cell in crises.values()
                     if isinstance(cell, dict) and cell.get('d') is None)
        if total > 0 and n_none / total > 0.25:
            issues.append({
                'severity': 'WARNING',
                'check': 'nan_density',
                'message': f'{method}: {n_none}/{total} cells have null d-values ({n_none/total:.0%})',
                'details': {'method': method, 'null_count': n_none, 'total': total},
            })
    return issues


def _check_identical_values(results: dict) -> list[dict]:
    """Detect suspiciously identical d-values across crises."""
    issues = []
    for method, crises in results.items():
        d_values = [cell['d'] for cell in crises.values()
                    if isinstance(cell, dict) and cell.get('d') is not None]
        if len(d_values) >= 3:
            unique = len(set(round(d, 4) for d in d_values))
            if unique == 1:
                issues.append({
                    'severity': 'CRITICAL',
                    'check': 'identical_values',
                    'message': f'{method}: all {len(d_values)} d-values are identical ({d_values[0]:.4f})',
                    'details': {'method': method, 'd_values': d_values},
                })
    return issues


def _check_timing(results: dict) -> list[dict]:
    """Flag extreme timing outliers that might indicate computation issues."""
    issues = []
    all_timings = []
    for method, crises in results.items():
        for crisis, cell in crises.items():
            if isinstance(cell, dict) and cell.get('timing_s') is not None:
                all_timings.append((method, crisis, cell['timing_s']))

    if len(all_timings) < 3:
        return issues

    times = np.array([t[2] for t in all_timings])
    median_t = np.median(times)
    for method, crisis, t in all_timings:
        if median_t > 0 and t > median_t * 20:
            issues.append({
                'severity': 'INFO',
                'check': 'timing',
                'message': f'{method} x {crisis}: {t:.1f}s (20x median of {median_t:.1f}s)',
                'details': {'method': method, 'crisis': crisis, 'timing_s': t},
            })

    return issues


def validate_causal_ordering(dates: pd.DatetimeIndex, crisis_def: dict,
                              fit_end_idx: int, window_size: int = 10) -> list[dict]:
    """Verify scaler/PCA cutoff dates precede crisis starts.

    This catches the phase-19 data leakage bug.

    Args:
        dates: DatetimeIndex of the feature matrix.
        crisis_def: Dict with 'start' key.
        fit_end_idx: Index of the last data point used for fitting.
        window_size: Crisis window extension in trading days.

    Returns:
        List of issues (empty if ordering is correct).
    """
    issues = []
    crisis_start = pd.Timestamp(crisis_def['start'])
    cutoff_date = crisis_start - pd.Timedelta(days=window_size)

    if fit_end_idx >= len(dates):
        issues.append({
            'severity': 'CRITICAL',
            'check': 'causal_ordering',
            'message': f'fit_end_idx ({fit_end_idx}) >= len(dates) ({len(dates)})',
            'details': {'crisis_start': str(crisis_start), 'fit_end_idx': fit_end_idx},
        })
        return issues

    actual_cutoff = dates[fit_end_idx]
    if actual_cutoff >= crisis_start:
        issues.append({
            'severity': 'CRITICAL',
            'check': 'causal_ordering',
            'message': (f'Data leakage: fit cutoff ({actual_cutoff.date()}) >= '
                       f'crisis start ({crisis_start.date()})'),
            'details': {
                'crisis_start': str(crisis_start),
                'actual_cutoff': str(actual_cutoff),
                'fit_end_idx': fit_end_idx,
            },
        })

    return issues


def main():
    """CLI: validate a saved JSON results file."""
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Validate experiment results')
    parser.add_argument('json_path', help='Path to comparison JSON')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml for completeness checks')
    args = parser.parse_args()

    with open(args.json_path) as f:
        output = json.load(f)

    cfg = None
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    issues = validate_results(output, cfg)

    if not issues:
        print("All validation checks passed.")
    else:
        for issue in issues:
            marker = {'CRITICAL': 'X', 'WARNING': '!', 'INFO': 'i'}[issue['severity']]
            print(f"  [{marker}] {issue['severity']:8s} {issue['check']:20s} {issue['message']}")

        n_critical = sum(1 for i in issues if i['severity'] == 'CRITICAL')
        if n_critical:
            print(f"\n{n_critical} CRITICAL issues found. Do not integrate into paper.")
            return 1
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
