"""
Incremental experiment run engine with cell-level caching.

Decomposes comparisons into (method, crisis) cells, each independently
cacheable. Only recomputes cells whose parameters or source code changed.

Cache key = SHA256(method_params + crisis_def + library_hash)

Usage:
    from experiments.runner import ExperimentRunner
    from experiments.config import load_config

    cfg = load_config()
    runner = ExperimentRunner(cfg)
    results = runner.run_comparison('default')  # uses named experiment
    results = runner.run_comparison('quick')
"""

import hashlib
import json
import logging
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _get_library_hash() -> str:
    """Get a hash of the core library source code for cache invalidation.

    Uses git hash of qcml_geometry/ if in a git repo, otherwise hashes
    the source files directly.
    """
    qcml_dir = ROOT / 'qcml_geometry'
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%H', '--', 'qcml_geometry/'],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: hash source files directly
    hasher = hashlib.sha256()
    for py_file in sorted(qcml_dir.glob('**/*.py')):
        hasher.update(py_file.read_bytes())
    return hasher.hexdigest()[:12]


def _cell_cache_key(method_key: str, method_params: dict, crisis_key: str,
                    crisis_def: dict, lib_hash: str,
                    data_source: str = '', data_hash: str = '') -> str:
    """Compute SHA256 cache key for a (method, crisis) cell.

    Includes data_source and data_hash so that switching data sources
    (or getting different data from the same source) invalidates cache.
    """
    payload = json.dumps({
        'method': method_key,
        'params': method_params,
        'crisis': crisis_key,
        'crisis_def': crisis_def,
        'lib_hash': lib_hash,
        'data_source': data_source,
        'data_hash': data_hash,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ExperimentRunner:
    """Incremental experiment runner with (method, crisis) cell caching.

    Args:
        cfg: Loaded config dict from config.yaml.
        cache_dir: Directory for pickle cache files.
        force: If True, ignore all caches and recompute everything.
    """

    def __init__(self, cfg: dict, cache_dir: Optional[Path] = None, force: bool = False):
        self.cfg = cfg
        self.force = force
        self.cache_dir = cache_dir or (ROOT / cfg['data']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lib_hash = _get_library_hash()
        self._data_cache = {}

    def _load_data(self):
        """Load and cache the feature matrix and enriched features."""
        if 'X_enriched' in self._data_cache:
            return self._data_cache

        from experiments.data_loader import fetch_data, create_feature_matrix
        from qcml_geometry.observables import BaseRegimeDetector

        data_cfg = self.cfg['data']
        symbols = data_cfg['symbols']
        start = data_cfg['start_date']
        end = data_cfg['end_date']
        source = data_cfg.get('source', 'yfinance')

        logger.info(f"Fetching data: {symbols} {start} to {end}")
        raw = fetch_data(symbols, start, end, source=source)
        prices_df = raw['close'].unstack('symbol').dropna()
        X, dates = create_feature_matrix(prices_df)

        # Compute data hash for cache invalidation and provenance
        close_hash = hashlib.sha256(
            prices_df.values.tobytes()
        ).hexdigest()[:12]

        self._data_source = source
        self._data_hash = close_hash
        self._date_range = (str(dates[0].date()), str(dates[-1].date()))
        self._n_trading_days = len(dates)

        logger.info(f"Data source: {source}")
        logger.info(f"Data hash: {close_hash}")
        logger.info(f"Date range: {self._date_range[0]} to {self._date_range[1]}")
        logger.info(f"Trading days: {self._n_trading_days:,}")

        lookback = data_cfg.get('enrichment_lookback', 20)
        X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=lookback)
        dates_enriched = dates[lookback - 1:]

        self._data_cache = {
            'X': X,
            'dates': dates,
            'X_enriched': X_enriched,
            'dates_enriched': dates_enriched,
            'prices_df': prices_df,
        }
        logger.info(f"Feature matrix: {X.shape}, enriched: {X_enriched.shape}")
        return self._data_cache

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f'{cache_key}.pkl'

    def _load_cell(self, cache_key: str) -> Optional[dict]:
        """Load a cached cell result, or None if not cached."""
        if self.force:
            return None
        path = self._get_cache_path(cache_key)
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed for {cache_key}: {e}")
        return None

    def _save_cell(self, cache_key: str, result: dict):
        """Save a cell result to cache."""
        path = self._get_cache_path(cache_key)
        with open(path, 'wb') as f:
            pickle.dump(result, f)

    def _run_detector_cell(self, method_key: str, method_cfg: dict,
                           crisis_key: str, crisis_def: dict,
                           data: dict) -> dict:
        """Run a single (detector, crisis) cell.

        Returns:
            {'d': float, 'ci_lo': float, 'ci_hi': float, 'timing_s': float}
        """
        from experiments.config import resolve_class
        from experiments.evaluation import compute_cohens_d_with_ci

        eval_cfg = self.cfg['evaluation']
        window_size = eval_cfg['window_size']
        n_bootstrap = eval_cfg['n_bootstrap']

        X_enriched = data['X_enriched']
        dates_enriched = data['dates_enriched']

        crisis_start = pd.Timestamp(crisis_def['start'])
        cutoff_date = crisis_start - pd.Timedelta(days=window_size)
        fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))

        if fit_end_idx < eval_cfg.get('min_pre_crisis_rows', 100):
            return {'d': None, 'ci_lo': None, 'ci_hi': None,
                    'timing_s': 0, 'skipped': True,
                    'reason': f'insufficient pre-crisis data ({fit_end_idx} rows)'}

        crisis_end = pd.Timestamp(crisis_def['end'])
        cs = crisis_start - pd.Timedelta(days=window_size)
        ce = crisis_end + pd.Timedelta(days=window_size)
        crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        normal_mask = ~crisis_mask

        cls = resolve_class(method_cfg['class'])
        params = {**method_cfg['params'], 'causal_fit_length': fit_end_idx}

        t0 = time.time()
        det = cls(**params)
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)
        timing = time.time() - t0

        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap,
        )

        return {
            'd': float(d) if not np.isnan(d) else None,
            'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
            'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            'timing_s': round(timing, 2),
        }

    def _run_baseline_cell(self, method_key: str, method_cfg: dict,
                           crisis_key: str, crisis_def: dict,
                           data: dict) -> dict:
        """Run a single (baseline, crisis) cell."""
        from experiments.config import resolve_class
        from experiments.evaluation import compute_cohens_d_with_ci

        eval_cfg = self.cfg['evaluation']
        window_size = eval_cfg['window_size']
        n_bootstrap = eval_cfg['n_bootstrap']

        X_enriched = data['X_enriched']
        dates_enriched = data['dates_enriched']

        crisis_start = pd.Timestamp(crisis_def['start'])
        cutoff_date = crisis_start - pd.Timedelta(days=window_size)
        fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))

        if fit_end_idx < eval_cfg.get('min_pre_crisis_rows', 100):
            return {'d': None, 'ci_lo': None, 'ci_hi': None,
                    'timing_s': 0, 'skipped': True,
                    'reason': f'insufficient pre-crisis data ({fit_end_idx} rows)'}

        crisis_end = pd.Timestamp(crisis_def['end'])
        cs = crisis_start - pd.Timedelta(days=window_size)
        ce = crisis_end + pd.Timedelta(days=window_size)
        crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
        normal_mask = ~crisis_mask

        cls = resolve_class(method_cfg['class'])
        params = dict(method_cfg['params'])

        t0 = time.time()
        det = cls(**params)
        det.fit(X_enriched[:fit_end_idx])
        scores = det.compute_regime_scores(X_enriched)
        timing = time.time() - t0

        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap,
        )

        return {
            'd': float(d) if not np.isnan(d) else None,
            'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
            'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            'timing_s': round(timing, 2),
        }

    def _run_rf_loco(self, method_cfg: dict, crises: dict,
                     data: dict) -> dict[str, dict]:
        """Run Random Forest with leave-one-crisis-out evaluation.

        Returns:
            Dict mapping crisis_key -> cell result dict.
        """
        from experiments.config import resolve_class
        from experiments.evaluation import compute_cohens_d_with_ci

        eval_cfg = self.cfg['evaluation']
        window_size = eval_cfg['window_size']
        n_bootstrap = eval_cfg['n_bootstrap']

        X = data['X']
        dates = data['dates']
        X_enriched = data['X_enriched']
        dates_enriched = data['dates_enriched']

        cls = resolve_class(method_cfg['class'])
        params = dict(method_cfg['params'])

        rf_results = {}
        for held_out_key, crisis_def in crises.items():
            crisis_start = pd.Timestamp(crisis_def['start'])
            crisis_end = pd.Timestamp(crisis_def['end'])

            # Build labels excluding held-out crisis
            y = np.zeros(len(X))
            for train_ck, train_ci in crises.items():
                if train_ck == held_out_key:
                    continue
                tc_start = pd.Timestamp(train_ci['start'])
                tc_end = pd.Timestamp(train_ci['end'])
                mask = (dates >= tc_start) & (dates <= tc_end)
                y[mask] = 1.0

            cutoff_date = crisis_start - pd.Timedelta(days=window_size)
            fit_end_raw = int(np.searchsorted(dates, cutoff_date))
            if fit_end_raw < eval_cfg.get('min_pre_crisis_rows', 100):
                rf_results[held_out_key] = {
                    'd': None, 'ci_lo': None, 'ci_hi': None,
                    'timing_s': 0, 'skipped': True,
                    'reason': 'insufficient pre-crisis data',
                }
                continue

            t0 = time.time()
            rf = cls(**params)
            rf.fit_with_labels(X[:fit_end_raw], y[:fit_end_raw])
            scores = rf.compute_regime_scores(X)
            timing = time.time() - t0

            # Trim to enriched length
            lookback = self.cfg['data'].get('enrichment_lookback', 20)
            rf_scores = scores[lookback - 1:] if len(scores) > len(dates_enriched) else scores
            if len(rf_scores) > len(dates_enriched):
                rf_scores = rf_scores[:len(dates_enriched)]

            cs = crisis_start - pd.Timedelta(days=window_size)
            ce = crisis_end + pd.Timedelta(days=window_size)
            crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
            normal_mask = ~crisis_mask

            if len(rf_scores) == len(dates_enriched):
                d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                    rf_scores[crisis_mask], rf_scores[normal_mask],
                    n_bootstrap=n_bootstrap,
                )
            else:
                d, ci_lo, ci_hi = np.nan, np.nan, np.nan

            rf_results[held_out_key] = {
                'd': float(d) if not np.isnan(d) else None,
                'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
                'timing_s': round(timing, 2),
            }

        return rf_results

    def run_comparison(self, experiment_name: str = 'default') -> dict:
        """Run a named experiment with incremental caching.

        Args:
            experiment_name: Key from experiments section of config.yaml.

        Returns:
            Full results dict (same format as regime_comparison.py output).
        """
        from experiments.config import resolve_experiment, get_method_config
        from experiments.evaluation import friedman_test

        method_keys, crises = resolve_experiment(self.cfg, experiment_name)

        logger.info("=" * 70)
        logger.info(f"Experiment: {experiment_name} ({len(method_keys)} methods x {len(crises)} crises)")
        logger.info("=" * 70)

        data = self._load_data()

        results = {}
        total_cells = 0
        cache_hits = 0
        t_start = time.time()

        # Separate RF (LOCO) from other methods
        rf_key = None
        other_keys = []
        for mk in method_keys:
            mc = get_method_config(self.cfg, mk)
            if mc.get('evaluation') == 'leave_one_crisis_out':
                rf_key = mk
            else:
                other_keys.append(mk)

        # Run non-RF methods
        for mk in other_keys:
            mc = get_method_config(self.cfg, mk)
            display_name = mc['display_name']
            if mk in self.cfg['detectors']:
                method_section = self.cfg['detectors'][mk]
                is_detector = True
            elif mk in self.cfg.get('info_geometry', {}):
                method_section = self.cfg['info_geometry'][mk]
                is_detector = True  # info_geometry uses same pipeline as detectors
            else:
                method_section = self.cfg['baselines'][mk]
                is_detector = False

            results[display_name] = {}

            for ck, crisis_def in crises.items():
                total_cells += 1
                cache_key = _cell_cache_key(
                    mk, method_section['params'], ck, crisis_def, self._lib_hash,
                    self._data_source, self._data_hash,
                )

                cached = self._load_cell(cache_key)
                if cached is not None:
                    results[display_name][ck] = cached
                    cache_hits += 1
                    d_val = cached.get('d', 'N/A')
                    logger.debug(f"  [CACHE] {display_name} x {ck}: d={d_val}")
                    continue

                logger.info(f"  [RUN] {display_name} x {ck}")
                if is_detector:
                    cell = self._run_detector_cell(mk, method_section, ck, crisis_def, data)
                else:
                    cell = self._run_baseline_cell(mk, method_section, ck, crisis_def, data)

                results[display_name][ck] = cell
                self._save_cell(cache_key, cell)

                d_val = cell.get('d')
                if d_val is not None:
                    logger.info(f"    d = {d_val:.3f}")
                else:
                    logger.info(f"    d = N/A ({cell.get('reason', 'unknown')})")

        # Run RF with LOCO
        if rf_key:
            mc = get_method_config(self.cfg, rf_key)
            display_name = mc['display_name']
            method_section = self.cfg['baselines'][rf_key]

            # RF LOCO can't be decomposed into independent cells easily
            # because training labels depend on ALL other crises.
            # Cache the entire RF result block.
            rf_cache_key = _cell_cache_key(
                rf_key, method_section['params'],
                'LOCO_ALL', {'crises': list(crises.keys())},
                self._lib_hash, self._data_source, self._data_hash,
            )
            cached = self._load_cell(rf_cache_key)
            if cached is not None:
                results[display_name] = cached
                cache_hits += len(crises)
                total_cells += len(crises)
                logger.info(f"  [CACHE] {display_name} (LOCO, {len(crises)} crises)")
            else:
                logger.info(f"  [RUN] {display_name} (leave-one-crisis-out)")
                total_cells += len(crises)
                rf_results = self._run_rf_loco(method_section, crises, data)
                results[display_name] = rf_results
                self._save_cell(rf_cache_key, rf_results)

        # Summary statistics
        method_names = list(results.keys())
        crisis_list = list(crises.keys())
        n_methods = len(method_names)
        n_crises = len(crisis_list)

        d_matrix = np.full((n_crises, n_methods), np.nan)
        for j, mname in enumerate(method_names):
            for i, ck in enumerate(crisis_list):
                val = results[mname].get(ck, {}).get('d')
                if val is not None:
                    d_matrix[i, j] = val

        chi_sq, p_val, mean_ranks = friedman_test(d_matrix)

        median_d = {}
        for j, mname in enumerate(method_names):
            col = d_matrix[:, j]
            valid = col[~np.isnan(col)]
            median_d[mname] = float(np.median(valid)) if len(valid) > 0 else None

        elapsed = time.time() - t_start

        # Log summary
        logger.info("\n" + "=" * 70)
        logger.info(f"RESULTS ({experiment_name}): {cache_hits}/{total_cells} cache hits, "
                     f"{elapsed:.1f}s elapsed")
        logger.info("=" * 70)
        sorted_methods = sorted(median_d.items(), key=lambda x: x[1] or -1, reverse=True)
        for rank, (mname, md) in enumerate(sorted_methods, 1):
            logger.info(f"  {rank:2d}. {mname:25s}  median d = {md:.3f}" if md else
                        f"  {rank:2d}. {mname:25s}  median d = N/A")

        # Build output
        output = {
            'timestamp': datetime.now().isoformat(),
            'experiment': experiment_name,
            'config': {
                'causal': True,
                'window_size': self.cfg['evaluation']['window_size'],
                'n_bootstrap': self.cfg['evaluation']['n_bootstrap'],
                'n_crises': n_crises,
                'n_methods': n_methods,
                'cache_hits': cache_hits,
                'total_cells': total_cells,
                'elapsed_s': round(elapsed, 1),
                'lib_hash': self._lib_hash,
                'data_source': self._data_source,
                'data_hash': self._data_hash,
                'n_trading_days': self._n_trading_days,
                'date_range': self._date_range,
            },
            'results': results,
            'summary': {
                'median_d': median_d,
                'friedman_chi_sq': float(chi_sq) if not np.isnan(chi_sq) else None,
                'friedman_p': float(p_val) if not np.isnan(p_val) else None,
                'mean_ranks': (
                    {mname: float(mean_ranks[j]) for j, mname in enumerate(method_names)}
                    if mean_ranks is not None and not np.any(np.isnan(mean_ranks)) else None
                ),
            },
        }

        # Save
        out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f'causal_comparison_{ts}.json'
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nResults saved to {out_path}")

        return output

    def clear_cache(self):
        """Remove all cached cell results."""
        count = 0
        for pkl in self.cache_dir.glob('*.pkl'):
            pkl.unlink()
            count += 1
        logger.info(f"Cleared {count} cached cells from {self.cache_dir}")


def main():
    """CLI entry point for the incremental runner."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        force=True,
    )

    parser = argparse.ArgumentParser(description='Incremental experiment runner')
    parser.add_argument('experiment', nargs='?', default='default',
                        help='Named experiment from config.yaml (default: default)')
    parser.add_argument('--force', action='store_true',
                        help='Ignore cache and recompute everything')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear all cached results and exit')
    parser.add_argument('--config', type=Path, default=None,
                        help='Path to config.yaml (default: experiments/config.yaml)')
    args = parser.parse_args()

    from experiments.config import load_config
    cfg = load_config(args.config) if args.config else load_config()

    runner = ExperimentRunner(cfg, force=args.force)

    if args.clear_cache:
        runner.clear_cache()
        return

    results = runner.run_comparison(args.experiment)

    # Run validation
    from experiments.validate import validate_results
    issues = validate_results(results, cfg)
    if issues:
        logger.warning(f"\nValidation found {len(issues)} issues:")
        for issue in issues:
            logger.warning(f"  [{issue['severity']}] {issue['message']}")


if __name__ == '__main__':
    main()
