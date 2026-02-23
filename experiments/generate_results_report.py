"""
Generate comprehensive results report from all experiment outputs.

Loads latest JSON results from experiments/outputs/regime_detection/,
compiles strategy and regime detection metrics, and outputs a markdown
or LaTeX report.

Usage:
    python experiments/generate_results_report.py
    python experiments/generate_results_report.py --format latex
    python experiments/generate_results_report.py --format both
    python experiments/generate_results_report.py --output custom_report.md

Outputs:
    experiments/outputs/regime_detection/RESULTS_REPORT.md
    experiments/outputs/regime_detection/RESULTS_REPORT.tex  (with --format latex/both)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / 'experiments' / 'outputs' / 'regime_detection'


# =============================================================================
# JSON Loading Utilities
# =============================================================================

def _latest_json(directory: Path, prefix: str) -> dict | None:
    """Load the most recent JSON file matching prefix in directory."""
    candidates = sorted(directory.glob(f'{prefix}*.json'))
    if not candidates:
        logger.warning(f'No files matching {prefix}*.json in {directory}')
        return None
    path = candidates[-1]
    logger.info(f'Loading {path.name}')
    with open(path) as f:
        return json.load(f)


def _load_json(path: Path) -> dict | None:
    """Load a single JSON file."""
    if not path.exists():
        logger.warning(f'File not found: {path}')
        return None
    logger.info(f'Loading {path.name}')
    with open(path) as f:
        return json.load(f)


def load_all_results() -> dict:
    """Load all latest experiment results."""
    backtest_dir = OUTPUT_DIR / 'backtest'
    return {
        'backtest': _latest_json(backtest_dir, 'backtest_'),
        'sensitivity': _latest_json(backtest_dir, 'sensitivity_'),
        'online': _latest_json(OUTPUT_DIR, 'online_detection_'),
        'walk_forward': _latest_json(OUTPUT_DIR / 'walk_forward', 'walk_forward_'),
        'operator_ablation': _latest_json(
            OUTPUT_DIR / 'operator_ablation', 'operator_ablation_'
        ),
        'numerical_stability': _latest_json(
            OUTPUT_DIR / 'numerical_stability', 'stability_ablation_'
        ),
        'interaction_test': _latest_json(
            OUTPUT_DIR / 'interaction_test', 'interaction_test_'
        ),
        'window_sensitivity': _latest_json(
            OUTPUT_DIR / 'window_sensitivity', 'window_sensitivity_'
        ),
        'fixed_hp_ablation': _latest_json(
            OUTPUT_DIR / 'fixed_hp_ablation', 'fixed_hp_ablation_'
        ),
    }


# =============================================================================
# Report Section Generators
# =============================================================================

CRISIS_DISPLAY = {
    '2007_quant': 'Quant 2007',
    '2008_gfc': 'GFC 2008',
    '2010_flash': 'Flash Crash',
    '2011_euro': 'Euro Crisis',
    '2015_china': 'China 2015',
    '2018_volmageddon': 'Volmageddon',
    '2018_q4': 'Q4 2018',
    '2019_repo': 'Repo 2019',
    '2020_covid': 'COVID',
    '2022_rates': 'Rates 2022',
    '2023_svb': 'SVB 2023',
    '2024_carry': 'Carry 2024',
}


def _fmt(val, decimals=2):
    """Format a numeric value, handling None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '—'
    return f'{val:.{decimals}f}'


def _pct(val, decimals=2):
    """Format a value as percentage."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '—'
    return f'{val * 100:.{decimals}f}%'


def section_header(data: dict) -> str:
    """Generate report header with timestamp info."""
    lines = [
        '# QCML Geometric SDE — Comprehensive Results Report',
        '',
        f'**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '',
    ]
    bt = data.get('backtest')
    if bt:
        lines.append(f'**Backtest timestamp**: {bt.get("timestamp", "unknown")}')
        cfg = bt.get('config', {})
        lines.append(f'**Symbols**: {", ".join(cfg.get("symbols", []))}')
        lines.append(
            f'**Config**: vol_target={cfg.get("target_vol")}, '
            f'crisis_threshold={cfg.get("crisis_threshold")}'
        )
    lines.append('')
    lines.append('---')
    lines.append('')
    return '\n'.join(lines)


def section_backtest_full(data: dict) -> str:
    """Full-period strategy comparison table."""
    bt = data.get('backtest')
    if not bt:
        return ''
    results = bt.get('results', {})
    lines = [
        '## 1. Strategy Performance — Full Period',
        '',
        '| Strategy | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD | Calmar | Skew | Kurt |',
        '|----------|-----------|---------|--------|---------|--------|--------|------|------|',
    ]

    strategy_order = [
        'GeometricLongFlat', 'GeometricLongShort', 'GeometricMultiAsset',
        'BuyHoldSPY', 'SixtyForty', 'ConstantVolSPY',
        'BuyHoldEqualWeight', 'ConstantVolMultiAsset',
    ]

    for name in strategy_order:
        r = results.get(name, {})
        fp = r.get('full_period', {})
        if not fp:
            continue
        bold = '**' if name.startswith('Geometric') else ''
        lines.append(
            f'| {bold}{name}{bold} '
            f'| {_pct(fp.get("annual_return"))} '
            f'| {_pct(fp.get("annual_vol"))} '
            f'| {_fmt(fp.get("sharpe"))} '
            f'| {_fmt(fp.get("sortino"))} '
            f'| {_pct(fp.get("max_drawdown"))} '
            f'| {_fmt(fp.get("calmar"))} '
            f'| {_fmt(fp.get("skewness"))} '
            f'| {_fmt(fp.get("kurtosis"))} |'
        )

    lines.append('')
    return '\n'.join(lines)


def section_backtest_oos(data: dict) -> str:
    """In-sample vs out-of-sample comparison."""
    bt = data.get('backtest')
    if not bt:
        return ''
    results = bt.get('results', {})
    lines = [
        '## 2. In-Sample vs Out-of-Sample Performance',
        '',
        '| Strategy | IS Sharpe | OOS Sharpe | IS MaxDD | OOS MaxDD | IS Return | OOS Return |',
        '|----------|----------|-----------|---------|----------|----------|-----------|',
    ]

    for name in results:
        r = results[name]
        is_data = r.get('in_sample', {})
        oos = r.get('out_of_sample', {})
        if not is_data or not oos:
            continue
        bold = '**' if name.startswith('Geometric') else ''
        lines.append(
            f'| {bold}{name}{bold} '
            f'| {_fmt(is_data.get("sharpe"))} '
            f'| {_fmt(oos.get("sharpe"))} '
            f'| {_pct(is_data.get("max_drawdown"))} '
            f'| {_pct(oos.get("max_drawdown"))} '
            f'| {_pct(is_data.get("annual_return"))} '
            f'| {_pct(oos.get("annual_return"))} |'
        )

    # Sharpe CI
    lines.append('')
    lines.append('### Sharpe Ratio Confidence Intervals')
    lines.append('')
    lines.append('| Strategy | Sharpe | 95% CI |')
    lines.append('|----------|--------|--------|')
    for name in results:
        r = results[name]
        ci = r.get('sharpe_ci', {})
        if ci:
            lines.append(
                f'| {name} | {_fmt(ci.get("point"))} '
                f'| [{_fmt(ci.get("ci_lo"))}, {_fmt(ci.get("ci_hi"))}] |'
            )

    # Statistical comparisons
    stats_cmp = bt.get('statistical_comparisons', {})
    if stats_cmp:
        lines.append('')
        lines.append('### Statistical Comparisons (Ledoit-Wolf)')
        lines.append('')
        lines.append('| Strategy | vs Benchmark | Delta Sharpe | p-value | Significant | Break-even (bps) |')
        lines.append('|----------|-------------|-------------|---------|------------|-----------------|')
        for name, cmp in stats_cmp.items():
            sig = 'Yes' if cmp.get('significant') else 'No'
            lines.append(
                f'| {name} | {cmp.get("vs")} '
                f'| {_fmt(cmp.get("delta_sharpe"), 3)} '
                f'| {_fmt(cmp.get("p_value"), 4)} '
                f'| {sig} '
                f'| {cmp.get("breakeven_bps")} |'
            )

    lines.append('')
    return '\n'.join(lines)


def section_crisis_returns(data: dict) -> str:
    """Crisis-by-crisis return protection table."""
    bt = data.get('backtest')
    if not bt:
        return ''
    results = bt.get('results', {})

    geo = results.get('GeometricLongFlat', {}).get('crisis_returns', {})
    spy = results.get('BuyHoldSPY', {}).get('crisis_returns', {})
    cvol = results.get('ConstantVolSPY', {}).get('crisis_returns', {})

    if not geo or not spy:
        return ''

    lines = [
        '## 3. Crisis-by-Crisis Returns',
        '',
        '| Crisis | GeometricLF | SPY | ConstantVol | Geo Protection vs SPY |',
        '|--------|------------|-----|------------|----------------------|',
    ]

    for key in CRISIS_DISPLAY:
        g = geo.get(key)
        s = spy.get(key)
        c = cvol.get(key)
        if g is None or s is None:
            continue
        protection = g - s
        sign = '+' if protection > 0 else ''
        lines.append(
            f'| {CRISIS_DISPLAY[key]} '
            f'| {_pct(g)} '
            f'| {_pct(s)} '
            f'| {_pct(c) if c is not None else "—"} '
            f'| {sign}{protection * 100:.1f}pp |'
        )

    lines.append('')
    return '\n'.join(lines)


def section_sensitivity(data: dict) -> str:
    """Sensitivity analysis heatmap table."""
    sens = data.get('sensitivity')
    if not sens:
        return ''

    grid = sens.get('grid', {})
    results = sens.get('results', {})
    vol_targets = grid.get('vol_targets', [])
    thresholds = grid.get('thresholds', [])

    lines = [
        '## 4. Sensitivity Analysis (Sharpe)',
        '',
        f'Grid: {len(vol_targets)} vol targets × {len(thresholds)} crisis thresholds',
        '',
    ]

    # Sharpe heatmap
    header = '| vol \\ threshold | ' + ' | '.join(f'{t:.1f}' for t in thresholds) + ' |'
    sep = '|' + '---|' * (len(thresholds) + 1)
    lines.extend([header, sep])

    for vt in vol_targets:
        row = f'| {vt:.2f} '
        for ct in thresholds:
            key = f'vt{vt:.2f}_ct{ct:.2f}'
            r = results.get(key, {})
            sharpe = r.get('net_sharpe')
            row += f'| {_fmt(sharpe)} '
        row += '|'
        lines.append(row)

    # Best config
    best_key = max(results, key=lambda k: results[k].get('net_sharpe', -999))
    best = results[best_key]
    lines.extend([
        '',
        f'**Best config**: vol_target={best.get("vol_target")}, '
        f'threshold={best.get("crisis_threshold")} → '
        f'Sharpe={_fmt(best.get("net_sharpe"))}, '
        f'Alpha={_fmt(best.get("alpha_sharpe"), 3)}',
        '',
    ])

    # OOS Sharpe heatmap
    lines.append('### OOS Sharpe')
    lines.append('')
    header = '| vol \\ threshold | ' + ' | '.join(f'{t:.1f}' for t in thresholds) + ' |'
    lines.extend([header, sep])

    for vt in vol_targets:
        row = f'| {vt:.2f} '
        for ct in thresholds:
            key = f'vt{vt:.2f}_ct{ct:.2f}'
            r = results.get(key, {})
            oos = r.get('oos_sharpe')
            row += f'| {_fmt(oos)} '
        row += '|'
        lines.append(row)

    lines.append('')
    return '\n'.join(lines)


def section_online_detection(data: dict) -> str:
    """Online detection AUC and FAR summary."""
    online = data.get('online')
    if not online:
        return ''

    results = online.get('results', {})
    lines = [
        '## 5. Online Detection Performance',
        '',
        '| Method | AUC-ROC | AUC-PR | FAR@0.5/yr | Det@0.5 | FAR@1.0/yr | Det@1.0 | FAR@2.0/yr | Det@2.0 |',
        '|--------|---------|--------|-----------|---------|-----------|---------|-----------|---------|',
    ]

    for method, r in sorted(results.items()):
        m = r.get('metrics', {})
        far_05 = m.get('far_analysis', {}).get('far_0.5', {})
        far_10 = m.get('far_analysis', {}).get('far_1.0', {})
        far_20 = m.get('far_analysis', {}).get('far_2.0', {})
        lines.append(
            f'| {method} '
            f'| {_fmt(m.get("auc_roc"), 3)} '
            f'| {_fmt(m.get("auc_pr"), 3)} '
            f'| {_fmt(far_05.get("achieved_far"), 2)} '
            f'| {_pct(far_05.get("detection_rate"))} '
            f'| {_fmt(far_10.get("achieved_far"), 2)} '
            f'| {_pct(far_10.get("detection_rate"))} '
            f'| {_fmt(far_20.get("achieved_far"), 2)} '
            f'| {_pct(far_20.get("detection_rate"))} |'
        )

    # Per-crisis breakdown for best method
    best_method = max(results, key=lambda k: results[k].get('metrics', {}).get('auc_roc', 0))
    best_m = results[best_method].get('metrics', {})
    pc = best_m.get('per_crisis', {})
    if pc:
        lines.extend([
            '',
            f'### Per-Crisis Detection: {best_method} (best AUC-ROC)',
            '',
            '| Crisis | Mean P(crisis) | Max P(crisis) | % Above 50% |',
            '|--------|---------------|--------------|-------------|',
        ])
        for key in CRISIS_DISPLAY:
            c = pc.get(key)
            if not c:
                continue
            lines.append(
                f'| {CRISIS_DISPLAY[key]} '
                f'| {_fmt(c.get("mean_p_crisis"), 3)} '
                f'| {_fmt(c.get("max_p_crisis"), 3)} '
                f'| {_pct(c.get("pct_above_50"))} |'
            )

    lines.append('')
    return '\n'.join(lines)


def section_walk_forward(data: dict) -> str:
    """Walk-forward validation summary."""
    wf = data.get('walk_forward')
    if not wf:
        return ''

    summary = wf.get('method_summary', {})
    lines = [
        '## 6. Walk-Forward Validation',
        '',
        '| Method | Median d | Det. Rate | Median Delay | Median FAR | N Detected |',
        '|--------|---------|----------|-------------|-----------|-----------|',
    ]

    for method in sorted(summary):
        s = summary[method]
        lines.append(
            f'| {method} '
            f'| {_fmt(s.get("median_d"))} '
            f'| {_pct(s.get("detection_rate"))} '
            f'| {_fmt(s.get("median_delay"), 0)} '
            f'| {_fmt(s.get("median_far"))} '
            f'| {s.get("n_detected")}/{s.get("n_total")} |'
        )

    lines.append('')
    return '\n'.join(lines)


def section_operator_ablation(data: dict) -> str:
    """Operator ablation results."""
    ab = data.get('operator_ablation')
    if not ab:
        return ''

    results = ab.get('results', {})
    lines = [
        '## 7. Operator Ablation',
        '',
    ]

    # Aggregate by method × condition
    agg = {}
    for key, r in results.items():
        method = r['method']
        cond = r['condition']
        d = r['d']
        agg.setdefault((method, cond), []).append(d)

    lines.extend([
        '| Method | Condition | Mean d | Median d | Std d | N |',
        '|--------|----------|--------|---------|-------|---|',
    ])

    for (method, cond) in sorted(agg):
        vals = agg[(method, cond)]
        lines.append(
            f'| {method} | {cond} '
            f'| {_fmt(np.mean(vals))} '
            f'| {_fmt(np.median(vals))} '
            f'| {_fmt(np.std(vals))} '
            f'| {len(vals)} |'
        )

    lines.append('')
    return '\n'.join(lines)


def section_numerical_stability(data: dict) -> str:
    """Numerical stability ablation."""
    ns = data.get('numerical_stability')
    if not ns:
        return ''

    results = ns.get('results', {})
    epsilons = ns.get('epsilons', [])
    pca_dims = ns.get('pca_dims', [])
    crises = ns.get('crises', [])

    lines = [
        '## 8. Numerical Stability',
        '',
        '### Epsilon Sensitivity',
        '',
        '| Method | ' + ' | '.join(f'eps={e}' for e in epsilons) + ' |',
        '|--------|' + '|'.join('------' for _ in epsilons) + '|',
    ]

    method_map = {'berry': 'Berry Phase Rate', 'qfi_det': 'QFI Determinant'}
    for short, display in method_map.items():
        r = results.get(short, {})
        row = f'| {display} '
        for e in epsilons:
            vals = r.get(f'eps_{e}', {})
            mean_d = np.mean([v for v in vals.values() if v is not None]) if vals else float('nan')
            row += f'| {_fmt(mean_d)} '
        row += '|'
        lines.append(row)

    lines.extend([
        '',
        '### PCA Dimension Sensitivity',
        '',
        '| Method | ' + ' | '.join(f'p={p}' for p in pca_dims) + ' |',
        '|--------|' + '|'.join('------' for _ in pca_dims) + '|',
    ])

    full_map = {'berry': 'Berry Phase Rate', 'qfi_det': 'QFI Determinant', 'mlf': 'Multi-Lag Fidelity'}
    for short, display in full_map.items():
        r = results.get(short, {})
        row = f'| {display} '
        for p in pca_dims:
            vals = r.get(f'pca_{p}', {})
            mean_d = np.mean([v for v in vals.values() if v is not None]) if vals else float('nan')
            row += f'| {_fmt(mean_d)} '
        row += '|'
        lines.append(row)

    lines.append('')
    return '\n'.join(lines)


def section_window_sensitivity(data: dict) -> str:
    """Window sensitivity analysis."""
    ws = data.get('window_sensitivity')
    if not ws:
        return ''

    window_sizes = ws.get('window_sizes', [])
    results = ws.get('results', {})
    rank_corr = ws.get('rank_correlations', {})

    lines = [
        '## 9. Window Size Sensitivity',
        '',
        '| Method | ' + ' | '.join(f'w={w}' for w in window_sizes) + ' |',
        '|--------|' + '|'.join('------' for _ in window_sizes) + '|',
    ]

    # Get method names from first window
    first_window = results.get(str(window_sizes[0]), {})
    methods = sorted(first_window.keys())

    for method in methods:
        row = f'| {method} '
        for w in window_sizes:
            w_data = results.get(str(w), {}).get(method, {})
            vals = [v for v in w_data.values() if v is not None]
            mean_d = np.mean(vals) if vals else float('nan')
            row += f'| {_fmt(mean_d)} '
        row += '|'
        lines.append(row)

    if rank_corr:
        lines.extend([
            '',
            '### Rank Correlations (Kendall tau)',
            '',
            '| Comparison | tau | p-value |',
            '|-----------|-----|---------|',
        ])
        for pair, rc in sorted(rank_corr.items()):
            lines.append(f'| {pair} | {_fmt(rc.get("tau"), 3)} | {_fmt(rc.get("p"), 4)} |')

    lines.append('')
    return '\n'.join(lines)


def section_interaction_test(data: dict) -> str:
    """Method type × crisis type interaction test (ANOVA)."""
    it = data.get('interaction_test')
    if not it:
        return ''

    anova = it.get('anova', {})
    cell_means = it.get('cell_means', {})

    lines = [
        '## 10. Interaction Test (Geometric vs Classical × Novel vs Conventional)',
        '',
        '### ANOVA Results',
        '',
        '| Effect | F | p-value | eta² |',
        '|--------|---|---------|------|',
    ]

    for effect in ['method_type', 'crisis_type', 'interaction']:
        a = anova.get(effect, {})
        lines.append(
            f'| {effect.replace("_", " ").title()} '
            f'| {_fmt(a.get("F"))} '
            f'| {_fmt(a.get("p"), 4)} '
            f'| {_fmt(a.get("eta2"), 3)} |'
        )

    lines.extend([
        '',
        '### Cell Means (Mean Cohen\'s d)',
        '',
        '| | Conventional | Novel |',
        '|---------|-------------|-------|',
        f'| Classical | {_fmt(cell_means.get("classical_conventional"))} '
        f'| {_fmt(cell_means.get("classical_novel"))} |',
        f'| Geometric | {_fmt(cell_means.get("geometric_conventional"))} '
        f'| {_fmt(cell_means.get("geometric_novel"))} |',
        '',
    ])

    return '\n'.join(lines)


def section_fixed_hp(data: dict) -> str:
    """Fixed hyperparameter ablation (per-crisis d-values)."""
    fhp = data.get('fixed_hp_ablation')
    if not fhp:
        return ''

    results = fhp.get('results', {})

    # Parse into method → crisis → d
    method_crisis = {}
    for key, r in results.items():
        method = r['method']
        crisis = r['crisis']
        method_crisis.setdefault(method, {})[crisis] = r['d']

    methods = sorted(method_crisis.keys())
    lines = [
        '## 11. Fixed Hyperparameter Regime Detection (Per-Crisis)',
        '',
        f'Config: h={fhp.get("config", {}).get("hilbert_dim")}, '
        f'p={fhp.get("config", {}).get("n_pca_components")}, '
        f'op={fhp.get("config", {}).get("operator_method")}, '
        f'w={fhp.get("config", {}).get("rolling_window")}',
        '',
        '| Crisis | ' + ' | '.join(methods) + ' |',
        '|--------|' + '|'.join('------' for _ in methods) + '|',
    ]

    for key in CRISIS_DISPLAY:
        row = f'| {CRISIS_DISPLAY[key]} '
        for method in methods:
            d = method_crisis.get(method, {}).get(key)
            row += f'| {_fmt(d)} '
        row += '|'
        lines.append(row)

    # Method means
    lines.append('| **Mean** ')
    for method in methods:
        vals = [v for v in method_crisis[method].values() if v is not None]
        lines[-1] += f'| **{_fmt(np.mean(vals))}** '
    lines[-1] += '|'

    lines.append('')
    return '\n'.join(lines)


def section_key_findings(data: dict) -> str:
    """Executive summary of key findings."""
    bt = data.get('backtest')
    if not bt:
        return ''

    results = bt.get('results', {})
    stats_cmp = bt.get('statistical_comparisons', {})

    # Extract key numbers
    geo_lf_full = results.get('GeometricLongFlat', {}).get('full_period', {})
    geo_lf_oos = results.get('GeometricLongFlat', {}).get('out_of_sample', {})
    spy_full = results.get('BuyHoldSPY', {}).get('full_period', {})
    spy_oos = results.get('BuyHoldSPY', {}).get('out_of_sample', {})
    cvol_full = results.get('ConstantVolSPY', {}).get('full_period', {})
    cvol_oos = results.get('ConstantVolSPY', {}).get('out_of_sample', {})

    # Crisis protection
    geo_crisis = results.get('GeometricLongFlat', {}).get('crisis_returns', {})
    spy_crisis = results.get('BuyHoldSPY', {}).get('crisis_returns', {})
    gfc_protection = None
    if geo_crisis.get('2008_gfc') is not None and spy_crisis.get('2008_gfc') is not None:
        gfc_protection = (geo_crisis['2008_gfc'] - spy_crisis['2008_gfc']) * 100

    lines = [
        '## Key Findings',
        '',
        '### Strengths',
        '',
    ]

    # OOS Sharpe comparison
    geo_oos_sharpe = geo_lf_oos.get('sharpe')
    spy_oos_sharpe = spy_oos.get('sharpe')
    cvol_oos_sharpe = cvol_oos.get('sharpe')
    if geo_oos_sharpe and spy_oos_sharpe:
        lines.append(
            f'1. **OOS Sharpe superiority**: GeometricLongFlat OOS Sharpe = {_fmt(geo_oos_sharpe)} '
            f'vs SPY {_fmt(spy_oos_sharpe)} '
            f'(+{(geo_oos_sharpe - spy_oos_sharpe):.2f})'
        )

    # Drawdown protection
    geo_full_dd = geo_lf_full.get('max_drawdown')
    spy_full_dd = spy_full.get('max_drawdown')
    if geo_full_dd and spy_full_dd:
        lines.append(
            f'2. **Drawdown protection**: Max DD {_pct(geo_full_dd)} vs SPY {_pct(spy_full_dd)} '
            f'({(1 - geo_full_dd / spy_full_dd) * 100:.0f}% reduction)'
        )

    if gfc_protection:
        lines.append(
            f'3. **GFC protection**: +{gfc_protection:.1f}pp vs SPY during 2008-09'
        )

    lines.append('')
    lines.append('### Limitations')
    lines.append('')

    # Full-period Sharpe gap
    geo_full_sharpe = geo_lf_full.get('sharpe')
    cvol_full_sharpe = cvol_full.get('sharpe')
    if geo_full_sharpe and cvol_full_sharpe:
        lines.append(
            f'1. **Full-period Sharpe gap**: {_fmt(geo_full_sharpe)} vs '
            f'ConstantVol {_fmt(cvol_full_sharpe)} '
            f'(delta = {geo_full_sharpe - cvol_full_sharpe:+.2f})'
        )

    # Alpha vs benchmark
    lf_cmp = stats_cmp.get('GeometricLongFlat', {})
    if lf_cmp:
        lines.append(
            f'2. **Alpha vs {lf_cmp.get("vs")}**: '
            f'delta Sharpe = {lf_cmp.get("delta_sharpe"):+.3f} '
            f'(p = {_fmt(lf_cmp.get("p_value"), 4)}, '
            f'{"significant" if lf_cmp.get("significant") else "not significant"})'
        )

    lines.append('')
    return '\n'.join(lines)


def section_signal_stats(data: dict) -> str:
    """Signal generation statistics."""
    bt = data.get('backtest')
    if not bt:
        return ''

    ss = bt.get('signal_stats', {})
    if not ss:
        return ''

    lines = [
        '## Signal Statistics',
        '',
        f'- Valid signal days: {ss.get("n_valid")}',
        f'- Mean P(crisis): {_fmt(ss.get("mean_p_crisis"), 4)}',
        f'- Std P(crisis): {_fmt(ss.get("std_p_crisis"), 4)}',
        f'- % Above threshold: {_pct(ss.get("pct_above_threshold"))}',
        '',
    ]
    return '\n'.join(lines)


# =============================================================================
# LaTeX Generation Utilities
# =============================================================================

def _esc(s: str) -> str:
    """Escape special LaTeX characters in a string."""
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('^', r'\textasciicircum{}'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def _latex_table(
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
    col_fmt: str | None = None,
    bold_first_col: bool = False,
    notes: str = '',
) -> str:
    """Render a booktabs LaTeX table."""
    n = len(headers)
    if col_fmt is None:
        col_fmt = 'l' + 'r' * (n - 1)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        rf'\caption{{{_esc(caption)}}}',
        rf'\label{{tab:{label}}}',
        rf'\begin{{tabular}}{{{col_fmt}}}',
        r'\toprule',
        ' & '.join(rf'\textbf{{{_esc(h)}}}' for h in headers) + r' \\',
        r'\midrule',
    ]

    for i, row in enumerate(rows):
        cells = []
        for j, cell in enumerate(row):
            cell_str = str(cell)
            if j == 0 and bold_first_col:
                cells.append(rf'\textbf{{{_esc(cell_str)}}}')
            else:
                cells.append(_esc(cell_str))
        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    if notes:
        lines.append(rf'\par\smallskip{{\footnotesize\textit{{Note:}} {_esc(notes)}}}')
    lines.append(r'\end{table}')
    lines.append('')
    return '\n'.join(lines)


def _latex_midrule_at(rows: list[list[str]], idx: int) -> list[list[str]]:
    """Insert a sentinel row that renders as \\midrule at position idx."""
    result = rows[:idx] + [['__MIDRULE__']] + rows[idx:]
    return result


def _render_table_with_midrules(
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
    col_fmt: str | None = None,
    midrule_before: list[int] | None = None,
    notes: str = '',
    wide: bool = False,
) -> str:
    """Render booktabs table with optional mid-section rules.

    Args:
        wide: If True, wrap in \\resizebox{\\linewidth}{!}{...} for wide tables.
    """
    n = len(headers)
    if col_fmt is None:
        col_fmt = 'l' + 'r' * (n - 1)

    tabular_lines = [
        rf'\begin{{tabular}}{{{col_fmt}}}',
        r'\toprule',
        ' & '.join(rf'\textbf{{{_esc(h)}}}' for h in headers) + r' \\',
        r'\midrule',
    ]

    midrule_set = set(midrule_before or [])
    for i, row in enumerate(rows):
        if i in midrule_set:
            tabular_lines.append(r'\midrule')
        tabular_lines.append(' & '.join(_esc(str(c)) for c in row) + r' \\')

    tabular_lines.append(r'\bottomrule')
    tabular_lines.append(r'\end{tabular}')
    tabular_str = '\n'.join(tabular_lines)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        rf'\caption{{{_esc(caption)}}}',
        rf'\label{{tab:{label}}}',
    ]

    if wide:
        lines.append(r'\resizebox{\linewidth}{!}{%')
        lines.append(tabular_str)
        lines.append(r'}')
    else:
        lines.append(tabular_str)

    if notes:
        lines.append(rf'\par\smallskip{{\footnotesize\textit{{Note:}} {_esc(notes)}}}')
    lines.append(r'\end{table}')
    lines.append('')
    return '\n'.join(lines)
    lines.append(r'\end{table}')
    lines.append('')
    return '\n'.join(lines)


# =============================================================================
# LaTeX Section Generators
# =============================================================================

def latex_preamble(data: dict) -> str:
    bt = data.get('backtest', {})
    cfg = bt.get('config', {}) if bt else {}
    ts = bt.get('timestamp', 'unknown') if bt else 'unknown'
    return r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{siunitx}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue}

\title{QCML Geometric SDE --- Comprehensive Results Report}
\author{Will Hammond \\ Pitzer College \\ \texttt{whammond@pitzer.edu}}
\date{""" + f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} \\\\ Backtest: {ts}' + r"""}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""


def latex_postamble() -> str:
    return r'\end{document}' + '\n'


def latex_key_findings(data: dict) -> str:
    bt = data.get('backtest')
    if not bt:
        return ''
    results = bt.get('results', {})
    stats_cmp = bt.get('statistical_comparisons', {})

    geo_lf_full = results.get('GeometricLongFlat', {}).get('full_period', {})
    geo_lf_oos = results.get('GeometricLongFlat', {}).get('out_of_sample', {})
    spy_oos = results.get('BuyHoldSPY', {}).get('out_of_sample', {})
    cvol_full = results.get('ConstantVolSPY', {}).get('full_period', {})
    geo_crisis = results.get('GeometricLongFlat', {}).get('crisis_returns', {})
    spy_crisis = results.get('BuyHoldSPY', {}).get('crisis_returns', {})

    geo_oos_sharpe = geo_lf_oos.get('sharpe')
    spy_oos_sharpe = spy_oos.get('sharpe')
    geo_full_dd = geo_lf_full.get('max_drawdown')
    spy_full_dd = results.get('BuyHoldSPY', {}).get('full_period', {}).get('max_drawdown')
    geo_full_sharpe = geo_lf_full.get('sharpe')
    cvol_full_sharpe = cvol_full.get('sharpe')
    lf_cmp = stats_cmp.get('GeometricLongFlat', {})
    gfc_prot = None
    if geo_crisis.get('2008_gfc') is not None and spy_crisis.get('2008_gfc') is not None:
        gfc_prot = (geo_crisis['2008_gfc'] - spy_crisis['2008_gfc']) * 100

    lines = [
        r'\section{Key Findings}',
        '',
        r'\subsection*{Strengths}',
        r'\begin{itemize}',
    ]
    if geo_oos_sharpe and spy_oos_sharpe:
        delta = geo_oos_sharpe - spy_oos_sharpe
        lines.append(
            rf'  \item \textbf{{OOS Sharpe superiority:}} GeometricLongFlat OOS Sharpe $= {geo_oos_sharpe:.2f}$ '
            rf'vs SPY ${spy_oos_sharpe:.2f}$ ($+{delta:.2f}$).'
        )
    if geo_full_dd and spy_full_dd:
        pct_red = (1 - geo_full_dd / spy_full_dd) * 100
        lines.append(
            rf'  \item \textbf{{Drawdown protection:}} Maximum drawdown {geo_full_dd * 100:.1f}\% '
            rf'vs SPY {spy_full_dd * 100:.1f}\% ({pct_red:.0f}\% reduction).'
        )
    if gfc_prot:
        lines.append(
            rf'  \item \textbf{{GFC protection:}} $+{gfc_prot:.1f}$\,pp vs SPY during 2008--09.'
        )
    lines.append(r'\end{itemize}')
    lines.append('')
    lines.append(r'\subsection*{Limitations}')
    lines.append(r'\begin{itemize}')
    if geo_full_sharpe and cvol_full_sharpe:
        delta = geo_full_sharpe - cvol_full_sharpe
        lines.append(
            rf'  \item \textbf{{Full-period Sharpe gap:}} ${geo_full_sharpe:.2f}$ vs ConstantVol '
            rf'${cvol_full_sharpe:.2f}$ ($\Delta = {delta:+.2f}$).'
        )
    if lf_cmp:
        sig = 'statistically significant' if lf_cmp.get('significant') else 'not significant'
        lines.append(
            rf'  \item \textbf{{Alpha vs {_esc(lf_cmp.get("vs", ""))}}}: '
            rf'$\Delta$Sharpe $= {lf_cmp.get("delta_sharpe", 0):+.3f}$ '
            rf'($p = {lf_cmp.get("p_value", 0):.4f}$, {sig}).'
        )
    lines.append(r'\end{itemize}')
    lines.append('')
    return '\n'.join(lines)


def latex_backtest_full(data: dict) -> str:
    bt = data.get('backtest')
    if not bt:
        return ''
    results = bt.get('results', {})

    strategy_order = [
        'GeometricLongFlat', 'GeometricLongShort', 'GeometricMultiAsset',
        'BuyHoldSPY', 'SixtyForty', 'ConstantVolSPY',
        'BuyHoldEqualWeight', 'ConstantVolMultiAsset',
    ]

    headers = ['Strategy', r'Ann.\ Ret.', r'Ann.\ Vol.', 'Sharpe', 'Sortino', 'Max DD', 'Calmar', 'Skew', 'Kurt']
    rows = []
    for i, name in enumerate(strategy_order):
        r = results.get(name, {})
        fp = r.get('full_period', {})
        if not fp:
            continue
        display = name.replace('Geometric', r'\textbf{Geometric}') if name.startswith('Geometric') else name
        rows.append([
            display,
            _pct(fp.get('annual_return')),
            _pct(fp.get('annual_vol')),
            _fmt(fp.get('sharpe')),
            _fmt(fp.get('sortino')),
            _pct(fp.get('max_drawdown')),
            _fmt(fp.get('calmar')),
            _fmt(fp.get('skewness')),
            _fmt(fp.get('kurtosis')),
        ])

    col_fmt = 'l' + 'S[table-format=2.2]' * (len(headers) - 1)
    # Use plain rendering (siunitx alignment would need numeric-only cells)
    col_fmt = 'l' + 'r' * (len(headers) - 1)

    section = r'\section{Strategy Performance --- Full Period}' + '\n\n'
    section += _render_table_with_midrules(
        headers, rows,
        caption='Strategy performance over full period (2005--2024, $\\sim$20 years). '
                'Geometric strategies shown in bold.',
        label='backtest_full',
        col_fmt=col_fmt,
        midrule_before=[3],  # separator before benchmarks
        notes='Ann. Ret.: annualised return. Ann. Vol.: annualised volatility. '
              'Max DD: maximum drawdown. Calmar: annualised return / max drawdown.',
        wide=True,
    )
    return section


def latex_backtest_oos(data: dict) -> str:
    bt = data.get('backtest')
    if not bt:
        return ''
    results = bt.get('results', {})
    stats_cmp = bt.get('statistical_comparisons', {})

    headers = ['Strategy', 'IS Sharpe', 'OOS Sharpe', 'IS MaxDD', 'OOS MaxDD', 'IS Ret.', 'OOS Ret.']
    rows = []
    for name, r in results.items():
        is_d = r.get('in_sample', {})
        oos = r.get('out_of_sample', {})
        if not is_d or not oos:
            continue
        rows.append([
            name,
            _fmt(is_d.get('sharpe')),
            _fmt(oos.get('sharpe')),
            _pct(is_d.get('max_drawdown')),
            _pct(oos.get('max_drawdown')),
            _pct(is_d.get('annual_return')),
            _pct(oos.get('annual_return')),
        ])

    section = r'\section{In-Sample vs Out-of-Sample Performance}' + '\n\n'
    section += _render_table_with_midrules(
        headers, rows,
        caption='In-sample (IS, 2005--2019) vs out-of-sample (OOS, 2020--2024) performance.',
        label='backtest_oos',
        wide=True,
    )

    # Sharpe CI table
    ci_headers = ['Strategy', 'Sharpe', '95\\% CI Lower', '95\\% CI Upper']
    ci_rows = []
    for name, r in results.items():
        ci = r.get('sharpe_ci', {})
        if ci:
            ci_rows.append([
                name,
                _fmt(ci.get('point')),
                _fmt(ci.get('ci_lo')),
                _fmt(ci.get('ci_hi')),
            ])
    section += _render_table_with_midrules(
        ci_headers, ci_rows,
        caption='Bootstrap Sharpe ratio confidence intervals (10{,}000 bootstrap samples).',
        label='sharpe_ci',
    )

    # Statistical comparisons
    if stats_cmp:
        cmp_headers = ['Strategy', 'Benchmark', r'$\Delta$Sharpe', '$p$-value', 'Sig.', 'Break-even (bps)']
        cmp_rows = []
        for name, cmp in stats_cmp.items():
            sig = 'Yes' if cmp.get('significant') else 'No'
            cmp_rows.append([
                name,
                cmp.get('vs', ''),
                _fmt(cmp.get('delta_sharpe'), 3),
                _fmt(cmp.get('p_value'), 4),
                sig,
                str(cmp.get('breakeven_bps', '')),
            ])
        section += _render_table_with_midrules(
            cmp_headers, cmp_rows,
            caption='Ledoit--Wolf Sharpe ratio comparison tests.',
            label='stat_compare',
        )

    return section


def latex_crisis_returns(data: dict) -> str:
    bt = data.get('backtest')
    if not bt:
        return ''
    results = bt.get('results', {})

    geo = results.get('GeometricLongFlat', {}).get('crisis_returns', {})
    spy = results.get('BuyHoldSPY', {}).get('crisis_returns', {})
    cvol = results.get('ConstantVolSPY', {}).get('crisis_returns', {})
    if not geo or not spy:
        return ''

    headers = ['Crisis', 'Geometric LF', 'SPY', r'Const.\ Vol.', 'Protection vs SPY']
    rows = []
    for key in CRISIS_DISPLAY:
        g = geo.get(key)
        s = spy.get(key)
        c = cvol.get(key)
        if g is None or s is None:
            continue
        protection = g - s
        sign = '+' if protection >= 0 else ''
        rows.append([
            CRISIS_DISPLAY[key],
            _pct(g),
            _pct(s),
            _pct(c) if c is not None else '---',
            f'{sign}{protection * 100:.1f}pp',
        ])

    section = r'\section{Crisis-by-Crisis Returns}' + '\n\n'
    section += _render_table_with_midrules(
        headers, rows,
        caption='Strategy returns during labelled crisis periods. Protection = Geometric LF minus SPY return (percentage points).',
        label='crisis_returns',
        col_fmt='lrrrr',
    )
    return section


def latex_sensitivity(data: dict) -> str:
    sens = data.get('sensitivity')
    if not sens:
        return ''

    grid = sens.get('grid', {})
    results = sens.get('results', {})
    vol_targets = grid.get('vol_targets', [])
    thresholds = grid.get('thresholds', [])

    # Full-period Sharpe heatmap
    headers = [r'$\sigma^*$ \textbackslash\ $\tau$'] + [f'{t:.1f}' for t in thresholds]
    rows = []
    for vt in vol_targets:
        row = [f'{vt:.2f}']
        for ct in thresholds:
            key = f'vt{vt:.2f}_ct{ct:.2f}'
            r = results.get(key, {})
            row.append(_fmt(r.get('net_sharpe')))
        rows.append(row)

    section = r'\section{Sensitivity Analysis}' + '\n\n'
    section += _render_table_with_midrules(
        headers, rows,
        caption=r'Full-period net Sharpe ratio across $6\times6$ grid of volatility target $\sigma^*$ '
                r'and crisis threshold $\tau$.',
        label='sensitivity_sharpe',
        col_fmt='l' + 'r' * len(thresholds),
    )

    # OOS heatmap
    oos_rows = []
    for vt in vol_targets:
        row = [f'{vt:.2f}']
        for ct in thresholds:
            key = f'vt{vt:.2f}_ct{ct:.2f}'
            r = results.get(key, {})
            row.append(_fmt(r.get('oos_sharpe')))
        oos_rows.append(row)

    section += _render_table_with_midrules(
        headers, oos_rows,
        caption=r'OOS (2020--2024) Sharpe across $6\times6$ sensitivity grid.',
        label='sensitivity_oos',
        col_fmt='l' + 'r' * len(thresholds),
    )

    best_key = max(results, key=lambda k: results[k].get('net_sharpe', -999))
    best = results[best_key]
    section += (
        rf'Best configuration: $\sigma^* = {best.get("vol_target")}$, '
        rf'$\tau = {best.get("crisis_threshold")}$ '
        rf'$\Rightarrow$ Sharpe $= {_fmt(best.get("net_sharpe"))}$, '
        rf'$\alpha = {_fmt(best.get("alpha_sharpe"), 3)}$.' + '\n\n'
    )
    return section


def latex_online_detection(data: dict) -> str:
    online = data.get('online')
    if not online:
        return ''
    results = online.get('results', {})

    headers = ['Method', 'AUC-ROC', 'AUC-PR',
               r'FAR\textsubscript{0.5}', r'Det\textsubscript{0.5}',
               r'FAR\textsubscript{1.0}', r'Det\textsubscript{1.0}',
               r'FAR\textsubscript{2.0}', r'Det\textsubscript{2.0}']
    rows = []
    for method in sorted(results):
        m = results[method].get('metrics', {})
        f05 = m.get('far_analysis', {}).get('far_0.5', {})
        f10 = m.get('far_analysis', {}).get('far_1.0', {})
        f20 = m.get('far_analysis', {}).get('far_2.0', {})
        rows.append([
            method,
            _fmt(m.get('auc_roc'), 3),
            _fmt(m.get('auc_pr'), 3),
            _fmt(f05.get('achieved_far'), 2),
            _pct(f05.get('detection_rate')),
            _fmt(f10.get('achieved_far'), 2),
            _pct(f10.get('detection_rate')),
            _fmt(f20.get('achieved_far'), 2),
            _pct(f20.get('detection_rate')),
        ])

    section = r'\section{Online Detection Performance}' + '\n\n'
    section += _render_table_with_midrules(
        headers, rows,
        caption='Online regime detection performance. '
                'FAR: false alarm rate per year at target. Det: detection rate at that FAR.',
        label='online_detection',
        col_fmt='l' + 'r' * (len(headers) - 1),
        wide=True,
    )
    return section


def latex_walk_forward(data: dict) -> str:
    wf = data.get('walk_forward')
    if not wf:
        return ''
    summary = wf.get('method_summary', {})

    headers = ['Method', r'Median $d$', r'Det.\ Rate', 'Median Delay (d)', 'Median FAR', 'N Detected']
    rows = []
    for method in sorted(summary):
        s = summary[method]
        rows.append([
            method,
            _fmt(s.get('median_d')),
            _pct(s.get('detection_rate')),
            _fmt(s.get('median_delay'), 0),
            _fmt(s.get('median_far')),
            f'{s.get("n_detected")}/{s.get("n_total")}',
        ])

    section = r'\section{Walk-Forward Validation}' + '\n\n'
    section += _render_table_with_midrules(
        headers, rows,
        caption="Walk-forward leave-one-crisis-out results. Cohen's $d$ computed per fold; "
                "detection delay in calendar days.",
        label='walk_forward',
    )
    return section


def latex_operator_ablation(data: dict) -> str:
    ab = data.get('operator_ablation')
    if not ab:
        return ''
    results = ab.get('results', {})

    agg = {}
    for key, r in results.items():
        method, cond = r['method'], r['condition']
        agg.setdefault((method, cond), []).append(r['d'])

    headers = ['Method', 'Condition', r'Mean $d$', r'Median $d$', r'Std $d$', '$N$']
    rows = []
    prev_method = None
    midrule_before = []
    for i, (method, cond) in enumerate(sorted(agg)):
        if method != prev_method and prev_method is not None:
            midrule_before.append(i)
        prev_method = method
        vals = agg[(method, cond)]
        rows.append([
            method, cond,
            _fmt(np.mean(vals)), _fmt(np.median(vals)), _fmt(np.std(vals)), str(len(vals)),
        ])

    section = r'\section{Operator Ablation}' + '\n\n'
    section += _render_table_with_midrules(
        headers, rows,
        caption='Operator ablation: mean Cohen\'s $d$ across 11 crises for each method--condition pair.',
        label='operator_ablation',
        midrule_before=midrule_before,
    )
    return section


def latex_numerical_stability(data: dict) -> str:
    ns = data.get('numerical_stability')
    if not ns:
        return ''
    results = ns.get('results', {})
    epsilons = ns.get('epsilons', [])
    pca_dims = ns.get('pca_dims', [])

    # Epsilon table
    eps_headers = ['Method'] + [rf'$\varepsilon = {e}$' for e in epsilons]
    eps_rows = []
    for short, display in [('berry', 'Berry Phase Rate'), ('qfi_det', 'QFI Determinant')]:
        r = results.get(short, {})
        row = [display]
        for e in epsilons:
            vals = r.get(f'eps_{e}', {})
            row.append(_fmt(np.mean([v for v in vals.values() if v is not None]) if vals else float('nan')))
        eps_rows.append(row)

    # PCA dim table
    pca_headers = ['Method'] + [f'$p = {p}$' for p in pca_dims]
    pca_rows = []
    for short, display in [('berry', 'Berry Phase Rate'), ('qfi_det', 'QFI Determinant'), ('mlf', 'Multi-Lag Fidelity')]:
        r = results.get(short, {})
        row = [display]
        for p in pca_dims:
            vals = r.get(f'pca_{p}', {})
            row.append(_fmt(np.mean([v for v in vals.values() if v is not None]) if vals else float('nan')))
        pca_rows.append(row)

    section = r'\section{Numerical Stability}' + '\n\n'
    section += _render_table_with_midrules(
        eps_headers, eps_rows,
        caption=r'Mean Cohen\'s $d$ (4 crises) varying regularisation $\varepsilon$.',
        label='stability_epsilon',
    )
    section += _render_table_with_midrules(
        pca_headers, pca_rows,
        caption=r'Mean Cohen\'s $d$ (4 crises) varying number of PCA components $p$.',
        label='stability_pca',
    )
    return section


def latex_window_sensitivity(data: dict) -> str:
    ws = data.get('window_sensitivity')
    if not ws:
        return ''
    window_sizes = ws.get('window_sizes', [])
    results = ws.get('results', {})
    rank_corr = ws.get('rank_correlations', {})

    first_window = results.get(str(window_sizes[0]), {})
    methods = sorted(first_window.keys())

    headers = ['Method'] + [f'$w = {w}$' for w in window_sizes]
    rows = []
    for method in methods:
        row = [method]
        for w in window_sizes:
            w_data = results.get(str(w), {}).get(method, {})
            vals = [v for v in w_data.values() if v is not None]
            row.append(_fmt(np.mean(vals) if vals else float('nan')))
        rows.append(row)

    section = r'\section{Window Size Sensitivity}' + '\n\n'
    section += _render_table_with_midrules(
        headers, rows,
        caption=r'Mean Cohen\'s $d$ (12 crises) for each method across rolling-window sizes.',
        label='window_sensitivity',
    )

    if rank_corr:
        rc_headers = ['Comparison', r'Kendall $\tau$', '$p$-value']
        rc_rows = []
        for pair in sorted(rank_corr):
            rc = rank_corr[pair]
            rc_rows.append([pair, _fmt(rc.get('tau'), 3), _fmt(rc.get('p'), 4)])
        section += _render_table_with_midrules(
            rc_headers, rc_rows,
            caption='Rank correlations of method rankings across window sizes.',
            label='rank_corr',
        )
    return section


def latex_interaction_test(data: dict) -> str:
    it = data.get('interaction_test')
    if not it:
        return ''
    anova = it.get('anova', {})
    cell_means = it.get('cell_means', {})

    anova_headers = ['Effect', '$F$', '$p$-value', r'$\eta^2$']
    anova_rows = []
    for effect in ['method_type', 'crisis_type', 'interaction']:
        a = anova.get(effect, {})
        anova_rows.append([
            effect.replace('_', ' ').title(),
            _fmt(a.get('F')),
            _fmt(a.get('p'), 4),
            _fmt(a.get('eta2'), 3),
        ])

    section = r'\section{Interaction Test: Geometric vs Classical $\times$ Novel vs Conventional}' + '\n\n'
    section += _render_table_with_midrules(
        anova_headers, anova_rows,
        caption='Two-way ANOVA: method type (Geometric/Classical) $\\times$ crisis type (Novel/Conventional).',
        label='anova',
    )

    # Cell means as a 2×2 table
    cm_rows = [
        ['Classical', _fmt(cell_means.get('classical_conventional')), _fmt(cell_means.get('classical_novel'))],
        ['Geometric', _fmt(cell_means.get('geometric_conventional')), _fmt(cell_means.get('geometric_novel'))],
    ]
    section += _render_table_with_midrules(
        ['Method Type', 'Conventional', 'Novel'],
        cm_rows,
        caption="Cell means (mean Cohen's $d$) for the $2\\times2$ interaction.",
        label='cell_means',
        col_fmt='lrr',
    )
    return section


def latex_fixed_hp(data: dict) -> str:
    fhp = data.get('fixed_hp_ablation')
    if not fhp:
        return ''
    results = fhp.get('results', {})
    cfg = fhp.get('config', {})

    method_crisis = {}
    for key, r in results.items():
        method_crisis.setdefault(r['method'], {})[r['crisis']] = r['d']

    methods = sorted(method_crisis.keys())
    headers = ['Crisis'] + methods
    rows = []
    for key in CRISIS_DISPLAY:
        row = [CRISIS_DISPLAY[key]]
        for method in methods:
            row.append(_fmt(method_crisis.get(method, {}).get(key)))
        rows.append(row)

    # Mean row
    mean_row = ['\\textbf{Mean}']
    for method in methods:
        vals = [v for v in method_crisis[method].values() if v is not None]
        mean_row.append(rf'\textbf{{{_fmt(np.mean(vals))}}}')
    rows.append(mean_row)

    section = r'\section{Fixed Hyperparameter Regime Detection (Per-Crisis)}' + '\n\n'
    section += (
        rf'Configuration: $h = {cfg.get("hilbert_dim")}$, '
        rf'$p = {cfg.get("n_pca_components")}$, '
        rf'operator = \texttt{{{_esc(str(cfg.get("operator_method", "")))}}},'
        rf' window $= {cfg.get("rolling_window")}$.' + '\n\n'
    )
    section += _render_table_with_midrules(
        headers, rows,
        caption="Cohen's $d$ per crisis under fixed hyperparameters. "
                "Bold row: mean across all crises.",
        label='fixed_hp',
        midrule_before=[len(rows) - 1],
    )
    return section


def latex_signal_stats(data: dict) -> str:
    bt = data.get('backtest')
    if not bt:
        return ''
    ss = bt.get('signal_stats', {})
    if not ss:
        return ''

    rows = [
        ['Valid signal days', str(ss.get('n_valid', ''))],
        [r'Mean $P(\text{crisis})$', _fmt(ss.get('mean_p_crisis'), 4)],
        [r'Std $P(\text{crisis})$', _fmt(ss.get('std_p_crisis'), 4)],
        [r'\% above threshold', _pct(ss.get('pct_above_threshold'))],
    ]

    section = r'\section{Signal Statistics}' + '\n\n'
    section += _render_table_with_midrules(
        ['Statistic', 'Value'], rows,
        caption='Summary statistics of the geometric regime-probability signal.',
        label='signal_stats',
        col_fmt='lr',
    )
    return section


# =============================================================================
# Main Report Assembly
# =============================================================================

def generate_report(data: dict) -> str:
    """Assemble the full markdown report."""
    sections = [
        section_header(data),
        section_key_findings(data),
        section_backtest_full(data),
        section_backtest_oos(data),
        section_crisis_returns(data),
        section_sensitivity(data),
        section_online_detection(data),
        section_walk_forward(data),
        section_operator_ablation(data),
        section_numerical_stability(data),
        section_window_sensitivity(data),
        section_interaction_test(data),
        section_fixed_hp(data),
        section_signal_stats(data),
    ]
    return '\n'.join(s for s in sections if s)


def generate_latex_report(data: dict) -> str:
    """Assemble the full LaTeX report."""
    body_sections = [
        latex_key_findings(data),
        latex_backtest_full(data),
        latex_backtest_oos(data),
        latex_crisis_returns(data),
        latex_sensitivity(data),
        latex_online_detection(data),
        latex_walk_forward(data),
        latex_operator_ablation(data),
        latex_numerical_stability(data),
        latex_window_sensitivity(data),
        latex_interaction_test(data),
        latex_fixed_hp(data),
        latex_signal_stats(data),
    ]
    body = '\n'.join(s for s in body_sections if s)
    return latex_preamble(data) + body + latex_postamble()


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive results report')
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output file path (overrides default)',
    )
    parser.add_argument(
        '--format', '-f', choices=['markdown', 'latex', 'both'], default='markdown',
        help='Output format (default: markdown)',
    )
    args = parser.parse_args()

    logger.info('Loading all experiment results...')
    data = load_all_results()

    loaded = sum(1 for v in data.values() if v is not None)
    logger.info(f'Loaded {loaded}/{len(data)} result files')

    if loaded == 0:
        logger.error('No result files found. Run experiments first.')
        sys.exit(1)

    fmt = args.format
    outputs = []

    if fmt in ('markdown', 'both'):
        logger.info('Generating markdown report...')
        md_report = generate_report(data)
        if args.output and fmt != 'both':
            md_path = Path(args.output)
        else:
            md_path = OUTPUT_DIR / 'RESULTS_REPORT.md'
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_report)
        logger.info(f'Markdown written to {md_path}')
        outputs.append(('Markdown', md_path, md_report))

    if fmt in ('latex', 'both'):
        logger.info('Generating LaTeX report...')
        tex_report = generate_latex_report(data)
        if args.output and fmt == 'latex':
            tex_path = Path(args.output)
        else:
            tex_path = OUTPUT_DIR / 'RESULTS_REPORT.tex'
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(tex_report)
        logger.info(f'LaTeX written to {tex_path}')
        outputs.append(('LaTeX', tex_path, tex_report))

    print('\n' + '=' * 60)
    print('RESULTS REPORT GENERATED')
    print('=' * 60)
    for label, path, content in outputs:
        print(f'{label}: {path}  ({len(content.splitlines())} lines)')
    print('=' * 60)

    # Quick console summary
    bt = data.get('backtest')
    if bt:
        results = bt.get('results', {})
        geo = results.get('GeometricLongFlat', {})
        oos = geo.get('out_of_sample', {})
        fp = geo.get('full_period', {})
        print(f'\nGeometricLongFlat:')
        print(f'  Full:  Sharpe={_fmt(fp.get("sharpe"))}, MaxDD={_pct(fp.get("max_drawdown"))}')
        print(f'  OOS:   Sharpe={_fmt(oos.get("sharpe"))}, MaxDD={_pct(oos.get("max_drawdown"))}')
        spy_oos = results.get('BuyHoldSPY', {}).get('out_of_sample', {})
        print(f'\nBuy&Hold SPY OOS: Sharpe={_fmt(spy_oos.get("sharpe"))}, MaxDD={_pct(spy_oos.get("max_drawdown"))}')


if __name__ == '__main__':
    main()
