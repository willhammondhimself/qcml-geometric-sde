"""
Unified publication-quality style for all QCML figures.

Provides a centralized color palette, rcParams, colormaps, and helper
functions so that every figure—paper, poster, notebook—shares one
consistent "clean academic white" aesthetic (Physical Review / PNAS style).

Optionally layers on top of SciencePlots base styles for journal-specific
formatting (Nature, IEEE, APS).

Usage:
    from experiments.plot_style import apply_style, NAVY, TEAL, BURGUNDY, ...
    apply_style()                    # Default QCML style
    apply_style(journal='nature')    # Nature-style with QCML colors
    apply_style(journal='ieee')      # IEEE two-column style
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

try:
    import scienceplots  # noqa: F401
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False

# =============================================================================
# Color Palette — Physical Review / PNAS inspired
# =============================================================================

NAVY = '#1B2A4A'
TEAL = '#2D6A6A'
BURGUNDY = '#8B2252'
GOLD = '#C9A84C'
INDIGO = '#4B3F72'
SLATE = '#64748B'
LIGHT = '#F8F6F0'

# Semantic mapping: observable family → color
SEMANTIC_COLORS = {
    'metric': NAVY,
    'curvature': BURGUNDY,
    'spectral': TEAL,
    'purity': INDIGO,
    'topological': GOLD,
    'baseline': SLATE,
    'price': NAVY,
    'crisis': BURGUNDY,
}

# Method-level color mapping (for bar charts, legend consistency)
METHOD_COLORS = {
    'Berry Phase Rate': BURGUNDY,
    'QFI Determinant': NAVY,
    'Multi-Lag Fidelity': TEAL,
    'Spectral Gap': TEAL,
    'Reduced Purity': INDIGO,
    'Dim. Collapse': INDIGO,
    'QCML Chern': GOLD,
    'Geodesic Velocity': NAVY,
    'Speed Limit Ratio': TEAL,
    'Sect. Curv. Sign': BURGUNDY,
    'Spectral Entropy': TEAL,
    'Hamiltonian Sensitivity': NAVY,
    'Random Forest': SLATE,
    'Rolling Vol Z': SLATE,
    'CUSUM': SLATE,
    'GARCH(1,1)': SLATE,
    'Hamilton MS': SLATE,
    'EWMA': SLATE,
    'Mahalanobis': SLATE,
}

# Ordered cycle for when you need N distinct colors
COLOR_CYCLE = [NAVY, BURGUNDY, TEAL, GOLD, INDIGO, SLATE]

# =============================================================================
# Custom Colormaps
# =============================================================================

CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    'qcml_seq', ['#FFFFFF', '#C5D5E4', '#6B8EAD', NAVY], N=256,
)

CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    'qcml_div', [BURGUNDY, '#D4A0B0', '#FFFFFF', '#8FBFBF', TEAL], N=256,
)

# =============================================================================
# rcParams — "Clean Academic White"
# =============================================================================

RCPARAMS = {
    'font.family': 'serif',
    'font.size': 10,
    'mathtext.fontset': 'cm',
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'axes.edgecolor': NAVY,
    'axes.labelcolor': NAVY,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': SLATE,
    'grid.alpha': 0.15,
    'grid.linewidth': 0.5,
    'xtick.color': NAVY,
    'ytick.color': NAVY,
    'text.color': NAVY,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'figure.figsize': (6.5, 4.0),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}

FIGURE_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'


# Journal-specific SciencePlots style mappings
_JOURNAL_STYLES = {
    'default': ['science', 'no-latex'],
    'nature': ['science', 'nature', 'no-latex'],
    'ieee': ['science', 'ieee', 'no-latex'],
    'aps': ['science', 'no-latex'],  # APS uses science base
}


def apply_style(journal='default'):
    """Set matplotlib rcParams to the unified publication style.

    Args:
        journal: Base style from SciencePlots. Options:
            'default' — SciencePlots base + QCML overrides
            'nature'  — Nature journal formatting
            'ieee'    — IEEE two-column formatting
            'aps'     — APS (Physical Review) formatting
            If SciencePlots is not installed, falls back to QCML-only style.
    """
    if _HAS_SCIENCEPLOTS and journal in _JOURNAL_STYLES:
        plt.style.use(_JOURNAL_STYLES[journal])
    # QCML overrides always applied on top
    plt.rcParams.update(RCPARAMS)


# =============================================================================
# Helper Functions
# =============================================================================

def crisis_figure(n_rows, n_cols, crisis_start=None, crisis_end=None,
                  figsize=None, sharex=True, **kwargs):
    """Create a figure with optional crisis-band shading on every axes.

    Args:
        n_rows: Number of subplot rows.
        n_cols: Number of subplot columns.
        crisis_start: pandas.Timestamp for crisis start (optional).
        crisis_end: pandas.Timestamp for crisis end (optional).
        figsize: Explicit (w, h). Defaults to (6.5, 2.5*n_rows).
        sharex: Share x-axis across rows (default True).
        **kwargs: Forwarded to plt.subplots.

    Returns:
        (fig, axes) tuple.
    """
    apply_style()
    if figsize is None:
        figsize = (6.5, 2.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex, **kwargs)
    if crisis_start is not None and crisis_end is not None:
        ax_flat = np.atleast_1d(axes).flat
        for ax in ax_flat:
            ax.axvspan(crisis_start, crisis_end, alpha=0.12, color=GOLD, zorder=0)
    return fig, axes


def format_date_axis(ax, interval_months=3):
    """Format x-axis with month ticks rotated 45 degrees.

    Args:
        ax: Matplotlib axes.
        interval_months: Months between major ticks (default 3).
    """
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval_months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')


def save_figure(fig, name, formats=('pdf', 'png'), output_dir=None):
    """Save figure to paper/figures/ in the given formats.

    Args:
        fig: Matplotlib figure.
        name: Filename stem (no extension).
        formats: Iterable of format strings.
        output_dir: Override output directory (default paper/figures/).
    """
    out = Path(output_dir) if output_dir else FIGURE_DIR
    out.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out / f'{name}.{fmt}'
        fig.savefig(path)
    plt.close(fig)


def comparison_barchart(methods, values, ci_low=None, ci_high=None,
                        title=None, xlabel=None, figsize=(6.5, 5)):
    """Ranked horizontal bar chart with optional CI whiskers.

    Args:
        methods: List of method names (sorted by values descending).
        values: Corresponding values (e.g. median Cohen's d).
        ci_low: Lower error bar (value - ci_low). Optional.
        ci_high: Upper error bar (ci_high - value). Optional.
        title: Plot title.
        xlabel: X-axis label.
        figsize: Figure size.

    Returns:
        (fig, ax) tuple.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    colors = [METHOD_COLORS.get(m, SLATE) for m in methods]

    if ci_low is not None and ci_high is not None:
        ax.barh(range(len(methods)), values,
                xerr=[ci_low, ci_high],
                color=colors, edgecolor='white', linewidth=0.5,
                capsize=3, alpha=0.85)
    else:
        ax.barh(range(len(methods)), values,
                color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.invert_yaxis()
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    return fig, ax


def heatmap_figure(data, row_labels, col_labels, title=None,
                   cmap=None, vmin=None, vmax=None, figsize=None,
                   annotate=True, cbar_label=None):
    """Publication-quality heatmap.

    Args:
        data: 2D numpy array.
        row_labels: List of row labels.
        col_labels: List of column labels.
        title: Plot title.
        cmap: Colormap (default CMAP_SEQUENTIAL).
        vmin, vmax: Color scale bounds.
        figsize: Figure size.
        annotate: If True, write values in cells.
        cbar_label: Colorbar label.

    Returns:
        (fig, ax) tuple.
    """
    apply_style()
    if cmap is None:
        cmap = CMAP_SEQUENTIAL
    if figsize is None:
        figsize = (max(6.5, len(col_labels) * 0.9), max(4, len(row_labels) * 0.45))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    if annotate:
        thresh = (data[np.isfinite(data)].max() + data[np.isfinite(data)].min()) / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    color = 'white' if val > thresh else NAVY
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if cbar_label:
        cbar.set_label(cbar_label)
    if title:
        ax.set_title(title)
    return fig, ax
