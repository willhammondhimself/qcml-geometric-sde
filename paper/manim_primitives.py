"""
QCML Manim Primitives — Reusable building blocks for explainer videos.
=====================================================================
Scene primitives, color palette, and data binding helpers used by
generated Manim scripts. Not standalone scenes — these are imported
by scene files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    Arrow,
    BraceBetweenPoints,
    Create,
    Dot,
    Ellipse,
    Circle,
    FadeIn,
    GrowArrow,
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    VGroup,
    Axes,
    BOLD,
)

# ── Color palette ─────────────────────────────────────────────────────────
QCML_COLORS = {
    "CALM_BLUE": "#2196F3",
    "CRISIS_RED": "#F44336",
    "HIGHLIGHT_GOLD": "#FFD700",
    "DARK_BG": "#1a1a2e",
    "ACCENT_TEAL": "#00BCD4",
    "SOFT_WHITE": "#E0E0E0",
    "DIM_GRAY": "#666666",
}

# Module-level aliases for direct import
CALM_BLUE = QCML_COLORS["CALM_BLUE"]
CRISIS_RED = QCML_COLORS["CRISIS_RED"]
HIGHLIGHT_GOLD = QCML_COLORS["HIGHLIGHT_GOLD"]
DARK_BG = QCML_COLORS["DARK_BG"]
ACCENT_TEAL = QCML_COLORS["ACCENT_TEAL"]
SOFT_WHITE = QCML_COLORS["SOFT_WHITE"]
DIM_GRAY = QCML_COLORS["DIM_GRAY"]


# ── Scene setup ───────────────────────────────────────────────────────────

def dark_scene_setup(scene, title: str, title_color: str = HIGHLIGHT_GOLD) -> Text:
    """Set dark background and create animated section header.

    Args:
        scene: The Manim Scene instance.
        title: Header text to display.
        title_color: Color for the header (default: HIGHLIGHT_GOLD).

    Returns:
        The header Text mobject (already added to scene).
    """
    scene.camera.background_color = DARK_BG
    header = Text(title, font_size=36, color=title_color)
    header.to_edge(UP, buff=0.4)
    scene.play(FadeIn(header), run_time=0.6)
    return header


# ── Formula reveal ────────────────────────────────────────────────────────

def formula_reveal(
    scene,
    tex_strings: list[str],
    annotations: list[dict] | None = None,
    title: str | None = None,
    position=None,
    font_size: int = 44,
    run_time: float = 2.0,
) -> MathTex:
    """Animate a formula appearing with optional labeled annotations.

    Args:
        scene: The Manim Scene instance.
        tex_strings: List of LaTeX strings (separate for coloring individual parts).
        annotations: List of dicts with keys:
            - index (int): Index into tex_strings to annotate.
            - text (str): Annotation text.
            - color (str): Color for annotation and arrow.
        title: Optional title text above the formula.
        position: Optional position (default: UP * 1.5).
        font_size: Font size for the formula.
        run_time: Duration of the Write animation.

    Returns:
        The MathTex mobject.
    """
    from manim import Write

    formula = MathTex(*tex_strings, font_size=font_size)
    if position is not None:
        formula.move_to(position)
    else:
        formula.shift(UP * 1.5)

    scene.play(Write(formula), run_time=run_time)
    scene.wait(0.5)

    if annotations:
        ann_mobjects = []
        for ann in annotations:
            idx = ann["index"]
            color = ann.get("color", ACCENT_TEAL)
            formula[idx].set_color(color)

            note = Text(ann["text"], font_size=16, color=color)
            note.next_to(formula[idx], DOWN, buff=0.4)
            arr = Arrow(
                note.get_top(),
                formula[idx].get_bottom(),
                buff=0.05,
                color=color,
                stroke_width=1,
                max_tip_length_to_length_ratio=0.2,
            )
            scene.play(FadeIn(note), GrowArrow(arr), run_time=0.7)
            ann_mobjects.extend([note, arr])

        scene.wait(1)
        from manim import FadeOut

        scene.play(*[FadeOut(m) for m in ann_mobjects], run_time=0.5)

    return formula


# ── Eigenvalue bars ───────────────────────────────────────────────────────

def eigenvalue_bars(
    scene,
    values: list[float],
    position=None,
    highlight_idx: int | None = 0,
    highlight_label: str | None = None,
    bar_color: str = ACCENT_TEAL,
    highlight_color: str = HIGHLIGHT_GOLD,
    bar_width: float = 2.4,
    y_scale: float = 0.5,
) -> tuple[VGroup, VGroup]:
    """Create animated horizontal energy level bars.

    Args:
        scene: The Manim Scene instance.
        values: List of eigenvalue positions (y-axis).
        position: Base position for the bar stack.
        highlight_idx: Index of bar to highlight (default: 0 = ground state).
        highlight_label: LaTeX label for the highlighted bar.
        bar_color: Color for normal bars.
        highlight_color: Color for highlight box and label.
        bar_width: Width of each bar line.
        y_scale: Vertical scaling factor.

    Returns:
        Tuple of (bars VGroup, bar_labels VGroup).
    """
    if position is None:
        position = LEFT * 3.5

    bars = VGroup()
    bar_labels = VGroup()
    base_y = -2.8

    for i, ev in enumerate(values):
        y = base_y + ev * y_scale
        color = DIM_GRAY if i >= 2 else bar_color
        width = 3 if i >= 2 else 4
        bar = Line(
            LEFT * (bar_width / 2),
            RIGHT * (bar_width / 2),
            color=color,
            stroke_width=width,
        )
        bar.move_to(position + UP * y)
        label = MathTex(f"E_{i}", font_size=18, color=SOFT_WHITE)
        label.next_to(bar, RIGHT, buff=0.2)
        bars.add(bar)
        bar_labels.add(label)

    scene.play(
        LaggedStart(*[Create(b) for b in bars], lag_ratio=0.15),
        LaggedStart(*[FadeIn(lbl) for lbl in bar_labels], lag_ratio=0.15),
        run_time=1.5,
    )

    if highlight_idx is not None and highlight_idx < len(bars):
        gs_box = SurroundingRectangle(
            bars[highlight_idx], color=highlight_color, buff=0.1, stroke_width=2
        )
        if highlight_label:
            gs_label = MathTex(highlight_label, font_size=22, color=highlight_color)
            gs_label.next_to(gs_box, LEFT, buff=0.3)
            scene.play(Create(gs_box), FadeIn(gs_label), run_time=0.8)
        else:
            scene.play(Create(gs_box), run_time=0.8)
        scene.wait(0.5)

    return bars, bar_labels


# ── Spectral gap bracket ─────────────────────────────────────────────────

def spectral_gap_bracket(
    scene,
    bar0,
    bar1,
    label_tex: str = r"\Delta = E_1 - E_0",
    color: str = CRISIS_RED,
    direction=RIGHT,
    offset: float = 1.5,
) -> tuple:
    """Animate a brace between two bars showing the spectral gap.

    Args:
        scene: The Manim Scene instance.
        bar0: Lower bar mobject.
        bar1: Upper bar mobject.
        label_tex: LaTeX for the gap label.
        color: Color for brace and label.
        direction: Direction for the brace (LEFT or RIGHT).
        offset: How far from center to place the brace.

    Returns:
        Tuple of (brace, label) mobjects.
    """
    brace = BraceBetweenPoints(
        bar1.get_center() + direction * offset,
        bar0.get_center() + direction * offset,
        direction=direction,
        color=color,
    )
    label = MathTex(label_tex, font_size=22, color=color)
    label.next_to(brace, direction, buff=0.2)
    scene.play(Create(brace), FadeIn(label), run_time=0.8)
    return brace, label


# ── Pipeline diagram ──────────────────────────────────────────────────────

def pipeline_box(text: str, pos, color: str = ACCENT_TEAL) -> VGroup:
    """Create a rounded rectangle box with centered label.

    Args:
        text: Label text.
        pos: Center position.
        color: Border color.

    Returns:
        VGroup containing the box and label.
    """
    box = RoundedRectangle(
        width=2.0,
        height=0.7,
        corner_radius=0.15,
        color=color,
        fill_opacity=0.15,
        stroke_width=1.5,
    ).move_to(pos)
    label = Text(text, font_size=15, color=WHITE).move_to(pos)
    return VGroup(box, label)


def pipeline_diagram(
    scene,
    steps: list[tuple[str, str]],
    start_pos=None,
    spacing: float = 2.5,
) -> VGroup:
    """Animate a horizontal boxes-and-arrows pipeline.

    Args:
        scene: The Manim Scene instance.
        steps: List of (label, color) tuples for each box.
        start_pos: Position of first box center.
        spacing: Horizontal spacing between box centers.

    Returns:
        VGroup of all boxes and arrows.
    """
    if start_pos is None:
        start_pos = LEFT * 4.0

    arr_kw = dict(
        color=DIM_GRAY,
        stroke_width=1.5,
        max_tip_length_to_length_ratio=0.15,
    )

    all_parts = VGroup()
    prev_box = None

    for i, (label, color) in enumerate(steps):
        pos = start_pos + RIGHT * (i * spacing)
        box = pipeline_box(label, pos, color)

        if prev_box is not None:
            arr = Arrow(prev_box.get_right(), box.get_left(), buff=0.1, **arr_kw)
            scene.play(GrowArrow(arr), FadeIn(box), run_time=0.7)
            all_parts.add(arr)
        else:
            scene.play(FadeIn(box), run_time=0.7)

        all_parts.add(box)
        prev_box = box

    return all_parts


# ── Price chart ───────────────────────────────────────────────────────────

def price_chart(
    scene,
    n: int = 200,
    position=None,
    seed: int = 42,
    crisis_frac: tuple[float, float] = (0.6, 0.75),
) -> tuple[Axes, VGroup]:
    """Create an animated price chart with crisis coloring.

    Args:
        scene: The Manim Scene instance.
        n: Number of data points.
        position: Center position for the axes.
        seed: Random seed for reproducibility.
        crisis_frac: (start, end) fractions defining the crisis period.

    Returns:
        Tuple of (axes, lines VGroup).
    """
    if position is None:
        position = LEFT * 4.5 + DOWN * 0.5

    np.random.seed(seed)
    t = np.linspace(0, 1, n)
    cs, ce = crisis_frac
    log_price = np.cumsum(
        np.where(t < cs, 0.002, np.where(t < ce, -0.015, 0.004))
        + 0.01 * np.random.randn(n)
    )
    price = np.exp(log_price - log_price[0]) * 100

    ax = Axes(
        x_range=[0, n, 50],
        y_range=[70, 130, 10],
        x_length=4,
        y_length=2.5,
        tips=False,
        axis_config={"color": SOFT_WHITE, "stroke_width": 1},
    ).move_to(position)

    crisis_start, crisis_end = int(cs * n), int(ce * n)
    calm1 = ax.plot_line_graph(
        x_values=list(range(crisis_start)),
        y_values=price[:crisis_start].tolist(),
        add_vertex_dots=False,
        line_color=CALM_BLUE,
        stroke_width=2,
    )
    crisis_line = ax.plot_line_graph(
        x_values=list(range(crisis_start, crisis_end)),
        y_values=price[crisis_start:crisis_end].tolist(),
        add_vertex_dots=False,
        line_color=CRISIS_RED,
        stroke_width=2,
    )
    calm2 = ax.plot_line_graph(
        x_values=list(range(crisis_end, n)),
        y_values=price[crisis_end:].tolist(),
        add_vertex_dots=False,
        line_color=CALM_BLUE,
        stroke_width=2,
    )

    price_label = Text("SPY price", font_size=16, color=SOFT_WHITE)
    price_label.next_to(ax, UP, buff=0.15)

    scene.play(Create(ax), FadeIn(price_label), run_time=0.8)
    scene.play(Create(calm1), run_time=1.0)
    scene.play(Create(crisis_line), run_time=0.6)
    scene.play(Create(calm2), run_time=0.6)

    lines = VGroup(calm1, crisis_line, calm2, price_label)
    return ax, lines


# ── Sphere projection ────────────────────────────────────────────────────

def sphere_projection(
    scene,
    center=None,
    n_dots: int = 40,
    crisis_frac: tuple[float, float] = (0.55, 0.75),
    radius: float = 1.5,
) -> VGroup:
    """Create a 2D sphere projection with colored dots.

    Args:
        scene: The Manim Scene instance.
        center: Center position for the sphere.
        n_dots: Number of dots to place.
        crisis_frac: Fraction range for crisis-colored dots.
        radius: Radius of the circle.

    Returns:
        VGroup of circle, ellipse, dots, labels, and legend.
    """
    if center is None:
        center = np.array([1.0, -2.0, 0.0])

    circle = Circle(radius=radius, color=SOFT_WHITE, stroke_width=1).move_to(center)
    ellipse = Ellipse(
        width=radius * 2, height=radius * 0.53, color=SOFT_WHITE, stroke_width=0.5
    ).move_to(center)
    sphere_label = Text("S^{n-1}", font_size=20, color=HIGHLIGHT_GOLD)
    sphere_label.next_to(circle, DOWN, buff=0.2)

    scene.play(Create(circle), Create(ellipse), FadeIn(sphere_label), run_time=1.0)

    angles = np.linspace(0.3, 5.5, n_dots)
    dots = VGroup()
    cs, ce = crisis_frac
    for i, angle in enumerate(angles):
        frac = i / n_dots
        r = radius * (0.85 + 0.15 * np.sin(3 * angle))
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle) * 0.5
        color = CRISIS_RED if cs < frac < ce else CALM_BLUE
        dot = Dot(point=[x, y, 0], radius=0.04, color=color)
        dots.add(dot)

    scene.play(
        LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.04),
        run_time=2.0,
    )

    legend = VGroup(
        VGroup(
            Dot(color=CALM_BLUE, radius=0.06),
            Text("calm", font_size=14, color=CALM_BLUE),
        ).arrange(RIGHT, buff=0.15),
        VGroup(
            Dot(color=CRISIS_RED, radius=0.06),
            Text("crisis", font_size=14, color=CRISIS_RED),
        ).arrange(RIGHT, buff=0.15),
    ).arrange(RIGHT, buff=0.6)
    legend.next_to(circle, RIGHT, buff=0.5).shift(DOWN * 0.3)
    scene.play(FadeIn(legend), run_time=0.5)

    return VGroup(circle, ellipse, sphere_label, dots, legend)


# ── Multi-panel time series ──────────────────────────────────────────────

def multi_panel_timeseries(
    scene,
    families: list[tuple[str, list[float]]],
    cols: int = 3,
    x_spacing: float = 3.5,
    y_spacing: float = 2.5,
    origin_offset=None,
) -> VGroup:
    """Create a grid of mini time series panels with crisis shading.

    Args:
        scene: The Manim Scene instance.
        families: List of (name, values) tuples.
        cols: Number of columns in the grid.
        x_spacing: Horizontal spacing.
        y_spacing: Vertical spacing.
        origin_offset: Top-left offset.

    Returns:
        VGroup of all panels.
    """
    if origin_offset is None:
        origin_offset = np.array([-3.5, 0.5, 0.0])

    panels = VGroup()
    for i, (name, vals) in enumerate(families):
        row = i // cols
        col = i % cols
        center = origin_offset + np.array([col * x_spacing, -row * y_spacing, 0])

        n_pts = len(vals)
        ax = Axes(
            x_range=[0, n_pts - 1, 1],
            y_range=[0, 1.1, 0.5],
            x_length=2.8,
            y_length=1.5,
            tips=False,
            axis_config={"color": DIM_GRAY, "stroke_width": 0.8},
        ).move_to(center)

        # Crisis shading (middle portion)
        shade_width = 2.8 * 4 / (n_pts - 1) if n_pts > 1 else 2.8
        shade = Rectangle(
            width=shade_width,
            height=1.5,
            fill_color=CRISIS_RED,
            fill_opacity=0.08,
            stroke_width=0,
        ).move_to(ax.c2p((n_pts - 1) / 2, 0.55))

        line = ax.plot_line_graph(
            x_values=list(range(n_pts)),
            y_values=vals,
            add_vertex_dots=False,
            line_color=ACCENT_TEAL,
            stroke_width=2,
        )

        label = Text(name, font_size=15, color=HIGHLIGHT_GOLD)
        label.next_to(ax, UP, buff=0.08)

        panel = VGroup(shade, ax, line, label)
        panels.add(panel)

    scene.play(
        LaggedStart(*[FadeIn(p, shift=UP * 0.2) for p in panels], lag_ratio=0.15),
        run_time=3.0,
    )

    return panels


# ── Animate gap closure ──────────────────────────────────────────────────

def animate_gap_closure(
    scene,
    bar0,
    bar1,
    label_e1,
    target_gap: float = 0.15,
    base_y: float = -1.5,
    e0_val: float = 0.5,
    bar_x=None,
    run_time: float = 2.5,
) -> tuple:
    """Smoothly animate E1 bar closing toward E0.

    Args:
        scene: The Manim Scene instance.
        bar0: Ground state bar.
        bar1: First excited state bar.
        label_e1: Label mobject for E1.
        target_gap: Final gap between E0 and E1.
        base_y: Base y position.
        e0_val: E0 eigenvalue.
        bar_x: Horizontal position of bars.
        run_time: Animation duration.

    Returns:
        Tuple of (new_brace, new_label) for the closed gap.
    """
    from manim import Transform

    if bar_x is None:
        bar_x = LEFT * 3.0

    e1_target_y = base_y + (e0_val + target_gap) * 0.8
    new_bar1_pos = bar_x + UP * e1_target_y

    new_brace = BraceBetweenPoints(
        new_bar1_pos + LEFT * 1.3 + UP * 0.08,
        bar0.get_center() + LEFT * 1.3,
        direction=LEFT,
        color=CRISIS_RED,
    )
    tiny_delta = MathTex(r"\Delta \to 0", font_size=22, color=CRISIS_RED)
    tiny_delta.next_to(new_brace, LEFT, buff=0.15)

    return new_bar1_pos, new_brace, tiny_delta


# ── Data binding ──────────────────────────────────────────────────────────

def load_canonical_values(json_path: str, bindings: list[dict]) -> dict[str, str]:
    """Load values from canonical JSON for storyboard data_bindings.

    Args:
        json_path: Path to the canonical JSON file.
        bindings: List of binding dicts with keys:
            - field (str): Dot-separated path into the JSON (e.g.,
              "method_results.Berry Phase Rate.median_d").
            - display (str): Format template (e.g., "d = {value:.3f}").
            - key (str, optional): Key name for the returned dict.
              Defaults to the field name.

    Returns:
        Dict mapping binding keys to formatted display strings.

    Example:
        >>> bindings = [
        ...     {"field": "method_results.Berry Phase Rate.median_d",
        ...      "display": "d = {value:.3f}", "key": "berry_d"}
        ... ]
        >>> vals = load_canonical_values("path/to/results.json", bindings)
        >>> vals["berry_d"]
        'd = 0.437'
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Canonical JSON not found: {json_path}")

    with open(path) as f:
        data = json.load(f)

    result = {}
    for binding in bindings:
        field = binding["field"]
        display_fmt = binding.get("display", "{value}")
        key = binding.get("key", field)

        # Navigate dot-separated path
        value = _resolve_json_path(data, field)
        if value is not None:
            result[key] = display_fmt.format(value=value)
        else:
            result[key] = f"[missing: {field}]"

    return result


def _resolve_json_path(data: dict, path: str) -> Any | None:
    """Navigate a dot-separated path through nested dicts/lists."""
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if current is None:
            return None
    return current


def data_bound_text(
    template: str,
    json_path: str,
    field: str,
    font_size: int = 24,
    color: str = SOFT_WHITE,
    use_math: bool = False,
) -> Text | MathTex:
    """Create a Text or MathTex mobject with a live value from canonical JSON.

    Args:
        template: Format string with {value} placeholder.
        json_path: Path to canonical JSON.
        field: Dot-separated field path.
        font_size: Font size.
        color: Text color.
        use_math: If True, return MathTex instead of Text.

    Returns:
        Text or MathTex with the resolved value.
    """
    vals = load_canonical_values(json_path, [{"field": field, "display": template}])
    text = vals[field]

    if use_math:
        return MathTex(text, font_size=font_size, color=color)
    return Text(text, font_size=font_size, color=color)
