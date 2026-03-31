"""
QCML Geometric Observatory — Manim Explainer Animation
=======================================================
6 scenes (~4-5 min total) explaining the geometric regime detection framework.

Render:
    manim -pql paper/manim_qcml_explainer.py           # low quality preview
    manim -pqh paper/manim_qcml_explainer.py           # high quality
    manim -pqh paper/manim_qcml_explainer.py SceneName # single scene
"""

from manim import *
import numpy as np
import sys
from pathlib import Path

# Ensure the paper/ directory is importable when manim loads this file directly
_paper_dir = str(Path(__file__).resolve().parent)
if _paper_dir not in sys.path:
    sys.path.insert(0, _paper_dir)

from manim_primitives import (
    CALM_BLUE,
    CRISIS_RED,
    HIGHLIGHT_GOLD,
    DARK_BG,
    ACCENT_TEAL,
    SOFT_WHITE,
    DIM_GRAY,
    dark_scene_setup,
    formula_reveal,
    eigenvalue_bars,
    spectral_gap_bracket,
    pipeline_diagram,
    price_chart,
    sphere_projection,
    multi_panel_timeseries,
    animate_gap_closure,
    pipeline_box,
)


# ═══════════════════════════════════════════════════════════════════════════
# Scene 1: Title
# ═══════════════════════════════════════════════════════════════════════════
class TitleScene(Scene):
    """Title card — ~10s."""

    def construct(self):
        self.camera.background_color = DARK_BG

        title = Text(
            "The Geometry of Financial Crises",
            font_size=52,
            color=WHITE,
            weight=BOLD,
        )
        subtitle = Text(
            "How curvature detects regime transitions",
            font_size=28,
            color=ACCENT_TEAL,
        )
        subtitle.next_to(title, DOWN, buff=0.5)

        rule = Line(LEFT * 4, RIGHT * 4, color=HIGHLIGHT_GOLD, stroke_width=2)
        rule.next_to(subtitle, DOWN, buff=0.4)

        attribution = Text(
            "Hammond & QCML  •  2026",
            font_size=18,
            color=DIM_GRAY,
        )
        attribution.next_to(rule, DOWN, buff=0.4)

        self.play(FadeIn(title, shift=UP * 0.3), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=1.0)
        self.play(Create(rule), FadeIn(attribution), run_time=0.8)
        self.wait(2)
        self.play(FadeOut(Group(title, subtitle, rule, attribution)), run_time=1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Scene 2: Pipeline — raw prices → features → sphere
# ═══════════════════════════════════════════════════════════════════════════
class PipelineScene(Scene):
    """Data pipeline: prices → features → PCA → unit sphere. ~45s."""

    def construct(self):
        header = dark_scene_setup(self, "The Data Pipeline")

        # ── Price chart (left) ────────────────────────────────────────
        ax_price, price_lines = price_chart(self, position=LEFT * 4.5 + DOWN * 0.5)

        # ── Pipeline arrows and boxes ─────────────────────────────────
        arr_kw = dict(
            color=DIM_GRAY, stroke_width=1.5, max_tip_length_to_length_ratio=0.15,
        )
        a1_start = ax_price.get_right() + RIGHT * 0.2
        box1 = pipeline_box("13 features", a1_start + RIGHT * 1.3)
        arr1 = Arrow(a1_start, box1.get_left(), buff=0.1, **arr_kw)

        box2 = pipeline_box("PCA → 15 dim", box1.get_center() + RIGHT * 2.5)
        arr2 = Arrow(box1.get_right(), box2.get_left(), buff=0.1, **arr_kw)

        box3 = pipeline_box("Unit sphere", box2.get_center() + RIGHT * 2.5, color=HIGHLIGHT_GOLD)
        arr3 = Arrow(box2.get_right(), box3.get_left(), buff=0.1, **arr_kw)

        self.play(GrowArrow(arr1), FadeIn(box1), run_time=0.7)
        self.play(GrowArrow(arr2), FadeIn(box2), run_time=0.7)
        self.play(GrowArrow(arr3), FadeIn(box3), run_time=0.7)

        # ── Sphere projection ─────────────────────────────────────────
        sphere_parts = sphere_projection(self, center=np.array([1.0, -2.0, 0.0]))

        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════════
# Scene 3: Hamiltonian + spectral gap
# ═══════════════════════════════════════════════════════════════════════════
class HamiltonianScene(Scene):
    """H(x) formula, eigenvalue bars, spectral gap. ~60s."""

    def construct(self):
        header = dark_scene_setup(self, "The Data Hamiltonian")

        # ── Formula ───────────────────────────────────────────────────
        formula = formula_reveal(
            self,
            tex_strings=[
                r"H(\mathbf{x})", r"=", r"\frac{1}{2}",
                r"\sum_{k=1}^{K}", r"(A_k", r"-", r"x_k I)^2",
            ],
            annotations=[
                {"index": 0, "text": "", "color": HIGHLIGHT_GOLD},
                {"index": 4, "text": "observable matrices", "color": ACCENT_TEAL},
                {"index": 6, "text": "market data point", "color": CALM_BLUE},
            ],
        )

        # ── Eigenvalue bars ───────────────────────────────────────────
        eig_label = Text("Energy levels (eigenvalues)", font_size=18, color=SOFT_WHITE)
        eig_label.shift(DOWN * 0.5 + LEFT * 3.5)
        self.play(FadeIn(eig_label), run_time=0.5)

        bars, bar_labels = eigenvalue_bars(
            self,
            values=[0.3, 1.1, 2.0, 3.2, 4.8],
            highlight_idx=0,
            highlight_label=r"|\psi_0(\mathbf{x})\rangle",
        )

        # ── Spectral gap bracket ──────────────────────────────────────
        gap_brace, gap_label = spectral_gap_bracket(self, bars[0], bars[1])

        # ── Explanation text ──────────────────────────────────────────
        explain = Text(
            "The spectral gap controls sensitivity to regime change",
            font_size=18,
            color=SOFT_WHITE,
        ).shift(DOWN * 0.2 + RIGHT * 1.5)
        self.play(FadeIn(explain), run_time=0.8)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════════
# Scene 4: Ground state evolution on the sphere
# ═══════════════════════════════════════════════════════════════════════════
class GroundStateEvolutionScene(ThreeDScene):
    """Ground state traces a path; curvature spikes in crisis. ~60s."""

    def construct(self):
        self.camera.background_color = DARK_BG

        # ── 2D header (before 3D camera move) ─────────────────────────
        header = Text("Ground State Evolution", font_size=36, color=HIGHLIGHT_GOLD)
        header.to_corner(UL, buff=0.4)
        self.add_fixed_in_frame_mobjects(header)
        self.play(FadeIn(header), run_time=0.6)

        # ── Build sphere ──────────────────────────────────────────────
        sphere = Surface(
            lambda u, v: np.array([
                1.8 * np.cos(u) * np.cos(v),
                1.8 * np.cos(u) * np.sin(v),
                1.8 * np.sin(u),
            ]),
            u_range=[-PI / 2, PI / 2],
            v_range=[0, TAU],
            resolution=(24, 48),
            fill_opacity=0.08,
            stroke_width=0.3,
            stroke_color=DIM_GRAY,
            checkerboard_colors=[DARK_BG, "#222244"],
        )

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(sphere), run_time=1.5)

        # ── Ground state trajectory ───────────────────────────────────
        n_pts = 300
        t = np.linspace(0, 4 * PI, n_pts)
        crisis_mask = (t > 7.5) & (t < 9.5)

        theta_path = 0.5 * t + np.where(crisis_mask, 2.5 * np.sin(8 * t), 0)
        phi_path = 0.3 * np.sin(0.4 * t) + np.where(crisis_mask, 1.2 * np.sin(6 * t), 0)

        points = []
        for i in range(n_pts):
            r = 1.8
            x = r * np.cos(phi_path[i]) * np.cos(theta_path[i])
            y = r * np.cos(phi_path[i]) * np.sin(theta_path[i])
            z = r * np.sin(phi_path[i])
            points.append([x, y, z])

        calm_pts_1 = [points[i] for i in range(n_pts) if not crisis_mask[i] and t[i] <= 7.5]
        crisis_pts = [points[i] for i in range(n_pts) if crisis_mask[i]]
        calm_pts_2 = [points[i] for i in range(n_pts) if not crisis_mask[i] and t[i] > 9.5]

        def make_curve(pts, color, width=2.5):
            if len(pts) < 2:
                return VGroup()
            return VMobject(color=color, stroke_width=width).set_points_smoothly(
                [np.array(p) for p in pts]
            )

        calm_curve1 = make_curve(calm_pts_1, CALM_BLUE)
        crisis_curve = make_curve(crisis_pts, CRISIS_RED, width=3.5)
        calm_curve2 = make_curve(calm_pts_2, CALM_BLUE)

        self.play(Create(calm_curve1), run_time=2.5)
        self.play(Create(crisis_curve), run_time=1.5)
        self.play(Create(calm_curve2), run_time=1.5)

        # ── Curvature side panel (fixed in frame) ─────────────────────
        curv_ax = Axes(
            x_range=[0, n_pts, 100],
            y_range=[0, 5, 1],
            x_length=3.5,
            y_length=1.8,
            tips=False,
            axis_config={"color": SOFT_WHITE, "stroke_width": 1},
        )
        curv_ax.to_corner(DR, buff=0.5)

        curv_label = Text("Berry curvature", font_size=14, color=CRISIS_RED)
        curv_label.next_to(curv_ax, UP, buff=0.1)

        curvature = np.zeros(n_pts)
        for i in range(1, n_pts - 1):
            v1 = np.array(points[i]) - np.array(points[i - 1])
            v2 = np.array(points[i + 1]) - np.array(points[i])
            curvature[i] = np.linalg.norm(v2 - v1) * 30

        curv_line = curv_ax.plot_line_graph(
            x_values=list(range(n_pts)),
            y_values=np.clip(curvature, 0, 5).tolist(),
            add_vertex_dots=False,
            line_color=CRISIS_RED,
            stroke_width=1.5,
        )

        self.add_fixed_in_frame_mobjects(curv_ax, curv_label, curv_line)
        self.play(Create(curv_ax), FadeIn(curv_label), run_time=0.8)
        self.play(Create(curv_line), run_time=2.0)

        explanation = Text(
            "Berry curvature = how sharply\nthe quantum state path bends",
            font_size=14,
            color=SOFT_WHITE,
            line_spacing=1.3,
        )
        explanation.next_to(curv_ax, LEFT, buff=0.4)
        self.add_fixed_in_frame_mobjects(explanation)
        self.play(FadeIn(explanation), run_time=0.8)

        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════════
# Scene 5: Observatory — 6 channel families
# ═══════════════════════════════════════════════════════════════════════════
class ObservatoryScene(Scene):
    """19 channels across 6 families, each lighting up at different times. ~45s."""

    def construct(self):
        header = dark_scene_setup(self, "The Geometric Observatory")

        subtitle = Text(
            "19 channels — each sees a different geometric property",
            font_size=18,
            color=SOFT_WHITE,
        )
        subtitle.next_to(header, DOWN, buff=0.25)
        self.play(FadeIn(subtitle), run_time=0.5)

        # ── 6 mini time series panels ─────────────────────────────────
        families = [
            ("Holonomy", [0.1, 0.1, 0.2, 0.8, 1.0, 0.7, 0.2, 0.1]),
            ("Metric", [0.1, 0.2, 0.6, 1.0, 0.9, 0.4, 0.1, 0.1]),
            ("State", [0.1, 0.1, 0.1, 0.3, 0.9, 1.0, 0.6, 0.2]),
            ("Kinematics", [0.2, 0.5, 1.0, 0.8, 0.3, 0.1, 0.1, 0.1]),
            ("Spectral", [0.1, 0.1, 0.3, 0.7, 1.0, 0.8, 0.3, 0.1]),
            ("Curvature", [0.1, 0.3, 0.8, 1.0, 0.6, 0.2, 0.1, 0.1]),
        ]

        panels = multi_panel_timeseries(self, families)
        self.wait(1)

        # ── Highlight complementarity ─────────────────────────────────
        for panel in panels:
            box = SurroundingRectangle(panel, color=HIGHLIGHT_GOLD, buff=0.1, stroke_width=2)
            self.play(Create(box), run_time=0.3)
            self.play(FadeOut(box), run_time=0.2)

        complementary = Text(
            "Different channels spike at different times → complementary detection",
            font_size=18,
            color=HIGHLIGHT_GOLD,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(complementary), run_time=0.8)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════════
# Scene 6: Dead signals — spectral gap closure
# ═══════════════════════════════════════════════════════════════════════════
class DeadSignalScene(Scene):
    """Gap closure kills geometry; theory predicts which channels fail. ~40s."""

    def construct(self):
        header = dark_scene_setup(self, "When Geometry Breaks", title_color=CRISIS_RED)

        # ── Eigenvalue bars (animated gap closure) ────────────────────
        bar_x = LEFT * 3.0
        base_y = -1.5
        e0_start, e1_start = 0.5, 2.0

        bar0 = Line(LEFT * 1.0, RIGHT * 1.0, color=CALM_BLUE, stroke_width=4)
        bar0.move_to(bar_x + UP * (base_y + e0_start * 0.8))
        bar1 = Line(LEFT * 1.0, RIGHT * 1.0, color=ACCENT_TEAL, stroke_width=4)
        bar1.move_to(bar_x + UP * (base_y + e1_start * 0.8))
        bar2 = Line(LEFT * 1.0, RIGHT * 1.0, color=DIM_GRAY, stroke_width=2)
        bar2.move_to(bar_x + UP * (base_y + 3.5 * 0.8))
        bar3 = Line(LEFT * 1.0, RIGHT * 1.0, color=DIM_GRAY, stroke_width=2)
        bar3.move_to(bar_x + UP * (base_y + 4.5 * 0.8))

        label_e0 = MathTex("E_0", font_size=20, color=CALM_BLUE).next_to(bar0, RIGHT, buff=0.2)
        label_e1 = MathTex("E_1", font_size=20, color=ACCENT_TEAL).next_to(bar1, RIGHT, buff=0.2)

        gap_brace = BraceBetweenPoints(
            bar1.get_center() + LEFT * 1.3,
            bar0.get_center() + LEFT * 1.3,
            direction=LEFT,
            color=HIGHLIGHT_GOLD,
        )
        gap_text = MathTex(r"\Delta", font_size=24, color=HIGHLIGHT_GOLD)
        gap_text.next_to(gap_brace, LEFT, buff=0.15)

        self.play(
            Create(bar0), Create(bar1), Create(bar2), Create(bar3),
            FadeIn(label_e0), FadeIn(label_e1),
            Create(gap_brace), FadeIn(gap_text),
            run_time=1.2,
        )
        self.wait(0.5)

        healthy_label = Text("Healthy gap", font_size=16, color=HIGHLIGHT_GOLD)
        healthy_label.next_to(gap_brace, DOWN, buff=0.3)
        self.play(FadeIn(healthy_label), run_time=0.5)
        self.wait(1)

        # ── Animate gap closing ───────────────────────────────────────
        closing_label = Text("Gap closing...", font_size=16, color=CRISIS_RED)
        closing_label.move_to(healthy_label.get_center())

        new_bar1_pos, new_brace, tiny_delta = animate_gap_closure(
            self, bar0, bar1, label_e1,
            target_gap=0.15, base_y=base_y, e0_val=e0_start, bar_x=bar_x,
        )

        self.play(
            bar1.animate.move_to(new_bar1_pos),
            label_e1.animate.next_to(new_bar1_pos + RIGHT * 1.2, RIGHT, buff=0.0),
            Transform(gap_brace, new_brace),
            Transform(gap_text, tiny_delta),
            Transform(healthy_label, closing_label),
            run_time=2.5,
        )
        self.wait(0.5)

        # ── Curvature bound formula ───────────────────────────────────
        bound_formula = MathTex(
            r"|F_{ij}|", r"\leq", r"\frac{C}{\Delta^2}",
            font_size=40,
        ).shift(RIGHT * 2.5 + UP * 1.0)
        bound_formula[0].set_color(ACCENT_TEAL)
        bound_formula[2].set_color(CRISIS_RED)

        self.play(Write(bound_formula), run_time=1.5)

        bound_explain = Text(
            "Berry curvature diverges\nas the gap vanishes",
            font_size=16,
            color=SOFT_WHITE,
            line_spacing=1.3,
        ).next_to(bound_formula, DOWN, buff=0.4)
        self.play(FadeIn(bound_explain), run_time=0.8)
        self.wait(1)

        # ── Consequence text ──────────────────────────────────────────
        consequence = VGroup(
            Text("Theory predicts exactly which", font_size=18, color=SOFT_WHITE),
            Text("channels fail — and why", font_size=18, color=HIGHLIGHT_GOLD),
        ).arrange(DOWN, buff=0.1).shift(RIGHT * 2.5 + DOWN * 1.5)
        self.play(FadeIn(consequence), run_time=0.8)

        # ── Noisy path sketch ─────────────────────────────────────────
        np.random.seed(7)
        noisy_pts = []
        for i in range(50):
            angle = i * 0.15
            r = 0.6 + 0.25 * np.random.randn()
            x = 2.5 + r * np.cos(angle)
            y = -2.5 + r * np.sin(angle)
            noisy_pts.append([x, y, 0])

        noisy_path = VMobject(color=CRISIS_RED, stroke_width=1.5, stroke_opacity=0.6)
        noisy_path.set_points_smoothly([np.array(p) for p in noisy_pts])

        noisy_label = Text("erratic state path", font_size=13, color=DIM_GRAY)
        noisy_label.move_to([2.5, -3.3, 0])

        self.play(Create(noisy_path), FadeIn(noisy_label), run_time=1.5)

        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)
