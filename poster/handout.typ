// APS Global Physics Summit 2026 — 1-Page Research Handout (v2)
// Will Hammond | Pitzer College | Physics & Mathematics

#set page(
  paper: "us-letter",
  margin: (x: 0.5in, y: 0.35in),
  fill: white,
)

#set text(font: "Helvetica Neue", size: 9pt)
#set par(leading: 0.5em, justify: true)

// Colors
#let navy = rgb("#1B2838")
#let gold = rgb("#F5A623")
#let teal = rgb("#26C6DA")
#let green = rgb("#4CAF50")

// -- Header --
#block(width: 100%, inset: (x: 10pt, y: 8pt), radius: 6pt, fill: navy)[
  #grid(
    columns: (1fr, auto),
    [
      #text(size: 16pt, weight: "bold", fill: white)[
        Finding Hidden Structure in Complex Systems
      ] \
      #text(size: 10pt, fill: rgb("#B0BEC5"))[
        From Differential Geometry to Quantitative Finance
      ]
    ],
    align(right)[
      #text(size: 10pt, fill: white, weight: "bold")[Will Hammond] \
      #text(size: 9pt, fill: rgb("#B0BEC5"))[
        Pitzer College #sym.dot.op Physics & Math, Minor in Data Science #sym.dot.op GPA 3.8 \
        Research Advisor: Professor Trung "Average" Phan \
        whammond\@pitzer.edu #sym.dot.op github.com/willhammondhimself #sym.dot.op linkedin.com/in/willhammond
      ]
    ],
  )
]

#v(4pt)

// -- Section helper --
#let section(title, accent: navy) = {
  v(2pt)
  text(size: 12pt, weight: "bold", fill: accent)[#title]
  v(-2pt)
  line(length: 100%, stroke: 0.8pt + accent.lighten(50%))
  v(2pt)
}

// ================================================
// RESEARCH HIGHLIGHT
// ================================================

#section("Research Highlight: Geometric ML for Financial Regime Detection", accent: navy)

#grid(
  columns: (1fr, 1fr),
  gutter: 16pt,

  [
    *Problem.* Market regime shifts (2008 GFC, COVID-19, SVB collapse) cause
    catastrophic portfolio losses. Traditional detection methods require labeled
    historical crisis data, making them blind to novel crisis types. Unsupervised
    alternatives lack the sensitivity to distinguish true regime changes from noise.

    *Approach.* I developed a geometric machine learning framework --- QCML
    (Quantum Cognitive Machine Learning) --- a quantum-inspired framework that lifts
    financial time-series into a projective Hilbert space. While the name references
    quantum mechanics, the detection signals are rooted in differential geometry and
    spectral theory applied to classical financial data. In this space, market
    states become points on a manifold. From this representation, I extract three
    novel geometric observables:

    - *Berry phase rate* --- measures the rate of geometric rotation in state space
    - *Quantum Fisher information* --- quantifies the distinguishability between nearby market states
    - *Multi-lag fidelity* --- measures state overlap decay across multiple time horizons

    These observables serve as unsupervised regime-change indicators: they spike when
    the geometric structure of market returns changes, without requiring any crisis labels.

    *Validation.* Benchmarked against 17 detection methods (including Random Forest,
    HMM, BOCPD, Isolation Forest, CUSUM) across 16 historical crises on 5 ETFs
    (SPY, QQQ, IWM, EFA, DIA), spanning 1997--2024. All comparisons use walk-forward
    validation with bootstrap confidence intervals and permutation tests.
  ],

  [
    *Key Results.*

    #block(inset: 8pt, radius: 4pt, fill: rgb("#FFF8E1"), stroke: 0.5pt + gold)[
      #grid(
        columns: (1fr, 1fr),
        gutter: 8pt,
        [
          #align(center)[
            #text(size: 20pt, weight: "bold", fill: navy)[d = 0.36] \
            #text(size: 8pt, fill: navy)[Best geometric detector (QFI) \
            (+71% over supervised RF)]
          ]
        ],
        [
          #align(center)[
            #text(size: 20pt, weight: "bold", fill: navy)[5 ETFs] \
            #text(size: 8pt, fill: navy)[Multi-asset generalization \
            SPY, QQQ, IWM, EFA, DIA]
          ]
        ],
      )
    ]

    #v(4pt)

    - Best geometric detector (QFI Determinant) achieves Cohen's d = 0.36 (median across 16 crises), exceeding supervised Random Forest (d = 0.21) by 71%
    - Unsupervised: requires *no crisis labels* --- detects novel regime types
    - Geometric signals are *faster* to compute than RF (0.26s vs 1.07s per window)
    - Statistically significant (Friedman $chi^2$ = 66.5, $p < 10^(-10)$)
    - Granger causality: geometric signals predict absolute returns ($F$ = 9.5, $p$ = 0.0004)

    *Publication.* 34-page paper with 3 theorems, 1 proposition, and 33 peer-reviewed
    references. Available upon request.

    *Tools.* Python, PyTorch, NumPy/SciPy, scikit-learn, Optuna, Polygon.io API, LaTeX

    *So What?* This work demonstrates:
    - Ability to design novel ML features from first principles (differential geometry)
    - Rigorous experimental methodology (17-method benchmark, 16 crises, 5 assets)
    - Production-quality statistical validation (bootstrap, permutation, Bayesian)
    - Independent end-to-end research execution (solo author, sophomore year)
  ],
)

#v(2pt)

// ================================================
// ADDITIONAL EXPERIENCE
// ================================================

#section("Additional Experience", accent: navy)

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    *Quanta Ventures Fund --- Production Quant Research* \
    5-sleeve architecture aggregating 133 ML signals (XGBoost, Ridge, mean-reversion)
    with covariance-based risk budgeting. Walk-forward validation with 30-day embargo
    and 11-test validation suite.
    *Sharpe 2.92* | *Calmar 5.02* (true out-of-sample, 2022--2025).

    *Adaptive Vol Arb Platform --- Systems Engineering* \
    Built delta-neutral variance-risk-premium strategy with optimized C++ FFT pricer
    (pybind11). *40x latency reduction* (150ms to \<5ms). 38,460 lines across
    Python (20,760), C++ (1,130), React (4,678) with 24 test files + 24 validation
    scripts and CI/CD.  *Sharpe 1.32* (in-sample).
  ],
  [
    *Honors & Qualifications* \
    #sym.bullet Incoming Quant Research Intern, Steadfast Financial LP (Summer 2026) \
    #sym.bullet Carpe Diem Endowed Scholar, Pitzer College \
    #sym.bullet Harvey Mudd Putnam Math Competition Team \
    #sym.bullet CS Teaching Assistant --- mentoring 60+ students \
    #sym.bullet Graduate-level coursework: Mathematical Finance, Stochastic Calculus \
    #sym.bullet Languages: Python, C++, SQL, TypeScript, Mathematica, LaTeX \
    #sym.bullet Methods: Stochastic calculus, Monte Carlo, Bayesian inference, options pricing, game theory
  ],
)

#v(1fr)
#align(center)[
  #block(inset: (x: 12pt, y: 5pt), radius: 4pt, fill: navy)[
    #text(size: 8.5pt, fill: white)[
      *Seeking: Quantitative Research Internship --- Summer 2026/2027* #h(16pt)
      whammond\@pitzer.edu #h(16pt)
      github.com/willhammondhimself #h(16pt)
      linkedin.com/in/willhammond
    ]
  ]
]
