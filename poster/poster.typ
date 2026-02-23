// APS Global Physics Summit 2026 — Job Seeker Poster v3
// Will Hammond | Pitzer College | Physics & Mathematics
// 48 x 36 inches, landscape — Dark navy Bloomberg terminal aesthetic

// --- Color Palette (Dark Theme) ---
#let bg = rgb("#0D1117")
#let card-bg = rgb("#161B22")
#let card-border = rgb("#30363D")
#let text-primary = rgb("#F0F0F0")
#let text-secondary = rgb("#8B949E")
#let text-muted = rgb("#6E7681")
#let gold = rgb("#C8850F")
#let green = rgb("#4CAF50")
#let white = rgb("#FFFFFF")
#let red-accent = rgb("#E53935")
#let teal = rgb("#00897B")

// --- Page Setup ---
#set page(
  width: 48in,
  height: 36in,
  margin: (x: 0.5in, y: 0.35in),
  fill: bg,
)

#set text(font: "Helvetica Neue", fill: text-primary, size: 22pt)
#set par(leading: 0.45em)

// --- Helper Functions ---
#let section-title(body, accent: gold) = {
  block(width: 100%, inset: (bottom: 4pt))[
    #text(size: 34pt, weight: "bold", fill: accent)[#body]
    #v(-2pt)
    #line(length: 100%, stroke: 2.5pt + accent.lighten(20%))
  ]
}

#let sub-heading(body) = {
  text(size: 25pt, weight: "bold", fill: text-primary)[#body]
}

#let metric-card(value, label, accent: gold) = {
  block(
    width: 100%,
    inset: (x: 10pt, y: 8pt),
    radius: 6pt,
    fill: bg,
    stroke: 1.5pt + accent.lighten(10%),
  )[
    #align(center)[
      #text(size: 40pt, weight: "bold", fill: accent)[#value] \
      #text(size: 17pt, fill: text-secondary)[#label]
    ]
  ]
}

#let bullet-item(body) = {
  block(inset: (left: 18pt, y: 1pt))[
    #text(fill: text-primary, size: 21pt)[#sym.bullet.op #h(4pt) #body]
  ]
}

#let dark-card(body, accent: card-border) = {
  block(
    width: 100%,
    inset: (x: 14pt, y: 12pt),
    radius: 6pt,
    fill: card-bg,
    stroke: 1pt + accent,
  )[#body]
}

#let skill-tag(body, accent: teal) = {
  box(
    inset: (x: 9pt, y: 4pt),
    radius: 4pt,
    fill: accent.darken(60%),
    stroke: 1pt + accent.darken(20%),
  )[#text(size: 17pt, fill: accent.lighten(60%), weight: "medium")[#body]]
}

#let tool-tag(body) = {
  box(
    inset: (x: 8pt, y: 3pt),
    radius: 4pt,
    fill: rgb("#21262D"),
    stroke: 1pt + card-border,
  )[#text(size: 16pt, fill: text-secondary)[#body]]
}


// ================================================================
// HEADER
// ================================================================

#block(width: 100%, inset: (x: 24pt, y: 10pt))[
  #grid(
    columns: (auto, 1fr, auto),
    gutter: 20pt,
    align: horizon,

    // Left: Name + credentials
    [
      #text(size: 36pt, weight: "bold", fill: white)[Will Hammond] #h(10pt)
      #box(inset: (x: 10pt, y: 5pt), radius: 4pt, fill: gold.darken(30%), stroke: 1pt + gold)[
        #text(size: 21pt, weight: "bold", fill: gold.lighten(60%))[Incoming Quant Research Intern — Steadfast Financial LP]
      ]
      #v(2pt)
      #text(size: 22pt, fill: text-secondary)[
        Pitzer College #sym.dot.op B.A. Physics & Mathematics #sym.dot.op GPA 3.8 #sym.dot.op Advisor: Prof. Trung Phan
      ]
      #v(1pt)
      #text(size: 19pt, fill: text-muted)[
        whammond\@pitzer.edu #h(14pt) github.com/willhammondhimself #h(14pt) linkedin.com/in/willhammond
      ]
    ],

    // Center: Title
    align(center)[
      #text(size: 60pt, weight: "bold", fill: white)[
        Finding Hidden Structure in Complex Systems
      ]
      #v(1pt)
      #text(size: 32pt, fill: gold, weight: "bold")[
        From Differential Geometry to Quantitative Finance
      ]
    ],

    // Right: Seeking badge + QR
    align(right)[
      #block(inset: (x: 16pt, y: 10pt), radius: 6pt, fill: gold, stroke: none)[
        #align(center)[
          #text(size: 24pt, weight: "bold", fill: white)[Seeking:] \
          #text(size: 21pt, fill: white)[Quant Research Internship] \
          #text(size: 19pt, fill: white.darken(10%))[Summer 2026 / 2027]
        ]
      ]
      #v(6pt)
      #block(width: 1.5in, height: 1.5in, radius: 6pt, fill: white, inset: 8pt)[
        #align(center + horizon)[
          #text(size: 18pt, fill: bg, weight: "bold")[QR Code \
          #text(size: 14pt, weight: "regular", fill: rgb("#444"))[Portfolio & GitHub]]
        ]
      ]
    ],
  )
]

#v(2pt)
#line(length: 100%, stroke: 1pt + card-border)
#v(4pt)


// ================================================================
// TWO-COLUMN LAYOUT
// ================================================================

#grid(
  columns: (1.05fr, 0.95fr),
  gutter: 18pt,

  // ── COLUMN 1: QCML RESEARCH (Gold accent) ─────────────────────
  dark-card(accent: gold.darken(20%))[
    #section-title("Geometric ML for Regime Detection", accent: gold)
    #text(size: 19pt, fill: gold.lighten(20%), style: "italic")[
      QCML Framework #sym.dot.op Independent Research #sym.dot.op Solo-authored 34-page paper
    ]
    #v(4pt)

    // Problem
    #sub-heading[Problem]
    #v(2pt)
    #text(size: 23pt, fill: text-primary)[
      Market crashes cause catastrophic losses. Can we detect regime shifts _without labeled crisis data?_
    ]
    #v(6pt)

    // Approach
    #sub-heading[Approach]
    #v(3pt)
    #bullet-item[Embed financial time series into a *geometric space* (projective Hilbert space) using spectral metric learning]
    #bullet-item[Extract *3 geometric observables* that measure how the data manifold deforms during stress]
    #bullet-item[*Fully unsupervised* — no crisis labels, no regime annotations, no look-ahead]
    #v(2pt)
    #text(size: 18pt, fill: text-muted, style: "italic")[
      #h(18pt) Note: "QCML" is a historical name — the math is differential geometry and spectral theory, not quantum computing.
    ]
    #v(6pt)

    // Metric callouts — ABOVE figure so numbers hit first
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 12pt,
      metric-card("d = 0.68", "QFI vs calm periods\n(unsupervised, no labels)", accent: gold),
      metric-card("12 / 12", "Crises detected by\nbest geometric observable", accent: gold),
      metric-card("0.26s", "Multi-Lag Fidelity\nfaster than RF (1.07s)", accent: gold),
    )
    #v(6pt)

    // Hero figure
    #block(width: 100%, radius: 6pt, clip: true, stroke: 1pt + card-border)[
      #image("figures/crisis_timeline_dark.png", width: 96%)
    ]
    #v(6pt)

    // Key Results
    #sub-heading[Key Results]
    #v(2pt)
    #bullet-item[*Unsupervised* QFI (d = 0.68) outperforms *supervised* Random Forest (d = 0.24) — no crisis labels needed]
    #bullet-item[CUSUM (d = 0.87) strongest overall; geometric methods are *competitive and complementary*]
    #bullet-item[Each observable captures different crisis types — QFI on novel mechanisms, Berry on structural transitions]
    #bullet-item[*Multi-asset generalization*: tested across 5 ETFs (SPY, QQQ, IWM, EFA, DIA) × 4 major crises]
    #v(4pt)

    #tool-tag[Python] #h(3pt) #tool-tag[PyTorch] #h(3pt) #tool-tag[NumPy/SciPy] #h(3pt) #tool-tag[scikit-learn] #h(3pt) #tool-tag[Optuna] #h(3pt) #tool-tag[Polygon.io] #h(3pt) #tool-tag[LaTeX]
  ],


  // ── COLUMN 2: QUANTA VENTURES (Green accent) ──────────────────
  dark-card(accent: green.darken(20%))[
    #section-title("Production Quantitative Research", accent: green)
    #text(size: 19pt, fill: green.lighten(20%), style: "italic")[
      Quanta Ventures Fund #sym.dot.op Systematic equity strategies
    ]
    #v(4pt)

    // Problem
    #sub-heading[Problem]
    #v(2pt)
    #text(size: 23pt, fill: text-primary)[
      How do you combine 133 uncorrelated ML signals into a single portfolio that survives regime changes and transaction costs?
    ]
    #v(8pt)

    // Approach
    #sub-heading[Approach]
    #v(4pt)
    #bullet-item[*5-sleeve architecture* aggregating 133 signals across XGBoost, Ridge, and mean-reversion]
    #v(2pt)
    #bullet-item[Two-stage volatility targeting with *VIX/VVIX regime diagnostics*]
    #v(2pt)
    #bullet-item[Walk-forward validation with 30-day embargo and 11-test robustness suite]
    #v(8pt)

    // Metric callouts
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 12pt,
      metric-card("2.92", "Sharpe Ratio\n(true OOS, 2022–2025)", accent: green),
      metric-card("5.02", "Calmar Ratio\n(return / max drawdown)", accent: green),
      metric-card("133", "Signals across\n5 strategy sleeves", accent: green),
    )
    #v(8pt)

    // Pipeline architecture diagram
    #sub-heading[System Architecture]
    #v(4pt)
    #block(width: 100%, inset: (x: 10pt, y: 12pt), radius: 6pt, fill: bg, stroke: 1pt + green.darken(20%))[
      #align(center)[
        #grid(
          columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
          align: center + horizon,
          gutter: 4pt,
          block(inset: (x: 6pt, y: 8pt), radius: 4pt, fill: green.darken(60%), stroke: 1pt + green.darken(20%))[
            #text(size: 18pt, fill: green.lighten(50%), weight: "bold")[133 Signals] \
            #text(size: 14pt, fill: text-muted)[XGB / Ridge / MR]
          ],
          text(size: 30pt, fill: green.lighten(30%))[→],
          block(inset: (x: 6pt, y: 8pt), radius: 4pt, fill: green.darken(60%), stroke: 1pt + green.darken(20%))[
            #text(size: 18pt, fill: green.lighten(50%), weight: "bold")[5 Sleeves] \
            #text(size: 14pt, fill: text-muted)[Cov-weighted]
          ],
          text(size: 30pt, fill: green.lighten(30%))[→],
          block(inset: (x: 6pt, y: 8pt), radius: 4pt, fill: green.darken(60%), stroke: 1pt + green.darken(20%))[
            #text(size: 18pt, fill: green.lighten(50%), weight: "bold")[Risk Engine] \
            #text(size: 14pt, fill: text-muted)[VIX/VVIX]
          ],
          text(size: 30pt, fill: green.lighten(30%))[→],
          block(inset: (x: 6pt, y: 8pt), radius: 4pt, fill: green.darken(40%), stroke: 1.5pt + green)[
            #text(size: 18pt, fill: green.lighten(60%), weight: "bold")[Portfolio] \
            #text(size: 14pt, fill: text-muted)[Dynamic sizing]
          ],
        )
      ]
    ]
    #v(8pt)

    // Signal Types
    #sub-heading[Signal Types]
    #v(4pt)
    #grid(
      columns: (1fr, 1fr),
      gutter: 10pt,
      dark-card(accent: green.darken(30%))[
        #text(size: 22pt, fill: green.lighten(30%), weight: "bold")[Momentum] \
        #text(size: 18pt, fill: text-secondary)[XGBoost cross-sectional signals]
      ],
      dark-card(accent: green.darken(30%))[
        #text(size: 22pt, fill: green.lighten(30%), weight: "bold")[Mean Reversion] \
        #text(size: 18pt, fill: text-secondary)[Z-score convergence strategies]
      ],
      dark-card(accent: green.darken(30%))[
        #text(size: 22pt, fill: green.lighten(30%), weight: "bold")[Statistical] \
        #text(size: 18pt, fill: text-secondary)[Ridge regression factor models]
      ],
      dark-card(accent: green.darken(30%))[
        #text(size: 22pt, fill: green.lighten(30%), weight: "bold")[Risk Budget] \
        #text(size: 18pt, fill: text-secondary)[Covariance-weighted allocation]
      ],
    )
    #v(8pt)

    // Key Results
    #sub-heading[Key Results]
    #v(3pt)
    #bullet-item[*Sharpe 2.92* and *Calmar 5.02* on true out-of-sample data (2022–2025)]
    #v(2pt)
    #bullet-item[*133 signals* with covariance-based risk budgeting across 5 sleeves]
    #v(2pt)
    #bullet-item[Passed *11-test validation suite* (parameter stability, bootstrap, walk-forward)]
    #v(2pt)
    #bullet-item[Dynamic regime-aware leverage reduced peak drawdown via VIX/VVIX classification]
    #v(8pt)

    // Risk Management
    #sub-heading[Risk Management]
    #v(3pt)
    #bullet-item[VIX/VVIX regime classification for *dynamic leverage adjustment*]
    #v(2pt)
    #bullet-item[30-day embargo prevents lookahead bias contamination]
    #v(2pt)
    #bullet-item[Heston stochastic vol for tail-risk monitoring]
    #v(6pt)

    #tool-tag[Python] #h(3pt) #tool-tag[XGBoost] #h(3pt) #tool-tag[Optuna] #h(3pt) #tool-tag[Polygon.io] #h(3pt) #tool-tag[PostgreSQL] #h(3pt) #tool-tag[FRED API]
  ],
)

#v(4pt)

// ================================================================
// BOTTOM SKILLS BAR
// ================================================================

#block(
  width: 100%,
  inset: (x: 20pt, y: 12pt),
  radius: 6pt,
  fill: card-bg,
  stroke: 1pt + card-border,
)[
  #grid(
    columns: (1.1fr, 1.1fr, 1fr, 1.2fr),
    gutter: 16pt,

    // Technical Skills
    [
      #text(size: 24pt, weight: "bold", fill: white)[Technical Skills]
      #v(4pt)
      #skill-tag[Python (NumPy, Pandas, PyTorch)] #h(3pt)
      #skill-tag[C++ (pybind11, FFT)] #v(2pt)
      #skill-tag[SQL (PostgreSQL)] #h(3pt)
      #skill-tag[TypeScript / React] #v(2pt)
      #skill-tag[Mathematica] #h(3pt)
      #skill-tag[Git / CI/CD] #h(3pt)
      #skill-tag[LaTeX]
    ],

    // Quantitative Methods
    [
      #text(size: 24pt, weight: "bold", fill: white)[Quantitative Methods]
      #v(4pt)
      #skill-tag(accent: gold)[Stochastic Calculus] #h(3pt)
      #skill-tag(accent: gold)[Monte Carlo] #v(2pt)
      #skill-tag(accent: gold)[Options Pricing (BS, Heston)] #v(2pt)
      #skill-tag(accent: gold)[Bayesian Inference] #h(3pt)
      #skill-tag(accent: gold)[Time Series] #v(2pt)
      #skill-tag(accent: gold)[Walk-Forward Validation] #v(2pt)
      #skill-tag(accent: gold)[Risk Metrics (Sharpe, VaR)]
    ],

    // Communication
    [
      #text(size: 24pt, weight: "bold", fill: white)[Communication]
      #v(4pt)
      #text(size: 20pt, fill: text-primary, weight: "medium")[CS Teaching Assistant]
      #text(size: 17pt, fill: text-secondary)[Mentoring 60+ students in intro CS]
      #v(3pt)
      #text(size: 20pt, fill: text-primary, weight: "medium")[Independent Researcher]
      #text(size: 17pt, fill: text-secondary)[Solo-authored 34-page paper, 3 theorems]
      #v(3pt)
      #text(size: 20pt, fill: text-primary, weight: "medium")[Technical Communicator]
      #text(size: 17pt, fill: text-secondary)[Translating complex math for diverse audiences]
    ],

    // Honors & Experience
    [
      #text(size: 24pt, weight: "bold", fill: white)[Honors & Experience]
      #v(4pt)
      #block(inset: (left: 4pt))[
        #text(size: 21pt, fill: gold, weight: "bold")[Incoming Quant Research Intern] \
        #text(size: 18pt, fill: text-secondary)[Steadfast Financial LP — Summer 2026]
        #v(3pt)
        #text(size: 20pt, fill: text-primary, weight: "bold")[Vol Arb Platform] \
        #text(size: 17pt, fill: text-secondary)[Real-time C++/Python/React, 40x latency reduction]
        #v(3pt)
        #text(size: 20pt, fill: text-primary, weight: "bold")[Carpe Diem Endowed Scholar] \
        #text(size: 17pt, fill: text-secondary)[Pitzer College academic distinction]
        #v(3pt)
        #text(size: 20pt, fill: text-primary, weight: "bold")[Harvey Mudd Putnam Math] \
        #text(size: 17pt, fill: text-secondary)[Competitive mathematics]
        #v(3pt)
        #text(size: 20pt, fill: text-primary, weight: "bold")[Graduate-Level Coursework] \
        #text(size: 17pt, fill: text-secondary)[Math Finance, Stochastic Calc (sophomore)]
      ]
    ],
  )
]
