// APS Global Physics Summit 2026 — Job Seeker Poster v4
// Will Hammond | Pitzer College | Physics & Mathematics
// 48 x 36 inches, landscape — Cream & Navy theme
//
// This is a PROFESSIONAL POSITIONING poster, not a research poster.
// Goal: Make recruiters think "I want to talk to this kid."

// ================================================================
// COLOR PALETTE — Cream & Navy (warm stationery with navy ink)
// ================================================================
#let bg = rgb("#FAF7F2")            // Warm cream background
#let primary = rgb("#1B2A4A")       // Deep navy
#let secondary = rgb("#2C3E5A")     // Medium navy (card 2 accent)
#let accent-slate = rgb("#3D5068")  // Slate navy (card 3 accent)
#let card-bg = rgb("#FFFFFF")       // Clean white
#let card-border = rgb("#D6CFC5")   // Warm taupe border
#let text-primary = rgb("#1A1814")  // Warm near-black
#let text-secondary = rgb("#5C564E")// Warm dark gray
#let text-muted = rgb("#8A837A")    // Warm medium gray
#let tag-bg = rgb("#EDE8E1")        // Light warm tan (all tags)
#let tag-text = rgb("#1B2A4A")      // Navy (all tags)
#let footer-bg = rgb("#1B2A4A")     // Navy footer
#let footer-text = rgb("#FAF7F2")   // Cream on navy

// ================================================================
// PAGE SETUP (48 x 36 landscape)
// ================================================================
#set page(
  width: 48in,
  height: 36in,
  margin: (x: 0.6in, y: 0.5in),
  fill: bg,
)
#set text(font: "Georgia", fill: text-primary, size: 24pt)
#set par(leading: 0.5em)

// ================================================================
// HELPER FUNCTIONS
// ================================================================

#let section-title(body, accent: primary) = {
  block(width: 100%, inset: (bottom: 6pt))[
    #text(size: 36pt, weight: "bold", fill: accent, font: "Helvetica Neue")[#body]
    #v(-2pt)
    #line(length: 100%, stroke: 3pt + accent.lighten(30%))
  ]
}

#let project-card(title, accent, body) = {
  block(
    width: 100%,
    inset: (x: 18pt, y: 16pt),
    radius: 6pt,
    fill: card-bg,
    stroke: (left: 6pt + accent, rest: 1.5pt + card-border),
  )[
    #section-title(title, accent: accent)
    #v(4pt)
    #body
  ]
}

#let skill-tag(body) = {
  box(
    inset: (x: 10pt, y: 5pt),
    radius: 5pt,
    fill: tag-bg,
    stroke: 1pt + tag-bg.darken(15%),
  )[#text(size: 19pt, fill: tag-text, weight: "medium")[#body]]
}

#let tool-tag(body) = {
  box(
    inset: (x: 8pt, y: 4pt),
    radius: 4pt,
    fill: tag-bg,
    stroke: 1pt + card-border,
  )[#text(size: 17pt, fill: text-secondary)[#body]]
}

#let bullet-item(body) = {
  block(inset: (left: 18pt, y: 2pt))[
    #text(fill: text-primary, size: 22pt)[#sym.bullet #h(6pt) #body]
  ]
}


// ================================================================
// HEADER
// ================================================================

#block(width: 100%, inset: (x: 16pt, y: 6pt))[
  #grid(
    columns: (1.8in, 1fr, 2.6in),
    gutter: 20pt,
    align: horizon,

    // ── Headshot placeholder ──
    // Replace with: #image("headshot.jpg", width: 1.6in)
    block(width: 1.6in, height: 2in, radius: 6pt, fill: rgb("#FAF7F2"), stroke: 3pt + primary)[
      #align(center + horizon)[
        #text(size: 16pt, fill: text-muted, weight: "bold")[YOUR \
        HEADSHOT \
        HERE]
      ]
    ],

    // ── Name, affiliation, title ──
    [
      #text(size: 58pt, weight: "bold", fill: primary, font: "Helvetica Neue")[WILL ] #text(size: 58pt, weight: "bold", fill: primary, font: "Helvetica Neue")[HAMMOND]
      #v(-6pt)
      #text(size: 28pt, fill: text-secondary)[
        Pitzer College #sym.dot.c B.A. Physics & Mathematics #sym.dot.c Expected 2028
      ]
      #v(1pt)
      #text(size: 21pt, fill: text-muted)[
        whammond\@pitzer.edu #h(14pt) #sym.bar.v #h(14pt) linkedin.com/in/willhammond #h(14pt) #sym.bar.v #h(14pt) github.com/willhammondhimself
      ]

      #v(8pt)

      // Title quote
      #text(size: 38pt, weight: "bold", fill: secondary, style: "italic", font: "Helvetica Neue")[
        "Quantitative Research at the Intersection of Physics, Math, and Finance"
      ]

      #v(6pt)

      // About Me
      #text(size: 22pt, fill: text-primary)[
        I'm a sophomore studying physics and mathematics with a minor in data science, taking graduate-level coursework across the Claremont Colleges. I love applying mathematical tools to complex, uncertain systems — from detecting financial market crises using differential geometry to building systematic trading strategies. Advised by Prof. Trung Phan, Keck Science Department.
      ]
    ],

    // ── QR codes ──
    // Replace placeholders with: #image("qr_linkedin.png", width: 1in)
    align(right)[
      #grid(
        columns: (1fr,),
        gutter: 8pt,
        block(width: 1.2in, height: 1.2in, radius: 6pt, fill: rgb("#FAF7F2"), stroke: 1.5pt + primary, inset: 6pt)[
          #align(center + horizon)[
            #text(size: 13pt, fill: primary, weight: "bold")[LinkedIn] \
            #text(size: 11pt, fill: text-muted)[QR Code]
          ]
        ],
        block(width: 1.2in, height: 1.2in, radius: 6pt, fill: rgb("#FAF7F2"), stroke: 1.5pt + secondary, inset: 6pt)[
          #align(center + horizon)[
            #text(size: 13pt, fill: secondary, weight: "bold")[GitHub] \
            #text(size: 11pt, fill: text-muted)[QR Code]
          ]
        ],
        block(width: 1.2in, height: 1.2in, radius: 6pt, fill: rgb("#FAF7F2"), stroke: 1.5pt + accent-slate, inset: 6pt)[
          #align(center + horizon)[
            #text(size: 13pt, fill: accent-slate, weight: "bold")[Portfolio] \
            #text(size: 11pt, fill: text-muted)[QR Code]
          ]
        ],
      )
    ],
  )
]

#v(4pt)
#line(length: 100%, stroke: 2pt + card-border)
#v(6pt)


// ================================================================
// THREE PROJECT CARDS
// ================================================================

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 14pt,


  // ── CARD 1: QCML RESEARCH (Blue accent) ────────────────────────
  project-card("QCML Research", primary)[
    #text(size: 20pt, fill: primary, style: "italic", weight: "medium")[
      I applied physics math to find hidden patterns in financial data
    ]
    #v(6pt)

    #text(size: 24pt, weight: "bold", fill: text-primary)[Problem]
    #v(2pt)
    #text(size: 21pt)[Can we detect market regime shifts _without labeled crisis data?_]
    #v(6pt)

    #text(size: 24pt, weight: "bold", fill: text-primary)[Approach]
    #v(2pt)
    #bullet-item[Embed time series into geometric space using spectral metric learning]
    #bullet-item[Extract 3 geometric observables measuring manifold deformation during stress]
    #bullet-item[*Fully unsupervised* — no crisis labels, no look-ahead]
    #v(6pt)

    // Figure: QCML vs RF comparison
    #block(width: 100%, height: 3.0in, clip: true)[
      #image("pptx_images/image_0.png", width: 100%)
    ]
    #v(6pt)

    #text(size: 24pt, weight: "bold", fill: text-primary)[Outcome]
    #v(2pt)
    // KEY METRICS — update these if numbers change
    #bullet-item[Multi-Lag Fidelity (*d=1.44*) beat Random Forest (*d=1.13*, p=0.002)]
    #bullet-item[Tested across *5 ETFs* and *4 out-of-sample crises*]
    #bullet-item[Solo-authored *34-page research paper* (in revision)]
    #v(6pt)

    #tool-tag[Python] #h(3pt) #tool-tag[PyTorch] #h(3pt) #tool-tag[NumPy/SciPy] #h(3pt) #tool-tag[Optuna] #h(3pt) #tool-tag[LaTeX]
  ],


  // ── CARD 2: QUANTA VENTURES (Purple accent) ────────────────────
  project-card("Quanta Ventures", secondary)[
    #text(size: 20pt, fill: secondary, style: "italic", weight: "medium")[
      Developed systematic equity strategies for a quantitative fund
    ]
    #v(6pt)

    #text(size: 24pt, weight: "bold", fill: text-primary)[Problem]
    #v(2pt)
    #text(size: 21pt)[Building robust multi-strategy portfolios that survive regime changes]
    #v(6pt)

    #text(size: 24pt, weight: "bold", fill: text-primary)[Approach]
    #v(2pt)
    #bullet-item[Modular framework: multiple strategy sleeves with volatility targeting]
    #bullet-item[VIX/VVIX regime diagnostics for dynamic leverage]
    #bullet-item[Walk-forward validation with 30-day embargo and robustness suite]
    #v(6pt)

    // Figure: Quanta metrics
    #block(width: 100%, height: 3.0in, clip: true)[
      #image("pptx_images/image_1.png", width: 100%)
    ]
    #v(6pt)

    #text(size: 24pt, weight: "bold", fill: text-primary)[Outcome]
    #v(2pt)
    // KEY METRICS — update these if numbers change
    #bullet-item[Sharpe 2.92 and Calmar 5.02 on out-of-sample data]
    #bullet-item[Passed full robustness test suite]
    #bullet-item[Built C++ Heston pricer for options strategy component]
    #v(6pt)

    #tool-tag[Python] #h(3pt) #tool-tag[scikit-learn] #h(3pt) #tool-tag[Optuna] #h(3pt) #tool-tag[PostgreSQL] #h(3pt) #tool-tag[C++]
  ],


  // ── CARD 3: STUDENT QUANT FUND (Slate navy accent) ──────────────────
  project-card("Student Quant Fund", accent-slate)[
    #text(size: 20pt, fill: accent-slate, style: "italic", weight: "medium")[
      Finding alpha and building community
    ]
    #v(6pt)

    #text(size: 24pt, weight: "bold", fill: text-primary)[Problem]
    #v(2pt)
    #text(size: 21pt)[Giving underclassmen access to quantitative skills and a production trading platform]
    #v(6pt)

    #text(size: 24pt, weight: "bold", fill: text-primary)[Approach]
    #v(2pt)
    #bullet-item[Teaching momentum and mean-variance strategies]
    #bullet-item[Building production-ready platform for future cohorts]
    #bullet-item[Preparing for paper trading and capital deployment]
    #v(6pt)

    // Figure: Skills radar / fund visual
    #block(width: 100%, height: 3.0in, clip: true)[
      #image("pptx_images/image_2.png", width: 100%)
    ]
    #v(6pt)

    #text(size: 24pt, weight: "bold", fill: text-primary)[Outcome]
    #v(2pt)
    #bullet-item[Mentoring newer students in collaborative strategy development]
    #bullet-item[Building institutional knowledge across cohorts]
    #bullet-item[Platform designed for live paper trading]
    #v(6pt)

    #tool-tag[Python] #h(3pt) #tool-tag[Jupyter] #h(3pt) #tool-tag[Team Collaboration]
  ],
)

#v(8pt)


// ================================================================
// SKILLS SECTION
// ================================================================

#block(
  width: 100%,
  inset: (x: 20pt, y: 14pt),
  radius: 6pt,
  fill: card-bg,
  stroke: 1.5pt + card-border,
)[
  #text(size: 32pt, weight: "bold", fill: primary, font: "Helvetica Neue")[Skills]
  #v(6pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 16pt,

    // ── Hard Skills ──
    [
      #text(size: 25pt, weight: "bold", fill: text-primary)[Hard Skills]
      #v(6pt)

      #text(size: 21pt, weight: "bold", fill: primary)[Tech ]
      #skill-tag[Python (NumPy, Pandas, PyTorch)] #h(2pt)
      #skill-tag[SQL] #h(2pt)
      #skill-tag[Git/CI-CD] #h(2pt)
      #skill-tag[Docker] #h(2pt)
      #skill-tag[LaTeX] #h(2pt)
      #skill-tag[Mathematica]
      #v(4pt)

      #text(size: 21pt, weight: "bold", fill: primary)[CS ]
      #skill-tag[Data Structures & Algorithms] #h(2pt)
      #skill-tag[ML/PyTorch] #h(2pt)
      #skill-tag[Numerical Optimization]
      #v(4pt)

      #text(size: 21pt, weight: "bold", fill: secondary)[Finance ]
      #skill-tag[Stochastic Calculus] #h(2pt)
      #skill-tag[Options Pricing (BS, Heston)] #h(2pt)
      #skill-tag[Backtesting/Walk-Forward] #h(2pt)
      #skill-tag[Risk Metrics (Sharpe, VaR, CVaR)]
    ],

    // ── Soft Skills ──
    [
      #text(size: 25pt, weight: "bold", fill: text-primary)[Soft Skills]
      #v(6pt)

      #text(size: 22pt, weight: "bold", fill: text-primary)[CS Teaching Assistant] \
      #text(size: 19pt, fill: text-secondary)[Held weekly office hours, mentored 60+ students]
      #v(4pt)

      #text(size: 22pt, weight: "bold", fill: text-primary)[Independent Researcher] \
      #text(size: 19pt, fill: text-secondary)[Solo-authored 34-page research paper]
      #v(4pt)

      #text(size: 22pt, weight: "bold", fill: text-primary)[Team Collaboration] \
      #text(size: 19pt, fill: text-secondary)[Student Quant Fund leadership]
      #v(4pt)

      #text(size: 22pt, weight: "bold", fill: text-primary)[Technical Communication] \
      #text(size: 19pt, fill: text-secondary)[Translating complex math for diverse audiences]
    ],
  )
]

#v(6pt)


// ================================================================
// FOOTER
// ================================================================

#block(
  width: 100%,
  inset: (x: 20pt, y: 10pt),
  radius: 6pt,
  fill: footer-bg,
)[
  #grid(
    columns: (auto, 1fr, auto),
    gutter: 16pt,
    align: horizon,

    // Logos + honors
    [
      // Replace with: #image("pitzer_logo.png", height: 0.6in) #image("keck_logo.png", height: 0.6in)
      #text(size: 19pt, fill: footer-text, weight: "bold")[Pitzer College #h(10pt) #sym.bar.v #h(10pt) Keck Science Department]
      #v(2pt)
      #text(size: 17pt, fill: rgb("#B0A99F"))[Carpe Diem Endowed Scholar]
    ],

    // Seeking
    align(center)[
      #text(size: 25pt, fill: footer-text, weight: "bold")[
        Seeking: Quant Research/Trading #h(6pt) #sym.bar.v #h(6pt) ML/AI Engineering #h(6pt) #sym.bar.v #h(6pt) Physics Research
      ]
      #v(2pt)
      #text(size: 21pt, fill: rgb("#D6CFC5"))[Internships & Full-time — Available May 2028]
    ],

    // Interests
    align(right)[
      #text(size: 18pt, fill: rgb("#B0A99F"))[
        USA Climbing Divisionals \
        Poker #sym.dot.c Guitar & Drums
      ]
    ],
  )
]
