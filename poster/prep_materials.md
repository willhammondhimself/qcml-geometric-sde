# APS Global Physics Summit — Poster Prep Materials

## 60-Second Walkthrough Script

> Practice this until it's natural, not memorized. Hit the bold points.

"Hi, I'm **Will Hammond** — I'm a sophomore at **Pitzer College** studying physics and math.

I'm most excited about applying **mathematical tools to messy real-world problems**, especially in finance.

My main research project is **QCML** — a quantum-inspired framework where I embed financial time series into a projective Hilbert space and extract geometric observables that detect regime shifts, like market crashes — **without any labeled data**. My geometric observables are **competitive with supervised baselines** and provide **complementary signals** — a simple ensemble significantly outperforms any individual method. I solo-authored a **34-page paper** on this.

I also built a systematic trading system for **Quanta Ventures** — multi-strategy framework with walk-forward validation, achieving a **Sharpe of 2.92** and **Calmar of 5.02** out-of-sample.

And I co-lead a **Student Quant Fund** where I teach momentum and mean-variance strategies to younger students.

I'm looking for opportunities in **quant research, ML engineering, or physics research** — happy to dive deeper into anything here."

---

## Talking Points by Section

### QCML Research (Blue card)

**The 30-second pitch:**
"I took tools from differential geometry — the same math used in general relativity — and applied them to financial time series. By embedding price data into a projective Hilbert space, I can measure how the geometry of the data manifold deforms during stress. Three different geometric observables each capture different types of crises. The key result: these unsupervised geometric signals are competitive with supervised methods and provide genuinely complementary detection — they catch crises that classical methods miss, and vice versa."

**If they ask "What's the Hilbert space / why quantum?":**
"The name 'QCML' is historical — the actual math is spectral theory and differential geometry. I map feature vectors to quantum-like states because projective Hilbert space has rich geometric structure: there's a natural metric (Fubini-Study), Berry curvature, and topological invariants. The 'quantum' framing gives us a rigorous mathematical toolkit that happens to work really well for detecting nonlinear regime changes."

**If they ask about the comparison with Random Forest:**
"On the current walk-forward benchmark across 16 crises, QFI Determinant achieves Cohen's d = 0.36 vs RF's d = 0.21 — a 71% improvement. But be honest: CUSUM — a classical change-point method — scores d = 0.71, which is stronger than any individual geometric method on mean effect size. The geometric advantage is that it's unsupervised and complementary: it detects different crises than CUSUM or RF, so a simple ensemble nearly doubles the best individual method."

> **Honesty note:** CUSUM d=0.71 is the strongest baseline — be upfront about this at the poster. Don't claim geometric methods dominate; claim they are *complementary* and *unsupervised*, which is the real contribution.

**If they push on CUSUM beating geometric methods on mean d:**
"CUSUM (d=0.71) beats all of our geometric methods on median Cohen's d — that's true. The QCML advantage is threefold: (1) fully unsupervised, no labeled data needed, (2) the geometric signals fire on different crises than CUSUM, providing genuine complementarity, and (3) a simple average ensemble of geometric + classical methods significantly outperforms any individual method. The contribution is a new *class* of features, not a single dominant method."

**If they ask about the math:**
"The three observables are: (1) Berry phase rate — measures curvature of the data path on the manifold, (2) QFI determinant — quantum Fisher information quantifying distinguishability between nearby states, and (3) Multi-Lag Fidelity — measures how quickly the state space decorrelates across multiple time scales. Each captures a different geometric signature of market stress."

### Quanta Ventures (Purple card)

**The 30-second pitch:**
"I built a systematic equity framework with 5 strategy sleeves — momentum, mean-reversion, statistical, factor, and risk budget — validated with walk-forward testing and a 30-day embargo to prevent look-ahead bias. Achieved a Sharpe of 2.92 and Calmar of 5.02 out-of-sample. The risk engine uses VIX/VVIX regime classification for dynamic leverage."

**If they ask about the Sharpe/Calmar:**
"These are true out-of-sample numbers from 2022-2025, including the rate hiking cycle. Walk-forward with 30-day embargo between train and test. I also built an 11-test robustness suite — parameter stability, bootstrap, Monte Carlo permutation — to make sure these aren't overfit."

**If they ask about the C++ component:**
"I built a Heston stochastic volatility pricer in C++ with pybind11 bindings for Python integration. It's used for options strategy pricing and tail-risk hedging. The C++ implementation gives about 40x speedup over pure Python for Monte Carlo paths."

### Student Quant Fund (Teal card)

**The 30-second pitch:**
"I co-lead a student quant fund at the Claremont Colleges. We teach momentum and mean-variance strategies, and I'm building a production platform for paper trading. The goal is institutional knowledge — so future cohorts can build on our work rather than starting from scratch."

**If they ask what you've learned from teaching:**
"Teaching intro CS and leading the quant fund taught me that explaining complex ideas simply is the hardest and most valuable skill. In office hours, I learned to diagnose where someone is stuck — is it the concept, the syntax, or the mental model? That translates directly to how I write research and communicate technical ideas."

### Skills (Bottom section)

**If they ask about graduate-level coursework:**
"I'm taking Stochastic Calculus at Harvey Mudd as a sophomore — it's a grad-level course. I've also taken Quantum Mechanics, Quantitative Finance, and Real Analysis. The Claremont Colleges consortium lets me take courses at all 5 schools."

**If they ask about experience:**
"This summer I'll be a quant research intern at Steadfast Financial LP, working on systematic equity strategies. I've also done independent research under Prof. Trung Phan in the Keck Science Department."

---

## Anticipated Questions & Answers

### Technical Questions

**Q: "How do you handle look-ahead bias in your QCML research?"**
A: "Three ways: (1) all geometric observables use only past data in a rolling window, (2) the Random Forest baseline uses leave-one-crisis-out — it never sees the test crisis during training, and (3) we validated with a separate temporal out-of-sample split: calibrate on pre-2020 data, test on COVID and later crises."

**Q: "What's the computational cost?"**
A: "Berry Phase Rate processes 3,400 time steps in 0.77 seconds, Multi-Lag Fidelity in 0.26 seconds. Both are faster than Random Forest at 1.07 seconds. QFI is slower at 5.95 seconds. All single-threaded on a laptop."

**Q: "Why not use a neural network for regime detection?"**
A: "We benchmarked against LSTM and TCN — they perform worse than both our method and RF on this task, likely because regime transitions are rare events with limited training data. Our geometric approach doesn't require many examples because it's measuring structural properties of the data manifold, not learning a classification boundary."

**Q: "What programming languages do you use?"**
A: "Primarily Python for research and data science. C++ for performance-critical components — I built a Heston model pricer with pybind11. SQL for data management. I've also used TypeScript/React for a real-time trading dashboard."

### Career Questions

**Q: "What kind of role are you looking for?"**
A: "Quant research or trading, ML/AI engineering, or physics research. I'm particularly drawn to roles where I can apply mathematical tools to real-world problems with measurable outcomes. My Steadfast internship this summer is in systematic equity quant research."

**Q: "You're a sophomore — what's your timeline?"**
A: "I graduate in May 2028. I'm doing my first internship this summer at Steadfast Financial. I'm looking for internships for summer 2027 and full-time opportunities starting 2028."

**Q: "What do you want to learn next?"**
A: "I want to deepen my understanding of market microstructure and optimal execution. I'm also interested in applying geometric methods to higher-frequency data where the manifold structure might be even more informative. And I want to get more production engineering experience — my Steadfast internship will help with that."

**Q: "Why physics + finance?"**
A: "Physics trained me to build mathematical models of complex systems and validate them against data. Finance is one of the richest sources of high-quality, timestamped data about complex adaptive systems. The tools transfer surprisingly well — differential geometry, spectral theory, statistical mechanics. And the feedback loop is immediate: your model either makes money or it doesn't."

### Curveball Questions

**Q: "What's your biggest failure?"**
A: "My initial QCML approach had a subtle data leakage bug in the PCA pipeline — the scaler was fit on the full dataset instead of just the training window. When I caught it and fixed it, the results were honest but weaker. I rewrote the paper around 'competitive and complementary' instead of 'beats everything.' CUSUM actually has a higher mean effect size than any individual geometric method — but the geometric signals catch different crises, so the ensemble is strongest. That taught me more about research integrity than any class."

**Q: "What's the poker club about?"**
A: "I co-founded the poker club. Poker is applied decision theory under uncertainty — Bayesian updating, expected value, risk management, reading incomplete information. The skills transfer directly to quantitative thinking."

**Q: "How do you balance research, teaching, and extracurriculars?"**
A: "Aggressive prioritization and systems thinking. I block time for deep research work, automate what I can (CI/CD pipelines, automated testing), and recognize that teaching actually sharpens my understanding of the research. The climbing and music are non-negotiable — they're how I recharge."

---

## Do's and Don'ts

### Do
- Make eye contact and read the recruiter's name badge
- Gesture at the poster sections as you talk about them
- Ask what they're working on after your pitch
- Have your phone ready with LinkedIn QR code
- Bring business cards if you have them
- Stand to the side of the poster, not in front of it

### Don't
- Don't read from the poster verbatim
- Don't go over 60 seconds for the opening pitch
- Don't dive into math unless they ask
- Don't apologize for being "just a sophomore"
- Don't say "quantum" without immediately clarifying it's differential geometry
- Don't oversell — the honest framing (competitive, not dominant) is more credible
