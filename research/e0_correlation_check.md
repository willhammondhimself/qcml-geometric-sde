# E_0 ground state energy: correlation check

Date: 2026-04-04
Cross-pollination memo item #1

## Question

Is E_0(x) = min eigenvalue of H(x) just a proxy for realized volatility?

## Answer

No. Pearson r = -0.048 between E_0 and 20-day rolling realized vol (N = 6723).
Spearman r = -0.149. Essentially uncorrelated.

## Crisis separability (4 smoke crises, n=8, random operators, no HPO)

| Crisis | E_0 d | Vol Z d | Winner |
|--------|-------|---------|--------|
| 2008 GFC | 1.17 | 1.72 | Vol |
| 2020 COVID | 0.74 | 2.26 | Vol |
| 2018 Volmageddon | 0.35 | 0.18 | E_0 |
| 2022 Rates | 0.03 | 0.41 | Vol |

## Interpretation

E_0 captures structural changes in the Hamiltonian landscape that are orthogonal to volatility.
It's particularly sensitive to flash/structural events (Volmageddon) where vol is a poor indicator.

The ideation log (Q107) reported d=1.411 with HPO-tuned parameters on 4 crises. Our smoke test
with default random operators gives d=1.17 on GFC, d=0.74 on COVID — weaker but still strong,
and genuinely independent from vol.

## Next steps

- Run on full 17-crisis panel with HPO (proper comparison)
- If median d > 0.5 and |r_vol| < 0.3, promote to Paper 1 as a 5th observable
- If not, save for Paper 2 as an orthogonality demonstration

## Setup

- Hilbert dim: 8
- Operator method: random (seed 42)
- Features: 52-dimensional enriched feature matrix (SPY/DIA)
- No HPO — default parameters
