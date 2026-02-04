# References

This project builds upon the groundbreaking QCML (Quantum Cognition Machine Learning) framework developed by Qognitive, Inc. and collaborators. The following papers form the theoretical and empirical foundation for our work on geometric SDEs and topological regime detection.

## Core QCML Framework Papers

### 1. Foundational Theory - Intrinsic Dimension Estimation

**Candelori, L., Abanov, A. G., Berger, J., Hogan, C. J., Kirakosyan, V., Musaelian, K., Samson, R., Smith, J. E. T., Villani, D., Wells, M. T., & Xu, M.** (2025). Robust estimation of the intrinsic dimension of data sets with quantum cognition machine learning. *Scientific Reports*, *15*, Article 6933. https://doi.org/10.1038/s41598-025-91676-8

> **Key contribution**: Introduces spectral gap detection for robust intrinsic dimension estimation, avoiding "shadow dimensions" that plague classical methods. Demonstrates O(N²) complexity advantage over exponential scaling of classical approaches.

### 2. Quantum Geometry Theory

**Abanov, A. G., Candelori, L., Steinacker, H. C., Wells, M. T., Busemeyer, J. R., Hogan, C. J., Kirakosyan, V., Marzari, N., Pinnamaneni, S., Villani, D., Xu, M., & Musaelian, K.** (2025). Quantum geometry of data. *arXiv preprint* arXiv:2507.21135v1 [cs.LG]. https://arxiv.org/abs/2507.21135

> **Key contribution**: Establishes the mathematical connection between QCML and quantum geometry, introducing quantum metric tensor, Berry curvature, and Chern numbers as tools for data analysis. Demonstrates how QCML learns the geometric and topological structure of data manifolds.

### 3. QCML Framework Overview

**Musaelian, K., Abanov, A., Berger, J., Candelori, L., Kirakosyan, V., Samson, R., Smith, J., & Villani, D.** (2024, March 1). Quantum cognition machine learning: AI needs quantum. *Qognitive, Inc. White Paper*. Miami Beach, FL.

> **Key contribution**: Comprehensive overview of QCML's advantages over classical machine learning, including logarithmic economy of representation and ability to handle high-dimensional, low-sample-size data without curse of dimensionality.

---

## Financial Applications

### 4. Equity Forecasting

**Samson, R., Berger, J., Candelori, L., Kirakosyan, V., Musaelian, K., & Villani, D.** (2024, September 20). Quantum cognition machine learning: Financial forecasting. *arXiv preprint* arXiv:2409.XXXXX. Qognitive, Inc., Wayne State University, Duality Group.

> **Key contribution**: First application of QCML to equity forecasting, demonstrating superior Sharpe ratios (0.69-1.25) compared to classical methods on S&P 500 constituents.

### 5. Corporate Bond Similarity

**Rosaler, J., Candelori, L., Kirakosyan, V., Musaelian, K., Samson, R., Wells, M. T., Mehta, D., & Pasquali, S.** (2025). Supervised similarity for high-yield corporate bonds with quantum cognition machine learning. *arXiv preprint* arXiv:2502.01495v1 [q-fin.ST]. https://arxiv.org/abs/2502.01495

> **Key contribution**: QCML-based supervised similarity learning for corporate bonds outperforms random forest baselines in high-yield markets. Demonstrates robustness to noise and ability to handle sparse, illiquid data characteristic of fixed income markets.

### 6. Firm Linkages and Momentum Spillover

**Samson, R., Banner, A., Candelori, L., Cottrell, S., Di Matteo, T., Duchnowski, P., Kirakosyan, V., Marques, J., Musaelian, K., Pasquali, S., Stever, R., & Villani, D.** (2025). Supervised similarity for firm linkages. *arXiv preprint* arXiv:2506.19856v1 [q-fin.ST]. https://arxiv.org/abs/2506.19856

> **Key contribution**: Introduces Characteristic Vector Linkages (CVLs) for firm similarity, enabling profitable momentum spillover trading strategies (Sharpe ratio 1.42). Demonstrates QCML's superiority over Euclidean distance-based methods for capturing complex firm relationships.

---

## Biomedical Application (Demonstrating Generalizability)

### 7. Cancer Prediction

**Di Caro, G., Kirakosyan, V., Abanov, A. G., Busemeyer, J. R., Candelori, L., Hartmann, N., Lam, E. T., Musaelian, K., Samson, R., Steinacker, H., Villani, D., Wells, M. T., Wenstrup, R. J., & Xu, M.** (2025). Quantum cognition machine learning for forecasting chromosomal instability. *arXiv preprint* arXiv:2506.03199v2 [q-bio.QM]. https://arxiv.org/abs/2506.03199

> **Key contribution**: QCML achieves highest AUC-ROC score for predicting chromosomal instability in breast cancer CTCs, outperforming classical SVM and machine learning methods. Demonstrates QCML's generalizability beyond finance to high-dimensional biomedical data.

---

## Media Coverage

### 8. Industry Recognition

**Cesa, M., & Mannix, R.** (2023, December). AI model uses quantum maths to learn like a human. *Risk.net*. https://www.risk.net/

> **Overview**: Industry publication covering Qognitive's QCML framework and its application to financial markets, highlighting the quantum-inspired approach to machine learning.

---

## Our Contribution: Geometric SDEs and Topological Regime Detection

This repository (`qcml-geometric-sde`) extends the QCML framework in three groundbreaking directions:

1. **Geometric Stochastic Differential Equations**: We model financial dynamics as SDEs on the QCML-learned manifold, where drift and diffusion respect the quantum metric tensor: $dX^a = \mu^a(X)dt + \sigma^a_b(X)dW^b$ with $\Sigma^{ab} \propto g^{-1}$.

2. **Topological Regime Detection**: We use Chern numbers (topological invariants computed from Berry curvature) for robust market regime classification. The key insight: $\Delta C = 0$ indicates an extreme event within the same regime, while $\Delta C \neq 0$ signals a topological transition (true regime change).

3. **Trading Signal Generation**: We generate trading signals from geometric and topological features:
   - Berry curvature spikes → market stress indicators
   - Chern number transitions → regime change signals
   - Quantum metric divergence → volatility expansion signals
   - Spectral gap compression → instability warnings

**Novel contributions**:
- First combination of QCML geometry with stochastic dynamics
- First use of Chern numbers for financial regime detection
- Theoretical framework bridging quantum geometry, SDEs, and mathematical finance
- Empirical validation framework for historical crisis periods (2008, 2020, etc.)

---

## Citation Format

When citing this work, please use:

**APA Format**:
```
Hammond, W. (2025). qcml-geometric-sde: Topological regime detection via quantum
geometry and geometric SDEs [Software]. GitHub.
https://github.com/[username]/qcml-geometric-sde
```

**BibTeX Format**:
```bibtex
@software{hammond2025qcml,
  author = {Hammond, Will},
  title = {{qcml-geometric-sde}: Topological Regime Detection via Quantum Geometry
           and Geometric SDEs},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/[username]/qcml-geometric-sde},
  note = {Building upon QCML framework by Qognitive, Inc.}
}
```

---

## Acknowledgments

This work builds directly upon the QCML framework developed by:
- **Qognitive, Inc.** (Kharen Musaelian, Ryan Samson, Jeffrey Berger, Dario Villani, and team)
- **Academic collaborators**: Alexander G. Abanov (Stony Brook), Luca Candelori (Wayne State), Martin T. Wells (Cornell), Jerome R. Busemeyer (Indiana), Vahagn Kirakosyan (Wayne State), and others

We are grateful to the Qognitive team for pioneering the quantum cognition approach to machine learning and for sharing their research openly.

---

## Additional Resources

- **Qognitive, Inc.**: https://www.qognitive.io/
- **QCML Documentation**: Contact Qognitive for access to implementation details
- **arXiv QCML Papers**: Search for "quantum cognition machine learning" or "QCML"

---

*Last updated: February 2025*
