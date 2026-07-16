# Research Questions Tracker

This document tracks interesting methodological and theoretical questions that emerge during the internship. Rather than solving them immediately, they are recorded here to form the foundation of the final manuscript's Discussion section.

---

### RQ-01: Does $M(i,j)$ predict $S(i,j)$?
**Status:** Open
*Notes:* The core hypothesis of the project. We are testing whether the theoretical density interaction metric $M(i,j)$ has a strong positive correlation (Pearson $r > 0.6$) with the empirical search speedup $S(i,j)$ across PySR runs.

---

### RQ-02: How sensitive is $M(i,j)$ to grammar depth?
**Status:** Running
*Notes:* We currently stratify the Monte Carlo sampling across depths 2, 4, 6, and 8. Does the constraint density $\rho(C)$ collapse to zero for specific constraints at higher depths? How does this impact the interaction term?

---

### RQ-03: How stable are bootstrap intervals?
**Status:** Closed
*Notes:* Addressed during the metric calibration phase. Bootstrap CIs become extremely unstable or infinite when $\rho(C) < 1/N$. We established the invariant to flag $M$ as NaN when the base density is too low for the sampling budget.

---

### RQ-04: How much does PySR's search distribution differ from uniform sampling distribution?
**Status:** Resolved
*Notes:* Evolutionary programming (PySR) inherently biases the search away from uniform grammar sampling. Using a 1-iteration Hall-of-Fame proxy (distribution audit), we found the divergence between empirical HoF $\rho$ and Monte Carlo $\rho$ is strictly less than an order of magnitude (max 2.94x for C1a). This confirms $M(i,j)$ retains directional predictive power.
