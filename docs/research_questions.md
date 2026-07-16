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

---

### RQ-05: How severely does Julia JIT compilation skew search cost metrics?
**Status:** Resolved
*Notes:* During pipeline validation, we discovered that the first PySR scenario in any Python process incurs a massive Julia JIT compilation penalty, heavily inflating the wall-clock time (which we use as our proxy for $S_i$). This was resolved by adding a `exclude_jit=True` aggregation stage to drop the first run of a batch.

---

### RQ-06: What are the hidden discrepancies between theoretical constraints and GP implementations?
**Status:** Resolved
*Notes:* The theoretical Monte Carlo model easily restricted both nested trig (C1a) and consecutive binary operators (C1b). However, we realized PySR's internal grammar doesn't natively enforce C1b, only general tree depth and unary nesting. We had to officially decouple C1a and C1b to prevent $M(i,j)$ from making predictions on constraints PySR wasn't actually evaluating.

---

### RQ-07: Is structural equivalence a viable metric for SR success, or is numerical approximation strictly better?
**Status:** Resolved
*Notes:* We initially attempted to verify ground-truth recovery using SymPy (`check_semantic_exact`). However, evolutionary search often produces mathematically equivalent but structurally chaotic expressions that SymPy `simplify()` fails to reduce or times out on. We were forced to demote symbolic equivalence to an exploratory metric, relying solely on relative numerical MSE to declare a successful run.

---

### RQ-08: How do we handle right-censored search costs (timeouts) without biasing $S(i,j)$?
**Status:** Open
*Notes:* Highly restrictive constraints frequently cause PySR to fail to converge within the 300s timeout. If we simply drop these `timeout_hit` runs, we introduce survivorship bias into $S(i,j)$ (only fast runs are recorded). How much does this data attrition skew the empirical interaction effect, and is our statistical power robust up to the projected 30% attrition rate?
