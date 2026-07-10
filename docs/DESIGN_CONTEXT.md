# Design Context

**Version:** 1.0  
**Last Updated:** 2026-07-06  
**Author:** Biswajyoti Nath  
**Breaking Changes:** None (initial version)

> Living document. Every module author must read this before writing code.
> Update this file whenever a design decision is made or an assumption changes.

## 0. Probability Space

Let Ω denote the set of all expressions producible by the stochastic grammar G.
Let P be the probability measure induced by the recursive generation process
(defined in §1).

Every Monte Carlo estimate approximates:

    ρ(C) = P(E ∈ C)    for E ~ P

All reported interaction coefficients M(i,j) are therefore conditional on the
generator-induced distribution P. They do not represent properties of "all
possible expressions" — only of the distribution our generator defines.

## 1. Sampling Distribution

- `ρ(C)` is defined over the **stochastic grammar process**, not uniform over the
  language `L(G)`. The process is:
  - 50/50 binary vs unary at each internal node.
  - `leaf_probability` (config) chance of early leaf return at depth > 0.
  - 50/50 variable vs constant at each leaf.
  - Constants: uniform over `[min, max]`, 50% sign flip.
- **Changing `leaf_probability` changes ρ.** Document this in every results table.
- `M(i,j)` is only meaningful relative to this distribution. State this in the
  handbook and in any paper/report.

## 2. Expression Representation

- `sympify(evaluate=False)` preserves tree structure for constraint checking.
  Without `evaluate=False`, SymPy auto-simplifies `x + x` → `2*x`, which would
  destroy the AST that constraints inspect.
- Domain validity is checked **separately** via numeric probing (`_validate_domain`).
  This is a second pass that catches what `evaluate=False` misses: NaN, ∞, complex
  results from trig-of-large-values, etc.
- An expression that passes `sympify` but fails domain validation is **rejected**
  and counted in `stats['domain_rejected']`.

## 3. Depth Semantics

- **"depth d"** means: at least one root-to-leaf path has length exactly `d`.
- `generate(max_depth=d)` produces trees with depth **≤ d** (stochastic; actual
  depth varies due to early leaf returns).
- `generate_at_depth(d)` produces trees with depth **== d** (guaranteed; at least
  one forced path reaches depth 0 without early termination).
- Stratified sampling uses `generate_at_depth` to produce equal buckets per depth.

## 4. Stats Contract

- Every call to `generate_sympy` or `generate_sympy_at_depth` increments
  `stats['generated']`.
- Downstream code (`metric.py`, `experiments.py`) **must** call `reset_stats()`
  before each sampling run and **log stats after**, so rejection rates are traceable.
- Stats keys:
  - `generated`: total generation attempts
  - `sympify_failed`: expressions that didn't parse
  - `domain_rejected`: parsed but failed numeric probing
  - `returned_none`: exhausted all retries without a valid expression

## 5. Config as Single Source of Truth

- All tunable parameters live in `config.yaml`. No magic numbers in code.
- `leaf_probability`, `domain_validation` settings, `depths`, `N` — all in config.
- If a parameter is used in more than one module, it must come from config, not
  be duplicated.

## 6. Module Invariants

### 6.1 Generator (`expr_generator.py`)

Every returned expression must:
- Be syntactically valid (parseable by `sympify`)
- Pass domain validation (finite real values at test points)
- Satisfy the requested depth constraint (≤ d or == d)
- Be reproducible given an identical random seed
- Never be mutated after generation

### 6.2 Constraints (`constraints.py`) — Implemented

Each constraint C must:
- Be a pure function: `sympy.Expr → bool`
- Not mutate the input expression
- Be deterministic (fixed RNG seed if randomness is needed)
- Handle edge cases (constants, single-variable, deeply nested) 
  without raising — return False if invalid
- Be independent of other constraints (no shared mutable state)

> C2 is a documented exception to the §7 note on depth semantics. It uses SymPy `.args`-walk depth. This is the only depth computable from a pure `Expr → bool` function. The bias is bounded and measured empirically — see `test_depth_divergence_measured`.


### 6.3 Metric (`metric.py`) — Implemented

- Must call `generator.reset_stats()` before each sampling run (✓ enforced in `DensityEstimator.estimate()`).
- Must log `generator.stats` after each run (✓ stored in `RhoResult.stats`).
- M(i,j) must be symmetric: M[i][j] == M[j][i] (✓ explicit mirror loop in `InteractionMatrix.compute()`).
- Must report bootstrap CIs alongside point estimates (✓ `BootstrapCI.compute()`).
- Must handle ρ = 0 gracefully (✓ M → NaN, not crash; NaN mirrored to lower triangle).
- `N_actual` == requested `N` always (✓ while-loop retries None draws; see `RhoResult` docstring).
- `grammar_config` can be passed to `BootstrapCI.compute()` for §9 reproducibility snapshotting.

### 6.4 Experiments (`experiments.py`) — TODO

- Must log all reproducibility metadata (see §9)
- Must use fixed seeds from config
- Must not modify config during a run
- Results must be append-only (no overwriting previous runs)

## 7. Known Gotchas

- **SymPy `zoo` (complex infinity):** `sympify("1/0", evaluate=False)` does NOT
  raise — it returns `zoo`. The domain validation probe catches this numerically.
- **Large trig arguments:** `sin(10**20)` is technically valid but numerically
  meaningless. Domain validation with `test_range=[-10, 10]` won't catch expressions
  that only blow up outside this range. This is acceptable for our purposes.
- **Expression equivalence:** Two syntactically different trees can be semantically
  identical (e.g., `x + y` and `y + x`). We do **not** deduplicate — ρ is defined
  over the generative distribution, not over unique semantic expressions.
- **Floating-point overflow:** Deeply nested multiplication trees can produce
  values exceeding `float64` range. Domain validation catches this via
  `np.isfinite`, but be aware when interpreting rejection rates.
- **Monte Carlo estimates depend on grammar parameters.** Changing the grammar
  (operators, leaf_probability, depth) invalidates all previous ρ and M estimates.
  Re-run the full pipeline if grammar changes.
- **PySR search distribution ≠ generator distribution.** PySR's evolutionary
  search explores expressions differently from our uniform-random generator.
  M(i,j) predicts *structural* interactions in the grammar space, not exact
  search dynamics. Results should be interpreted as approximations.
- **SymPy tree depth ≠ generator depth.** `sympify("-x", evaluate=False)` returns
  `Mul(-1, x)` (depth 1 in SymPy), not `Integer(-7)` (depth 0). Always measure
  depth on the generated string, not on SymPy's internal tree.
- **C1b SymPy flattening:** `(a+b)+c` → `Add(a,b,c)` even under `evaluate=False` for repeated application. C1b checks `len(args)`, not tree depth.
- **C2 SymPy vs. generator depth:** SymPy depth ≥ generator depth for expressions with negative constants or subtraction (gap ≤ +2 per affected node). Measured by `test_depth_divergence_measured`.
- **C3 `x*x` vs. `x**2`:** Under `evaluate=False`, `x*x` stays `Mul(x,x)` (accepted). Only `x**2` literal produces `Pow` (rejected). Generator never produces `Pow` since it uses `evaluate=False`.
- **C4 non-negativity name mismatch:** Predicate is `>= 0` (Research Plan §5). Config key `positivity` implies `> 0`. Boundary case `f=0` → True. Comment in `config.yaml` clarifies.

## 8. Computational Complexity

| Operation | Complexity | Variables |
|-----------|-----------|-----------|
| Expression generation | O(2^d) worst case | d = depth |
| Single constraint evaluation | O(n) | n = AST nodes |
| Monte Carlo density estimation | O(N · k) | N = samples, k = constraints |
| Interaction matrix computation | O(N · k²) | Full pairwise |
| Bootstrap CI (B resamples) | O(B · k²) | B = 1000 default |
| PySR single run | O(pop · iters · n) | pop, iters from config |
| Full experiment matrix | O(scenarios · repeats · PySR) | |

For N = 100,000 and k = 5: the Monte Carlo pass is ~500k constraint
evaluations. At ~1ms per evaluation, this is ~8 minutes single-threaded,
~2 minutes on 4 cores with joblib.

## 9. Reproducibility Contract

Every experiment run must log:

| Field | Source | Example |
|-------|--------|---------|
| Git commit hash | `git rev-parse HEAD` | `a1b2c3d` |
| Config file hash | `hashlib.sha256(config_bytes)` | `e4f5...` |
| Random seed | `config.yaml → experiments.seeds` | `42` |
| Sample size N | `config.yaml → monte_carlo.N` | `100000` |
| Grammar version | `config.yaml → grammar` (full dict) | — |
| Timestamp | `datetime.utcnow()` | `2026-07-06T15:00:00Z` |
| Python version | `sys.version` | `3.12.3` |
| Key package versions | `sympy`, `numpy`, `pysr` | `1.12, 1.26, 0.19` |

Logs are written to `results/<run_id>/metadata.json`. This file is
created before the experiment starts and updated with results after.

## 10. Statistical Assumptions

1. **IID sampling.** Bootstrap and Monte Carlo estimates assume that sampled
   expressions are independent and identically distributed under the
   grammar-induced distribution P. This holds because each call to
   `generate_sympy` draws independently from the same stochastic process.

2. **Law of Large Numbers.** Monte Carlo estimates ρ̂(C) converge to ρ(C)
   as N → ∞. Convergence is verified empirically by comparing estimates
   at N = 1k, 10k, 100k (§7.3 of Research Plan).

3. **Finite-sample variance.** Reported via B = 1000 bootstrap resamples.
   95% confidence intervals are constructed using the percentile method.

4. **Multiple comparisons.** When testing multiple (i,j) pairs for
   significance, Benjamini–Hochberg FDR correction is applied at α = 0.05.
