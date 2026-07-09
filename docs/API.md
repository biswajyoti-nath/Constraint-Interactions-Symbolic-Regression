# API Reference

This document provides a comprehensive API reference for the public interfaces in the `Constraint-Interaction-SymREG` framework.

## Module: `src.expr_generator`

### Class: `GrammarGenerator`

The core engine for generating stochastic mathematical expressions (Abstract Syntax Trees) based on the definitions in `config.yaml`.

#### Instantiation

```python
from src.expr_generator import GrammarGenerator
generator = GrammarGenerator(config_path="config.yaml")
```

#### Properties

- **`stats`** (`dict`): Returns a dictionary containing the current generation statistics.
  - Keys: `generated`, `sympify_failed`, `domain_rejected`, `returned_none`.

#### Methods

##### `reset_stats()`
Resets all generation counters in the `stats` dictionary to zero. Should be called before starting a new batch of Monte Carlo sampling.

##### `generate(max_depth=None)`
Generates a random mathematical expression as a string. Produces a tree with `depth ≤ max_depth` (the actual depth varies stochastically due to early leaf returns).
- **Args**:
  - `max_depth` (int, optional): The maximum depth of the expression tree. Defaults to the value in `config.yaml`.
- **Returns**: `str` representing the mathematical expression.

##### `generate_at_depth(target_depth)`
Generates an expression whose tree depth is exactly `target_depth`. This guarantees that at least one root-to-leaf path has a length of exactly `target_depth`.
- **Args**:
  - `target_depth` (int): The exact depth to produce.
- **Returns**: `str` representing the mathematical expression.

##### `generate_sympy(max_depth=None, max_retries=10)`
Generates a valid SymPy expression (depth ≤ max_depth). Repeatedly generates strings, parses them into SymPy objects (`evaluate=False`), and numerically probes them for domain errors (e.g., division by zero, complex results) until a valid expression is found.
- **Args**:
  - `max_depth` (int, optional): Maximum depth of the expression tree.
  - `max_retries` (int): Maximum generation attempts before giving up. Defaults to 10.
- **Returns**: `sympy.Expr` or `None` if all retries failed.

##### `generate_sympy_at_depth(target_depth, max_retries=10)`
Generates a valid SymPy expression at exactly `target_depth`. Uses the same parsing and domain validation logic as `generate_sympy`.
- **Args**:
  - `target_depth` (int): The exact tree depth to produce.
  - `max_retries` (int): Maximum generation attempts. Defaults to 10.
- **Returns**: `sympy.Expr` or `None` if all retries failed.

##### `generate_stratified(depths=None, n_per_depth=None)`
Generates expressions stratified equally across target depths (e.g., 2, 4, 6, 8). This is the primary method for gathering uniformly distributed samples for Monte Carlo density estimation.
- **Args**:
  - `depths` (list[int], optional): The depth buckets to sample. Defaults to the values in `config.yaml`.
  - `n_per_depth` (int, optional): The number of valid samples to generate per bucket. Defaults to $N / |depths|$ from `config.yaml`.
- **Returns**: `dict[int, list[sympy.Expr]]` mapping each depth bucket to a list of valid SymPy expressions.

---

> **Note on Internal Helpers**: The `GrammarGenerator` class contains several internal helper methods (e.g., `_random_leaf`, `_generate_recursive`, `_generate_exact_depth`, `_validate_domain`). These are deliberately excluded from this public API reference to maintain a clean interface, but advanced users subclassing the generator may override them to implement custom generation behaviors.

---

## Module: `src.metric`

Implements Algorithm 1 from the Research Plan (§6). Provides Monte Carlo density estimation, the pairwise interaction matrix, and bootstrap confidence intervals. All estimates are conditional on the generator-induced distribution P (DESIGN_CONTEXT §0).

### Class: `RhoResult`

Data container returned by `DensityEstimator.estimate()`. A plain dataclass — read-only after creation.

| Attribute | Type | Description |
|---|---|---|
| `rho_i` | `np.ndarray` shape `(k,)` | `rho_i[i]` = ρ(C_i) |
| `rho_ij` | `np.ndarray` shape `(k, k)` | `rho_ij[i][j]` = ρ(C_i ∧ C_j); upper triangle filled by estimator |
| `N_actual` | `int` | Always equals requested `N` (while-loop guarantees this) |
| `stats` | `dict` | Generator stats snapshot: `generated`, `sympify_failed`, `domain_rejected`, `returned_none` |

### Class: `DensityEstimator`

Monte Carlo estimator for ρ(C_i) and ρ(C_i ∧ C_j). Single-pass over N expressions.

#### Instantiation

```python
from src.metric import DensityEstimator
est = DensityEstimator(generator, max_depth=4)
```

- `generator` (`GrammarGenerator`): A configured generator instance.
- `max_depth` (`int | None`): Depth passed to `generate_sympy()`. Uses `generator.max_depth` if `None`.

#### Methods

##### `estimate(constraints, N) → RhoResult`

Draw exactly `N` valid expressions and count per-constraint and pairwise hits.

- **Args**:
  - `constraints` (`list[Callable[[sympy.Expr], bool]]`): Pure constraint functions (DESIGN_CONTEXT §6.2).
  - `N` (`int`): Exact number of valid expressions to collect.
- **Returns**: `RhoResult`.
- **Side effects**: Calls `generator.reset_stats()` before and captures `generator.stats` after.

### Class: `InteractionMatrix`

Computes the symmetric pairwise interaction matrix M(i,j) from a `RhoResult`.

#### Methods

##### `compute(rho_result) → np.ndarray`

- **Args**: `rho_result` (`RhoResult`) — output of `DensityEstimator.estimate()`.
- **Returns**: `ndarray` of shape `(k, k)`. Diagonal is `NaN`. Off-diagonal: `M[i][j] = ρ(C_i∧C_j) / (ρ(C_i)·ρ(C_j))`. Lower triangle is explicitly mirrored from upper triangle.
- **NaN handling**: If either `ρ_i = 0` or `ρ_j = 0`, both `M[i][j]` and `M[j][i]` are set to `NaN` (not a crash).

### Class: `BootstrapCI`

95% bootstrap confidence intervals for M(i,j) using the percentile method.

#### Instantiation

```python
from src.metric import BootstrapCI
ci_estimator = BootstrapCI(B=1000, seed=42)
```

#### Methods

##### `compute(expressions, constraints, grammar_config=None) → np.ndarray`

- **Args**:
  - `expressions` (`list[sympy.Expr]`): Pre-drawn expression list.
  - `constraints` (`list`): Constraint callables.
  - `grammar_config` (`dict | None`): `generator.config` dict for §9 reproducibility snapshotting. Stored as `self.grammar_config` after the call. Pass `None` in unit tests.
- **Returns**: `ndarray` of shape `(k, k, 2)`. `[..., 0]` = lower bound, `[..., 1]` = upper bound of 95% CI.

### Function: `run_benchmark`

```python
from src.metric import run_benchmark
record = run_benchmark(generator, N=1000, max_depth=4, output_path="results/benchmark_1k.json")
```

Calibration utility. Runs the estimator with an always-true constraint and logs observed draw statistics to a JSON file. Does **not** assert hard thresholds (those are calibrated from real output per Research Plan §8.3).

- **Returns**: `dict` with keys `N`, `max_depth`, `elapsed_s`, `stats`, `rho_always_true`.

---

> **Note on Planned Modules**: `src/constraints.py` and `src/experiments.py` are not yet implemented. Their public APIs will be documented here as they are built.
