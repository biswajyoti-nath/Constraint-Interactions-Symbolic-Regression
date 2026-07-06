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
