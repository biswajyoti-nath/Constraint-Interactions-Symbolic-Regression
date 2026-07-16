# Project Logbook
**Project:** Constraint-Interaction Metric for Symbolic Regression
**Team Members:** Biswajyoti Nath & Subinoy Nath
**Program:** TIH IITG Summer Internship 2026

---

### Entry 01: Orientation and Ideation
**Date:** 02/07/2026
**Supervising Faculty:** Dr. Mahapara Khureshi

- Attended the official internship orientation session and collected the introductory kits.
- Participated in an introductory lecture on Artificial Intelligence and Machine Learning (AI/ML).
- Initiated preliminary discussions regarding potential topics for our summer research project.

---

### Entry 02: Topic Finalization
**Date:** 03/07/2026
**Supervising Faculty:** Dr. Mahapara Khureshi

- Attended a specialized session on "Bio-Signals and the Application of AI/ML."
- Successfully finalized our internship research topic: **"Constraint-Interaction Metric for Symbolic Regression."**

---

### Entry 03: Environment Setup and Code Initialization
**Date:** 06/07/2026

- Initialized the project's GitHub repository and established the standard directory structure (src, data, docs, tests).
- Configured the Python environment, including the installation of PySR, Julia, and other core scientific dependencies.
- Developed the foundational `expr_generator.py` script to generate grammar-based mathematical abstract syntax trees (ASTs).
- Conducted a preliminary literature review to inform our grammar and constraint definitions.

---

### Entry 04: Expression Generator — Hardening and Formal Documentation
**Date:** 07/07/2026

- **Domain Validation Fix**: Discovered that `sympify(evaluate=False)` silently passes mathematically invalid expressions (e.g., division by zero). Implemented a two-pass system: AST parsing followed by a numeric probe (`_validate_domain`) that rejects NaN, infinity, and complex results.
- **Stratified Sampling**: Implemented `generate_at_depth()` and `generate_stratified()` to guarantee uniform representation across depth buckets {2, 4, 6, 8}, preventing depth bias in Monte Carlo density estimates.
- **Reproducibility**: Isolated the global Python random state into a local `random.Random` instance seeded from `config.yaml`. All generation statistics (`generated`, `sympify_failed`, `domain_rejected`, `returned_none`) are tracked per-run.
- **Formal Specification**: Upgraded design notes to `DESIGN_CONTEXT.md` (v1.0), formally defining the probability space Ω, module invariants, reproducibility contract, and computational complexity bounds.
- **Documentation & Testing**: Applied Google-style docstrings with 8 embedded doctests. Generated `API.md` and `ARCHITECTURE.md`. Written 16-test suite; all pass. Codebase formatted with `ruff`.
- **Commit**: `a1856b0` — "chore: sanitize codebase, formalize design context, and document expression generator"

---

### Entry 05: Monte Carlo Estimator — Implementation
**Date:** 09/07/2026

- **`metric.py` implemented**: Built `DensityEstimator` (Algorithm 1 from Research Plan §6), `InteractionMatrix`, and `BootstrapCI`.
  - `DensityEstimator.estimate()`: Draws exactly N valid expressions via a retry loop; single-pass constraint evaluation in O(N·k²).
  - `InteractionMatrix.compute()`: Fills upper triangle then **explicitly mirrors** to lower triangle — guarantees M[i][j] == M[j][i] (DESIGN_CONTEXT §6.3). NaN-safe for zero-density constraints.
  - `BootstrapCI.compute()`: B=1000 resample percentile method; accepts `grammar_config` for §9 reproducibility snapshotting.
- **`run_benchmark()`**: Calibration utility that logs observed draw statistics to `results/benchmark_1k.json` without asserting guessed thresholds.
- **3 docstring/API fixes** after code review: corrected `N_actual` docstring (while-loop guarantees N_actual == N always), removed unused `generator` param from `BootstrapCI.compute()`, hardened doctest path resolution to use `__file__` instead of CWD.
- **Testing**: 33 tests pass (16 generator + 15 metric + 2 doctests). Includes `sympify(evaluate=False)` regression test.
- **Commits**: `499a1a3`, `ad55e4e`.

---

### Entry 06: Constraints Implementation (C1–C4)
**Date:** 10/07/2026

- **`constraints.py` implemented**: Built the core constraint factories `make_c1_structural`, `make_c2_depth`, `make_c3_operator`, `make_c4_positivity`, and the `build_constraints` aggregator.
  - C1 (Structural): Checks maximum nested trig depth and consecutive binary ops. Handled SymPy flattening by checking `len(args)`.
  - C2 (Depth): Uses SymPy `.args`-walk depth. Documented the divergence (bounded gap) for negative constants and subtraction vs. generator depth.
  - C3 (Operator Whitelist): Pure AST node type check. Avoids `x**2` / `x*x` edge cases thanks to `evaluate=False`.
  - C4 (Positivity): Validates f(x,y) >= 0. Uses a fast, vectorized `lambdify` path with a robust `.subs()` fallback.
- **Decisions Made & Documented**: Kept C4 name as `positivity` (implements >=0), C2 depth uses SymPy args-walk, and RNG seed for C4 is pulled from `config.yaml`.
- **Testing**: Added rigorous testing for boundary conditions, depth divergence measurement, and fallback pathways. The test suite now totals 60 passing tests.
- **Documentation**: Updated `DESIGN_CONTEXT.md` (§6.2 and §7 gotchas) and `API.md` (Constraints module).

---

### Entry 07: PySR Integration and Experiment Orchestration
**Date:** 16/07/2026

- **`experiments.py` implemented**: Implemented the four-layer PySR benchmarking module to produce raw search cost data $S(i,j)$.
  - **Dataset Loaders**: Created deterministic loaders for `feynman_ke`, `feynman_coulomb`, `polynomial`, and `srsd_dummy`.
  - **PySR Constraint Mapper & Composer**: Mapped active constraints dynamically to PySR config parameters (C1 nested trig limits, C2 maxdepth, C3 operator whitelist intersection, C4 Julia custom loss penalty).
  - **Scenario Runner**: Configured isolated serial deterministic PySR runs, logging wall-clock search timings, calculating MSE/relative MSE/NED (DP Levenshtein fallback), and running post-hoc compliance checks.
  - **Orchestrator**: Orchestrates the benchmark runs, writing Git/environment metadata to `metadata.json` and logging results incrementally to a run-specific CSV.
- **Dependencies**: Added `scikit-learn>=1.3` to `requirements.txt`.
- **Testing**: Implemented `tests/test_experiments.py` covering shapes, determinism, ground truth recovery, Levenshtein fallback, and constraint merging. Test suite expanded from 60 to 83 passing tests.
- **Documentation**: Updated `DESIGN_CONTEXT.md` (§6.4 invariants) and logged progress.

---

### Entry 08: Benchmark Validation & The "M ≠ S" Gap Live Observation
**Date:** 16/07/2026

- Ran a rapid benchmark test of the `feynman_ke` dataset using the baseline and `['C1']` scenarios to ensure cluster readiness.
- **Critical Finding for Paper/Handbook (Ch. 13)**: The baseline run *successfully* recovered the kinetic energy formula `0.5 * m * v**2`, represented by PySR as `((v * m) * (v * 0.5)) - 1.2630079e-7`.
- However, our `constraints.py` post-hoc compliance check flagged `C1_Satisfied = False`.
- **Why?** SymPy evaluates PySR's output as an n-ary flattened tree: `Add(Mul(v, m, v, 0.5), -1.263...)`. The `Mul` node has **4** arguments. Our structural constraint C1b dictates `max_consecutive_binary=3`. Since PySR lacks an internal mechanism to restrict sibling counts during its search, it output a mathematically correct but structurally violating equation.
- This perfectly validates the architectural decision to include strict post-hoc compliance checks and empirically confirms the "M ≠ S" (Measurement vs. Search) gap documented in the PRD. We now have real, reproducible data to write about in the handbook.

---

### Entry 09: Early-Stop Smoke Test — S(i,j) Signal Validation
**Date:** 16/07/2026

- **Blocking concern:** Without `early_stop_condition`, PySR terminates at `niterations` regardless of convergence quality. Wall-clock time would be constant across scenarios (~35s each), and $S(i,j)$ would collapse to ~1 for all pairs. This would silently invalidate the entire empirical arm.
- **Fix applied:** Wired `early_stop_condition = "f(loss, complexity) = (loss < 1e-6) && (complexity < 25)"` in `config.yaml` and `experiments.py`.
- **Empirical verification** on `polynomial` dataset (2x² + 3xy - y + 5):

| Scenario | Wall-clock (s) | Best Loss | Recovered |
|---|---|---|---|
| baseline | 11.68 | 7.16e-07 | True |
| C3 (+,-,* only) | 2.37 | 4.64e-12 | True |
| C2 (maxdepth=6) | 0.79 | 5.77e-12 | True |
| C2+C3 | 0.41 | 5.11e-12 | True |

- **Result:** 28.5× max/min ratio, 121% coefficient of variation in wall-clock. Early stop fires; constrained runs converge dramatically faster. The joint constraint (C2+C3 at 0.41s) is faster than either single constraint, which is exactly the synergy signal $S(i,j)$ is designed to capture.
- **Execution model confirmed:** The orchestrator loop runs all scenarios sequentially in a single process (`for dataset → for scenario → for seed`). No joblib/multiprocessing. JIT warmup is paid once at the first `model.fit()` and shared across all subsequent calls. This means seed-paired ratios do NOT perfectly cancel JIT warmup — the first scenario (baseline, first seed) absorbs most of the overhead.
- **JIT diagnostic recommended for `analysis.py`:** Add a check of wall-clock by execution order (row index in CSV), not just by seed, to empirically verify whether JIT distortion is concentrated in the first few rows.


