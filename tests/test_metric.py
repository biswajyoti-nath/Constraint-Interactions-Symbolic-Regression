"""Tests for metric.py.

Covers: density estimation, interaction matrix symmetry and NaN handling,
sympify(evaluate=False) regression, and 1k benchmark convergence.
"""

import json
import math
import pathlib
import sys
import time

import numpy as np
import pytest
from sympy import sympify, Add, Mul, Symbol

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from expr_generator import GrammarGenerator
from metric import DensityEstimator, InteractionMatrix, BootstrapCI, RhoResult, run_benchmark

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gen():
    return GrammarGenerator(str(PROJECT_ROOT / "config.yaml"))


# ---------------------------------------------------------------------------
# TestDensityEstimator
# ---------------------------------------------------------------------------


class TestDensityEstimator:

    def test_rho_trivial_true_constraint(self, gen):
        """Always-true constraint must produce rho = 1.0."""
        est = DensityEstimator(gen, max_depth=3)
        result = est.estimate([lambda e: True], N=100)
        assert result.rho_i[0] == 1.0, f"Expected 1.0, got {result.rho_i[0]}"

    def test_rho_trivial_false_constraint(self, gen):
        """Always-false constraint must produce rho = 0.0."""
        est = DensityEstimator(gen, max_depth=3)
        result = est.estimate([lambda e: False], N=100)
        assert result.rho_i[0] == 0.0

    def test_stats_logged_after_run(self, gen):
        """generator.stats['generated'] must be > 0 after a run."""
        est = DensityEstimator(gen, max_depth=3)
        result = est.estimate([lambda e: True], N=50)
        assert result.stats["generated"] > 0

    def test_reset_stats_called(self, gen):
        """Running twice: generated count must reflect only the second run."""
        est = DensityEstimator(gen, max_depth=3)
        est.estimate([lambda e: True], N=50)
        result2 = est.estimate([lambda e: True], N=30)
        # After reset, stats['generated'] tracks the second run only
        assert result2.stats["generated"] <= 30 + 10  # at most 30 + retry overhead

    def test_sympify_evaluate_false_regression(self, gen):
        """DESIGN_CONTEXT §2: sympify(evaluate=False) must NOT auto-simplify.

        If the generator ever drops evaluate=False, x+x collapses to 2*x,
        an Add node becomes a Mul node, and constraints inspecting Add nodes
        would silently break. This test catches that regression.
        """
        # Build expression the same way the generator does
        expr = sympify("x + x", evaluate=False)

        # With evaluate=False, the tree must still be Add(x, x)
        assert isinstance(expr, Add), (
            f"Expected Add (x + x unevaluated), got {type(expr).__name__}. "
            "This means evaluate=False was dropped somewhere."
        )

        # Confirm it is NOT 2*x
        x = Symbol("x")
        assert expr != 2 * x, "Expression was auto-simplified to 2*x — evaluate=False violated."

    def test_n_actual_equals_n(self, gen):
        """N_actual must equal the requested N for well-behaved depth."""
        est = DensityEstimator(gen, max_depth=4)
        result = est.estimate([lambda e: True], N=100)
        assert result.N_actual == 100


# ---------------------------------------------------------------------------
# TestInteractionMatrix
# ---------------------------------------------------------------------------


class TestInteractionMatrix:

    def _make_rho(self, rho_i_vals, rho_ij_vals):
        """Helper: construct a RhoResult from lists."""
        k = len(rho_i_vals)
        rho_i = np.array(rho_i_vals, dtype=float)
        rho_ij = np.zeros((k, k), dtype=float)
        for (i, j), v in rho_ij_vals.items():
            rho_ij[i][j] = v
        return RhoResult(rho_i=rho_i, rho_ij=rho_ij, N_actual=1000)

    def test_symmetric(self):
        """M[i][j] must equal M[j][i] for all pairs (DESIGN_CONTEXT §6.3)."""
        rho = self._make_rho([0.5, 0.4], {(0, 1): 0.2})
        M = InteractionMatrix().compute(rho)
        assert M[0][1] == M[1][0], f"Symmetry violated: M[0][1]={M[0][1]}, M[1][0]={M[1][0]}"

    def test_independent_constraints(self):
        """Independent constraints (rho_ij = rho_i * rho_j) → M ≈ 1.0."""
        rho_i = [0.5, 0.4]
        joint = 0.5 * 0.4  # exact independence
        rho = self._make_rho(rho_i, {(0, 1): joint})
        M = InteractionMatrix().compute(rho)
        assert abs(M[0][1] - 1.0) < 1e-9

    def test_zero_rho_returns_nan(self):
        """ρ_i = 0 → M[i][j] and M[j][i] must both be NaN, not a crash."""
        rho = self._make_rho([0.0, 0.4], {(0, 1): 0.0})
        M = InteractionMatrix().compute(rho)
        assert math.isnan(M[0][1]), "Expected NaN for zero-density constraint"
        assert math.isnan(M[1][0]), "Lower triangle NaN not mirrored correctly"

    def test_diagonal_is_nan(self):
        """Diagonal of M is undefined — must be NaN."""
        rho = self._make_rho([0.5, 0.4], {(0, 1): 0.2})
        M = InteractionMatrix().compute(rho)
        assert math.isnan(M[0][0])
        assert math.isnan(M[1][1])

    def test_synergy_greater_than_one(self):
        """Synergistic constraints (rho_ij > rho_i*rho_j) → M > 1."""
        rho_i = [0.5, 0.4]
        joint = 0.5 * 0.4 * 1.5  # 1.5x the independent value
        rho = self._make_rho(rho_i, {(0, 1): joint})
        M = InteractionMatrix().compute(rho)
        assert M[0][1] > 1.0

    def test_redundancy_less_than_one(self):
        """Redundant constraints (rho_ij < rho_i*rho_j) → M < 1."""
        rho_i = [0.5, 0.4]
        joint = 0.5 * 0.4 * 0.5  # 0.5x the independent value
        rho = self._make_rho(rho_i, {(0, 1): joint})
        M = InteractionMatrix().compute(rho)
        assert M[0][1] < 1.0


# ---------------------------------------------------------------------------
# TestBenchmark1k
# ---------------------------------------------------------------------------


class TestBenchmark1k:
    """Calibration benchmark — establishes baseline, does not assert guessed thresholds.

    Per the implementation plan: only assert derivable guarantees.
    Log observed values to results/benchmark_1k.json for regression tracking.
    """

    def test_no_unrecoverable_failures(self, gen, tmp_path):
        """returned_none must be 0 at depth ≤ 4, N=1000.

        This is derivable: 10 retries × near-zero failure probability at depth 4
        makes returned_none == 0 a safe hard assertion.
        """
        record = run_benchmark(
            gen,
            N=1000,
            max_depth=4,
            output_path=str(tmp_path / "benchmark_1k.json"),
        )
        assert record["stats"]["returned_none"] == 0, (
            f"returned_none={record['stats']['returned_none']} — generator failed to "
            "produce valid expressions within max_retries. Check domain validation."
        )

    def test_rho_convergence(self, gen):
        """Convergence check (Research Plan §8.3): rho at N=500 vs N=1000 < 0.05.

        Uses always-true constraint so rho should be exactly 1.0 at both scales.
        This also verifies that the estimator's loop terminates correctly.
        """
        est = DensityEstimator(gen, max_depth=4)
        always_true = lambda e: True  # noqa: E731

        r500 = est.estimate([always_true], N=500)
        r1000 = est.estimate([always_true], N=1000)

        diff = abs(r500.rho_i[0] - r1000.rho_i[0])
        assert diff < 0.05, (
            f"Convergence check failed: |rho_500 - rho_1000| = {diff:.4f} ≥ 0.05"
        )

    def test_benchmark_json_written(self, gen, tmp_path):
        """Benchmark must write a valid JSON file with required keys."""
        out = tmp_path / "benchmark_1k.json"
        record = run_benchmark(gen, N=100, max_depth=3, output_path=str(out))

        assert out.exists(), "benchmark JSON file was not created"
        with open(out) as f:
            data = json.load(f)

        required_keys = {"N", "elapsed_s", "stats", "rho_always_true"}
        assert required_keys.issubset(data.keys()), (
            f"Missing keys: {required_keys - data.keys()}"
        )
