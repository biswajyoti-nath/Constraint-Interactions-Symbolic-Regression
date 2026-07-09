"""Monte Carlo density estimator and interaction matrix for constraint analysis.

Implements Algorithm 1 from the Research Plan (§6). All invariants are
defined in DESIGN_CONTEXT.md §6.3 and §9.

Probability Space
-----------------
All estimates ρ(C) are conditional on the generator-induced distribution P.
See DESIGN_CONTEXT.md §0.
"""

import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# Allow running from project root without installing
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from expr_generator import GrammarGenerator


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RhoResult:
    """Holds raw Monte Carlo density estimates.

    Attributes:
        rho_i: 1-D array of shape (k,). rho_i[i] = ρ(C_i).
        rho_ij: 2-D array of shape (k, k). rho_ij[i][j] = ρ(C_i ∧ C_j).
            Upper triangle is filled by the estimator; lower triangle is
            mirrored by InteractionMatrix.
        N_actual: number of valid expressions that were drawn.
        stats: generator stats dict captured after sampling.
    """

    rho_i: np.ndarray
    rho_ij: np.ndarray
    N_actual: int
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DensityEstimator
# ---------------------------------------------------------------------------


class DensityEstimator:
    """Monte Carlo estimator for constraint densities ρ(C_i) and ρ(C_i ∧ C_j).

    Implements Algorithm 1 from the Research Plan exactly. Draws N expressions
    from the generator and counts constraint hits in a single pass.

    DESIGN_CONTEXT §6.3 invariants enforced here:
    - reset_stats() called before sampling.
    - generator.stats logged after sampling.
    - N_actual tracked separately from attempted draws.

    Args:
        generator: A configured GrammarGenerator instance.
        max_depth: Expression depth passed to generate_sympy(). If None,
            uses generator.max_depth from config.

    Example:
        >>> import pathlib, sys
        >>> sys.path.insert(0, str(pathlib.Path('.').resolve() / 'src'))
        >>> from expr_generator import GrammarGenerator
        >>> gen = GrammarGenerator()
        >>> est = DensityEstimator(gen, max_depth=3)
        >>> always_true = lambda e: True
        >>> result = est.estimate([always_true], N=50)
        >>> result.rho_i[0]  # doctest: +ELLIPSIS
        np.float64(1.0)
    """

    def __init__(self, generator: GrammarGenerator, max_depth: int | None = None):
        self.generator = generator
        self.max_depth = max_depth

    def estimate(self, constraints: list, N: int) -> RhoResult:
        """Draw N valid expressions and count constraint hits.

        Args:
            constraints: List of callables ``sympy.Expr → bool``.
                Each must be a pure function (DESIGN_CONTEXT §6.2).
            N: Target number of valid expressions. Actual count may be
                slightly lower if the generator returns None for a draw.

        Returns:
            RhoResult with rho_i, rho_ij, N_actual, and generator stats.
        """
        k = len(constraints)
        n_i = np.zeros(k, dtype=int)
        # Upper triangle only — mirrored by InteractionMatrix
        n_ij = np.zeros((k, k), dtype=int)

        # DESIGN_CONTEXT §6.3: reset before sampling
        self.generator.reset_stats()

        collected = []
        while len(collected) < N:
            expr = self.generator.generate_sympy(max_depth=self.max_depth)
            if expr is None:
                continue
            collected.append(expr)

        # Single pass: evaluate all constraints on each expression
        for expr in collected:
            hits = [bool(c(expr)) for c in constraints]
            for i in range(k):
                if hits[i]:
                    n_i[i] += 1
                    for j in range(i + 1, k):
                        if hits[j]:
                            n_ij[i][j] += 1

        # DESIGN_CONTEXT §6.3: log stats after sampling
        stats = self.generator.stats

        rho_i = n_i / N
        rho_ij = n_ij / N

        return RhoResult(
            rho_i=rho_i,
            rho_ij=rho_ij,
            N_actual=len(collected),
            stats=stats,
        )


# ---------------------------------------------------------------------------
# InteractionMatrix
# ---------------------------------------------------------------------------


class InteractionMatrix:
    """Compute the pairwise interaction coefficient matrix M(i,j).

    M(i,j) = ρ(C_i ∧ C_j) / [ρ(C_i) · ρ(C_j)]

    DESIGN_CONTEXT §6.3 invariants:
    - M must be symmetric: M[i][j] == M[j][i].
    - ρ_i = 0 or ρ_j = 0 → M[i][j] = NaN, not a crash.

    Example:
        >>> import numpy as np
        >>> result = type('R', (), {
        ...     'rho_i': np.array([1.0, 1.0]),
        ...     'rho_ij': np.array([[0.0, 1.0], [0.0, 0.0]])
        ... })()
        >>> M = InteractionMatrix().compute(result)
        >>> bool(M[0][1] == M[1][0])  # symmetry always holds
        True
    """

    def compute(self, rho_result: RhoResult) -> np.ndarray:
        """Compute symmetric M matrix from a RhoResult.

        Args:
            rho_result: Output of DensityEstimator.estimate().

        Returns:
            ndarray of shape (k, k). Diagonal is NaN. Off-diagonal is M(i,j).
            Lower triangle mirrors upper triangle explicitly.
        """
        rho_i = rho_result.rho_i
        rho_ij = rho_result.rho_ij
        k = len(rho_i)
        M = np.full((k, k), np.nan)

        # Fill upper triangle (Algorithm 1 structure)
        for i in range(k):
            for j in range(i + 1, k):
                denom = rho_i[i] * rho_i[j]
                if denom == 0.0:
                    M[i][j] = np.nan
                else:
                    M[i][j] = rho_ij[i][j] / denom

        # Explicit mirror step (DESIGN_CONTEXT §6.3: M[i][j] == M[j][i])
        for i in range(k):
            for j in range(i + 1, k):
                M[j][i] = M[i][j]  # NaN propagates correctly

        return M


# ---------------------------------------------------------------------------
# BootstrapCI
# ---------------------------------------------------------------------------


class BootstrapCI:
    """95% bootstrap confidence intervals for M(i,j).

    Uses the percentile method with B resamples. If 1 ∉ CI, the interaction
    is statistically significant at α=0.05 (Research Plan §4.3).

    Args:
        B: Number of bootstrap resamples. Default 1000.
        seed: NumPy RNG seed for reproducibility.
    """

    def __init__(self, B: int = 1000, seed: int = 42):
        self.B = B
        self.rng = np.random.default_rng(seed)

    def compute(
        self, expressions: list, constraints: list, generator: GrammarGenerator
    ) -> np.ndarray:
        """Compute 95% CI for each M(i,j) via bootstrap resampling.

        Args:
            expressions: Pre-drawn list of sympy.Expr objects.
            constraints: List of constraint callables.
            generator: Generator instance (used only for config access).

        Returns:
            ndarray of shape (k, k, 2) where [..., 0] is the lower bound
            and [..., 1] is the upper bound of the 95% CI.
        """
        k = len(constraints)
        n = len(expressions)
        m_samples = np.full((self.B, k, k), np.nan)

        for b in range(self.B):
            idxs = self.rng.integers(0, n, size=n)
            resample = [expressions[i] for i in idxs]

            n_i = np.zeros(k, dtype=int)
            n_ij = np.zeros((k, k), dtype=int)

            for expr in resample:
                hits = [bool(c(expr)) for c in constraints]
                for i in range(k):
                    if hits[i]:
                        n_i[i] += 1
                        for j in range(i + 1, k):
                            if hits[j]:
                                n_ij[i][j] += 1

            rho_i = n_i / n
            rho_ij = n_ij / n

            for i in range(k):
                for j in range(i + 1, k):
                    denom = rho_i[i] * rho_i[j]
                    if denom == 0.0:
                        m_samples[b, i, j] = np.nan
                        m_samples[b, j, i] = np.nan
                    else:
                        val = rho_ij[i][j] / denom
                        m_samples[b, i, j] = val
                        m_samples[b, j, i] = val

        ci = np.full((k, k, 2), np.nan)
        for i in range(k):
            for j in range(k):
                col = m_samples[:, i, j]
                valid = col[~np.isnan(col)]
                if len(valid) > 0:
                    ci[i, j, 0] = np.percentile(valid, 2.5)
                    ci[i, j, 1] = np.percentile(valid, 97.5)

        return ci


# ---------------------------------------------------------------------------
# Benchmark utility
# ---------------------------------------------------------------------------


def run_benchmark(
    generator: GrammarGenerator,
    N: int = 1000,
    max_depth: int = 4,
    output_path: str = "results/benchmark_1k.json",
) -> dict:
    """Run a calibration benchmark and log results to a JSON file.

    Does NOT assert hard thresholds for rejection rate or wall-clock time —
    those numbers are calibrated from real output, not guessed upfront.
    Logs observed values so future runs can detect regressions.

    Args:
        generator: GrammarGenerator instance.
        N: Number of expressions to draw.
        max_depth: Expression depth limit.
        output_path: Path for JSON output.

    Returns:
        dict with 'N', 'elapsed_s', 'stats', 'rho_always_true'.
    """
    generator.reset_stats()
    always_true = lambda e: True  # noqa: E731

    est = DensityEstimator(generator, max_depth=max_depth)

    t0 = time.perf_counter()
    result = est.estimate([always_true], N=N)
    elapsed = time.perf_counter() - t0

    record = {
        "N": N,
        "max_depth": max_depth,
        "elapsed_s": round(elapsed, 3),
        "stats": result.stats,
        "rho_always_true": float(result.rho_i[0]),
    }

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)

    return record


if __name__ == "__main__":
    gen = GrammarGenerator()
    record = run_benchmark(gen)
    print(json.dumps(record, indent=2))
