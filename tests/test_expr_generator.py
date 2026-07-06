"""Tests for GrammarGenerator.

Covers: exact-depth generation, domain validation, stats tracking,
stratified output shape, and config-driven leaf_probability.
"""

import sys
import pathlib
import pytest
from sympy import sympify

# Add src to path so we can import without installing
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from expr_generator import GrammarGenerator


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _string_depth(expr_str):
    """Compute the depth of a generated expression by its string structure.

    Our grammar produces strings like '(x + sin(y))'. Depth is defined by
    the generator's recursion level, not SymPy's internal tree (which may
    add nodes for things like -7 → Mul(-1, 7)).

    We count: each operator application adds 1 level. A leaf is depth 0.
    We detect operators by looking for our grammar patterns:
    - binary: '(' ... op ... ')'
    - unary: 'func(' ... ')'
    """
    expr_str = expr_str.strip()

    # Check for unary operator: func(...)
    for func in ("sin", "cos"):
        if expr_str.startswith(func + "(") and expr_str.endswith(")"):
            inner = expr_str[len(func) + 1 : -1]
            return 1 + _string_depth(inner)

    # Check for binary operator: (left op right)
    if expr_str.startswith("(") and expr_str.endswith(")"):
        # Find the operator at depth 0 inside the outer parens
        inner = expr_str[1:-1]
        paren_depth = 0
        for i, ch in enumerate(inner):
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
            elif paren_depth == 0 and ch in "+-*" and i > 0:
                # Check it's an operator, not a unary minus at the start
                left = inner[:i].strip()
                right = inner[i + 1 :].strip()
                if left and right:
                    return 1 + max(_string_depth(left), _string_depth(right))

    # It's a leaf (variable or constant, possibly negative like '-7')
    return 0


@pytest.fixture
def gen():
    """Return a GrammarGenerator using the project config.yaml."""
    config_path = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
    return GrammarGenerator(str(config_path))


# -------------------------------------------------------------------
# Fix 2: Exact-depth generation
# -------------------------------------------------------------------


class TestGenerateAtDepth:
    """generate_at_depth(d) must produce trees with depth == d."""

    @pytest.mark.parametrize("target_depth", [1, 2, 3, 4])
    def test_exact_depth(self, gen, target_depth):
        """String-level tree depth of output must equal the requested depth."""
        for _ in range(20):
            expr_str = gen.generate_at_depth(target_depth)
            depth = _string_depth(expr_str)
            assert depth == target_depth, (
                f"Expected depth {target_depth}, got {depth} for expr: {expr_str}"
            )

    def test_depth_zero_is_leaf(self, gen):
        """Depth 0 must always return a leaf (no operators)."""
        for _ in range(20):
            expr_str = gen.generate_at_depth(0)
            assert _string_depth(expr_str) == 0


# -------------------------------------------------------------------
# Fix 1: Domain validation
# -------------------------------------------------------------------


class TestDomainValidation:
    """_validate_domain must catch NaN/Inf/complex results."""

    def test_catches_division_by_zero(self, gen):
        """1/0 produces zoo in SymPy — domain validation should reject it."""
        expr = sympify("1/0", evaluate=False)
        assert gen._validate_domain(expr) is False

    def test_accepts_simple_polynomial(self, gen):
        """x + 1 is valid everywhere — should pass."""
        expr = sympify("x + 1", evaluate=False)
        assert gen._validate_domain(expr) is True

    def test_accepts_trig(self, gen):
        """sin(x) is valid everywhere — should pass."""
        expr = sympify("sin(x)", evaluate=False)
        assert gen._validate_domain(expr) is True

    def test_accepts_constant(self, gen):
        """Pure constant like 42 should pass."""
        expr = sympify("42", evaluate=False)
        assert gen._validate_domain(expr) is True


# -------------------------------------------------------------------
# Fix 4: Stats tracking
# -------------------------------------------------------------------


class TestStats:
    """Stats must track every generation attempt and rejection."""

    def test_generated_count(self, gen):
        """stats['generated'] must equal the number of attempts."""
        gen.reset_stats()
        n = 15
        for _ in range(n):
            gen.generate_sympy(max_depth=3)
        # At minimum, n attempts were made (could be more due to retries)
        assert gen.stats["generated"] >= n

    def test_reset_stats(self, gen):
        """reset_stats must zero all counters."""
        gen.generate_sympy(max_depth=3)
        gen.reset_stats()
        assert all(v == 0 for v in gen.stats.values())

    def test_stats_keys_present(self, gen):
        """All expected keys must be present."""
        expected = {"generated", "sympify_failed", "domain_rejected", "returned_none"}
        assert expected == set(gen.stats.keys())


# -------------------------------------------------------------------
# Fix 2: Stratified sampling output shape
# -------------------------------------------------------------------


class TestStratified:
    """generate_stratified must return correct structure."""

    def test_output_keys(self, gen):
        """Keys must match requested depths."""
        depths = [2, 4]
        result = gen.generate_stratified(depths=depths, n_per_depth=5)
        assert set(result.keys()) == set(depths)

    def test_output_count(self, gen):
        """Each bucket should have approximately n_per_depth entries."""
        result = gen.generate_stratified(depths=[2], n_per_depth=10)
        # Some may be rejected, but most should succeed
        assert len(result[2]) >= 5, (
            f"Expected ~10 exprs at depth 2, got {len(result[2])}"
        )


# -------------------------------------------------------------------
# Fix 3: Config-driven leaf_probability
# -------------------------------------------------------------------


class TestConfigDriven:
    """leaf_probability must come from config, not be hardcoded."""

    def test_leaf_probability_from_config(self, gen):
        """Generator's leaf_probability must match config value."""
        assert gen.leaf_probability == 0.2

    def test_domain_validation_enabled(self, gen):
        """domain_validation_enabled must match config."""
        assert gen.domain_validation_enabled is True
