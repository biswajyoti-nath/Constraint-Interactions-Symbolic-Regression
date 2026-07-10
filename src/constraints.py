import sympy
import numpy as np
from typing import Callable, List


def _sympy_depth(expr) -> int:
    if not expr.args:
        return 0
    return 1 + max(_sympy_depth(a) for a in expr.args)


def _nested_trig_depth(expr) -> int:
    is_trig = isinstance(expr, (sympy.sin, sympy.cos))
    child_max = max((_nested_trig_depth(a) for a in expr.args), default=0)
    return (1 + child_max) if is_trig else child_max


_OP_CLASS_MAP = {
    "+": sympy.Add,
    "-": None,  # no SymPy Sub class; handled by Add + Mul
    "*": sympy.Mul,
    "**": sympy.Pow,
    "sin": sympy.sin,
    "cos": sympy.cos,
}


def _allowed_op_classes(whitelist):
    classes = set()
    for op in whitelist:
        cls = _OP_CLASS_MAP.get(op)
        if cls is not None:
            classes.add(cls)
        elif op == "-":
            classes.add(sympy.Add)
            classes.add(sympy.Mul)
    return frozenset(classes)


def make_c1_structural(config) -> Callable:
    max_nested_trig = config["constraints"]["structural"]["max_nested_trig"]
    max_consecutive_binary = config["constraints"]["structural"][
        "max_consecutive_binary"
    ]

    def c1_structural(expr: sympy.Expr) -> bool:
        if _nested_trig_depth(expr) > max_nested_trig:
            return False

        for node in sympy.preorder_traversal(expr):
            if isinstance(node, (sympy.Add, sympy.Mul)):
                if len(node.args) > max_consecutive_binary:
                    return False
        return True

    c1_structural.__name__ = "c1_structural"
    c1_structural.__doc__ = (
        "C1: Structural constraint checking nested trig and consecutive binary ops."
    )
    return c1_structural


def make_c2_depth(config) -> Callable:
    limit = config["constraints"]["depth"]["limit"]

    def c2_depth(expr: sympy.Expr) -> bool:
        return _sympy_depth(expr) <= limit

    c2_depth.__name__ = "c2_depth"
    c2_depth.__doc__ = "C2: Depth constraint checking sympy .args walk depth."
    return c2_depth


def make_c3_operator(config) -> Callable:
    allowed = _allowed_op_classes(
        config["constraints"]["operator_whitelist"]["allowed"]
    )

    def c3_operator(expr: sympy.Expr) -> bool:
        for node in sympy.preorder_traversal(expr):
            if isinstance(
                node,
                (
                    sympy.Symbol,
                    sympy.Number,
                    sympy.Integer,
                    sympy.Float,
                    sympy.Rational,
                ),
            ):
                continue
            if type(node) not in allowed:
                return False
        return True

    c3_operator.__name__ = "c3_operator"
    c3_operator.__doc__ = "C3: Operator whitelist constraint."
    return c3_operator


def make_c4_positivity(config) -> Callable:
    cfg = config["constraints"]["positivity"]
    n_pts = cfg["n_test_points"]
    lo, hi = cfg["domain"]
    seed = cfg["rng_seed"]  # reads from config, NOT hardcoded

    def c4_positivity(expr: sympy.Expr) -> bool:
        symbols = sorted(expr.free_symbols, key=str)
        rng = np.random.default_rng(seed)

        if not symbols:
            try:
                val = complex(expr)
                return bool(
                    np.isfinite(val.real) and val.imag == 0.0 and val.real >= 0.0
                )
            except Exception:
                return False

        pts = rng.uniform(lo, hi, size=(n_pts, len(symbols)))

        # Path 1: lambdify (vectorised, fast)
        try:
            f = sympy.lambdify(symbols, expr, modules="numpy")
            vals = np.asarray(
                f(*[pts[:, i] for i in range(len(symbols))]), dtype=complex
            )
            return bool(
                np.all(np.isfinite(vals.real))
                and np.all(vals.imag == 0.0)
                and np.all(vals.real >= 0.0)
            )
        except Exception:
            pass  # fall through

        # Path 2: .subs() fallback (slower, handles exotic SymPy expressions)
        for i in range(n_pts):
            subs = {s: sympy.Float(pts[i, j]) for j, s in enumerate(symbols)}
            try:
                val = complex(expr.subs(subs))
                if not (np.isfinite(val.real) and val.imag == 0.0 and val.real >= 0.0):
                    return False
            except Exception:
                return False
        return True

    c4_positivity.__name__ = "c4_positivity"
    c4_positivity.__doc__ = (
        "C4: f(x,y) >= 0 at 200 random points in domain. "
        "Implements non-negativity per Research Plan §5. "
        "RNG seed from config['constraints']['positivity']['rng_seed']."
    )
    return c4_positivity


def build_constraints(config, include_c5=False) -> List[Callable]:
    constraints = [
        make_c1_structural(config),
        make_c2_depth(config),
        make_c3_operator(config),
        make_c4_positivity(config),
    ]
    return constraints
