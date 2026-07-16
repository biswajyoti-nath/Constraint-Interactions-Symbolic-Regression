"""PySR integration module running symbolic regression benchmarks under constraint scenarios.

Satisfies PRD §7.4 (FR-EXP-01 through FR-EXP-07) and DESIGN_CONTEXT §6.4.
"""

import csv
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import numpy as np
import sympy
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pysr import PySRRegressor

# Add current directory to path to allow importing constraints
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from constraints import (
    make_c1_structural,
    make_c2_depth,
    make_c3_operator,
    make_c4_positivity,
)


# ---------------------------------------------------------------------------
# Layer 1: Dataset Loaders
# ---------------------------------------------------------------------------

def load_feynman_ke(n_samples=200, seed=42):
    """Load kinetic energy dataset: KE = 0.5 * m * v^2."""
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.1, 10.0, n_samples)
    v = rng.uniform(0.1, 10.0, n_samples)
    X = np.stack([m, v], axis=1)
    y = 0.5 * m * v**2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return X_train, y_train, X_test, y_test, "0.5 * m * v**2"


def load_feynman_coulomb(n_samples=200, seed=42):
    """Load Coulomb's Law dataset: F = q1 * q2 / (4 * pi * eps0 * r^2)."""
    rng = np.random.default_rng(seed)
    q1 = rng.uniform(0.1, 5.0, n_samples)
    q2 = rng.uniform(0.1, 5.0, n_samples)
    r = rng.uniform(0.1, 5.0, n_samples)
    X = np.stack([q1, q2, r], axis=1)
    eps0 = 8.854e-12
    denom = 4 * np.pi * eps0
    y = (q1 * q2) / (denom * r**2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return X_train, y_train, X_test, y_test, f"q1 * q2 / ({denom} * r**2)"


def load_polynomial(n_samples=200, seed=42):
    """Load polynomial dataset: f(x,y) = 2x^2 + 3xy - y + 5."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, n_samples)
    y = rng.uniform(-5.0, 5.0, n_samples)
    X = np.stack([x, y], axis=1)
    y_vals = 2 * x**2 + 3 * x * y - y + 5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_vals, test_size=0.2, random_state=seed
    )
    return X_train, y_train, X_test, y_test, "2 * x**2 + 3 * x * y - y + 5"


def load_srsd_dummy(n_samples=200, seed=42):
    """Load SRSD dummy dataset: f(x,y) = x^2 + y, with z as noise variable."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, n_samples)
    y = rng.uniform(-5.0, 5.0, n_samples)
    z = rng.uniform(-5.0, 5.0, n_samples)
    X = np.stack([x, y, z], axis=1)
    y_vals = x**2 + y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_vals, test_size=0.2, random_state=seed
    )
    return X_train, y_train, X_test, y_test, "x**2 + y"


LOADERS = {
    "feynman_ke": load_feynman_ke,
    "feynman_coulomb": load_feynman_coulomb,
    "polynomial": load_polynomial,
    "srsd_dummy": load_srsd_dummy,
}

DATASET_FEATURE_NAMES = {
    "feynman_ke": ["m", "v"],
    "feynman_coulomb": ["q1", "q2", "r"],
    "polynomial": ["x", "y"],
    "srsd_dummy": ["x", "y", "z"],
}


# ---------------------------------------------------------------------------
# Layer 2: PySR Constraint Mapper & Composer
# ---------------------------------------------------------------------------

def merge_nested_constraints(d1, d2):
    """Deep merge two nested constraints dicts, taking min of limits for overlapping pairs."""
    if not d1:
        return d2
    if not d2:
        return d1
    res = {}
    all_keys = set(d1.keys()) | set(d2.keys())
    for k in all_keys:
        if k in d1 and k in d2:
            if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                res[k] = merge_nested_constraints(d1[k], d2[k])
            else:
                res[k] = min(d1[k], d2[k])
        elif k in d1:
            res[k] = d1[k]
        else:
            res[k] = d2[k]
    return res


def build_pysr_kwargs(config: dict, active_constraints: list[str]) -> dict:
    """Compose PySR Regressor keyword arguments by applying active constraints."""
    # 1. Start with base kwargs from config.yaml -> pysr section
    pysr_cfg = config.get("pysr", {})
    kwargs = {
        "binary_operators": list(pysr_cfg.get("binary_operators", ["+", "-", "*", "/"])),
        "unary_operators": list(pysr_cfg.get("unary_operators", ["sin", "cos", "exp", "log"])),
        "population_size": pysr_cfg.get("population_size", 33),
        "niterations": pysr_cfg.get("niterations", 100),
        "maxsize": pysr_cfg.get("maxsize", 25),
    }

    # Map timeout_seconds to timeout_in_seconds if present
    if "timeout_seconds" in pysr_cfg:
        kwargs["timeout_in_seconds"] = float(pysr_cfg["timeout_seconds"])

    # 2. Layer constraint-specific modifications
    nested_constraints = {}
    max_depth_val = None
    whitelists = []
    penalties = []

    for constraint in active_constraints:
        if constraint == "C1":
            # Structural trig nested constraints
            struct_cfg = config.get("constraints", {}).get("structural", {})
            max_nested_trig = struct_cfg.get("max_nested_trig", 1)
            c1_nested = {
                "sin": {"sin": max_nested_trig, "cos": max_nested_trig},
                "cos": {"sin": max_nested_trig, "cos": max_nested_trig},
            }
            nested_constraints = merge_nested_constraints(nested_constraints, c1_nested)
        elif constraint == "C2":
            # Depth limit
            depth_cfg = config.get("constraints", {}).get("depth", {})
            max_depth_val = depth_cfg.get("limit", 6)
        elif constraint == "C3":
            # Whitelist of operators
            op_cfg = config.get("constraints", {}).get("operator_whitelist", {})
            whitelists.append(set(op_cfg.get("allowed", ["+", "-", "*"])))
        elif constraint == "C4":
            # Positivity penalty
            penalties.append("10.0 * (prediction < 0.0 ? prediction^2 : 0.0)")

    # Apply nested_constraints if any
    if nested_constraints:
        kwargs["nested_constraints"] = nested_constraints

    # Apply min depth limit if any
    if max_depth_val is not None:
        kwargs["maxdepth"] = max_depth_val

    # Apply whitelists to binary and unary operators
    if whitelists:
        allowed_ops = whitelists[0]
        for w in whitelists[1:]:
            allowed_ops = allowed_ops.intersection(w)
        kwargs["binary_operators"] = [op for op in kwargs["binary_operators"] if op in allowed_ops]
        kwargs["unary_operators"] = [op for op in kwargs["unary_operators"] if op in allowed_ops]

    # Apply elementwise loss penalties
    if penalties:
        penalty_str = " + ".join(penalties)
        kwargs["elementwise_loss"] = f"loss(prediction, target) = (prediction - target)^2 + {penalty_str}"

    return kwargs


# ---------------------------------------------------------------------------
# Layer 3: Scenario Runner
# ---------------------------------------------------------------------------

def _levenshtein_distance_dp(s1: str, s2: str) -> int:
    """Compute exact Levenshtein distance between two strings using standard DP."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = min(dp[j - 1], dp[j], prev) + 1
            prev = temp
    return dp[n]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Compute normalized edit distance between two strings, ignoring spaces."""
    s1_clean = "".join(s1.split())
    s2_clean = "".join(s2.split())
    if not s1_clean and not s2_clean:
        return 0.0
    try:
        import Levenshtein
        dist = Levenshtein.distance(s1_clean, s2_clean)
    except ImportError:
        dist = _levenshtein_distance_dp(s1_clean, s2_clean)
    return dist / max(len(s1_clean), len(s2_clean))


def run_scenario(dataset_name, dataset, active_constraints, config, seed) -> dict:
    """Run a single experimental scenario."""
    X_train, y_train, X_test, y_test, ground_truth = dataset
    pysr_kwargs = build_pysr_kwargs(config, active_constraints)

    # Add determinism and run options
    pysr_kwargs.update({
        "deterministic": True,
        "parallelism": "serial",
        "random_state": seed,
        "verbosity": 0,
        "progress": False,
    })

    # Prepare features
    features = DATASET_FEATURE_NAMES.get(dataset_name)

    t0 = time.perf_counter()
    model = PySRRegressor(**pysr_kwargs)
    try:
        model.fit(X_train, y_train, variable_names=features)
        elapsed = time.perf_counter() - t0
        error_msg = None
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "dataset": dataset_name,
            "constraints": ",".join(sorted(active_constraints)) or "baseline",
            "seed": seed,
            "wall_clock_s": round(elapsed, 3),
            "evals": np.nan,
            "best_expression": "",
            "best_loss": np.nan,
            "complexity": np.nan,
            "mse": np.nan,
            "relative_mse": np.nan,
            "ned": np.nan,
            "recovered": False,
            "timeout_hit": False,
            "constraints_satisfied": {},
            "error": str(e),
        }

    # Extract best expression (last row in HOF / equations_ dataframe)
    best_row = model.equations_.iloc[-1]
    best_expr_str = str(best_row["equation"])
    best_sympy = best_row["sympy_format"]
    best_loss = float(best_row["loss"])
    complexity = int(best_row["complexity"])

    # Compute recovery metrics on test set
    try:
        y_pred = model.predict(X_test, index=len(model.equations_) - 1)
        mse = float(mean_squared_error(y_test, y_pred))
        var_y = float(np.var(y_test))
        relative_mse = mse / var_y if var_y > 0 else np.inf
    except Exception:
        mse = np.nan
        relative_mse = np.nan

    ned = normalized_edit_distance(best_expr_str, ground_truth)
    recovered = bool((ned < 0.1) or (relative_mse is not None and relative_mse < 1e-6))

    # Post-hoc constraint compliance checks
    compliance = {}
    constraint_fns = {
        "C1": make_c1_structural(config),
        "C2": make_c2_depth(config),
        "C3": make_c3_operator(config),
        "C4": make_c4_positivity(config),
    }

    for name in ["C1", "C2", "C3", "C4"]:
        try:
            compliance[name] = bool(constraint_fns[name](best_sympy))
        except Exception:
            compliance[name] = None

    # Determine if timeout was hit
    timeout_limit = float(config.get("pysr", {}).get("timeout_seconds", 300))
    timeout_hit = elapsed >= timeout_limit - 1.0

    return {
        "dataset": dataset_name,
        "constraints": ",".join(sorted(active_constraints)) or "baseline",
        "seed": seed,
        "wall_clock_s": round(elapsed, 3),
        "evals": np.nan,
        "best_expression": best_expr_str,
        "best_loss": best_loss,
        "complexity": complexity,
        "mse": mse,
        "relative_mse": relative_mse,
        "ned": ned,
        "recovered": recovered,
        "timeout_hit": timeout_hit,
        "constraints_satisfied": compliance,
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Layer 4: Experiment Orchestrator
# ---------------------------------------------------------------------------

def flatten_result(res: dict) -> dict:
    """Flatten results dict to map compliance checks to individual columns."""
    flat = res.copy()
    compliance = flat.pop("constraints_satisfied", {})
    flat["c1_satisfied"] = compliance.get("C1")
    flat["c2_satisfied"] = compliance.get("C2")
    flat["c3_satisfied"] = compliance.get("C3")
    flat["c4_satisfied"] = compliance.get("C4")
    return flat


def run_experiment_matrix(config_path="config.yaml") -> pathlib.Path:
    """Run full benchmark matrix and log results."""
    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seeds = config.get("experiments", {}).get("seeds", [42, 123, 456, 789, 1024])
    seed_hash = hashlib.sha256(str(seeds).encode("utf-8")).hexdigest()[:8]
    run_id = f"{datetime.datetime.utcnow():%Y%m%d_%H%M%S}_{seed_hash}"

    # 2. Create results directory
    results_dir = pathlib.Path("results") / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get Git commit hash
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        git_commit = "unknown"

    with open(config_path, "rb") as f:
        config_bytes = f.read()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    import pysr
    import scipy

    # 3. Write metadata.json before starting
    metadata = {
        "run_id": run_id,
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "seeds": seeds,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "packages": {
            "pysr": pysr.__version__,
            "sympy": sympy.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "pysr_deterministic": True,
        "pysr_parallelism": "serial",
        "wall_clock_proxy_note": (
            "PySR v1.5.10 does not expose eval counts. "
            "wall_clock_s is the search-cost proxy for S(i,j). "
            "See Ch.9 for justification."
        ),
        "known_gaps": {
            "C1b_not_enforced": (
                "PySR has no max_consecutive_binary parameter. "
                "Search enforces C1a only. See Ch.13."
            ),
            "C2_depth_convention": (
                "PySR maxdepth uses internal tree depth, "
                "not SymPy .args depth. See Ch.9."
            ),
        },
    }

    metadata_path = results_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Define matrix
    datasets = ["feynman_ke", "feynman_coulomb", "polynomial", "srsd_dummy"]
    scenarios = [
        [],
        ["C1"], ["C2"], ["C3"], ["C4"],
        ["C1", "C2"], ["C1", "C3"], ["C1", "C4"],
        ["C2", "C3"], ["C2", "C4"],
        ["C3", "C4"],
        ["C1", "C2", "C3", "C4"],
    ]

    csv_path = results_dir / "pysr_results.csv"
    headers = [
        "dataset",
        "constraints",
        "seed",
        "wall_clock_s",
        "evals",
        "best_expression",
        "best_loss",
        "complexity",
        "mse",
        "relative_mse",
        "ned",
        "recovered",
        "timeout_hit",
        "c1_satisfied",
        "c2_satisfied",
        "c3_satisfied",
        "c4_satisfied",
        "error",
    ]

    total_runs = len(datasets) * len(scenarios) * len(seeds)
    completed = 0

    print(f"Starting experiment run {run_id}. Total runs: {total_runs}")

    # Open CSV in append mode and flush after each row
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()

        for dataset_name in datasets:
            # Generate dataset once using the first seed (fixed data split)
            dataset = LOADERS[dataset_name](seed=seeds[0])
            for scenario in scenarios:
                for seed in seeds:
                    print(
                        f"[{completed + 1}/{total_runs}] Running {dataset_name} | "
                        f"scenario={scenario} | seed={seed}..."
                    )
                    res = run_scenario(dataset_name, dataset, scenario, config, seed)
                    flat_res = flatten_result(res)
                    writer.writerow(flat_res)
                    csv_file.flush()
                    completed += 1

    # Update metadata with end timestamp
    metadata["timestamp_completed_utc"] = datetime.datetime.utcnow().isoformat()
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Experiments complete. Results saved to {results_dir}")
    return csv_path
