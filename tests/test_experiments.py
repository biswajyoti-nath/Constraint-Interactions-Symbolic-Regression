import numpy as np
import pytest
import sympy
from sklearn.metrics import mean_squared_error
from src.experiments import (
    LOADERS,
    DATASET_FEATURE_NAMES,
    build_pysr_kwargs,
    normalized_edit_distance,
    _levenshtein_distance_dp,
)


# ---------------------------------------------------------------------------
# 1. Loader Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loader_name", ["feynman_ke", "feynman_coulomb", "polynomial", "srsd_dummy"]
)
def test_loader_determinism(loader_name):
    loader = LOADERS[loader_name]
    # Call twice with same seed
    X_train1, y_train1, X_test1, y_test1, gt1 = loader(n_samples=50, seed=42)
    X_train2, y_train2, X_test2, y_test2, gt2 = loader(n_samples=50, seed=42)

    assert np.array_equal(X_train1, X_train2)
    assert np.array_equal(y_train1, y_train2)
    assert np.array_equal(X_test1, X_test2)
    assert np.array_equal(y_test1, y_test2)
    assert gt1 == gt2


@pytest.mark.parametrize(
    "loader_name", ["feynman_ke", "feynman_coulomb", "polynomial", "srsd_dummy"]
)
def test_loader_shapes(loader_name):
    loader = LOADERS[loader_name]
    X_train, y_train, X_test, y_test, gt = loader(n_samples=100, seed=42)

    assert X_train.shape == (80, len(DATASET_FEATURE_NAMES[loader_name]))
    assert X_test.shape == (20, len(DATASET_FEATURE_NAMES[loader_name]))
    assert y_train.shape == (80,)
    assert y_test.shape == (20,)


@pytest.mark.parametrize(
    "loader_name", ["feynman_ke", "feynman_coulomb", "polynomial", "srsd_dummy"]
)
def test_loader_ground_truth(loader_name):
    loader = LOADERS[loader_name]
    X_train, y_train, X_test, y_test, gt = loader(n_samples=50, seed=42)
    features = DATASET_FEATURE_NAMES[loader_name]

    expr = sympy.sympify(gt)
    symbols = [sympy.Symbol(f) for f in features]
    f_num = sympy.lambdify(symbols, expr, modules="numpy")

    cols = [X_test[:, i] for i in range(X_test.shape[1])]
    y_pred = f_num(*cols)

    mse = mean_squared_error(y_test, y_pred)
    assert mse < 1e-8


# ---------------------------------------------------------------------------
# 2. Constraint Mapper Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    return {
        "pysr": {
            "population_size": 20,
            "niterations": 50,
            "timeout_seconds": 150,
            "maxsize": 15,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["sin", "cos", "exp"],
        },
        "constraints": {
            "structural": {
                "max_nested_trig": 2,
            },
            "depth": {
                "limit": 5,
            },
            "operator_whitelist": {
                "allowed": ["+", "-", "*"],
            },
        },
    }


def test_build_pysr_kwargs_baseline(mock_config):
    kwargs = build_pysr_kwargs(mock_config, [])
    assert kwargs["population_size"] == 20
    assert kwargs["niterations"] == 50
    assert kwargs["timeout_in_seconds"] == 150.0
    assert kwargs["maxsize"] == 15
    assert kwargs["binary_operators"] == ["+", "-", "*", "/"]
    assert kwargs["unary_operators"] == ["sin", "cos", "exp"]
    assert "nested_constraints" not in kwargs
    assert "maxdepth" not in kwargs
    assert "elementwise_loss" not in kwargs


def test_build_pysr_kwargs_c1(mock_config):
    kwargs = build_pysr_kwargs(mock_config, ["C1"])
    assert "nested_constraints" in kwargs
    assert kwargs["nested_constraints"] == {
        "sin": {"sin": 2, "cos": 2},
        "cos": {"sin": 2, "cos": 2},
    }


def test_build_pysr_kwargs_c2(mock_config):
    kwargs = build_pysr_kwargs(mock_config, ["C2"])
    assert kwargs["maxdepth"] == 5


def test_build_pysr_kwargs_c3(mock_config):
    kwargs = build_pysr_kwargs(mock_config, ["C3"])
    assert kwargs["binary_operators"] == ["+", "-", "*"]
    assert kwargs["unary_operators"] == []


def test_build_pysr_kwargs_c4(mock_config):
    kwargs = build_pysr_kwargs(mock_config, ["C4"])
    assert "elementwise_loss" in kwargs
    assert (
        "10.0 * (prediction < 0.0 ? prediction^2 : 0.0)" in kwargs["elementwise_loss"]
    )


def test_build_pysr_kwargs_c1_c3_merge(mock_config):
    kwargs = build_pysr_kwargs(mock_config, ["C1", "C3"])
    assert "nested_constraints" in kwargs
    assert kwargs["binary_operators"] == ["+", "-", "*"]
    assert kwargs["unary_operators"] == []


def test_build_pysr_kwargs_c1_c4_merge(mock_config):
    kwargs = build_pysr_kwargs(mock_config, ["C1", "C4"])
    assert "nested_constraints" in kwargs
    assert "elementwise_loss" in kwargs


def test_build_pysr_kwargs_c2_c4_merge(mock_config):
    kwargs = build_pysr_kwargs(mock_config, ["C2", "C4"])
    assert kwargs["maxdepth"] == 5
    assert "elementwise_loss" in kwargs


def test_build_pysr_kwargs_all(mock_config):
    kwargs = build_pysr_kwargs(mock_config, ["C1", "C2", "C3", "C4"])
    assert "nested_constraints" in kwargs
    assert kwargs["maxdepth"] == 5
    assert kwargs["binary_operators"] == ["+", "-", "*"]
    assert kwargs["unary_operators"] == []
    assert "elementwise_loss" in kwargs


# ---------------------------------------------------------------------------
# 3. NED Tests
# ---------------------------------------------------------------------------


def test_levenshtein_distance_dp():
    assert _levenshtein_distance_dp("", "") == 0
    assert _levenshtein_distance_dp("abc", "abc") == 0
    assert _levenshtein_distance_dp("abc", "ab") == 1
    assert _levenshtein_distance_dp("abc", "acb") == 2
    assert _levenshtein_distance_dp("kitten", "sitting") == 3


def test_normalized_edit_distance():
    assert normalized_edit_distance("x + y", "x+y") == 0.0
    assert normalized_edit_distance("x + y", "x * y") > 0.0
    assert normalized_edit_distance("", "") == 0.0
