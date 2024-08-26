import pytest
import jax.numpy as jnp
from nemos.solvers import _svrg_defaults
from contextlib import nullcontext as does_not_raise

# Define fixtures for X_sample and y_sample
@pytest.fixture
def X_sample():
    return jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

@pytest.fixture
def y_sample():
    return jnp.array([1.0, 2.0, 3.0])

# Test _convert_to_float decorator
def test_convert_to_float_decorator():
    @ _svrg_defaults._convert_to_float
    def sample_function(x):
        return x

    result = sample_function(1)
    assert isinstance(result, float)

def test_svrg_optimal_batch_and_stepsize(X_sample, y_sample):
    """Test calculation of optimal batch size and step size for SVRG."""
    result = _svrg_defaults.svrg_optimal_batch_and_stepsize(
        _svrg_defaults.glm_softplus_poisson_l_max_and_l, X_sample, y_sample, strong_convexity=0.1
    )
    assert "batch_size" in result
    assert "stepsize" in result
    assert result["batch_size"] > 0
    assert result["stepsize"] > 0
    assert isinstance(result["batch_size"], int)
    assert isinstance(result["stepsize"], float)

def test_softplus_poisson_l_smooth_multiply(X_sample, y_sample):
    """Test multiplication with X.T @ D @ X without forming the matrix."""
    v_sample = jnp.array([0.5, 1.0])
    result = _svrg_defaults._glm_softplus_poisson_l_smooth_multiply(X_sample, y_sample, v_sample)
    diag_mat = jnp.diag(y_sample * 0.17 + 0.25)
    expected_result = X_sample.T.dot(diag_mat).dot(X_sample.dot(v_sample)) / X_sample.shape[0]
    assert jnp.allclose(result, expected_result)

def test_softplus_poisson_l_smooth_with_power_iteration(X_sample, y_sample):
    """Test the power iteration method for finding the largest eigenvalue."""
    result = _svrg_defaults._glm_softplus_poisson_l_smooth_with_power_iteration(X_sample, y_sample, n_power_iters=20)
    # compute eigvals directly
    diag_mat = jnp.diag(y_sample * 0.17 + 0.25)
    XDX = X_sample.T.dot(diag_mat).dot(X_sample) / X_sample.shape[0]
    eigmax = jnp.linalg.eigvalsh(XDX).max()

    assert result > 0
    assert jnp.allclose(eigmax, result)

@pytest.mark.parametrize("batch_size", [1, 2, 10])
@pytest.mark.parametrize("num_samples", [10, 12, 100, 500])
@pytest.mark.parametrize("l_smooth_max", [0.1, 1., 10.])
@pytest.mark.parametrize("l_smooth", [0.01, 0.05, 2.])
def test_calculate_stepsize_svrg(batch_size, num_samples, l_smooth_max, l_smooth):
    """Test calculation of the optimal step size for SVRG."""
    stepsize = _svrg_defaults._calculate_stepsize_svrg(batch_size, num_samples, l_smooth_max, l_smooth)
    assert stepsize > 0
    assert isinstance(stepsize, float)

@pytest.mark.parametrize("batch_size", [1, 2, 10])
@pytest.mark.parametrize("num_samples", [10, 12, 100, 500])
@pytest.mark.parametrize("l_smooth_max", [0.1, 1., 10.])
@pytest.mark.parametrize("l_smooth", [0.01, 0.05, 2.])
def test_calculate_stepsize_saga(batch_size, num_samples, l_smooth_max, l_smooth):
    """Test calculation of the optimal step size for SAGA."""
    stepsize = _svrg_defaults._calculate_stepsize_saga(batch_size, num_samples, l_smooth_max, l_smooth)
    assert stepsize > 0
    assert isinstance(stepsize, float)

@pytest.mark.parametrize("num_samples", [12, 100, 500])
@pytest.mark.parametrize("l_smooth_max", [0.1, 1., 10.])
@pytest.mark.parametrize("l_smooth", [0.01, 0.05])
@pytest.mark.parametrize("strong_convexity", [0.01, 1., 10.])
def test_calculate_optimal_batch_size_svrg(num_samples, l_smooth_max, l_smooth, strong_convexity):
    """Test calculation of the optimal batch size for SVRG."""
    batch_size = _svrg_defaults._calculate_optimal_batch_size_svrg(num_samples, l_smooth_max, l_smooth, strong_convexity)
    assert batch_size > 0
    assert isinstance(batch_size, int)

@pytest.mark.parametrize(
    "num_samples, l_smooth_max, l_smooth, expected_b_hat",
    [
        (100, 10.0, 2.0, 2.8697202),
    ],
)
def test_calculate_b_hat(num_samples, l_smooth_max, l_smooth, expected_b_hat):
    """Test calculation of b_hat for SVRG."""
    b_hat = _svrg_defaults._calculate_b_hat(num_samples, l_smooth_max, l_smooth)
    assert jnp.isclose(b_hat, expected_b_hat)

@pytest.mark.parametrize(
    "num_samples, l_smooth_max, l_smooth, strong_convexity, expected_b_tilde",
    [
        (100, 10.0, 2.0, 0.1, 3.4146341463414633),
    ],
)
def test_calculate_b_tilde(num_samples, l_smooth_max, l_smooth, strong_convexity, expected_b_tilde):
    """Test calculation of b_tilde for SVRG."""
    b_tilde = _svrg_defaults._calculate_b_tilde(num_samples, l_smooth_max, l_smooth, strong_convexity)
    assert jnp.isclose(b_tilde, expected_b_tilde)

@pytest.mark.parametrize(
    "batch_size, stepsize, expected_batch_size, expected_stepsize",
    [
        (32, 0.01, 32, 0.01),  # Both batch_size and stepsize provided
        (32, None, 32, None),  # Only batch_size provided
        (None, 0.01, None, 0.01),  # Only stepsize provided
    ]
)
def test_svrg_optimal_batch_and_stepsize_with_provided_defaults(batch_size, stepsize, expected_batch_size,
                                                                expected_stepsize, X_sample, y_sample):
    """Test that provided defaults for batch_size and stepsize are returned as-is or computed correctly."""
    result = _svrg_defaults.svrg_optimal_batch_and_stepsize(
        _svrg_defaults.glm_softplus_poisson_l_max_and_l,
        X_sample,
        y_sample,
        batch_size=batch_size,
        stepsize=stepsize
    )
    if expected_batch_size is not None:
        assert result["batch_size"] == expected_batch_size, "Provided batch_size should be returned as-is."
    else:
        assert "batch_size" in result and result["batch_size"] > 0, "Batch size should be computed since it was not provided."
    if expected_stepsize is not None:
        assert result["stepsize"] == expected_stepsize, "Provided stepsize should be returned as-is."
    else:
        assert "stepsize" in result and result["stepsize"] > 0, "Stepsize should be computed since it was not provided."

@pytest.mark.parametrize(
    "batch_size, stepsize, strong_convexity, expectation",
    [
        (jnp.inf, None, 0.1, pytest.warns(UserWarning, match="Could not determine batch size automatically")),
        (32, None, 0.1, pytest.warns(UserWarning, match="Could not determine step size automatically")),
        (None, None, 0.1, does_not_raise()),
    ]
)
def test_warnigns_svrg_optimal_batch_and_stepsize(batch_size, stepsize, strong_convexity, expectation, X_sample, y_sample):
    """Test that warnings are correctly raised during SVRG optimization when appropriate."""
    with expectation:
        _svrg_defaults.svrg_optimal_batch_and_stepsize(
            _svrg_defaults.glm_softplus_poisson_l_max_and_l,
            X_sample,
            y_sample,
            batch_size=batch_size,
            stepsize=stepsize,
            strong_convexity=strong_convexity
        )
