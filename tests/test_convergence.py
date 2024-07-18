"""Tests for making sure that solution reached for proximal operator is that the same as using another
method with just penalized loss."""

import jax
import numpy as np
import nemos as nmo
from scipy.optimize import minimize


def test_unregularized_convergence():
    """
    Assert that solution found when using GradientDescent vs ProximalGradient with an
    unregularized GLM is the same.
    """
    # generate toy data
    np.random.seed(111)
    # random design tensor. Shape (n_time_points, n_features).
    X = 0.5 * np.random.normal(size=(100, 5))

    # log-rates & weights, shape (1, ) and (n_features, ) respectively.
    b_true = np.zeros((1,))
    w_true = np.random.normal(size=(5,))

    # sparsify weights
    w_true[1:4] = 0.0

    # generate counts
    rate = jax.numpy.exp(jax.numpy.einsum("k,tk->t", w_true, X) + b_true)
    y = np.random.poisson(rate)

    # instantiate and fit unregularized GLM with GradientDescent
    model_GD = nmo.glm.GLM()
    model_GD.fit(X, y)

    # instantiate and fit unregularized GLM with ProximalGradient
    model_PG = nmo.glm.GLM(solver_name="ProximalGradient")
    model_PG.fit(X, y)

    # assert weights are the same
    assert np.allclose(np.round(model_GD.coef_, 2), np.round(model_PG.coef_, 2))


def test_ridge_convergence():
    """
    Assert that solution found when using GradientDescent vs ProximalGradient with an
    ridge GLM is the same.
    """
    # generate toy data
    np.random.seed(111)
    # random design tensor. Shape (n_time_points, n_features).
    X = 0.5 * np.random.normal(size=(100, 5))

    # log-rates & weights, shape (1, ) and (n_features, ) respectively.
    b_true = np.zeros((1,))
    w_true = np.random.normal(size=(5,))

    # sparsify weights
    w_true[1:4] = 0.0

    # generate counts
    rate = jax.numpy.exp(jax.numpy.einsum("k,tk->t", w_true, X) + b_true)
    y = np.random.poisson(rate)

    # instantiate and fit ridge GLM with GradientDescent
    model_GD = nmo.glm.GLM(regularizer="ridge")
    model_GD.fit(X, y)

    # instantiate and fit ridge GLM with ProximalGradient
    model_PG = nmo.glm.GLM(regularizer="ridge", solver_name="ProximalGradient")
    model_PG.fit(X, y)

    # assert weights are the same
    assert np.allclose(np.round(model_GD.coef_, 2), np.round(model_PG.coef_, 2))


def test_lasso_convergence():
    """
    Assert that solution found when using ProximalGradient versus Nelder-Mead method using
    lasso GLM is the same.
    """
    # generate toy data
    num_samples, num_features, num_groups = 1000, 5, 3
    X = np.random.normal(size=(num_samples, num_features))  # design matrix
    w = [0, 0.5, 1, 0, -0.5]  # define some weights
    y = np.random.poisson(np.exp(X.dot(w)))  # observed counts

    # instantiate and fit GLM with ProximalGradient
    model_PG = nmo.glm.GLM(regularizer="lasso", solver_name="ProximalGradient")
    model_PG.regularizer.regularizer_strength = 0.1
    model_PG.fit(X, y)

    # use the penalized loss function to solve optimization via Nelder-Mead
    penalized_loss = lambda p, x, y: model_PG.regularizer.penalized_loss(
        model_PG._predict_and_compute_loss
    )(
        (
            p[1:],
            p[0].reshape(
                1,
            ),
        ),
        x,
        y,
    )
    res = minimize(penalized_loss, [0] + w, args=(X, y), method="Nelder-Mead")

    # assert absolute difference between the weights is less than 0.1
    a = np.abs(np.subtract(np.round(res.x[1:], 2), np.round(model_PG.coef_, 2))) < 1e-1
    assert a.all()


def test_group_lasso_convergence():
    """
    Assert that solution found when using ProximalGradient versus Nelder-Mead method using
    group lasso GLM is the same.
    """
    # generate toy data
    num_samples, num_features, num_groups = 1000, 5, 3
    X = np.random.normal(size=(num_samples, num_features))  # design matrix
    w = [0, 0.5, 1, 0, -0.5]  # define some weights
    y = np.random.poisson(np.exp(X.dot(w)))  # observed counts

    mask = np.zeros((num_groups, num_features))
    mask[0] = [1, 0, 0, 1, 0]  # Group 0 includes features 0 and 3
    mask[1] = [0, 1, 0, 0, 0]  # Group 1 includes features 1
    mask[2] = [0, 0, 1, 0, 1]  # Group 2 includes features 2 and 4

    # instantiate and fit GLM with ProximalGradient
    model_PG = nmo.glm.GLM(regularizer=nmo.regularizer.GroupLasso(mask=mask))
    model_PG.fit(X, y)

    # use the penalized loss function to solve optimization via Nelder-Mead
    penalized_loss = lambda p, x, y: model_PG.regularizer.penalized_loss(
        model_PG._predict_and_compute_loss
    )(
        (
            p[1:],
            p[0].reshape(
                1,
            ),
        ),
        x,
        y,
    )

    res = minimize(penalized_loss, [0] + w, args=(X, y), method="Nelder-Mead")

    # assert absolute difference between the weights is less than 0.5
    a = np.abs(np.subtract(np.round(res.x[1:], 2), np.round(model_PG.coef_, 2))) < 0.5
    assert a.all()
