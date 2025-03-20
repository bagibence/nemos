import jax
import jax.numpy as jnp
import optax
from optax._src import base as optax_base

from ..tree_utils import tree_add, tree_sub


def _prox_lasso_all_params(
    regularizer_strength: float, stepsize: float
) -> optax.GradientTransformation:
    """Applies the L1-proximal (soft-threshold) operator to parameters.

    In a chain of transformations, this should usually come *after* you've
    applied your gradient-based update (e.g. from Adam or SGD). It modifies
    the final update so that the resulting parameters are soft-thresholded
    with L1 penalty regularizer_strength * stepsize.

    Args:
      regularizer_strength: The L1 regularization coefficient (lambda).
      stepsize: The step size (learning rate) factor to scale the threshold.

    Returns:
      A GradientTransformation that applies soft-thresholding to p + u,
      returning the difference (new_p - p).
    """

    def init_fn(params):
        # No special state needed.
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax_base.NO_PARAMS_MSG)

        # We'll define a local soft-threshold function:
        def soft_threshold(x, threshold):
            return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)

        # For each leaf in the parameter tree:
        # 1) form y = p + u
        # 2) apply soft-threshold with tau = regularizer_strength * stepsize
        # 3) compute final_updates = new_p - p
        new_updates = jax.tree.map(
            lambda p, u: soft_threshold(p + u, regularizer_strength * stepsize) - p,
            params,
            updates,
            is_leaf=lambda x: x is None,
        )

        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def prox_lasso(regularizer_strength: float, stepsize: float):
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax_base.NO_PARAMS_MSG)

        def soft_threshold(x, threshold):
            return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)

        W_prev, b_prev = params
        dW, db = updates

        W_new = tree_add(W_prev, dW)
        b_new = tree_add(b_prev, db)

        # apply soft-threshold on the weights only
        threshold = regularizer_strength / b_prev.shape[0] * stepsize
        W_new = jax.tree.map(
            lambda p: soft_threshold(p, threshold),
            W_new,
            is_leaf=lambda x: x is None,
        )

        # updates are new params - old params
        W_update = tree_sub(W_new, W_prev)
        b_update = tree_sub(b_new, b_prev)

        new_updates = (W_update, b_update)

        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def prox_ridge_all_params(
    regularizer_strength: float, stepsize: float
) -> optax.GradientTransformation:
    """Applies the L2-proximal (ridge) operator to parameters.

    In a standard proximal gradient step for L2:
        theta_new = (1 / (1 + alpha * regularizer_strength)) * (theta_old + update).

    We return `theta_new - theta_old` as the final update, so Optax will
    effectively update parameters to that scaled value.

    Args:
      regularizer_strength: The L2 regularization coefficient (λ).
      stepsize: The step size (α). In a classical prox formulation, you
        would match this to the learning rate. If you're using an adaptive
        optimizer like Adam, this is often your *base* LR.

    Returns:
      A GradientTransformation that applies the ridge-prox operator to
      each parameter after the gradient-based update is computed.
    """

    def init_fn(params):
        # No special internal state needed.
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax_base.NO_PARAMS_MSG)

        # We'll define the ridge prox as a local function:
        def ridge_prox_op(x, alpha_reg):
            # alpha_reg = alpha * lambda
            scale = 1.0 / (1.0 + alpha_reg)
            return scale * x

        # For each parameter leaf:
        #   1) form candidate = p + u
        #   2) scale candidate by 1/(1 + αλ)
        #   3) final_updates = new_p - p
        new_updates = jax.tree_map(
            lambda p, u: ridge_prox_op(p + u, stepsize * regularizer_strength) - p,
            params,
            updates,
        )

        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def prox_ridge(regularizer_strength: float, stepsize: float):
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax_base.NO_PARAMS_MSG)

        def ridge_shrink(x, alpha_reg):
            # alpha_reg = alpha * lambda
            scale = 1.0 / (1.0 + alpha_reg)
            return scale * x

        W_prev, b_prev = params
        dW, db = updates

        W_new = tree_add(W_prev, dW)
        b_new = tree_add(b_prev, db)

        # apply soft-threshold on the weights only
        l2_reg = regularizer_strength / b_prev.shape[0] * stepsize
        W_new = jax.tree.map(
            lambda p: ridge_shrink(p, l2_reg),
            W_new,
            is_leaf=lambda x: x is None,
        )

        # updates are new params - old params
        W_update = tree_sub(W_new, W_prev)
        b_update = tree_sub(b_new, b_prev)

        new_updates = (W_update, b_update)

        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def prox_none(
    regularizer_strength: float, stepsize: float
) -> optax.GradientTransformation:
    """Returns an identity transformation that does no prox step.

    This transformation is a no-op and is intended for consistent usage in
    pipelines that expect a proximal operator.

    Returns:
      A GradientTransformation that returns the updates unchanged.
    """

    def init_fn(params):
        # No special internal state needed.
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax_base.NO_PARAMS_MSG)

        # Just pass updates through unchanged.
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _norm2_masked(weight_neuron: jnp.ndarray, mask: jnp.ndarray):
    """Utility: compute L2-norm of x restricted by mask, i.e. norm(x[mask != 0])."""
    # return jnp.sqrt(jnp.sum((weight_neuron * mask) ** 2))
    return jnp.linalg.norm(weight_neuron * mask, 2) / jnp.sqrt(mask.sum())


# vectorize the norm function
_vmap_norm2_masked_1 = jax.vmap(_norm2_masked, in_axes=(0, None), out_axes=0)
_vmap_norm2_masked_2 = jax.vmap(_vmap_norm2_masked_1, in_axes=(None, 0), out_axes=1)


def _prox_group_lasso_once(
    params: tuple[jnp.ndarray, jnp.ndarray],
    regularizer_strength: float,
    mask: jnp.ndarray,
    scaling: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply the group-lasso prox operator to a (weights, intercepts) tuple once.
    """
    weights, intercepts = params
    shape = weights.shape
    # divide reg strength by number of neurons
    regularizer_strength /= intercepts.shape[0]

    # ensure 2D: (n_features, n_neurons)
    weights_2d = jnp.atleast_2d(weights.T)
    # (n_neurons, n_groups)
    l2_norm = _vmap_norm2_masked_2(weights_2d, mask)
    factor = 1.0 - (regularizer_strength * scaling) / l2_norm
    factor = jax.nn.relu(factor)

    # For columns/groups that are not regularized at all (mask=0), no shrinkage
    not_regularized = jnp.outer(jnp.ones(factor.shape[0]), 1 - mask.sum(axis=0))

    # (n_features, n_neurons) again
    new_weights = (weights_2d * (factor @ mask + not_regularized)).T.reshape(shape)

    return new_weights, intercepts


def group_lasso_prox(
    regularizer_strength: float, mask: jnp.ndarray, stepsize: float
) -> optax.GradientTransformation:
    """
    Creates an Optax transformation that applies the group-lasso proximal operator
    to a (weights, intercepts) tuple of parameters.

    Args:
        regularizer_strength: The L2-group-lasso regularization coefficient (lambda).
        mask: (n_groups, n_features) array for grouping.
        stepsize: The step size alpha for the prox step.

    Returns:
        A GradientTransformation that applies group-lasso prox to each update.
    """

    def init_fn(params):
        # No internal state needed
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax_base.NO_PARAMS_MSG)

        # For each param leaf, form candidate = param + update, then apply prox
        def _apply_prox(p, u):
            candidate = (p[0] + u[0], p[1] + u[1])  # (weights, intercepts)
            new_p = _prox_group_lasso_once(
                candidate, regularizer_strength, mask, stepsize
            )
            return (new_p[0] - p[0], new_p[1] - p[1])

        # If you only store params as (weights, intercepts), you can just do:
        new_updates = jax.tree.map(_apply_prox, params, updates)
        # That yields the difference (new - old). These are the final updates.

        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)
