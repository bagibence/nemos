import jax
import jax.numpy as jnp

from ..tree_utils import tree_scalar_mul, tree_sub


def cauchy_termination(
    rtol: float,
    atol: float,
    y,
    y_prev,
    f,
    f_prev,
    norm,
):
    # x_scale = atol + rtol * jnp.abs(x)
    y_abs_scaled = tree_scalar_mul(rtol, jax.tree.map(jnp.abs, y))
    f_abs_scaled = tree_scalar_mul(rtol, jax.tree.map(jnp.abs, f))
    y_scale = jax.tree.map(lambda x: atol + x, y_abs_scaled)
    f_scale = jax.tree.map(lambda x: atol + x, f_abs_scaled)

    # diff = jnp.abs(x - x_prev)
    y_diff = jax.tree.map(jnp.abs, tree_sub(y, y_prev))
    f_diff = jax.tree.map(jnp.abs, tree_sub(f, f_prev))

    # x_converged = norm(x_diff / x_scale) < 1
    y_converged = norm(jax.tree.map(lambda a, b: a / b, y_diff, y_scale)) < 1
    f_converged = norm(jax.tree.map(lambda a, b: a / b, f_diff, f_scale)) < 1

    return y_converged, f_converged
