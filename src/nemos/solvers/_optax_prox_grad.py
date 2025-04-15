from typing import Callable, Optional, Any, NamedTuple


import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
import optimistix as optx

from ._optimistix_solvers import DEFAULT_MAX_STEPS, DEFAULT_ATOL, DEFAULT_RTOL
from ._optax_based_solvers import _make_rate_scaler

from jaxtyping import Bool, Scalar, Int, Float, PyTree

import equinox as eqx


class MonitorState(eqx.Module, strict=True):
    iter_num: Int[Scalar, ""]
    f_value: Float[Scalar, ""]
    y_converged: Bool
    f_converged: Bool


def step_counter():
    def init_fn(params):
        del params

        return MonitorState(
            jnp.array(0), jnp.array(jnp.inf), jnp.array(False), jnp.array(False)
        )

    def update_fn(updates, state, params=None):
        del params

        return updates, MonitorState(
            state.iter_num + 1, state.f_value, state.y_converged, state.f_converged
        )

    return optax.GradientTransformation(init_fn, update_fn)


class OptaxProximalGradient:
    def __init__(
        self,
        fun: Callable,
        prox: Callable,
        max_steps: int = DEFAULT_MAX_STEPS,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        stepsize: Optional[float] = None,
        linesearch_kwargs: Optional[dict[str, Any]] = None,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
    ):
        self.fun = fun
        self.prox = prox

        # might try the eqx.closure_to_pytree
        self.opt = optax.chain(
            step_counter(),
            optax.sgd(learning_rate=1.0, nesterov=True),
            _make_rate_scaler(stepsize, linesearch_kwargs),
        )

        self.max_steps = max_steps
        self.atol = atol
        self.rtol = rtol
        self.norm = norm

    def init(self, fn, y, args, options, tags):
        pass

    def update(self):
        pass

    def terminate(self):
        pass

    def run(self, init_params, regularizer_strength, *args):
        # copy jaxopt.ProximalGradient's way of getting the regularizer_strength

        def _fn(_params):
            return self.fun(_params, args)

        value_and_grad_fun = jax.jit(jax.value_and_grad(_fn))

        def step(carry):
            params, state = carry

            # take gradient step with optax's sgd with nesterov and linesearch
            value, grad = value_and_grad_fun(params)
            updates, new_state = self.opt.update(
                grad,
                state,
                params,
                value=value,
                grad=grad,
                value_fn=_fn,
            )
            new_params = optax.apply_updates(params, updates)

            monitor_state, sgd_state, linesearch_state = new_state

            # apply proximal operator
            new_params = self.prox(
                new_params,
                regularizer_strength,
                scaling=linesearch_state.learning_rate,
            )

            # reevaluate the function value at this point
            new_f_value = _fn(new_params)

            # update_norm = otu.tree_l2_norm(otu.tree_sub(new_params, params))

            # update the monitoring values
            # step count is done in the update step
            y_converged, f_converged = OptaxProximalGradient.cauchy_termination(
                self.rtol,
                self.atol,
                new_params,
                params,
                new_f_value,
                monitor_state.f_value,
                self.norm,
            )

            new_monitor_state = MonitorState(
                monitor_state.iter_num,
                new_f_value,
                y_converged,
                f_converged,
            )

            new_state = (new_monitor_state, sgd_state, linesearch_state)

            return new_params, new_state

        def continuing_criterion(carry):
            _, state = carry
            monitor_state, sgd_state, linesearch_state = state
            iter_num = monitor_state.iter_num

            # err = update_norm / linesearch_state.learning_rate
            # err = update_norm
            # not_converged = err >= self.tol

            not_converged = ~(monitor_state.y_converged & monitor_state.f_converged)

            return (iter_num == 0) | ((iter_num < self.max_steps) & not_converged)

        init_carry = (init_params, self.opt.init(init_params))
        final_params, final_state = jax.lax.while_loop(
            continuing_criterion, step, init_carry
        )

        return final_params, final_state

    @staticmethod
    def cauchy_termination(
        rtol: float,
        atol: float,
        y,
        y_prev,
        f,
        f_prev,
        norm,
    ):
        y_scale = jax.tree.map(
            lambda x: atol + x,
            otu.tree_scale(rtol, jax.tree.map(jnp.abs, y_prev)),
        )
        f_scale = jax.tree.map(
            lambda x: atol + x,
            otu.tree_scale(rtol, jax.tree.map(jnp.abs, f_prev)),
        )
        # f_scale = atol + rtol * jnp.abs(f)

        y_diff = jax.tree.map(jnp.abs, otu.tree_sub(y, y_prev))
        f_diff = jax.tree.map(jnp.abs, otu.tree_sub(f, f_prev))
        # f_diff = jnp.abs(f - f_prev)

        y_converged = norm(jax.tree.map(lambda a, b: a / b, y_diff, y_scale)) < 1
        f_converged = norm(jax.tree.map(lambda a, b: a / b, f_diff, f_scale)) < 1
        # f_converged = norm(f_diff / f_scale) < 1

        # return y_converged & f_converged
        return y_converged, f_converged
