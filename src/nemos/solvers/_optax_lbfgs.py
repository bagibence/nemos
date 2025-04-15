from typing import Callable, Optional, Union


import jax
import optax
import optax.tree_utils as otu

from ._optimistix_solvers import DEFAULT_MAX_STEPS, DEFAULT_ATOL, DEFAULT_RTOL


class OptaxLBFGS:
    """
    L-BFGS implementation using only optax.lbfgs and a jax.lax.scan adapted from
    the Optax documentation.

    Convergence criterion is the same as in JAXopt: l2_norm(grad) <= abs_tol.

    Parameters default to the same as in optax.lbfgs.
    """

    def __init__(
        self,
        fun: Callable,
        max_steps: int = DEFAULT_MAX_STEPS,
        tol: float = DEFAULT_ATOL,
        stepsize: Optional[optax.ScalarOrSchedule] = None,
        memory_size: int = 10,
        scale_init_precond: bool = True,
        linesearch: Optional[
            Union[optax.GradientTransformationExtraArgs, optax.GradientTransformation]
        ] = optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20,
            initial_guess_strategy="one",
        ),
    ):
        # might have to adapt the signature
        self.fun = fun

        # might try the eqx.closure_to_pytree
        self.opt = optax.lbfgs(
            learning_rate=stepsize,
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
            linesearch=linesearch,
        )

        self.max_steps = max_steps
        # self.atol = atol
        # self.rtol = rtol
        self.tol = tol

    def init(self, fn, y, args, options, tags):
        pass

    def update(self):
        pass

    def terminate(self):
        pass

    def run(self, init_params, *args):
        def _fn(params):
            return self.fun(params, args)

        value_and_grad_fun = optax.value_and_grad_from_state(_fn)

        def step(carry):
            params, state = carry
            value, grad = value_and_grad_fun(params, state=state)
            updates, state = self.opt.update(
                grad, state, params, value=value, grad=grad, value_fn=_fn
            )
            params = optax.apply_updates(params, updates)
            return params, state

        def continuing_criterion(carry):
            _, state = carry
            iter_num = otu.tree_get(state, "count")
            grad = otu.tree_get(state, "grad")
            err = otu.tree_l2_norm(grad)
            return (iter_num == 0) | ((iter_num < self.max_steps) & (err >= self.tol))

        init_carry = (init_params, self.opt.init(init_params))
        final_params, final_state = jax.lax.while_loop(
            continuing_criterion, step, init_carry
        )

        return final_params, final_state
