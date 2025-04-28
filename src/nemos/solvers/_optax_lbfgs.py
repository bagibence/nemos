from typing import Callable, Optional, Union
from functools import partial


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
        self.fun = jax.jit(fun)

        # might try the eqx.closure_to_pytree
        self.opt = optax.lbfgs(
            learning_rate=stepsize,
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
            linesearch=linesearch,
        )

        self.has_linesearch = linesearch is not None

        self.max_steps = max_steps
        # self.atol = atol
        # self.rtol = rtol
        self.tol = tol

    def init(self, fn, y, args, options, tags):
        return self.opt.init(y)

    def update(self, params, state, *xy_args):
        def _fn(params):
            return self.fun(params, xy_args)

        value, grad = jax.value_and_grad(_fn)(params)
        updates, state = self.opt.update(
            grad, state, params, value=value, grad=grad, value_fn=_fn
        )
        params = optax.apply_updates(params, updates)

        return params, state

    def terminate(self):
        pass

    def run(self, init_params, *args):
        if self.has_linesearch:
            return self._run(init_params, *args)
        else:
            return self._run_without_store_grad(init_params, *args)

    @partial(jax.jit, static_argnums=0)
    def _run(self, init_params, *args):
        def _fn(params):
            return self.fun(params, args)

        value_and_grad_fun = jax.jit(optax.value_and_grad_from_state(_fn))
        # value_and_grad_fun = jax.jit(jax.value_and_grad(_fn))

        def step(carry):
            params, state = carry
            value, grad = value_and_grad_fun(params, state=state)
            # value, grad = value_and_grad_fun(params)
            updates, state = self.opt.update(
                grad, state, params, value=value, grad=grad, value_fn=_fn
            )
            params = optax.apply_updates(params, updates)

            # jax.debug.print(
            #    "iter:{iter_num} n_linesearch_steps:{n_linesearch_steps} learning_rate:{learning_rate} value:{value}",
            #    iter_num=state[0].count,
            #    n_linesearch_steps=state[-1].info.num_linesearch_steps,
            #    learning_rate=state[-1].learning_rate,
            #    value=state[-1].value,
            # )

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

    @partial(jax.jit, static_argnums=0)
    def _run_without_store_grad(self, init_params, *args):
        def _fn(params):
            return self.fun(params, args)

        # value_and_grad_fun = jax.jit(optax.value_and_grad_from_state(_fn))
        value_and_grad_fun = jax.jit(jax.value_and_grad(_fn))

        def step(carry):
            params, state = carry
            # value, grad = value_and_grad_fun(params, state=state)
            value, grad = value_and_grad_fun(params)
            updates, state = self.opt.update(
                grad, state, params, value=value, grad=grad, value_fn=_fn
            )
            params = optax.apply_updates(params, updates)

            # jax.debug.print(
            #    "iter:{iter_num} n_linesearch_steps:{n_linesearch_steps} learning_rate:{learning_rate} value:{value}",
            #    iter_num=state[0].count,
            #    n_linesearch_steps=state[-1].info.num_linesearch_steps,
            #    learning_rate=state[-1].learning_rate,
            #    value=state[-1].value,
            # )

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


class OptaxLBFGSScan:
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

        # prepare stateful grad
        value_and_grad = optax.value_and_grad_from_state(_fn)

        # one pure-JAX step
        def step(carry, _):
            params, state = carry
            v, g = value_and_grad(params, state=state)
            u, state = self.opt.update(g, state, params, value=v, grad=g, value_fn=_fn)
            params = optax.apply_updates(params, u)

            # after k≥1, we still want to skip computation once tol is met:
            def cont(c):
                # here, `c` is the carry; reuse step logic
                return step(c, None)[0]  # compute next carry

            # mask out future updates if we’re already converged:
            # is_done = otu.tree_l2_norm(state.updates) < self.tol
            is_done = otu.tree_l2_norm(otu.tree_get(state, "grad")) < self.tol
            params, state = jax.lax.cond(
                is_done, lambda c: c, lambda c: (params, state), (params, state)
            )

            return (params, state), None

        init_state = self.opt.init(init_params)
        init_carry = (init_params, init_state)

        # now scan *statically* over max_steps
        scan_fn = jax.jit(
            lambda c: jax.lax.scan(step, c, None, length=self.max_steps)[0]
        )
        final_params, final_state = scan_fn(init_carry)
        return final_params, final_state
