from typing import Union, Callable, Optional, Any

from ._optimistix_solvers import GradientDescent, DEFAULT_MAX_STEPS

from jaxtyping import PyTree, Scalar, ArrayLike

import optimistix as optx
import jax
import jax.numpy as jnp

from optimistix._misc import (
    lin_to_grad,
    cauchy_termination,
    filter_cond,
)
from optimistix._solver.gradient_methods import _GradientDescentState
from optimistix._search import FunctionInfo
from equinox.internal import ω
from optimistix._solution import RESULTS


class ProximalGradient(GradientDescent):
    """
    Fully Optimistix-based proximal gradient implementation using
    nemos._optimistix_solvers.GradientDescent
    """

    fun: Callable
    fun_with_aux: Callable
    prox: Callable

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]

    descent: optx.AbstractDescent
    search: optx.AbstractSearch

    _stepsize: Optional[Union[float, jax.Array]]

    stats: dict[str, PyTree[ArrayLike]]

    def __init__(
        self,
        fun: Callable,
        prox: Callable,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        stepsize: Optional[float] = None,
        linesearch_kwargs: Optional[dict] = None,
    ):
        self.prox = prox

        super().__init__(
            fun=fun,
            rtol=rtol,
            atol=atol,
            norm=norm,
            stepsize=stepsize,
            linesearch_kwargs=linesearch_kwargs,
        )

    def run(
        self,
        init_params,
        hyperparams_prox,
        *args,
        options: Optional[dict[str, Any]] = None,
        has_aux: bool = False,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,
        adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint(),
        throw: bool = True,
        tags: frozenset[object] = frozenset(),
    ):
        # del hyperparams_prox
        if options is None:
            options = {}

        options["regularizer_strength"] = hyperparams_prox

        return super().run(
            init_params,
            *args,
            options=options,
            has_aux=has_aux,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            tags=tags,
        )

    # This one copies the optimistix.GradientDescent.step and applies the proximal operator
    def step(
        self,
        fn,
        y,
        args: PyTree,
        options: dict[str, Any],
        state,
        tags: frozenset[object],
    ):
        autodiff_mode = options.get("autodiff_mode", "bwd")
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            FunctionInfo.Eval(f_eval),
            state.search_state,
        )

        def accepted(descent_state):
            grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode=autodiff_mode)

            f_eval_info = FunctionInfo.EvalGrad(f_eval, grad)
            descent_state = self.descent.query(state.y_eval, f_eval_info, descent_state)
            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step
            return state.y_eval, f_eval_info, aux_eval, descent_state, terminate

        def rejected(descent_state):
            return y, state.f_info, state.aux, descent_state, jnp.array(False)

        y, f_info, aux, descent_state, terminate = filter_cond(
            accept, accepted, rejected, state.descent_state
        )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        y_eval = (y**ω + y_descent**ω).ω

        y_eval = self.prox(
            y_eval,
            options["regularizer_strength"],
            search_state.step_size,
        )

        result = RESULTS.where(
            search_result == RESULTS.successful, descent_result, search_result
        )

        state = _GradientDescentState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
        )
        return y, state, aux

    #    def step(
    #        self,
    #        fn,
    #        y,
    #        args,
    #        options,
    #        state,
    #        tags,
    #    ):
    #        new_params, new_state, new_aux = super().step(fn, y, args, options, state, tags)
    #
    #        new_params_eval = self.prox(
    #            new_params,
    #            options["regularizer_strength"],
    #            self.get_learning_rate(new_state),
    #        )
    #
    #        new_state = eqx.tree_at(lambda s: s.y_eval, new_state, new_params_eval)
    #
    #        # TODO return new_params or new_params_eval?
    #
    #        # might need something like this?
    #        # new_params = jax.lax.cond(
    #        #    eqx.tree_equal(new_params, new_state.y_eval),
    #        #    lambda: new_params_eval,
    #        #    lambda: new_state.y_eval,
    #        # )
    #
    #        return new_params, new_state, new_aux
    #
