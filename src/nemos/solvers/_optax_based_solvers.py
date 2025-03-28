from collections.abc import Callable
from jaxtyping import PyTree, Scalar, ArrayLike
from typing import Optional, Union, Any, cast

import optimistix as optx
import optax

import jax
import equinox as eqx
from optimistix._custom_types import Aux, Fn, Out, SolverState, Y


from ._optimistix_solvers import OptimistixSolverMixin, DEFAULT_MAX_STEPS


def _make_rate_scaler(
    stepsize: float | None,
    linesearch_kwargs: dict[str, Any] | None,
):
    if stepsize is None:
        if linesearch_kwargs is None:
            linesearch_kwargs = {
                # "approx_dec_rtol" : None, # setting this to none might be useful
            }

        if "max_linesearch_steps" not in linesearch_kwargs:
            linesearch_kwargs["max_linesearch_steps"] = 15

        return optax.scale_by_zoom_linesearch(**linesearch_kwargs)
    else:
        return optax.scale_by_learning_rate(stepsize)


class GradientDescent(optx.OptaxMinimiser, OptimistixSolverMixin):
    fun: Callable
    fun_with_aux: Callable

    stats: dict[str, PyTree[ArrayLike]]

    def __init__(
        self,
        fun: Callable,
        rtol: float,
        atol: float,
        stepsize: Optional[float] = None,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        verbose: frozenset[str] = frozenset(),
        linesearch_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.fun = fun
        self.fun_with_aux = lambda params, args: (fun(params, args), None)

        self.stats = {}

        _optax_gd = optax.chain(
            optax.sgd(1.0, nesterov=True),
            _make_rate_scaler(stepsize, linesearch_kwargs),
        )

        super().__init__(
            optim=_optax_gd,
            rtol=rtol,
            atol=atol,
            norm=norm,
            verbose=verbose,
        )


class LBFGS(optx.OptaxMinimiser, OptimistixSolverMixin):
    fun: Callable
    fun_with_aux: Callable

    stats: dict[str, PyTree[ArrayLike]]

    def __init__(
        self,
        fun: Callable,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        verbose: frozenset[str] = frozenset(),
        stepsize: Optional[optax.ScalarOrSchedule] = None,
        memory_size: int = 10,
        scale_init_precond: bool = True,
        linesearch: Optional[
            Union[optax.GradientTransformationExtraArgs, optax.GradientTransformation]
        ] = optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20, initial_guess_strategy="one"
        ),
    ):
        self.fun = fun
        self.fun_with_aux = lambda params, args: (fun(params, args), None)

        self.stats = {}

        _optax_lbfgs = optax.lbfgs(
            learning_rate=stepsize,
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
            linesearch=linesearch,
        )

        super().__init__(
            optim=_optax_lbfgs,
            rtol=rtol,
            atol=atol,
            norm=norm,
            verbose=verbose,
        )


class ProximalGradient(optx.OptaxMinimiser, OptimistixSolverMixin):
    fun: Callable
    fun_with_aux: Callable

    stats: dict[str, PyTree[ArrayLike]]

    def __init__(
        self,
        fun: Callable,
        rtol: float,
        atol: float,
        prox: optax.GradientTransformation,
        stepsize: Optional[float] = None,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        verbose: frozenset[str] = frozenset(),
        linesearch_kwargs: Optional[dict] = None,
    ):
        self.fun = fun
        self.fun_with_aux = lambda params, args: (fun(params, args), None)

        self.stats = {}

        _optax_proxgrad = optax.chain(
            optax.sgd(learning_rate=1.0, nesterov=True),
            _make_rate_scaler(stepsize, linesearch_kwargs),
            prox,
        )

        super().__init__(
            optim=_optax_proxgrad,
            rtol=rtol,
            atol=atol,
            norm=norm,
            verbose=verbose,
        )

    def run(
        self,
        init_params,
        hyperparams_prox,
        *args,
        options: Optional[dict[str, Any]] = None,
        has_aux: bool = False,
        max_steps: Optional[int] = 100_000,
        adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint(),
        throw: bool = True,
        tags: frozenset[object] = frozenset(),
    ):
        del hyperparams_prox
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

    def update(
        self,
        params,
        state,
        hyperparams_prox,
        *xy_args,
        options: Optional[dict[str, Any]] = None,
        tags: frozenset[object] = frozenset(),
    ):
        del hyperparams_prox
        return super().update(params, state, *xy_args, options=options, tags=tags)
