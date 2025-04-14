from collections.abc import Callable
from jaxtyping import PyTree, Scalar, ArrayLike
from typing import Optional, Union, Any, cast, NamedTuple

import optimistix as optx
import optax

import jax
import jax.numpy as jnp
import equinox as eqx
from optimistix._custom_types import Aux, Fn, Out, SolverState, Y


from ._optimistix_solvers import OptimistixSolverMixin, DEFAULT_MAX_STEPS

# FIXME This might be solved in a simpler way using
# https://optax.readthedocs.io/en/latest/getting_started.html#accessing-learning-rate


class ScaleByLearningRateState(NamedTuple):
    learning_rate: Union[float, jax.Array]


def stateful_scale_by_learning_rate(
    stepsize: float, flip_sign: bool = True
) -> optax.GradientTransformation:
    """
    Reimplementation of optax.scale_by_learning_rate, just
    storing the learning rate in the state.
    Required for setting the scaling appropriately when used with
    proximal gradient descent.
    """
    m = -1 if flip_sign else 1

    def init_fn(params):
        del params
        return ScaleByLearningRateState(jnp.array(stepsize))

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree.map(lambda g: m * stepsize * g, updates)

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _make_rate_scaler(
    stepsize: float | None,
    linesearch_kwargs: dict[str, Any] | None,
) -> optax.GradientTransformation:
    """
    Make an Optax transformation for setting the learning rate.
    If `stepsize` is not None, use it as a constant learning rate.
    If `stepsize` is None, create a zoom linesearch with `linesearch_kwargs`.
    """
    if stepsize is None:
        if linesearch_kwargs is None:
            linesearch_kwargs = {
                # "approx_dec_rtol" : None, # setting this to none might be useful
            }

        if "max_linesearch_steps" not in linesearch_kwargs:
            linesearch_kwargs["max_linesearch_steps"] = 15

        return optax.scale_by_zoom_linesearch(**linesearch_kwargs)
        # if "max_backtracking_steps" not in linesearch_kwargs:
        #    linesearch_kwargs["max_backtracking_steps"] = 15

        # return optax.scale_by_backtracking_linesearch(**linesearch_kwargs)
    else:
        # NOTE GradientDescent works with optax.scale_by_learning_rate as well
        # but for _prox_grad_learning_rate_from_optax.ProximalGradient
        # we need to be able to extract the current learning rate
        return stateful_scale_by_learning_rate(stepsize)


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
        norm: Callable[[PyTree], Scalar],
        verbose: frozenset[str] = frozenset(),
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
