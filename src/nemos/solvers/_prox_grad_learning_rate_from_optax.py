import equinox as eqx
import optax
import optimistix as optx
from typing import NamedTuple, Union, Any, Callable, Optional, cast

from nemos.solvers._optimistix_solvers import OptimistixSolverMixin, DEFAULT_MAX_STEPS

from jaxtyping import PyTree, Scalar, ArrayLike, Array

import jax
import jax.numpy as jnp

from optimistix._solver.optax import _OptaxState
from ..tree_utils import tree_sub


class ScaleByLearningRateState(NamedTuple):
    learning_rate: Union[float, jax.Array]


def scale_by_learning_rate(
    stepsize: float, flip_sign: bool = True
) -> optax.GradientTransformation:
    m = -1 if flip_sign else 1

    def init_fn(params):
        del params
        return ScaleByLearningRateState(jnp.array(stepsize))

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree.map(lambda g: m * stepsize * g, updates)

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


class ProximalGradient(optx.OptaxMinimiser, OptimistixSolverMixin):
    fun: Callable
    fun_with_aux: Callable
    prox: Callable

    stats: dict[str, PyTree[ArrayLike]]

    def __init__(
        self,
        fun: Callable,
        rtol: float,
        atol: float,
        prox: Callable,
        stepsize: Optional[float] = None,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        verbose: frozenset[str] = frozenset(),
        linesearch_kwargs: Optional[dict] = None,
    ):
        self.fun = fun
        self.fun_with_aux = lambda params, args: (fun(params, args), None)
        self.prox = prox

        self.stats = {}

        _optax_proxgrad = optax.chain(
            optax.sgd(learning_rate=1.0, nesterov=True),
            self._make_rate_scaler(stepsize, linesearch_kwargs),
        )

        super().__init__(
            optim=_optax_proxgrad,
            rtol=rtol,
            atol=atol,
            norm=norm,
            verbose=verbose,
        )

    @staticmethod
    def _make_rate_scaler(
        stepsize: float | None,
        linesearch_kwargs: dict[str, Any] | None,
    ):
        if stepsize is None:
            if linesearch_kwargs is None:
                linesearch_kwargs = {
                    "approx_dec_rtol": None,  # setting this to none might be useful
                }

            if "max_linesearch_steps" not in linesearch_kwargs:
                linesearch_kwargs["max_linesearch_steps"] = 15

            return optax.scale_by_zoom_linesearch(**linesearch_kwargs)
        else:
            return scale_by_learning_rate(stepsize)

    def get_learning_rate(self, state):
        return state.opt_state[-1].learning_rate

    # def step(
    #    self,
    #    fn,
    #    y,
    #    args,
    #    options,
    #    state,
    #    tags,
    # ):
    #    new_params, new_state, new_aux = super().step(fn, y, args, options, state, tags)

    #    new_params = self.prox(new_params, self.get_learning_rate(new_state))

    #    # TODO do I need something like this or not?
    #    # new_state = eqx.tree_at(lambda s: s.y_eval, new_state, new_params)

    #    return new_params, new_state, new_aux

    def step(
        self,
        fn,
        y,
        args: PyTree,
        options: dict[str, Any],
        state,
        tags: frozenset[object],
    ):
        (f, aux), grads = eqx.filter_value_and_grad(fn, has_aux=True)(y, args)
        f = cast(Array, f)
        # if len(self.verbose) > 0:
        #    verbose_print(
        #        ("step" in self.verbose, "Step", state.step),
        #        ("loss" in self.verbose, "Loss", f),
        #        ("y" in self.verbose, "y", y),
        #    )

        # fix args and discard aux
        _fn_for_optax = lambda y: fn(y, args)[0]

        updates, new_opt_state = self.optim.update(
            grads, state.opt_state, y, value=f, grad=grads, value_fn=_fn_for_optax
        )
        new_y = eqx.apply_updates(y, updates)

        new_y = self.prox(
            new_y,
            options["regularizer_strength"],
            new_opt_state[-1].learning_rate,
        )
        updates = tree_sub(new_y, y)

        terminate = optx._misc.cauchy_termination(
            self.rtol,
            self.atol,
            self.norm,
            y,
            updates,
            f,
            f - state.f,
        )
        new_state = _OptaxState(
            step=state.step + 1, f=f, opt_state=new_opt_state, terminate=terminate
        )
        return new_y, new_state, aux

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

    def update(
        self,
        params,
        state,
        hyperparams_prox,
        *xy_args,
        options: Optional[dict[str, Any]] = None,
        tags: frozenset[object] = frozenset(),
    ):
        if options is None:
            options = {}

        options["regularizer_strength"] = hyperparams_prox

        new_params, state, aux = self.step(
            fn=self.fun_with_aux,
            y=params,
            args=xy_args,
            state=state,
            options=options,
            tags=tags,
        )

        return new_params, state
