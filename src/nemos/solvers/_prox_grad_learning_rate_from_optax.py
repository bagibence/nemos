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

from ._optax_based_solvers import _make_rate_scaler


class ProximalGradient(optx.OptaxMinimiser, OptimistixSolverMixin):
    """
    ProximalGradient implementation combining Optax and Optimistix.

    Uses Optax's SGD with Nesterov acceleration combined with Optax's
    zoom linesearch or a constant learning rate.
    Then uses the learning rate given by Optax to scale the proximal
    operator's update and check for convergence using Optimistix's.

    Works with the same proximal operator functions as JAXopt did.

    Passes the regularizer strength (aka. `hyperparams_prox` following
    the JAXopt naming) to .step through the `options` dict as
    `options["regularizer_strength"]`.
    """

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

        _optax_gd = optax.chain(
            optax.sgd(learning_rate=1.0, nesterov=True),
            _make_rate_scaler(stepsize, linesearch_kwargs),
        )

        super().__init__(
            optim=_optax_gd,
            rtol=rtol,
            atol=atol,
            norm=norm,
            verbose=verbose,
        )

    def get_learning_rate(self, state) -> float:
        """
        Read out the learning rate for scaling within the proximal operator.
        This learning rate is either a static learning rate or was found by a linesearch.
        """
        return state.opt_state[-1].learning_rate

    def step(
        self,
        fn,
        y,
        args: PyTree,
        options: dict[str, Any],
        state,
        tags: frozenset[object],
    ):
        # take gradient step
        new_params, new_state, new_aux = super().step(fn, y, args, options, state, tags)

        # apply the proximal operator
        new_params = self.prox(
            new_params,
            options["regularizer_strength"],
            self.get_learning_rate(new_state),
        )

        # recheck convergence criteria with the projected point
        updates = tree_sub(new_params, y)
        terminate = optx._misc.cauchy_termination(
            self.rtol,
            self.atol,
            self.norm,
            y,
            updates,
            new_state.f,
            new_state.f - state.f,
        )

        new_state = eqx.tree_at(lambda s: s.terminate, new_state, terminate)

        return new_params, new_state, new_aux

    # THIS ONE WORKS
    # it's a bit complicated
    # but it doesn't calculate the convergence criteria twice
    # def step(
    #    self,
    #    fn,
    #    y,
    #    args: PyTree,
    #    options: dict[str, Any],
    #    state,
    #    tags: frozenset[object],
    # ):
    #    (f, aux), grads = eqx.filter_value_and_grad(fn, has_aux=True)(y, args)
    #    f = cast(Array, f)
    #    # if len(self.verbose) > 0:
    #    #    verbose_print(
    #    #        ("step" in self.verbose, "Step", state.step),
    #    #        ("loss" in self.verbose, "Loss", f),
    #    #        ("y" in self.verbose, "y", y),
    #    #    )

    #    # fix args and discard aux
    #    _fn_for_optax = lambda y: fn(y, args)[0]

    #    updates, new_opt_state = self.optim.update(
    #        grads, state.opt_state, y, value=f, grad=grads, value_fn=_fn_for_optax
    #    )
    #    new_y = eqx.apply_updates(y, updates)

    #    new_y = self.prox(
    #        new_y,
    #        options["regularizer_strength"],
    #        new_opt_state[-1].learning_rate,
    #    )
    #    updates = tree_sub(new_y, y)

    #    terminate = optx._misc.cauchy_termination(
    #        self.rtol,
    #        self.atol,
    #        self.norm,
    #        y,
    #        updates,
    #        f,
    #        f - state.f,
    #    )
    #    new_state = _OptaxState(
    #        step=state.step + 1, f=f, opt_state=new_opt_state, terminate=terminate
    #    )
    #    return new_y, new_state, aux

    def run(
        self,
        init_params,
        hyperparams_prox,
        *args,
        options: dict[str, Any],
        has_aux: bool,
        max_steps: int,
        adjoint: optx.AbstractAdjoint,
        throw: bool,
        tags: frozenset[object],
    ):
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
        options: dict[str, Any],
        tags: frozenset[object],
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
