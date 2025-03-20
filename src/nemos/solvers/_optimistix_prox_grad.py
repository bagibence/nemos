from typing import Union, Callable, Optional, Any

from ._optimistix_solvers import OptimistixSolverMixin

from jaxtyping import PyTree, Scalar, ArrayLike

import optimistix as optx
import jax
import equinox as eqx


# TODO Inherit from .optimistix_solvers.GradientDescent
class ProximalGradient(optx.AbstractGradientDescent, OptimistixSolverMixin):
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
        self.fun = fun
        self.fun_with_aux = lambda params, args: (fun(params, args), None)
        self.prox = prox

        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = optx.SteepestDescent()

        self.stats = {}

        if stepsize is not None:
            assert not linesearch_kwargs
            self._stepsize = stepsize
            self.search = optx.LearningRate(stepsize)
        else:
            if linesearch_kwargs is None:
                linesearch_kwargs = {}
            self._stepsize = None
            self.search = optx.BacktrackingArmijo(**linesearch_kwargs)

    def get_learning_rate(self, state):
        if self._stepsize is None:
            return state.search_state.step_size

        # return self.search.learning_rate
        return self._stepsize

    def step(
        self,
        fn,
        y,
        args,
        options,
        state,
        tags,
    ):
        new_params, new_state, new_aux = super().step(fn, y, args, options, state, tags)

        new_params_eval = self.prox(
            new_params,
            options["regularizer_strength"],
            self.get_learning_rate(new_state),
        )

        new_state = eqx.tree_at(lambda s: s.y_eval, new_state, new_params_eval)

        # TODO return new_params or new_params_eval?

        return new_params, new_state, new_aux

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
