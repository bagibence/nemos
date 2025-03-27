from collections.abc import Callable
from jaxtyping import PyTree, Scalar, ArrayLike
from typing import Optional, Union, Any, cast

import optimistix as optx
import optax

import jax
import equinox as eqx
from optimistix._custom_types import Aux, Fn, Out, SolverState, Y

DEFAULT_ATOL = 1e-8
DEFAULT_RTOL = 0.0


class OptimistixSolverMixin:
    def run(
        self,
        init_params,
        *args,
        options: Optional[dict[str, Any]] = None,
        has_aux: bool = False,
        max_steps: Optional[int] = 100_000,
        adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint(),
        throw: bool = True,
        tags: frozenset[object] = frozenset(),
    ):
        # for signature of optimistix.minimise look in
        # https://github.com/patrick-kidger/optimistix/blob/main/optimistix/_minimise.py#L40

        solution = optx.minimise(
            fn=self.fun,
            solver=self,
            y0=init_params,
            args=args,
            options=options,
            has_aux=has_aux,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            tags=tags,
        )

        self.stats.update(solution.stats)

        return solution.value, solution.state

    def update(
        self,
        params,
        state,
        *xy_args,
        options: Optional[dict[str, Any]] = None,
        tags: frozenset[object] = frozenset(),
    ):
        # for signature of solver.step look in
        # https://github.com/patrick-kidger/optimistix/blob/main/optimistix/_iterate.py#L76
        # https://github.com/patrick-kidger/optimistix/blob/main/optimistix/_solver/gradient_methods.py#L163

        new_params, state, aux = self.step(
            fn=self.fun_with_aux,
            y=params,
            args=xy_args,
            state=state,
            options=options,
            tags=tags,
        )

        return new_params, state

    ## TODO type annotations
    # def init(
    #    self,
    #    fn,
    #    y,
    #    args: PyTree,
    #    options: dict[str, Any],
    #    tags: frozenset[object] = frozenset(),
    # ):
    #    y = jax.tree.map(optx._misc.inexact_asarray, y)
    #    # if not has_aux:
    #    #    fn = NoneAux(fn)  # pyright: ignore
    #    fn = optx._misc.OutAsArray(fn)
    #    fn = eqx.filter_closure_convert(fn, y, args)  # pyright: ignore
    #    fn = cast(Fn[Y, Scalar, Aux], fn)
    #    f_struct, aux_struct = fn.out_struct
    #    if options is None:
    #        options = {}

    #    return super().init(
    #        fn,
    #        y,
    #        args,
    #        options=options,
    #        tags=tags,
    #        f_struc=f_struct,
    #        aux_struct=aux_struct,
    #    )

    # def init_state():
    #    # for signature of solver.init look in
    #    # https://github.com/patrick-kidger/optimistix/blob/main/optimistix/_iterate.py#L36
    #    # https://github.com/patrick-kidger/optimistix/blob/main/optimistix/_solver/gradient_methods.py#L140
    #    return solver.init(
    #        fn=loss,
    #        y=params,
    #        args=run_args,
    #        **solver_init_state_kwargs,
    #    )


class BFGS(optx.BFGS, OptimistixSolverMixin):
    fun: Callable
    fun_with_aux: Callable

    stats: dict[str, PyTree[ArrayLike]]

    def __init__(
        self,
        fun: Callable,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        use_inverse: bool = True,
        verbose: frozenset[str] = frozenset(),
    ):
        self.fun = fun
        self.fun_with_aux = lambda params, args: (fun(params, args), None)

        self.stats = {}

        super().__init__(rtol, atol, norm, use_inverse, verbose)


class NonlinearCG(optx.NonlinearCG, OptimistixSolverMixin):
    fun: Callable
    fun_with_aux: Callable

    stats: dict[str, PyTree[ArrayLike]]

    def __init__(
        self,
        fun: Callable,
        rtol: float,
        atol: float,
        norm: Callable = optx.max_norm,
        method: Callable = optx.polak_ribiere,
        search: optx.AbstractSearch = optx.BacktrackingArmijo(
            decrease_factor=0.5, slope=0.1
        ),
    ):
        self.fun = fun
        self.fun_with_aux = lambda params, args: (fun(params, args), None)

        self.stats = {}

        super().__init__(rtol, atol, norm, method, search)


class GradientDescent(optx.AbstractGradientDescent, OptimistixSolverMixin):
    fun: Callable
    fun_with_aux: Callable

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
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        stepsize: Optional[float] = None,
        linesearch_kwargs: Optional[dict] = None,
    ):
        self.fun = fun
        self.fun_with_aux = lambda params, args: (fun(params, args), None)

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
            if "decrease_factor" not in linesearch_kwargs:
                linesearch_kwargs["decrease_factor"] = 0.8
            self._stepsize = None
            self.search = optx.BacktrackingArmijo(**linesearch_kwargs)

    def get_learning_rate(self, state):
        if self._stepsize is None:
            return state.search_state.step_size

        # return self.search.learning_rate
        return self._stepsize
