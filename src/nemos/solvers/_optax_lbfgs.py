from typing import Callable, Optional, Union
from jaxtyping import PyTree
from functools import partial


import jax
import optax
import optax.tree_utils as otu

from ._optimistix_solvers import DEFAULT_MAX_STEPS, DEFAULT_ATOL, DEFAULT_RTOL


class OptaxLBFGS:
    """
    L-BFGS implementation using only optax.lbfgs and a jax.while_loop adapted from
    the Optax documentation.

    Convergence criterion is the same as in JAXopt: l2_norm(grad) <= abs_tol.
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
            increase_factor=2.0,
        ),
    ):
        """
        fun :
            Objective function.
        max_steps :
            Maximum number of optimization steps to take.
        tol :
            Absolute tolerance for the convergence criterion.
        stepsize :
            "optional global scaling factor" passed to `optax.lbfgs`.
            Default is None.
        memory_size :
            Memory or history size in the L-BFGS algorithm.
        scale_init_precond :
            "whether to use a scaled identity as the initial preconditioner"
            Always done in Optimistix, same as `use_gamma` in JAXopt (with a slightly modified formula).
        linesearch :
            Linesearch to use in the optimization.
            Needs to have `store_grad=True` set.
            Default is zoom linesearch.
        """
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

    # optax.ScaleByLBFGSState would be more precise but the checker is complaining
    def init(self, fn, y, args, options, tags) -> optax.OptState:
        """
        Initialize the optimizer state.

        y :
            Initial parameters.

        Rest of the arguments are included for consistency with Optimistix-based solvers.
        """
        return self.opt.init(y)

    def update(self, params, state, *xy_args) -> tuple[PyTree, optax.OptState]:
        """
        Perform a single step of the optimization.
        For running a whole optimization, see :meth:`run`.
        """

        def _fn(params):
            return self.fun(params, xy_args)

        value, grad = jax.value_and_grad(_fn)(params)
        updates, state = self.opt.update(
            grad, state, params, value=value, grad=grad, value_fn=_fn
        )
        params = optax.apply_updates(params, updates)

        return params, state

    def terminate(self) -> None:
        """
        Defined for consistency with Optimistix-based solvers.
        """
        pass

    def run(self, init_params, *args) -> tuple[PyTree, optax.OptState]:
        """
        Run a full optimization loop until convergence or until the max number of steps is reached.

        Convergence criterion is based on the gradient norm, consistent with JAXopt.

        Uses :meth:`_run` if the optimizer has a linesearch, :meth:`_run_without_store_grad` is used if it doesn't.
        """
        if self.has_linesearch:
            return self._run(init_params, *args)
        else:
            return self._run_without_store_grad(init_params, *args)

    @partial(jax.jit, static_argnums=0)
    def _run(self, init_params, *args) -> tuple[PyTree, optax.OptState]:
        """
        Run a full optimization loop.

        Called if the optimizer has a linesearch, otherwise :meth:`_run_without_store_grad` is used.

        The function value and gradient are stored in the linesearch state to save some computation.
        For background:
        - https://optax.readthedocs.io/en/stable/_collections/examples/lbfgs.html#linesearches-in-practice
        - https://optax.readthedocs.io/en/stable/api/utilities.html#optax.value_and_grad_from_state
        """

        def _fn(params):
            return self.fun(params, args)

        value_and_grad_fun = jax.jit(optax.value_and_grad_from_state(_fn))

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

    @partial(jax.jit, static_argnums=0)
    def _run_without_store_grad(
        self, init_params, *args
    ) -> tuple[PyTree, optax.OptState]:
        """
        Same as :meth:`_run`, but without reading the value and gradient from the search state.
        Called when the optimizer does not have a linesearch.
        """

        def _fn(params):
            return self.fun(params, args)

        value_and_grad_fun = jax.jit(jax.value_and_grad(_fn))

        def step(carry):
            params, state = carry
            value, grad = value_and_grad_fun(params)
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
