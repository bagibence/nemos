from typing import Optional, Union

import jaxopt
import optax

from ._optimistix_solvers import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_MAX_STEPS


class JaxoptOptaxLBFGS(jaxopt.OptaxSolver):
    """
    L-BFGS implementation using optax.lbfgs wrapped by JAXopt's.
    Useful for comparison with Optimistix's wrapper.

    Convergence criterion is implemented by JAXopt.

    Parameters default to the same as in optax.lbfgs.
    """

    def __init__(
        self,
        fun,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        stepsize: float = None,
        linesearch: Optional[
            Union[optax.GradientTransformationExtraArgs, optax.GradientTransformation]
        ] = optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20,
            initial_guess_strategy="one",
        ),
        implicit_diff: bool = False,
        verbose: bool = False,
        jit: bool = True,
    ):
        del rtol

        def _fun(params, *args):
            return fun(params, args)

        _optax_lbfgs = optax.lbfgs(
            learning_rate=stepsize,
            linesearch=linesearch,
        )

        super().__init__(
            fun=_fun,
            opt=_optax_lbfgs,
            tol=atol,
            maxiter=max_steps,
            implicit_diff=implicit_diff,
            verbose=verbose,
            jit=jit,
        )

    def init(self, fn, y, args):
        del fn
        return self.init_state(y, *args)

    def terminate(self):
        pass

    @property
    def max_steps(self):
        return self.maxiter
