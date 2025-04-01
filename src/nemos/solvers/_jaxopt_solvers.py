from typing import Optional

import jaxopt

from ._optimistix_solvers import DEFAULT_RTOL, DEFAULT_ATOL, DEFAULT_MAX_STEPS


class JaxoptProximalGradient(jaxopt.ProximalGradient):
    def __init__(
        self,
        fun,
        prox,
        max_steps: int = DEFAULT_MAX_STEPS,
        stepsize: float = -1.0,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
    ):
        del rtol

        def _fun(params, *args):
            return fun(params, args)

        super().__init__(
            fun=_fun,
            prox=prox,
            maxiter=max_steps,
            stepsize=stepsize,
            tol=atol,
        )

    def init(self, fn, y, args):
        del fn
        # args[0] is hyperparams_prox?
        return self.init_state(y, *args)

    def terminate(self):
        pass

    @property
    def max_steps(self):
        return self.maxiter


class JaxoptGradientDescent(jaxopt.GradientDescent):
    def __init__(
        self,
        fun,
        max_steps: int = DEFAULT_MAX_STEPS,
        stepsize: float = -1.0,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
    ):
        del rtol

        def _fun(params, *args):
            return fun(params, args)

        super().__init__(
            fun=_fun,
            maxiter=max_steps,
            stepsize=stepsize,
            tol=atol,
        )

    def init(self, fn, y, args):
        del fn
        return self.init_state(y, *args)

    def terminate(self):
        pass

    @property
    def max_steps(self):
        return self.maxiter


class JaxoptLBFGS(jaxopt.LBFGS):
    def __init__(
        self,
        fun,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
    ):
        del rtol

        def _fun(params, *args):
            return fun(params, args)

        super().__init__(
            fun=_fun,
            tol=atol,
        )

    def init(self, fn, y, args):
        del fn
        return self.init_state(y, *args)

    def terminate(self):
        pass

    @property
    def max_steps(self):
        return self.maxiter


class JaxoptBFGS(jaxopt.BFGS):
    pass


class JaxoptNonlinearCG(jaxopt.NonlinearCG):
    pass
