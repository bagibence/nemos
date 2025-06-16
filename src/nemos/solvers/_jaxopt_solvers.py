from typing import Optional

import jaxopt

from ._optimistix_solvers import DEFAULT_RTOL, DEFAULT_ATOL, DEFAULT_MAX_STEPS


class JaxoptProximalGradient(jaxopt.ProximalGradient):
    """
    jaxopt.ProximalGradient with its interface adapted to be compatible with
    nemos's new solver instantiation
    """

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
    """
    jaxopt.GradientDescent with its interface adapted to be compatible with
    nemos's new solver instantiation
    """

    def __init__(
        self,
        fun,
        max_steps: int = DEFAULT_MAX_STEPS,
        stepsize: float = -1.0,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        verbose: bool = False,
    ):
        del rtol

        def _fun(params, *args):
            return fun(params, args)

        super().__init__(
            fun=_fun,
            maxiter=max_steps,
            stepsize=stepsize,
            tol=atol,
            verbose=verbose,
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
    """
    jaxopt.LBFGS with its interface adapted to be compatible with
    nemos's new solver instantiation
    """

    def __init__(
        self,
        fun,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        stepsize: float = -1.0,
        history_size: int = 10,
        linesearch: str = "zoom",
        linesearch_init: str = "max",
        increase_factor: float = 2.0,
        maxls: int = 20,
        verbose: bool = False,
        unroll: bool = "auto",
        jit: bool = True,
    ):
        del rtol

        def _fun(params, *args):
            return fun(params, args)

        super().__init__(
            fun=_fun,
            tol=atol,
            maxiter=max_steps,
            stepsize=stepsize,
            history_size=history_size,
            linesearch=linesearch,
            linesearch_init=linesearch_init,
            increase_factor=increase_factor,
            maxls=maxls,
            verbose=verbose,
            unroll=unroll,
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


class JaxoptBFGS(jaxopt.BFGS):
    def __init__(
        self,
        fun,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        del rtol

        def _fun(params, *args):
            return fun(params, args)

        super().__init__(
            fun=_fun,
            tol=atol,
            maxiter=max_steps,
        )

    def init(self, fn, y, args):
        del fn
        return self.init_state(y, *args)

    def terminate(self):
        pass

    @property
    def max_steps(self):
        return self.maxiter


class JaxoptNonlinearCG(jaxopt.NonlinearCG):
    pass
