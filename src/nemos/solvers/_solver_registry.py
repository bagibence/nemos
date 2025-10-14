"""Registry for mapping from solver name to concrete implementation."""

from typing import Type

from ._jaxopt_solvers import (
    JaxoptBFGS,
    JaxoptGradientDescent,
    JaxoptLBFGS,
    JaxoptNonlinearCG,
    JaxoptProximalGradient,
)
from ._svrg import WrappedProxSVRG, WrappedSVRG

_solver_registry: dict[str, Type] = {
    "GradientDescent": JaxoptGradientDescent,
    #
    "ProximalGradient": JaxoptProximalGradient,
    #
    "LBFGS": JaxoptLBFGS,
    #
    "BFGS": JaxoptBFGS,
    #
    "SVRG": WrappedSVRG,
    "ProxSVRG": WrappedProxSVRG,
    #
    "NonlinearCG": JaxoptNonlinearCG,
}


def register_solver(name: str, cls: Type | None = None, replace: bool = False):
    """
    register_solver an optimizer class under a given name.

    Pass `replace` to overwrite an existing entry.

    Usage:
        @register_solver("sgd")
        class SGD:
            ...

    or:
        register_solver("adam", Adam)
    """

    def decorator(c: Type):
        # TODO: validate the solver here
        if name in _solver_registry and not replace:
            raise ValueError(f"Optimizer '{name}' already registered.")
        _solver_registry[name] = c
        return c

    # support both decorator and function styles
    if cls is None:
        return decorator
    else:
        return decorator(cls)


def get_solver(name: str) -> Type:
    """Get a solver by its name in the registry."""
    if name not in _solver_registry:
        raise KeyError(
            f"""No solver with the name {name} is available.
            The following solvers are available: {list_available_solvers()}"""
        )

    return _solver_registry[name]


def list_available_solvers():
    """
    List the available solvers that can be used for fitting models.

    To access an extended documentation about a specific solver,
    see `get_solver_documentation`.

    Example
    -------
    >>> import nemos as nmo
    >>> nmo.solvers.list_available_solvers()
    ['GradientDescent', 'ProximalGradient', 'LBFGS', 'BFGS', 'SVRG', 'ProxSVRG', 'NonlinearCG']
    >>> print(nmo.solvers.get_solver_documentation("SVRG"))
    Showing docstring of nemos.solvers._svrg.WrappedSVRG.
    For potentially more info, use `show_help=True`.
    <BLANKLINE>
    Adapter for NeMoS's implementation of SVRG following the AbstractSolver interface.
    <BLANKLINE>
    ...
    """
    return list(_solver_registry.keys())
