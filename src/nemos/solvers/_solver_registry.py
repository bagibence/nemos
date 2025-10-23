"""Registry for mapping from solver name to concrete implementation."""

from dataclasses import dataclass, field
from typing import Type

from ._jaxopt_solvers import (
    JaxoptBFGS,
    JaxoptGradientDescent,
    JaxoptLBFGS,
    JaxoptNonlinearCG,
    JaxoptProximalGradient,
)
from ._svrg import WrappedProxSVRG, WrappedSVRG


@dataclass
class SolverSpec:
    algo_name: str
    backend: str
    implementation: Type

    @property
    def full_name(self) -> str:
        return f"{self.algo_name}[{self.backend}]"

    def __repr__(self) -> str:
        return (
            f"{self.full_name!r} - "
            f"{self.__class__.__name__}("
            f"algo_name={self.algo_name!r}, "
            f"backend={self.backend!r}, "
            f"implementation={self.implementation})"
        )


@dataclass
class SolverRegistry:
    _registry: dict[str, dict[str, SolverSpec]] = field(default_factory=dict)
    _defaults: dict[str, str] = field(default_factory=dict)

    def _parse_name(self, name: str) -> tuple[str, str | None]:
        """Parse an algo_name[backend] string."""
        algo_name = name
        backend = None
        if "[" in name and name.endswith("]"):
            algo_name = name[: name.index("[")]
            backend = name[name.index("[") + 1 : -1]

        return algo_name, backend

    def _raise_if_not_in_registry(self, algo_name: str):
        """Raise an error if an algorithm is not in the registry."""
        if algo_name not in self._registry:
            raise ValueError(f"No solver registered for algorithm {algo_name}.")

    # TODO: Should the return type be SolverProtocol instead?
    def get_solver(self, name: str) -> Type:
        """Fetch the solver implementation from the registry."""
        algo_name, backend = self._parse_name(name)

        # make sure we have the algorithm
        self._raise_if_not_in_registry(algo_name)
        algo_versions = self._registry[algo_name]

        # if not specified, try getting the default backend for the algorithm
        if backend is None:
            backend = self._defaults.get(algo_name, None)
        if backend is None:
            if len(algo_versions) == 1:
                backend = next(iter(algo_versions.keys()))
            else:
                raise ValueError(
                    f"Multiple backends and no default found for {algo_name}. Please specify or set a default backend."
                )
        if backend not in algo_versions:
            raise ValueError(
                f"{backend} backend not available for {algo_name}. Available backends: {self.list_algo_backends(algo_name)}"
            )

        return algo_versions[backend].implementation

    def __getitem__(self, name: str) -> Type:
        """Fetch the solver implementation with nicer syntax."""
        return self.get_solver(name)

    def register(
        self,
        algo_name: str,
        implementation: Type,
        backend: str = "custom",
        replace: bool = False,
        default: bool = False,
    ):
        """
        Register a solver implementation in the registry.

        algo_name:
            Name of the optimization algorithm.
        implementation:
            Class implementing the solver.
            Has to adhere to the AbstractSolver interface.
        backend:
            Backend name. Defaults to "custom".
            When wrapping and registering an existing solver from an external
            package, this would be the package name.
        replace:
            If an implementation for the given algorithm and backend names
            is already present in the registry, overwrite it.
        default:
            Set this implementation as the default for the algorithm.
            Can also be done with `set_default`.
        """
        if not replace and backend in self._registry.get(algo_name, {}):
            raise ValueError(
                f"{algo_name}[{backend}] already registered. Use replace=True to overwrite."
            )
        if algo_name not in self._registry:
            self._registry[algo_name] = {}

        self._registry[algo_name][backend] = SolverSpec(
            algo_name, backend, implementation
        )

        if default:
            self.set_default(algo_name, backend)

    def set_default(self, algo_name: str, backend: str):
        """Set the default backend for a given algorithm."""
        self._raise_if_not_in_registry(algo_name)

        if backend not in self._registry[algo_name]:
            raise ValueError(
                f"{backend} backend not available for {algo_name}. Available backends: {self.list_algo_backends(algo_name)}"
            )
        self._defaults[algo_name] = backend

    def list_algo_backends(self, algo_name) -> list[str]:
        """List the available backend for an algorithm."""
        return list(self._registry[algo_name].keys())

    # TODO: Add doctest
    @property
    def available_solvers(self) -> list[SolverSpec]:
        """List all available solvers."""
        return [
            spec
            for algo_versions in self._registry.values()
            for spec in algo_versions.values()
        ]

    # TODO: Add doctest
    @property
    def available_algorithms(self) -> list[str]:
        """
        List the available algorithms that can be used for fitting models.

        To list the available backends for a given algorithm,
        see `list_algo_backends`.

        To access an extended documentation about a specific solver,
        see `nemos.solvers.get_solver_documentation`.
        """
        return list(self._registry.keys())


solver_registry = SolverRegistry()
solver_registry.register(
    "GradientDescent", JaxoptGradientDescent, "jaxopt", default=True
)
solver_registry.register(
    "ProximalGradient", JaxoptProximalGradient, "jaxopt", default=True
)
solver_registry.register("LBFGS", JaxoptLBFGS, "jaxopt", default=True)
solver_registry.register("BFGS", JaxoptBFGS, "jaxopt", default=True)
solver_registry.register("NonlinearCG", JaxoptNonlinearCG, "jaxopt", default=True)
solver_registry.register("SVRG", WrappedSVRG, "nemos", default=True)
solver_registry.register("ProxSVRG", WrappedProxSVRG, "nemos", default=True)
