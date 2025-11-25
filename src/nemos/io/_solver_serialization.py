from __future__ import annotations

import inspect
from typing import Any, Tuple

from ..solvers import solver_registry
from ..utils import _get_name


def serialize_solver_spec(spec) -> Tuple[bool, Any]:
    """
    Convert a ``SolverSpec`` into a pickle-free, JSON-like dict.

    Parameters
    ----------
    spec :
        The solver specification to serialize.

    Returns
    -------
    handled, value :
        Tuple flagging whether the value was handled, and the serialized value if so.
        If not handled, returns (False, spec).
    """
    if not isinstance(spec, solver_registry.SolverSpec):
        return False, spec

    return True, {
        "nemos_solver_spec": True,
        "algo_name": spec.algo_name,
        "backend": spec.backend,
        "implementation": _get_name(spec.implementation),
    }


def deserialize_solver_spec(
    solver: dict[str, str], filename, mapping_dict: dict | None = None
):
    """
    Rebuild a SolverSpec saved as primitives, mirroring the regularizer path.

    Parameters
    ----------
    solver :
        Object stored under the solver key after applying custom maps.
    filename :
        Path to the saved npz, for error messages.
    mapping_dict :
        Optional user mapping, used only for messaging when reconstruction fails.
    """
    if not isinstance(solver, dict) or not solver.get("nemos_solver_spec", False):
        raise ValueError(
            "solver has to be a dict containing key 'nemos_solver_spec' with value True"
        )

    algo_name = solver.get("algo_name")
    backend = solver.get("backend")
    implementation = solver.get("implementation")

    # If user provided a mapping, honor it first
    if mapping_dict is not None and "solver" in mapping_dict:
        mapped_solver = mapping_dict["solver"]
        if not inspect.isclass(mapped_solver):
            raise ValueError(
                "Invalid map parameter types detected. "
                "Only classes or callables can be mapped for 'solver'."
            )
        return solver_registry.SolverSpec(
            mapped_solver.__name__, "custom", mapped_solver
        )

    try:
        spec = solver_registry.get_solver(f"{algo_name}[{backend}]")
    except Exception as exc:
        raise ValueError(
            "Failed to reconstruct solver from saved file. "
            "If you saved a custom solver, please either register it with "
            "`nemos.solvers.solver_registry.register` before loading or "
            "provide `mapping_dict={'solver': CustomSolverClass}` when calling "
            f"`nmo.load_model('{filename}')`."
        ) from exc

    # guard against silently loading a different implementation than the saved one
    saved_impl_name = implementation
    spec_impl_name = _get_name(spec.implementation)
    if saved_impl_name != spec_impl_name:
        raise ValueError(
            "Mismatch between saved solver implementation and registry entry: "
            f"saved '{saved_impl_name}', registry '{spec_impl_name}'. "
            "Consider registering the expected solver or passing it via mapping_dict."
        )

    return spec
