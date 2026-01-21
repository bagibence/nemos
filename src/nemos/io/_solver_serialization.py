from __future__ import annotations

import inspect
import warnings
from typing import Any, Tuple

from ..solvers import _solver_registry
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
    if not isinstance(spec, _solver_registry.SolverSpec):
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
        Optional user mapping. If provided, it overrides the registry lookup and
        is used to construct the SolverSpec directly. If the mapped solver class
        does not match the saved implementation name, a warning is emitted and
        the mapping still takes precedence. The saved implementation name uses
        the full module path (``module.ClassName``), and the comparison uses
        this full path rather than only the class name.
    """
    if not isinstance(solver, dict) or not solver.get("nemos_solver_spec", False):
        raise ValueError(
            "solver has to be a dict containing key 'nemos_solver_spec' with value True"
        )

    algo_name = solver["algo_name"]
    backend = solver["backend"]
    implementation = solver["implementation"]

    # If user provided a mapping, honor it first
    if mapping_dict is not None and "solver" in mapping_dict:
        mapped_solver = mapping_dict["solver"]
        if not inspect.isclass(mapped_solver):
            raise ValueError(
                "Invalid map parameter types detected. "
                "Only classes can be mapped for 'solver'."
            )
        mapped_impl_name = _get_name(mapped_solver)
        if mapped_impl_name != implementation:
            warnings.warn(
                "Mismatch between saved solver implementation and mapping_dict: "
                f"saved '{implementation}', mapped '{mapped_impl_name}'. "
                "Proceeding with mapping_dict solver.",
                UserWarning,
            )
        return _solver_registry.SolverSpec(algo_name, backend, mapped_solver)

    try:
        spec = _solver_registry.get_solver(f"{algo_name}[{backend}]")
    except (ValueError, KeyError) as exc:
        raise ValueError(
            "Failed to reconstruct solver from saved file. "
            "If you saved a custom solver, please either register it with "
            "`nemos.solvers.register` before loading or "
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
