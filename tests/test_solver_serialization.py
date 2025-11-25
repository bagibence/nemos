"""Testing that custom solvers can be saved and loaded."""

import pytest

import nemos as nmo
from nemos.solvers._abstract_solver import OptimizationInfo

# include in solver-related tests
pytestmark = pytest.mark.solver_related


class _BaseDummySolver:
    """Minimal solver implementation for serialization tests."""

    def __init__(self, unregularized_loss, regularizer, regularizer_strength, **kwargs):
        self.fun = unregularized_loss
        self.regularizer = regularizer
        self.regularizer_strength = regularizer_strength

    @classmethod
    def get_accepted_arguments(cls):
        return set()

    def init_state(self, init_params, *args):
        return None

    def update(self, params, state, *args):
        return params, state

    def run(self, init_params, *args):
        return init_params, None

    def get_optim_info(self, state):
        return OptimizationInfo(
            function_val=None, num_steps=0, converged=True, reached_max_steps=False
        )


class RegisteredDummySolver(_BaseDummySolver):
    pass


class MappedDummySolver(_BaseDummySolver):
    pass


class UnregisteredDummySolver(_BaseDummySolver):
    pass


class RegistryMismatchSolverV1(_BaseDummySolver):
    pass


class RegistryMismatchSolverV2(_BaseDummySolver):
    pass


@pytest.fixture(autouse=True)
def allow_dummy_solvers(monkeypatch):
    """Temporarily allow dummy solvers and clean registry after tests."""

    # monkeypatch UnRegularized to allow the dummy solvers
    extra = {
        RegisteredDummySolver.__name__,
        MappedDummySolver.__name__,
        UnregisteredDummySolver.__name__,
        RegistryMismatchSolverV1.__name__,
        RegistryMismatchSolverV2.__name__,
    }
    original = nmo.regularizer.UnRegularized._allowed_solvers
    monkeypatch.setattr(
        nmo.regularizer.UnRegularized,
        "_allowed_solvers",
        original + tuple(s for s in extra if s not in original),
    )

    yield
    # setup was until here
    # on teardown remove the custom dummy solvers from the registry

    from nemos.solvers._solver_registry import _registry, _defaults

    for name in extra:
        if name in _registry and "custom" in _registry[name]:
            # remove custom dummy implementation
            _registry[name].pop("custom", None)
            # if there are no other implementations, remove the algo name
            if not _registry[name]:
                _registry.pop(name, None)
        # if the default implementation was the custom dummy one, remove it
        if _defaults.get(name) == "custom":
            _defaults.pop(name, None)


@pytest.mark.parametrize("glm_class_type", ["glm_class", "population_glm_class"])
def test_solver_serialization_with_registered_solver(tmp_path, request, glm_class_type):
    """Custom solver is reconstructed when registered in the solver registry."""

    glm_class = request.getfixturevalue(glm_class_type)
    nmo.solvers._solver_registry.register(
        RegisteredDummySolver.__name__,
        RegisteredDummySolver,
        backend="custom",
        replace=True,
        default=True,
    )

    model = glm_class(solver=RegisteredDummySolver)
    save_path = tmp_path / "model_solver_registered.npz"
    model.save_params(save_path)

    loaded = nmo.load_model(save_path)
    assert loaded.solver.implementation is RegisteredDummySolver
    assert loaded.solver.algo_name == RegisteredDummySolver.__name__
    assert loaded.solver.backend == "custom"


@pytest.mark.parametrize("glm_class_type", ["glm_class", "population_glm_class"])
def test_solver_serialization_with_mapping_dict(tmp_path, request, glm_class_type):
    """Custom solver is reconstructed using mapping_dict without registry entry."""

    glm_class = request.getfixturevalue(glm_class_type)
    model = glm_class(solver=MappedDummySolver)
    save_path = tmp_path / "model_solver_mapping.npz"
    model.save_params(save_path)

    mapping = {"solver": MappedDummySolver}
    loaded = nmo.load_model(save_path, mapping_dict=mapping)

    assert loaded.solver.implementation is MappedDummySolver
    assert loaded.solver.algo_name == MappedDummySolver.__name__
    assert loaded.solver.backend == "custom"


@pytest.mark.parametrize("glm_class_type", ["glm_class", "population_glm_class"])
def test_solver_serialization_unregistered_raises(tmp_path, request, glm_class_type):
    """Loading an unregistered custom solver without mapping raises a clear error."""

    glm_class = request.getfixturevalue(glm_class_type)
    model = glm_class(solver=UnregisteredDummySolver)
    save_path = tmp_path / "model_solver_unregistered.npz"
    model.save_params(save_path)

    with pytest.raises(ValueError, match="Failed to reconstruct solver"):
        nmo.load_model(save_path)


@pytest.mark.parametrize("glm_class_type", ["glm_class", "population_glm_class"])
def test_solver_serialization_registry_impl_mismatch(tmp_path, request, glm_class_type):
    """Loading should fail if registry implementation differs from saved metadata."""

    glm_class = request.getfixturevalue(glm_class_type)
    nmo.solvers._solver_registry.register(
        RegistryMismatchSolverV1.__name__,
        RegistryMismatchSolverV1,
        backend="custom",
        replace=True,
        default=True,
    )
    model = glm_class(solver=RegistryMismatchSolverV1)
    save_path = tmp_path / "model_solver_mismatch.npz"
    model.save_params(save_path)

    nmo.solvers._solver_registry.register(
        RegistryMismatchSolverV1.__name__,
        RegistryMismatchSolverV2,
        backend="custom",
        replace=True,
        default=True,
    )

    with pytest.raises(
        ValueError,
        match="Mismatch between saved solver implementation and registry entry",
    ):
        nmo.load_model(save_path)
