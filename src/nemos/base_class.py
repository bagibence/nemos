"""Abstract class for estimators."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import abc
import inspect
import warnings
from collections import defaultdict
from typing import Any, NamedTuple, Optional, Tuple, Union, TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import jaxopt
from numpy.typing import ArrayLike, NDArray

from . import validation, utils
from .pytrees import FeaturePytree
from ._regularizer_builder import create_regularizer

DESIGN_INPUT_TYPE = Union[jnp.ndarray, FeaturePytree]

if TYPE_CHECKING:
    from regularizer import Regularizer

SolverRun = Callable[
    [
        Any,  # parameters, could be any pytree
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    jaxopt.OptStep,
]

SolverInit = Callable[
    [
        Any,  # parameters, could be any pytree
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    NamedTuple,
]

SolverUpdate = Callable[
    [
        Any,  # parameters, could be any pytree
        NamedTuple,
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    jaxopt.OptStep,
]


class Base:
    """Base class for NeMoS estimators.

    A base class for estimators with utilities for getting and setting parameters,
    and for interacting with specific devices like CPU, GPU, and TPU.

    This class provides utilities for:
    - Getting and setting parameters using introspection.
    - Sending arrays to target devices (CPU, GPU, TPU).

    Notes
    -----
    The class provides helper methods mimicking scikit-learn's get_params and set_params.
    Additionally, it has methods for selecting target devices and sending arrays to them.
    """

    def get_params(self, deep=True):
        """
        From scikit-learn, get parameters by inspecting init.

        Parameters
        ----------
        deep

        Returns
        -------
            out:
                A dictionary containing the parameters. Key is the parameter
                name, value is the parameter value.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params: Any):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        nested_params: defaultdict = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            # TODO(1.4): remove specific handling of "base_estimator".
            # The "base_estimator" key is special. It was deprecated and
            # renamed to "estimator" for several estimators. This means we
            # need to translate it here and set sub-parameters on "estimator",
            # but only if the user did not explicitly set a value for
            # "base_estimator".
            if (
                    key == "base_estimator"
                    and valid_params[key] == "deprecated"
                    and self.__module__.startswith("sklearn.")
            ):
                warnings.warn(
                    (
                        f"Parameter 'base_estimator' of {self.__class__.__name__} is"
                        " deprecated in favor of 'estimator'. See"
                        f" {self.__class__.__name__}'s docstring for more details."
                    ),
                    FutureWarning,
                    stacklevel=2,
                )
                key = "estimator"
            valid_params[key].set_params(**sub_params)

        return self

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator."""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "GLM estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )

        # Consider the constructor parameters excluding 'self'
        parameters = [
            p.name for p in init_signature.parameters.values() if p.name != "self"
        ]

        # remove kwargs
        if "kwargs" in parameters:
            parameters.remove("kwargs")
        # Extract and sort argument names excluding 'self'
        return sorted(parameters)


class BaseRegressor(Base, abc.ABC):
    """Abstract base class for GLM regression models.

    This class encapsulates the common functionality for Generalized Linear Models (GLM)
    regression models. It provides an abstraction for fitting the model, making predictions,
    scoring the model, simulating responses, and preprocessing data. Concrete classes
    are expected to provide specific implementations of the abstract methods defined here.

    Parameters
    ----------
    regularizer :
        Regularization to use for model optimization. Defines the regularization scheme
        and related parameters.
        Default is UnRegularized regression.
    solver_name :
        Solver to use for model optimization. Defines the optimization scheme and related parameters.
        The solver must be an appropriate match for the chosen regularizer.
        Default is `None`. If no solver specified, one will be chosen based on the regularizer.
        Please see table below forregularizer/optimizer pairings.

    +---------------+------------------+-------------------------------------------------------------+
    | Regularizer   | Default Solver   | Available Solvers                                           |
    +===============+==================+=============================================================+
    | UnRegularized | GradientDescent  | GradientDescent, BFGS, LBFGS, ScipyMinimize, NonlinearCG,   |
    |               |                  | ScipyBoundedMinimize, LBFGSB                                |
    +---------------+------------------+-------------------------------------------------------------+
    | Ridge         | GradientDescent  | GradientDescent, BFGS, LBFGS, ScipyMinimize, NonlinearCG,   |
    |               |                  | ScipyBoundedMinimize, LBFGSB                                |
    +---------------+------------------+-------------------------------------------------------------+
    | Lasso         | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+
    | GroupLasso    | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+

    See Also
    --------
    Concrete models:

    - [`GLM`](../glm/#nemos.glm.GLM): A feed-forward GLM implementation.
    - [`GLMRecurrent`](../glm/#nemos.glm.GLMRecurrent): A recurrent GLM implementation.
    """

    def __init__(
            self,
            regularizer: str | Regularizer = "unregularized",
            solver_name: str = None,
            solver_kwargs: Optional[dict] = None
    ):
        self.regularizer = regularizer

        if solver_name is None:
            self.solver_name = self.regularizer.default_solver
        else:
            self.solver_name = solver_name

        if solver_kwargs is None:
            solver_kwargs = dict()
        self.solver_kwargs = solver_kwargs

    @property
    def regularizer(self) -> Union[None, Regularizer]:
        """Getter for the regularizer attribute."""
        return self._regularizer

    @regularizer.setter
    def regularizer(self, regularizer: str | Regularizer):
        """Setter for the regularizer attribute."""
        # instantiate regularizer
        if isinstance(regularizer, str):
            self._regularizer = create_regularizer(name=regularizer)
        else:
            self._regularizer = regularizer

    @property
    def solver_name(self) -> str:
        return self._solver_name

    @solver_name.setter
    def solver_name(self, solver_name: str):
        # check if solver str passed is valid for regularizer
        if solver_name not in self._regularizer.allowed_solvers:
            raise ValueError(f"The solver: {solver_name} is not allowed for "
                             f"{self._regularizer.__class__} regularizaration. Allowed solvers are "
                             f"{self._regularizer.allowed_solvers}.")
        self._solver_name = solver_name

    @property
    def solver_kwargs(self):
        return self._solver_kwargs

    @solver_kwargs.setter
    def solver_kwargs(self, solver_kwargs: dict):
        self._check_solver_kwargs(self.solver_name, solver_kwargs)
        self._solver_kwargs = solver_kwargs

    @staticmethod
    def _check_solver_kwargs(solver_name, solver_kwargs):
        """
        Check if provided solver keyword arguments are valid.

        Parameters
        ----------
        solver_name :
            Name of the solver.
        solver_kwargs :
            Additional keyword arguments for the solver.

        Raises
        ------
        NameError
            If any of the solver keyword arguments are not valid.
        """
        solver_args = inspect.getfullargspec(getattr(jaxopt, solver_name)).args
        undefined_kwargs = set(solver_kwargs.keys()).difference(solver_args)
        if undefined_kwargs:
            raise NameError(
                f"kwargs {undefined_kwargs} in solver_kwargs not a kwarg for jaxopt.{solver_name}!"
            )

    def instantiate_solver(
        self,
        loss: Callable,
        *args: Any,
        prox: Optional[Callable] = None,
        **kwargs: Any
    ) -> Tuple[SolverInit, SolverUpdate, SolverRun]:
        """
        Instantiate the solver with the provided loss function.

        Instantiate the solver with the provided loss function, and return callable functions
        that initialize the solver state, update the model parameters, and run the optimization.

        This method creates a solver instance from jaxopt library, tailored to the specific loss
        function and regularization approach defined by the Regularizer instance. It also handles
        the proximal operator if required for the optimization method. The returned functions are
         directly usable in optimization loops, simplifying the syntax by pre-setting
        common arguments like regularization strength and other hyperparameters.

        Parameters
        ----------
        loss :
            The loss function to be optimized.

        *args:
            Positional arguments for the jaxopt `solver.run` method, e.g. the regularizing
            strength for proximal gradient methods.

        prox:
            Optional, the proximal projection operator.

        *kwargs:
            Keyword arguments for the jaxopt `solver.run` method.

        Returns
        -------
        :
            A tuple containing three callable functions:
            - solver_init_state: Function to initialize the solver's state, necessary before starting the optimization.
            - solver_update: Function to perform a single update step in the optimization process,
            returning new parameters and state.
            - solver_run: Function to execute the optimization process, applying multiple updates until a
            stopping criterion is met.
        """
        # check that the loss is Callable
        utils.assert_is_callable(loss, "loss")

        # final check that solver is valid for chosen regularizer
        if self.solver_name not in self.regularizer.allowed_solvers:
            raise ValueError(f"The solver: {self.solver_name} is not allowed for "
                             f"{self._regularizer.__class__} regularizaration. Allowed solvers are "
                             f"{self._regularizer.allowed_solvers}.")

        # get the solver with given arguments.
        # The "fun" argument is not always the first one, but it is always KEYWORD
        # see jaxopt.EqualityConstrainedQP for example. The most general way is to pass it as keyword.
        # The proximal gradient is added to the kwargs if passed. This avoids issues with over-writing
        # the proximal operator.
        if "prox" in self.solver_kwargs:
            if prox is None:
                raise ValueError(
                    f"Regularizer of type {self.regularizer.__class__.__name__} "
                    f"does not require a proximal operator!"
                )
            else:
                warnings.warn(
                    "Overwritten the user-defined proximal operator! "
                    "There is only one valid proximal operator for each regularizer type.",
                    UserWarning,
                )
        # update the kwargs if prox is passed
        if prox is not None:
            solver_kwargs = self.solver_kwargs.copy()
            solver_kwargs.update(prox=prox)
        else:
            solver_kwargs = self.solver_kwargs
        solver = getattr(jaxopt, self._solver_name)(fun=loss, **solver_kwargs)

        def solver_run(
            init_params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray], *run_args: jnp.ndarray
        ) -> jaxopt.OptStep:
            return solver.run(init_params, *args, *run_args, **kwargs)

        def solver_update(params, state, *run_args, **run_kwargs) -> jaxopt.OptStep:
            return solver.update(
                params, state, *args, *run_args, **kwargs, **run_kwargs
            )

        def solver_init_state(params, state, *run_args, **run_kwargs) -> NamedTuple:
            return solver.init_state(
                params, state, *args, *run_args, **kwargs, **run_kwargs
            )

        return solver_init_state, solver_update, solver_run

    @abc.abstractmethod
    def fit(self, X: DESIGN_INPUT_TYPE, y: Union[NDArray, jnp.ndarray]):
        """Fit the model to neural activity."""
        pass

    @abc.abstractmethod
    def predict(self, X: DESIGN_INPUT_TYPE) -> jnp.ndarray:
        """Predict rates based on fit parameters."""
        pass

    @abc.abstractmethod
    def score(
            self,
            X: DESIGN_INPUT_TYPE,
            y: Union[NDArray, jnp.ndarray],
            # may include score_type or other additional model dependent kwargs
            **kwargs,
    ) -> jnp.ndarray:
        """Score the predicted firing rates (based on fit) to the target neural activity."""
        pass

    @abc.abstractmethod
    def simulate(
            self,
            random_key: jax.Array,
            feed_forward_input: DESIGN_INPUT_TYPE,
    ):
        """Simulate neural activity in response to a feed-forward input and recurrent activity."""
        pass

    @staticmethod
    @abc.abstractmethod
    def _check_params(
            params: Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike],
            data_type: Optional[jnp.dtype] = None,
    ) -> Tuple[DESIGN_INPUT_TYPE, jnp.ndarray]:
        """
        Validate the dimensions and consistency of parameters and data.

        This function checks the consistency of shapes and dimensions for model
        parameters.
        It ensures that the parameters and data are compatible for the model.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _check_input_dimensionality(
            X: Optional[Union[DESIGN_INPUT_TYPE, jnp.ndarray]] = None,
            y: Optional[jnp.ndarray] = None,
    ):
        pass

    @abc.abstractmethod
    def _get_coef_and_intercept(self) -> Tuple[Any, Any]:
        """Pack coef_ and intercept_  into a params pytree."""
        pass

    @abc.abstractmethod
    def _set_coef_and_intercept(self, params: Any):
        """Unpack and store params pytree to coef_ and intercept_."""
        pass

    @staticmethod
    @abc.abstractmethod
    def _check_input_and_params_consistency(
            params: Tuple[Union[DESIGN_INPUT_TYPE, jnp.ndarray], jnp.ndarray],
            X: Optional[Union[DESIGN_INPUT_TYPE, jnp.ndarray]] = None,
            y: Optional[jnp.ndarray] = None,
    ):
        """Validate the number of features in model parameters and input arguments.

        Raises
        ------
        ValueError
            - if the number of features is inconsistent between params[1] and X
              (when provided).

        """
        pass

    @staticmethod
    def _check_input_n_timepoints(
            X: Union[DESIGN_INPUT_TYPE, jnp.ndarray], y: jnp.ndarray
    ):
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "The number of time-points in X and y must agree. "
                f"X has {X.shape[0]} time-points, "
                f"y has {y.shape[0]} instead!"
            )

    def _validate(
            self,
            X: Union[DESIGN_INPUT_TYPE, jnp.ndarray],
            y: Union[NDArray, jnp.ndarray],
            init_params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
    ):
        # check input dimensionality
        self._check_input_dimensionality(X, y)
        self._check_input_n_timepoints(X, y)

        # error if all samples are invalid
        validation.error_all_invalid(X, y)

        # validate input and params consistency
        init_params = self._check_params(init_params)

        # validate input and params consistency
        self._check_input_and_params_consistency(init_params, X=X, y=y)

    @abc.abstractmethod
    def update(
            self,
            params: Tuple[jnp.ndarray, jnp.ndarray],
            opt_state: NamedTuple,
            X: DESIGN_INPUT_TYPE,
            y: jnp.ndarray,
            *args,
            **kwargs,
    ) -> jaxopt.OptStep:
        """Run a single update step of the jaxopt solver."""
        pass

    @abc.abstractmethod
    def initialize_solver(
            self,
            X: DESIGN_INPUT_TYPE,
            y: jnp.ndarray,
            *args,
            params: Optional = None,
            **kwargs,
    ) -> Tuple[Any, NamedTuple]:
        """Initialize the solver's state and optionally sets initial model parameters for the optimization."""
        pass
