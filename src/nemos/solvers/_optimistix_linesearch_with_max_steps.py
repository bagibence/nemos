from typing import cast, Union
from typing_extensions import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, Scalar, ScalarLike

from optimistix._custom_types import Y
from optimistix._search import AbstractSearch, FunctionInfo
from optimistix._solution import RESULTS

from optimistix._solver.backtracking import _FnInfo, _FnEvalInfo, _BacktrackingState

import optimistix as optx


class _MaxStepsBacktrackingState(eqx.Module, strict=True):
    step_size: Scalar
    iter_num: Scalar


# might want to have a linesearch inside
class MaxStepsBacktrackingArmijo(optx.BacktrackingArmijo):
    decrease_factor: ScalarLike = 0.5
    slope: ScalarLike = 1e-4
    step_init: ScalarLike = 1.0
    max_linesearch_steps: int = 25

    def __post_init__(self):
        super().__post_init__()
        self.step_init = eqx.error_if(
            self.max_linesearch_steps,
            self.max_linesearch_steps <= 0,  # pyright: ignore
            "`BacktrackingArmoji(max_linesearch_steps=...)` must be strictly greater than 0",
        )

    def init(self, y: Y, f_info_struct: _FnInfo) -> _MaxStepsBacktrackingState:
        del y, f_info_struct
        return _MaxStepsBacktrackingState(
            step_size=jnp.array(self.step_init, jnp.float64),
            iter_num=jnp.array(0),
        )

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        state: _MaxStepsBacktrackingState,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, _MaxStepsBacktrackingState]:
        # _state only holds the same step_size as is returned here
        step_size, accept, res, _state = super().step(
            first_step,
            y,
            y_eval,
            f_info,
            f_eval_info,
            _BacktrackingState(step_size=state.step_size),
        )

        # satisfy the criteria or reached the max steps
        accept = accept | (state.iter_num >= self.max_linesearch_steps)

        # after acceptance, reset the iter_num to zero and the step size to its default
        # otherwise increment iter_num and leave step_size
        new_iter_num = jnp.where(
            accept,
            jnp.array(0),
            state.iter_num + 1,
        )

        new_step_size = jnp.where(
            accept,
            jnp.array(self.step_init),
            step_size,
        )

        new_step_size = cast(Scalar, new_step_size)

        new_state = _MaxStepsBacktrackingState(
            step_size=new_step_size,
            iter_num=new_iter_num,
        )

        return (
            new_step_size,
            accept,
            res,
            new_state,
        )


MaxStepsBacktrackingArmijo.__init__.__doc__ += """
max_linesearch_steps: The maximal number of linesearch steps to take.
"""
