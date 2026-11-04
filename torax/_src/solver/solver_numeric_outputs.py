# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data structure for numerical solver outputs."""

import dataclasses

import jax
from torax._src import array_typing
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SolverNumericOutputs:
  """Numerical quantities related to the solver.

  Attributes:
    outer_solver_iterations: Number of iterations performed in the outer loop of
      the solver.
    solver_error_state: 0 if solver converged with fine tolerance for this step
      1 if solver did not converge for this step (was above coarse tol) 2 if
      solver converged within coarse tolerance. Allowed to pass with a warning.
      Occasional error=2 has low impact on final sim state.
    inner_solver_iterations: Total number of iterations performed in the solver
      across all iterations of the solver.
    sawtooth_crash: True if a sawtooth model is active and the solver step
      corresponds to a sawtooth crash step.
  """

  outer_solver_iterations: array_typing.IntScalar
  solver_error_state: array_typing.IntScalar
  inner_solver_iterations: array_typing.IntScalar
  sawtooth_crash: array_typing.BoolScalar

  def to_output_dict(
      self, context: output_grid_context.OutputGridContext
  ) -> dict[str, output_grid_context.OutputVar]:
    """Serializes stacked SolverNumericOutputs into an OutputVar mapping."""
    return {
        output_keys.OUTER_SOLVER_ITERATIONS: context.pack(
            output_keys.OUTER_SOLVER_ITERATIONS, self.outer_solver_iterations
        ),
        output_keys.INNER_SOLVER_ITERATIONS: context.pack(
            output_keys.INNER_SOLVER_ITERATIONS, self.inner_solver_iterations
        ),
        output_keys.SOLVER_ERROR_STATE: context.pack(
            output_keys.SOLVER_ERROR_STATE, self.solver_error_state
        ),
        output_keys.SAWTOOTH_CRASH: context.pack(
            output_keys.SAWTOOTH_CRASH, self.sawtooth_crash
        ),
    }
