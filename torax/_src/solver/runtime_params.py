# Copyright 2024 DeepMind Technologies Limited
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
"""Runtime params for the solver."""

import dataclasses

import jax
from torax._src import tridiagonal


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Input params for the solver which can be used as compiled args."""

  theta_implicit: float = jax.tree.static()
  use_predictor_corrector: bool = jax.tree.static()
  n_corrector_steps: int = jax.tree.static()
  use_pereverzev: bool = jax.tree.static()
  implicit_solver_type: tridiagonal.SolverType = jax.tree.static()
  chi_pereverzev: float
  D_pereverzev: float  # pylint: disable=invalid-name
  fixed_point_atol: float
  fixed_point_rtol: float
  fixed_point_termination_criterion: str = jax.tree.static()
  fixed_point_sufficient_decrease: float
  fixed_point_use_backtracking: bool = jax.tree.static()
  delta_reduction_factor: float
