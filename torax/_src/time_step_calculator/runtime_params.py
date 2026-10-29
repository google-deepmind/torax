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

"""Runtime parameters for the time step calculator."""

import dataclasses

import jax
from torax._src import array_typing


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Runtime parameters for the time step calculator."""

  t_initial: float
  t_final: float
  tolerance: float
  min_dt: float
  dt_backtracking_factor: float
  exact_t_final: bool = dataclasses.field(metadata={'static': True})
  allow_backtracking: bool = dataclasses.field(metadata={'static': True})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ChiRuntimeParams(RuntimeParams):
  """Runtime parameters for the chi time step calculator."""

  max_dt: float
  chi_timestep_prefactor: float


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FixedRuntimeParams(RuntimeParams):
  """Runtime parameters for the fixed time step calculator."""

  dt: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FromPreviousRuntimeParams(RuntimeParams):
  """Runtime parameters for the from_previous_dt time step calculator."""

  initial_dt: float
  max_dt: float

