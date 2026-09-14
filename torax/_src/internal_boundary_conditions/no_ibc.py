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

"""Internal boundary condition model for no internal boundary conditions."""

import dataclasses

import jax
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import base_model
from torax._src.internal_boundary_conditions import internal_boundary_conditions
from torax._src.internal_boundary_conditions import runtime_params as ibc_runtime_params
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(ibc_runtime_params.RuntimeParams):
  """Runtime parameters when no IBC is active."""

  pass


@dataclasses.dataclass(frozen=True, eq=False)
class NoIBCModel(base_model.InternalBoundaryConditionModel):
  """A model for when there are no internal boundary conditions."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_transition_state: (
          pedestal_transition_state_lib.PedestalTransitionState
      ),
  ) -> internal_boundary_conditions.InternalBoundaryConditions:
    del runtime_params, core_profiles, pedestal_transition_state
    return internal_boundary_conditions.InternalBoundaryConditions.empty(geo)
