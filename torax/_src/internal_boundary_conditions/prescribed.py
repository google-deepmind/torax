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

"""Prescribed internal boundary conditions model."""

import dataclasses

import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import base_model
from torax._src.internal_boundary_conditions import internal_boundary_conditions
from torax._src.internal_boundary_conditions import runtime_params as ibc_runtime_params
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(ibc_runtime_params.RuntimeParams):
  """Runtime parameters for prescribed internal boundary conditions."""

  T_i: array_typing.FloatVectorCell
  T_e: array_typing.FloatVectorCell
  n_e: array_typing.FloatVectorCell


@dataclasses.dataclass(frozen=True, eq=False)
class PrescribedIBCModel(base_model.InternalBoundaryConditionModel):
  """Prescribed internal boundary condition model."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_transition_state: (
          pedestal_transition_state_lib.PedestalTransitionState
      ),
  ) -> internal_boundary_conditions.InternalBoundaryConditions:
    del geo, core_profiles, pedestal_transition_state
    params = runtime_params.profile_conditions.internal_boundary_conditions
    assert isinstance(params, RuntimeParams), (
        f'Expected params to be prescribed.RuntimeParams, got {type(params)}.'
    )
    return internal_boundary_conditions.InternalBoundaryConditions(
        T_i=params.T_i,
        T_e=params.T_e,
        n_e=params.n_e,
    )
