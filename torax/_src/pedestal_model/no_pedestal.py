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
"""A pedestal model for when there is no pedestal."""
import dataclasses
from jax import numpy as jnp
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib


@dataclasses.dataclass(frozen=True, eq=False)
class NoPedestal(pedestal_model.PedestalModel):
  """A pedestal model for when there is no pedestal.

  This is a placeholder pedestal model that is used when there is no pedestal.
  It returns infinite pedestal location and zero temperature and density.
  Assuming set_pedestal is set to False properly this will not be used, but
  this is a safe fallback in case set_pedestal is not set properly and is needed
  for the jax cond to work.
  """

  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pedestal_model.PedestalModelOutput:
    if (
        runtime_params.pedestal.mode
        == pedestal_runtime_params_lib.Mode.ADAPTIVE_SOURCE
    ):
      return pedestal_model.AdaptiveSourcePedestalModelOutput(
          rho_norm_ped_top=jnp.inf,
          rho_norm_ped_top_nearest_cell_idx=geo.torax_mesh.nx,
          rho_norm_ped_top_nearest_face_idx=geo.torax_mesh.nx + 1,
          T_i_ped=0.0,
          T_e_ped=0.0,
          n_e_ped=0.0,
      )
    elif (
        runtime_params.pedestal.mode
        == pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT
    ):
      return pedestal_model.AdaptiveTransportPedestalModelOutput(
          rho_norm_ped_top=jnp.inf,
          rho_norm_ped_top_nearest_cell_idx=geo.torax_mesh.nx,
          rho_norm_ped_top_nearest_face_idx=geo.torax_mesh.nx + 1,
          chi_e_multiplier=1.0,
          chi_i_multiplier=1.0,
          D_e_multiplier=1.0,
          v_e_multiplier=1.0,
      )
    else:
      raise ValueError(
          f'Unsupported pedestal model mode: {runtime_params.pedestal.mode}'
      )
