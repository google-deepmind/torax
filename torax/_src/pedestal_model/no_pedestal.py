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
    return pedestal_model.PedestalModelOutput(
        rho_norm_ped_top=jnp.inf,
        T_i_ped=0.0,
        T_e_ped=0.0,
        n_e_ped=0.0,
        rho_norm_ped_top_idx=geo.torax_mesh.nx,
    )
