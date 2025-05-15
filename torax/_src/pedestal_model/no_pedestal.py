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
from jax import numpy as jnp
from torax import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model


class NoPedestal(pedestal_model.PedestalModel):
  """A pedestal model for when there is no pedestal.

  This is a placeholder pedestal model that is used when there is no pedestal.
  It returns infinite pedestal location and zero temperature and density.
  Assuming set_pedestal is set to False properly this will not be used, but
  this is a safe fallback in case set_pedestal is not set properly and is needed
  for the jax cond to work.
  """

  def __init__(self):
    super().__init__()
    self._frozen = True

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles
  ) -> pedestal_model.PedestalModelOutput:
    return pedestal_model.PedestalModelOutput(
        rho_norm_ped_top=jnp.inf, T_i_ped=0.0, T_e_ped=0.0, n_e_ped=0.0,
        rho_norm_ped_top_idx=geo.torax_mesh.nx,
    )

  def __hash__(self):
    return hash('NoPedestal')

  def __eq__(self, other) -> bool:
    return isinstance(other, NoPedestal)
