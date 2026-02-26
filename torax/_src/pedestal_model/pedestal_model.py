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

"""The PedestalModel abstract base class.

The pedestal model calculates quantities relevant to the pedestal.
"""

import abc
import dataclasses

import jax
import jax.numpy as jnp
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.sources import source_profiles as source_profiles_lib

# pylint: disable=invalid-name
# Using physics notation naming convention


@dataclasses.dataclass(frozen=True, eq=False)
class PedestalModel(static_dataclass.StaticDataclass, abc.ABC):
  """Calculates temperature and density of the pedestal."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
  ) -> pedestal_model_output.PedestalModelOutput:
    return jax.lax.cond(
        runtime_params.pedestal.set_pedestal,
        lambda: self._call_implementation(runtime_params, geo, core_profiles),
        lambda: pedestal_model_output.PedestalModelOutput(
            rho_norm_ped_top=jnp.inf,
            rho_norm_ped_top_idx=geo.torax_mesh.nx,
            T_i_ped=0.0,
            T_e_ped=0.0,
            n_e_ped=0.0,
        ),
    )

  @abc.abstractmethod
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pedestal_model_output.PedestalModelOutput:
    """Calculate the pedestal properties."""
