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
"""A basic version of the pedestal model that uses direct specification."""
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params as runtime_params_lib
from typing_extensions import override


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime params for the BgB transport model."""

  n_e_ped: array_typing.FloatScalar
  T_i_ped: array_typing.FloatScalar
  T_e_ped: array_typing.FloatScalar
  rho_norm_ped_top: array_typing.FloatScalar
  n_e_ped_is_fGW: array_typing.BoolScalar


class SetTemperatureDensityPedestalModel(pedestal_model.PedestalModel):
  """A basic version of the pedestal model that uses direct specification."""

  def __init__(
      self,
  ):
    super().__init__()
    self._frozen = True

  @override
  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pedestal_model.PedestalModelOutput:
    pedestal_params = dynamic_runtime_params_slice.pedestal
    assert isinstance(pedestal_params, DynamicRuntimeParams)
    nGW = (
        dynamic_runtime_params_slice.profile_conditions.Ip
        / 1e6  # Convert to MA.
        / (jnp.pi * geo.a_minor**2)
        * 1e20
    )
    # Calculate n_e_ped in m^-3.
    n_e_ped = jnp.where(
        pedestal_params.n_e_ped_is_fGW,
        pedestal_params.n_e_ped * nGW,
        pedestal_params.n_e_ped,
    )
    return pedestal_model.PedestalModelOutput(
        n_e_ped=n_e_ped,
        T_i_ped=pedestal_params.T_i_ped,
        T_e_ped=pedestal_params.T_e_ped,
        rho_norm_ped_top=pedestal_params.rho_norm_ped_top,
        rho_norm_ped_top_idx=jnp.abs(
            geo.rho_norm - pedestal_params.rho_norm_ped_top
        ).argmin(),
    )

  def __hash__(self) -> int:
    return hash('SetTemperatureDensityPedestalModel')

  def __eq__(self, other) -> bool:
    return isinstance(other, SetTemperatureDensityPedestalModel)
