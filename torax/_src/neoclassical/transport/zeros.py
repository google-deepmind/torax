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
"""Zeros model for neoclassical transport."""
from typing import Literal

import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.transport import base
from torax._src.neoclassical.transport import runtime_params


class ZerosModel(base.NeoclassicalTransportModel):
  """Zeros model for neoclassical transport."""

  def calculate_neoclassical_transport(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.NeoclassicalTransport:
    """Calculates neoclassical transport."""
    return base.NeoclassicalTransport(
        chi_neo_i=jnp.zeros_like(geometry.rho_face),
        chi_neo_e=jnp.zeros_like(geometry.rho_face),
        D_neo_e=jnp.zeros_like(geometry.rho_face),
        V_neo_e=jnp.zeros_like(geometry.rho_face),
        V_ware_e=jnp.zeros_like(geometry.rho_face),
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class ZerosModelConfig(base.NeoclassicalTransportModelConfig):
  """Config for the Zeros model implementation of neoclassical transport."""

  model_name: Literal['zeros'] = 'zeros'

  def build_dynamic_params(self) -> runtime_params.DynamicRuntimeParams:
    return runtime_params.DynamicRuntimeParams()

  def build_model(self) -> ZerosModel:
    return ZerosModel()
