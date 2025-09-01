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
"""Zeros model for bootstrap current."""
from typing import Annotated, Literal

import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.bootstrap_current import base
from torax._src.neoclassical.bootstrap_current import runtime_params as bootstrap_runtime_params
from torax._src.torax_pydantic import torax_pydantic


class ZerosModel(base.BootstrapCurrentModel):
  """Zeros model for bootstrap current."""

  def calculate_bootstrap_current(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.BootstrapCurrent:
    """Calculates bootstrap current."""
    return base.BootstrapCurrent(
        j_bootstrap=jnp.zeros_like(geometry.rho),
        j_bootstrap_face=jnp.zeros_like(geometry.rho_face),
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class ZerosModelConfig(base.BootstrapCurrentModelConfig):
  """Config for the Zeros model implementation of bootstrap current."""

  model_name: Annotated[Literal['zeros'], torax_pydantic.JAX_STATIC] = 'zeros'

  def build_runtime_params(self) -> bootstrap_runtime_params.RuntimeParams:
    return bootstrap_runtime_params.RuntimeParams()

  def build_model(self) -> ZerosModel:
    return ZerosModel()
