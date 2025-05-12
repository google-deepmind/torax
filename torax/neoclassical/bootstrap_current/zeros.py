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
from typing import Literal

import jax.numpy as jnp
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry as geometry_lib
from torax.neoclassical.bootstrap_current import base
from torax.neoclassical.bootstrap_current import runtime_params


class ZerosModel(base.BootstrapCurrentModel):
  """Zeros model for bootstrap current."""

  def calculate_bootstrap_current(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.BootstrapCurrent:
    """Calculates bootstrap current."""
    return base.BootstrapCurrent(
        j_bootstrap=jnp.zeros_like(geometry.rho),
        j_bootstrap_face=jnp.zeros_like(geometry.rho_face),
    )


class ZerosModelConfig(base.BootstrapCurrentModelConfig):
  """Config for the Zeros model implementation of bootstrap current."""

  model_name: Literal['zeros'] = 'zeros'

  def build_dynamic_params(self) -> runtime_params.DynamicRuntimeParams:
    return runtime_params.DynamicRuntimeParams()

  def build_model(self) -> ZerosModel:
    return ZerosModel()
