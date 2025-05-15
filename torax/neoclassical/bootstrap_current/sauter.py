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
"""Sauter model for bootstrap current."""

from typing import Literal

import chex
from torax import state
from torax._src.config import runtime_params_slice
from torax.geometry import geometry as geometry_lib
from torax.neoclassical.bootstrap_current import base
from torax.neoclassical.bootstrap_current import runtime_params
from torax.sources import bootstrap_current_source


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params.DynamicRuntimeParams):
  """Dynamic runtime params for the Sauter model."""

  bootstrap_multiplier: float


class SauterModel(base.BootstrapCurrentModel):
  """Sauter model for bootstrap current."""

  def calculate_bootstrap_current(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.BootstrapCurrent:
    """Calculates bootstrap current."""
    bootstrap_params = (
        dynamic_runtime_params_slice.neoclassical.bootstrap_current
    )
    assert isinstance(bootstrap_params, DynamicRuntimeParams)
    result = bootstrap_current_source.calc_sauter_model(
        bootstrap_multiplier=bootstrap_params.bootstrap_multiplier,
        density_reference=dynamic_runtime_params_slice.numerics.density_reference,
        Z_eff_face=dynamic_runtime_params_slice.plasma_composition.Z_eff_face,
        Z_i_face=core_profiles.Z_i_face,
        n_e=core_profiles.n_e,
        n_i=core_profiles.n_i,
        T_e=core_profiles.T_e,
        T_i=core_profiles.T_i,
        psi=core_profiles.psi,
        q_face=core_profiles.q_face,
        geo=geometry,
    )
    return base.BootstrapCurrent(
        j_bootstrap=result.j_bootstrap,
        j_bootstrap_face=result.j_bootstrap_face,
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class SauterModelConfig(base.BootstrapCurrentModelConfig):
  """Config for the Sauter model implementation of bootstrap current.

  Attributes:
    bootstrap_multiplier: Multiplication factor for bootstrap current.
  """

  model_name: Literal['sauter'] = 'sauter'
  bootstrap_multiplier: float = 1.0

  def build_dynamic_params(self) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(bootstrap_multiplier=self.bootstrap_multiplier)

  def build_model(self) -> SauterModel:
    return SauterModel()
