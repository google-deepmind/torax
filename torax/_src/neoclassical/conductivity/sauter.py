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
"""Sauter conductivity model."""

from typing import Literal

from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.conductivity import base
from torax._src.sources import bootstrap_current_source


class SauterModel(base.ConductivityModel):
  """Sauter conductivity model."""

  def calculate_conductivity(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.Conductivity:
    """Calculates conductivity."""
    # Bootstrap multiplier is not used in calculating conductivity.
    # TODO(b/314287587): Refactor calc_sauter_model into two functions
    result = bootstrap_current_source.calc_sauter_model(
        bootstrap_multiplier=1.0,
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
    return base.Conductivity(
        sigma=result.sigma,
        sigma_face=result.sigma_face,
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class SauterModelConfig(base.ConductivityModelConfig):
  """Sauter conductivity model config."""
  model_name: Literal['sauter'] = 'sauter'

  def build_model(self) -> SauterModel:
    return SauterModel()
