# Copyright 2026 DeepMind Technologies Limited
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
"""Zeros model for neoclassical poloidal velocity."""
from typing import Annotated, Literal
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.poloidal_velocity import base
from torax._src.neoclassical.poloidal_velocity import runtime_params as poloidal_velocity_runtime_params
from torax._src.torax_pydantic import torax_pydantic


class ZerosModel(base.PoloidalVelocityModel):
  """Zeros model for neoclassical poloidal velocity."""

  def calculate_poloidal_velocity(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      analytical_cache: formulas.AnalyticalCache | None = None,
  ) -> base.PoloidalVelocity:
    """Returns zero poloidal velocity."""
    del runtime_params, core_profiles, analytical_cache
    return base.PoloidalVelocity.zeros(geometry)

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class ZerosModelConfig(base.PoloidalVelocityModelConfig):
  """Config for the Zeros model implementation of poloidal velocity."""

  model_name: Annotated[Literal['zeros'], torax_pydantic.JAX_STATIC] = 'zeros'

  def build_runtime_params(
      self,
  ) -> poloidal_velocity_runtime_params.RuntimeParams:
    return poloidal_velocity_runtime_params.RuntimeParams(
        poloidal_velocity_multiplier=self.poloidal_velocity_multiplier,
    )

  def build_model(self) -> ZerosModel:
    return ZerosModel()
