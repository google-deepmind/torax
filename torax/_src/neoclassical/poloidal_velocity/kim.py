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
"""Kim model for neoclassical poloidal velocity."""
from typing import Annotated, Literal
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.poloidal_velocity import base
from torax._src.neoclassical.poloidal_velocity import runtime_params as poloidal_velocity_runtime_params
from torax._src.physics import psi_calculations
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class KimModel(base.PoloidalVelocityModel):
  """Kim (1991) model for neoclassical poloidal velocity."""

  def calculate_poloidal_velocity(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.PoloidalVelocity:
    """Calculates poloidal velocity according to the Kim (1991) model."""
    B_tor_face = geometry.F_face / geometry.R_major_profile_face
    B_pol_squared_face = psi_calculations.calc_bpol_squared(
        geometry, core_profiles.psi
    )
    B_total_squared_face = B_pol_squared_face + B_tor_face**2

    poloidal_velocity_params = runtime_params.neoclassical.poloidal_velocity
    v_pol = formulas.calculate_poloidal_velocity(
        T_i=core_profiles.T_i,
        n_i=core_profiles.n_i.face_value(),
        q=core_profiles.q_face,
        Z_eff=core_profiles.Z_eff_face,
        Z_i=core_profiles.Z_i_face,
        B_tor=B_tor_face,
        B_total_squared=B_total_squared_face,
        geo=geometry,
        poloidal_velocity_multiplier=poloidal_velocity_params.poloidal_velocity_multiplier,
    )
    return base.PoloidalVelocity(
        v_pol=v_pol,
        v_pol_face=v_pol.face_value(),
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class KimModelConfig(base.PoloidalVelocityModelConfig):
  """Config for the Kim (1991) model implementation of poloidal velocity."""

  model_name: Annotated[Literal['kim'], torax_pydantic.JAX_STATIC] = 'kim'

  def build_runtime_params(
      self,
  ) -> poloidal_velocity_runtime_params.RuntimeParams:
    return poloidal_velocity_runtime_params.RuntimeParams(
        poloidal_velocity_multiplier=self.poloidal_velocity_multiplier,
    )

  def build_model(self) -> KimModel:
    return KimModel()
