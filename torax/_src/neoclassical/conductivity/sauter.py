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

import chex
import jax.numpy as jnp
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.conductivity import base
from torax._src.neoclassical.conductivity import runtime_params
from torax._src.physics import collisions


# TODO(b/425750357): Add neoclassical correciton flag (default to True)
@jax_utils.jax_dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params.DynamicRuntimeParams):
  """Dynamic runtime params for the Sauter model."""


class SauterModel(base.ConductivityModel):
  """Sauter conductivity model."""

  def calculate_conductivity(
      self,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.Conductivity:
    """Calculates conductivity."""
    result = _calculate_conductivity(
        Z_eff_face=core_profiles.Z_eff_face,
        n_e=core_profiles.n_e,
        T_e=core_profiles.T_e,
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

  def build_dynamic_params(self) -> DynamicRuntimeParams:
    return DynamicRuntimeParams()

  def build_model(self) -> SauterModel:
    return SauterModel()


@jax_utils.jit
def _calculate_conductivity(
    *,
    Z_eff_face: chex.Array,
    n_e: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
    q_face: chex.Array,
    geo: geometry_lib.Geometry,
) -> base.Conductivity:
  """Calculates sigma and sigma_face using the Sauter model."""
  # pylint: disable=invalid-name

  # Formulas from Sauter PoP 1999. Future work can include Redl PoP 2021
  # corrections.

  # # local r/R0 on face grid
  epsilon = (geo.R_out_face - geo.R_in_face) / (geo.R_out_face + geo.R_in_face)
  epseff = (
      0.67 * (1.0 - 1.4 * jnp.abs(geo.delta_face) * geo.delta_face) * epsilon
  )
  aa = (1.0 - epsilon) / (1.0 + epsilon)
  ftrap = 1.0 - jnp.sqrt(aa) * (1.0 - epseff) / (1.0 + 2.0 * jnp.sqrt(epseff))

  # Spitzer conductivity
  NZ = 0.58 + 0.74 / (0.76 + Z_eff_face)
  lambda_ei = collisions.calculate_lambda_ei(T_e.face_value(), n_e.face_value())

  sigsptz = (
      1.9012e04 * (T_e.face_value() * 1e3) ** 1.5 / Z_eff_face / NZ / lambda_ei
  )

  nu_e_star = (
      6.921e-18
      * q_face
      * geo.R_major
      * n_e.face_value()
      * Z_eff_face
      * lambda_ei
      / (
          ((T_e.face_value() * 1e3) ** 2)
          * (epsilon + constants.CONSTANTS.eps) ** 1.5
      )
  )

  # Neoclassical correction to spitzer conductivity
  ft33 = ftrap / (
      1.0
      + (0.55 - 0.1 * ftrap) * jnp.sqrt(nu_e_star)
      + 0.45 * (1.0 - ftrap) * nu_e_star / (Z_eff_face**1.5)
  )
  signeo_face = 1.0 - ft33 * (
      1.0
      + 0.36 / Z_eff_face
      - ft33 * (0.59 / Z_eff_face - 0.23 / Z_eff_face * ft33)
  )
  sigma_face = sigsptz * signeo_face

  sigmaneo_cell = geometry_lib.face_to_cell(sigma_face)

  return base.Conductivity(
      sigma=sigmaneo_cell,
      sigma_face=sigma_face,
  )
