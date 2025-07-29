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
import dataclasses
from typing import Annotated, Literal

import chex
import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import formulas
from torax._src.neoclassical.conductivity import base
from torax._src.neoclassical.conductivity import runtime_params
from torax._src.physics import collisions
from torax._src.torax_pydantic import torax_pydantic


# TODO(b/425750357): Add neoclassical correciton flag (default to True)
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
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

  model_name: Annotated[Literal['sauter'], torax_pydantic.JAX_STATIC] = 'sauter'

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

  # Formulas from Sauter PoP 1999.

  # Effective trapped particle fraction
  f_trap = formulas.calculate_f_trap(geo)

  # Spitzer conductivity
  NZ = 0.58 + 0.74 / (0.76 + Z_eff_face)
  log_lambda_ei = collisions.calculate_log_lambda_ei(
      T_e.face_value(), n_e.face_value()
  )

  sigsptz = (
      1.9012e04
      * (T_e.face_value() * 1e3) ** 1.5
      / Z_eff_face
      / NZ
      / log_lambda_ei
  )

  nu_e_star_face = formulas.calculate_nu_e_star(
      q=q_face,
      geo=geo,
      n_e=n_e.face_value(),
      T_e=T_e.face_value(),
      Z_eff=Z_eff_face,
      log_lambda_ei=log_lambda_ei,
  )

  # Neoclassical correction to spitzer conductivity
  ft33 = f_trap / (
      1.0
      + (0.55 - 0.1 * f_trap) * jnp.sqrt(nu_e_star_face)
      + 0.45 * (1.0 - f_trap) * nu_e_star_face / (Z_eff_face**1.5)
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
