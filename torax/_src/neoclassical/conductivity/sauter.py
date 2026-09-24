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

import jax
from torax._src import array_typing
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.conductivity import base
from torax._src.neoclassical.conductivity import runtime_params as conductivity_runtime_params
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.formulas import sauter as sauter_formulas
from torax._src.torax_pydantic import torax_pydantic


# TODO(b/425750357): Add neoclassical correciton flag (default to True)
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(conductivity_runtime_params.RuntimeParams):
  """Runtime params for the Sauter model."""


class SauterModel(base.ConductivityModel):
  """Sauter conductivity model."""

  def calculate_conductivity(
      self,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      analytical_cache: formulas.AnalyticalCache | None = None,
  ) -> base.Conductivity:
    """Calculates conductivity."""
    if analytical_cache is None:
      analytical_cache = formulas.compute_analytical_cache(
          geometry, core_profiles
      )
    result = _calculate_conductivity(
        Z_eff_face=core_profiles.Z_eff_face,
        T_e=core_profiles.T_e,
        f_trap=analytical_cache.f_trap,
        log_lambda_ei=analytical_cache.log_lambda_ei,
        nu_e_star=analytical_cache.nu_e_star,
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

  def build_runtime_params(self) -> RuntimeParams:
    return RuntimeParams()

  def build_model(self) -> SauterModel:
    return SauterModel()


@jax.jit
def _calculate_conductivity(
    *,
    Z_eff_face: array_typing.FloatVectorFace,
    T_e: cell_variable.CellVariable,
    f_trap: array_typing.FloatVectorFace,
    log_lambda_ei: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
) -> base.Conductivity:
  """Calculates sigma and sigma_face using the Sauter model."""
  # pylint: disable=invalid-name

  # Formulas from Sauter PoP 1999.

  # Spitzer conductivity
  NZ = 0.58 + 0.74 / (0.76 + Z_eff_face)
  sigsptz = (
      1.9012e04
      * (T_e.face_value() * 1e3) ** 1.5
      / Z_eff_face
      / NZ
      / log_lambda_ei
  )

  # Neoclassical correction to spitzer conductivity
  signeo_face = sauter_formulas.calculate_L33(
      f_trap=f_trap,
      nu_e_star=nu_e_star,
      Z_eff=Z_eff_face,
  )
  sigma_face = sigsptz * signeo_face

  sigmaneo_cell = geometry_lib.face_to_cell(sigma_face)

  return base.Conductivity(
      sigma=sigmaneo_cell,
      sigma_face=sigma_face,
  )
