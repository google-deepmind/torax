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
"""Redl conductivity model.

Based on Redl et al., Physics of Plasmas 28, 022502 (2021).
"A new set of analytical formulae for the computation of the bootstrap
current and the neoclassical conductivity in tokamaks"
https://doi.org/10.1063/5.0012664
"""

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
from torax._src.neoclassical.formulas import redl as redl_formulas
from torax._src.torax_pydantic import torax_pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(conductivity_runtime_params.RuntimeParams):
  """Runtime params for the Redl conductivity model."""


class RedlModel(base.ConductivityModel):
  """Redl conductivity model."""

  def calculate_conductivity(
      self,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      analytical_cache: formulas.AnalyticalCache | None = None,
  ) -> base.Conductivity:
    """Calculates conductivity using the Redl model."""
    if analytical_cache is None:
      analytical_cache = formulas.compute_analytical_cache(
          geometry, core_profiles
      )
    return _calculate_conductivity(
        Z_eff_face=core_profiles.Z_eff_face,
        T_e=core_profiles.T_e,
        f_trap=analytical_cache.f_trap,
        log_lambda_ei=analytical_cache.log_lambda_ei,
        nu_e_star=analytical_cache.nu_e_star,
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class RedlModelConfig(base.ConductivityModelConfig):
  """Redl conductivity model config."""

  model_name: Annotated[Literal['redl'], torax_pydantic.JAX_STATIC] = 'redl'

  def build_runtime_params(self) -> RuntimeParams:
    return RuntimeParams()

  def build_model(self) -> RedlModel:
    return RedlModel()


@jax.jit
def _calculate_conductivity(
    *,
    Z_eff_face: array_typing.FloatVectorFace,
    T_e: cell_variable.CellVariable,
    f_trap: array_typing.FloatVectorFace,
    log_lambda_ei: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
) -> base.Conductivity:
  """Calculates sigma and sigma_face using the Redl model."""
  # pylint: disable=invalid-name

  # Spitzer conductivity
  NZ = 0.58 + 0.74 / (0.76 + Z_eff_face)
  sigsptz = (
      1.9012e04
      * (T_e.face_value() * 1e3) ** 1.5
      / Z_eff_face
      / NZ
      / log_lambda_ei
  )

  # Neoclassical correction to Spitzer conductivity (Redl PoP 2021 Eqs. 17-18)
  signeo_face = redl_formulas.calculate_L33(
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
