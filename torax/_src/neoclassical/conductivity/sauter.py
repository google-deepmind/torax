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
import functools
from typing import Annotated, Literal

import jax
from torax._src import array_typing
from torax._src.core_profiles import profiles
from torax._src.core_profiles import solver_profiles
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.conductivity import base
from torax._src.neoclassical.conductivity import runtime_params as conductivity_runtime_params
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.formulas import sauter as sauter_formulas
from torax._src.torax_pydantic import torax_pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(conductivity_runtime_params.RuntimeParams):
  """Runtime params for the Sauter model."""


class SauterModel(base.ConductivityModel):
  """Sauter conductivity model."""

  def calculate_conductivity(
      self,
      runtime_params: conductivity_runtime_params.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: solver_profiles.SolverProfiles,
      neoclassical_intermediates: formulas.AnalyticalNeoclassicalIntermediates,
  ) -> profiles.Conductivity:
    """Calculates conductivity."""
    del geometry
    return _calculate_conductivity(
        conductivity_multiplier=runtime_params.conductivity_multiplier,
        neoclassical_correction=runtime_params.neoclassical_correction,
        Z_eff_face=core_profiles.ions.Z_eff_face,
        T_e=core_profiles.kinetic.T_e,
        f_trap=neoclassical_intermediates.f_trap,
        log_lambda_ei=neoclassical_intermediates.log_lambda_ei,
        nu_e_star=neoclassical_intermediates.nu_e_star,
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class SauterModelConfig(base.ConductivityModelConfig):
  """Sauter conductivity model config."""

  model_name: Annotated[Literal['sauter'], torax_pydantic.JAX_STATIC] = 'sauter'

  def build_model(self) -> SauterModel:
    return SauterModel()


@functools.partial(jax.jit, static_argnames=['neoclassical_correction'])
def _calculate_conductivity(
    *,
    conductivity_multiplier: float,
    neoclassical_correction: bool,
    Z_eff_face: array_typing.FloatVectorFace,
    T_e: cell_variable.CellVariable,
    f_trap: array_typing.FloatVectorFace,
    log_lambda_ei: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
) -> profiles.Conductivity:
  """Calculates sigma and sigma_face using the Sauter model."""
  # pylint: disable=invalid-name

  # Formulas from Sauter PoP 1999.
  sigsptz = formulas.calculate_spitzer_conductivity(
      Z_eff=Z_eff_face,
      T_e=T_e.face_value(),
      log_lambda_ei=log_lambda_ei,
  )

  sigma_face = conductivity_multiplier * sigsptz
  if neoclassical_correction:
    # Neoclassical (trapped-particle) correction to Spitzer conductivity
    sigma_face *= sauter_formulas.calculate_L33(
        f_trap=f_trap,
        nu_e_star=nu_e_star,
        Z_eff=Z_eff_face,
    )

  sigma_cell = geometry_lib.face_to_cell(sigma_face)

  return profiles.Conductivity(
      sigma=sigma_cell,
      sigma_face=sigma_face,
  )
