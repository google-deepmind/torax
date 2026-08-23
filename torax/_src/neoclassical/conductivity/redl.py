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
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.conductivity import base
from torax._src.neoclassical.conductivity import runtime_params as conductivity_runtime_params
from torax._src.neoclassical.formulas import formulas
from torax._src.physics import collisions
from torax._src.torax_pydantic import torax_pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(conductivity_runtime_params.RuntimeParams):
  """Runtime params for the Redl model."""


class RedlModel(base.ConductivityModel):
  """Redl conductivity model.

  Spitzer conductivity, ``ν_e*``, and the neoclassical correction use
  ``Z_eff = Σ_s n_s Z_s² / n_e`` from per-species thermal ion/impurity
  densities (NEO multi-species ``zeff``), subtracting fast ions like
  bootstrap. The trapped fraction uses
  ``runtime_params.neoclassical.f_trap_model``.
  """

  def calculate_conductivity(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.Conductivity:
    """Calculates conductivity."""
    ion_species = formulas.build_ion_species_from_core_profiles(
        core_profiles,
        subtract_fast_ions=True,
    )
    Z_eff_face = formulas.calculate_Z_eff_from_ion_species(
        core_profiles, ion_species
    )
    # Computed outside the JIT helper so f_trap_model can use Python control flow.
    f_trap = formulas.calculate_f_trap(
        geometry,
        runtime_params.neoclassical.f_trap_model,
        q_face=core_profiles.q_face,
    )
    result = _calculate_conductivity(
        Z_eff_face=Z_eff_face,
        n_e=core_profiles.n_e,
        T_e=core_profiles.T_e,
        q_face=core_profiles.q_face,
        geo=geometry,
        f_trap=f_trap,
    )
    return base.Conductivity(
        sigma=result.sigma,
        sigma_face=result.sigma_face,
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
    n_e: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
    q_face: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
    f_trap: array_typing.FloatVectorFace,
) -> base.Conductivity:
  """Calculates sigma and sigma_face using the Redl model."""
  # pylint: disable=invalid-name

  # Spitzer conductivity (same normalization as Sauter PoP 1999 / TORAX).
  NZ = 0.58 + 0.74 / (0.76 + Z_eff_face)
  log_lambda_ei = collisions.calculate_log_lambda_ei(
      T_e.face_value(), n_e.face_value()  # pyrefly: ignore[bad-argument-type]
  )

  sigma_spitzer = (
      1.9012e04
      * (T_e.face_value() * 1e3) ** 1.5
      / Z_eff_face
      / NZ
      / log_lambda_ei
  )

  nu_e_star_face = formulas.calculate_nu_e_star(
      q=q_face,
      geo=geo,
      n_e=n_e.face_value(),  # pyrefly: ignore[bad-argument-type]
      T_e=T_e.face_value(),  # pyrefly: ignore[bad-argument-type]
      Z_eff=Z_eff_face,
      log_lambda_ei=log_lambda_ei,
  )

  # Neoclassical correction: Redl PoP 2021 conductivity fit (X33).
  X33 = f_trap / (
      1.0
      + 0.25
      * (1.0 - 0.7 * f_trap)
      * jnp.sqrt(nu_e_star_face)
      * (1.0 + 0.45 * jnp.sqrt(Z_eff_face - 1.0))
      + (1.0 - 0.41 * f_trap)
      * 0.61
      * nu_e_star_face
      / jnp.sqrt(Z_eff_face)
  )
  signeo_face = (
      1.0
      - (1.0 + 0.21 / Z_eff_face) * X33
      + (0.54 / Z_eff_face) * X33**2
      - (0.33 / Z_eff_face) * X33**3
  )
  sigma_face = sigma_spitzer * signeo_face

  return base.Conductivity(
      sigma=geometry_lib.face_to_cell(sigma_face),
      sigma_face=sigma_face,
  )
