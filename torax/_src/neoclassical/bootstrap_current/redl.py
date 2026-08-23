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
"""Redl model for bootstrap current.

Based on Redl et al., Physics of Plasmas 28, 022502 (2021).
"A new set of analytical formulae for the computation of the bootstrap
current and the neoclassical conductivity in tokamaks"
https://doi.org/10.1063/5.0012664

This model provides improved accuracy over the Sauter model, particularly
at higher collisionalities typical of tokamak edge pedestals and in the
presence of impurities.
"""

from collections.abc import Sequence
from typing import Annotated, Literal

import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.bootstrap_current import runtime_params as bootstrap_current_runtime_params
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.formulas import redl as redl_formulas
from torax._src.physics import collisions
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class RedlModel(bootstrap_current_base.BootstrapCurrentModel):
  """Redl model for bootstrap current.

  Multi-species treatment follows the same NEO drive assembly as Sauter
  (per-species ``L31`` and ``L34*α`` using each species' ``T_s`` and
  ``∇ln T_s``, ``ν_i*`` as ``nui_star_S ∝ Z_ion^4 * dens_sum`` with
  ``Z_ion``), using Redl L-coefficient fits.
  """

  def calculate_bootstrap_current(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> bootstrap_current_base.BootstrapCurrent:
    """Calculates bootstrap current using the Redl model."""
    bootstrap_params = runtime_params.neoclassical.bootstrap_current
    assert isinstance(
        bootstrap_params, bootstrap_current_runtime_params.RuntimeParams
    )
    ion_species = formulas.build_ion_species_from_core_profiles(
        core_profiles,
        subtract_fast_ions=True,
    )
    Z_eff_face = formulas.calculate_Z_eff_from_ion_species(
        core_profiles, ion_species
    )
    dens_sum_face = formulas.calculate_ion_density_sum_face(
        ion_species, placeholder=core_profiles.n_e.face_value()
    )
    # Computed outside the JIT helper so f_trap_model can use Python control flow.
    f_trap = formulas.calculate_f_trap(
        geometry,
        runtime_params.neoclassical.f_trap_model,
        q_face=core_profiles.q_face,
    )
    return _calculate_bootstrap_current(
        bootstrap_multiplier=bootstrap_params.bootstrap_multiplier,
        Z_eff_face=Z_eff_face,
        Z_i_face=core_profiles.Z_i_face,
        n_e=core_profiles.n_e,
        n_i=core_profiles.n_i,
        dens_sum_face=dens_sum_face,
        T_e=core_profiles.T_e,
        T_i=core_profiles.T_i,
        p_e=core_profiles.pressure_thermal_e,
        ion_species=ion_species,
        psi=core_profiles.psi,
        q_face=core_profiles.q_face,
        geo=geometry,
        f_trap=f_trap,
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class RedlModelConfig(bootstrap_current_base.BootstrapCurrentModelConfig):
  """Config for the Redl model implementation of bootstrap current."""

  model_name: Annotated[Literal['redl'], torax_pydantic.JAX_STATIC] = 'redl'

  def build_runtime_params(
      self,
  ) -> bootstrap_current_runtime_params.RuntimeParams:
    return bootstrap_current_runtime_params.RuntimeParams(
        bootstrap_multiplier=self.bootstrap_multiplier
    )

  def build_model(self) -> RedlModel:
    return RedlModel()


@jax.jit
def _calculate_bootstrap_current(
    *,
    bootstrap_multiplier: float,
    Z_eff_face: array_typing.FloatVectorFace,
    Z_i_face: array_typing.FloatVectorFace,
    n_e: cell_variable.CellVariable,
    n_i: cell_variable.CellVariable,
    dens_sum_face: array_typing.FloatVectorFace,
    T_e: cell_variable.CellVariable,
    T_i: cell_variable.CellVariable,
    p_e: cell_variable.CellVariable,
    ion_species: Sequence[formulas.IonSpeciesProfiles],
    psi: cell_variable.CellVariable,
    q_face: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
    f_trap: array_typing.FloatVectorFace,
) -> bootstrap_current_base.BootstrapCurrent:
  """Calculates j_parallel_bootstrap using the Redl model."""

  # Redl et al., PoP 28, 022502 (2021), with NEO multi-species drive assembly.

  log_lambda_ei = collisions.calculate_log_lambda_ei(
      T_e.face_value(), n_e.face_value()  # pyrefly: ignore[bad-argument-type]
  )
  log_lambda_ii = collisions.calculate_log_lambda_ii(
      T_i.face_value(), n_i.face_value(), Z_i_face  # pyrefly: ignore[bad-argument-type]
  )
  # n_i above is bundled main-ion density for Sauter lnΛ_ii (18e);
  # ν_i* density factor uses dens_sum (NEO), not n_i.
  nu_e_star = formulas.calculate_nu_e_star(
      q=q_face,
      geo=geo,
      n_e=n_e.face_value(),  # pyrefly: ignore[bad-argument-type]
      T_e=T_e.face_value(),  # pyrefly: ignore[bad-argument-type]
      Z_eff=Z_eff_face,
      log_lambda_ei=log_lambda_ei,
  )
  # NEO: nui_star_S ∝ Z_ion^4 * dens_sum
  # with Z = Z_ion in Sauter (18c) factor (not Z_eff).
  # dens_sum is passed as Sauter's n_i in Eq. (18c).
  nu_i_star = formulas.calculate_nu_i_star(
      q=q_face,
      geo=geo,
      n_i=dens_sum_face,
      T_i=T_i.face_value(),  # pyrefly: ignore[bad-argument-type]
      Z_i=Z_i_face,
      log_lambda_ii=log_lambda_ii,
  )

  L31 = redl_formulas.calculate_L31(f_trap, nu_e_star, Z_eff_face)
  L32 = redl_formulas.calculate_L32(f_trap, nu_e_star, Z_eff_face)
  # In Redl model, L34 is set equal to L31 (Eq. 19)
  L34 = L31
  alpha = redl_formulas.calculate_alpha(f_trap, nu_i_star, Z_eff_face)

  return formulas.calculate_analytic_bootstrap_current(
      bootstrap_multiplier=bootstrap_multiplier,
      n_e=n_e,
      T_e=T_e,
      p_e=p_e,
      ion_species=ion_species,
      psi=psi,
      geo=geo,
      L31=L31,
      L32=L32,
      L34=L34,
      alpha=alpha,
  )
