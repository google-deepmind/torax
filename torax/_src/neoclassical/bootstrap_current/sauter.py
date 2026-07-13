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
"""Sauter model for bootstrap current."""

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
from torax._src.neoclassical.formulas import sauter as sauter_formulas
from torax._src.physics import collisions
from torax._src.torax_pydantic import torax_pydantic


class SauterModel(bootstrap_current_base.BootstrapCurrentModel):
  """Sauter model for bootstrap current."""

  def calculate_bootstrap_current(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> bootstrap_current_base.BootstrapCurrent:
    """Calculates bootstrap current according to the Sauter model."""
    bootstrap_params = runtime_params.neoclassical.bootstrap_current
    assert isinstance(
        bootstrap_params, bootstrap_current_runtime_params.RuntimeParams
    )
    return _calculate_bootstrap_current(
        bootstrap_multiplier=bootstrap_params.bootstrap_multiplier,
        Z_eff_face=core_profiles.Z_eff_face,
        Z_i_face=core_profiles.Z_i_face,
        n_e=core_profiles.n_e,
        n_i=core_profiles.n_i,
        T_e=core_profiles.T_e,
        T_i=core_profiles.T_i,
        p_e=core_profiles.pressure_thermal_e,
        p_i=core_profiles.pressure_thermal_i,
        psi=core_profiles.psi,
        q_face=core_profiles.q_face,
        geo=geometry,
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class SauterModelConfig(bootstrap_current_base.BootstrapCurrentModelConfig):
  """Config for the Sauter model implementation of bootstrap current.

  Attributes:
    bootstrap_multiplier: Multiplication factor for bootstrap current.
  """

  model_name: Annotated[Literal['sauter'], torax_pydantic.JAX_STATIC] = 'sauter'

  def build_runtime_params(
      self,
  ) -> bootstrap_current_runtime_params.RuntimeParams:
    return bootstrap_current_runtime_params.RuntimeParams(
        bootstrap_multiplier=self.bootstrap_multiplier
    )

  def build_model(self) -> SauterModel:
    return SauterModel()


@jax.jit
def _calculate_bootstrap_current(
    *,
    bootstrap_multiplier: float,
    Z_eff_face: array_typing.FloatVectorFace,
    Z_i_face: array_typing.FloatVectorFace,
    n_e: cell_variable.CellVariable,
    n_i: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
    T_i: cell_variable.CellVariable,
    p_e: cell_variable.CellVariable,
    p_i: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    q_face: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
) -> bootstrap_current_base.BootstrapCurrent:
  """Calculates j_parallel_bootstrap using the Sauter model."""
  # pylint: disable=invalid-name

  # Formulas from Sauter PoP 1999. Future work can include Redl PoP 2021
  # corrections.

  # Effective trapped particle fraction
  f_trap = geo.trapped_fraction_face

  # Spitzer conductivity
  log_lambda_ei = collisions.calculate_log_lambda_ei(
      T_e.face_value(), n_e.face_value()  # pyrefly: ignore[bad-argument-type]
  )
  log_lambda_ii = collisions.calculate_log_lambda_ii(
      T_i.face_value(), n_i.face_value(), Z_i_face  # pyrefly: ignore[bad-argument-type]
  )
  nu_e_star = formulas.calculate_nu_e_star(
      q=q_face,
      geo=geo,
      n_e=n_e.face_value(),  # pyrefly: ignore[bad-argument-type]
      T_e=T_e.face_value(),  # pyrefly: ignore[bad-argument-type]
      Z_eff=Z_eff_face,
      log_lambda_ei=log_lambda_ei,
  )
  nu_i_star = formulas.calculate_nu_i_star(
      q=q_face,
      geo=geo,
      n_i=n_i.face_value(),  # pyrefly: ignore[bad-argument-type]
      T_i=T_i.face_value(),  # pyrefly: ignore[bad-argument-type]
      Z_eff=Z_eff_face,
      log_lambda_ii=log_lambda_ii,
  )

  # Terms for analytical fit
  L31 = sauter_formulas.calculate_L31(
      f_trap, nu_e_star, Z_eff_face
  )
  L32 = sauter_formulas.calculate_L32(
      f_trap, nu_e_star, Z_eff_face
  )
  L34 = sauter_formulas.calculate_L34(
      f_trap, nu_e_star, Z_eff_face
  )
  alpha = sauter_formulas.calculate_alpha(f_trap, nu_i_star)

  return formulas.calculate_analytic_bootstrap_current(
      bootstrap_multiplier=bootstrap_multiplier,
      n_e=n_e,
      n_i=n_i,
      T_e=T_e,
      T_i=T_i,
      p_e=p_e,
      p_i=p_i,
      psi=psi,
      geo=geo,
      L31=L31,
      L32=L32,
      L34=L34,
      alpha=alpha,
  )
