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
"""Redl model for bootstrap current.

Based on Redl et al., Physics of Plasmas 28, 022502 (2021).
"A new set of analytical formulae for the computation of the bootstrap
current and the neoclassical conductivity in tokamaks"
https://doi.org/10.1063/5.0012664

This model provides improved accuracy over the Sauter model, particularly
at higher collisionalities typical of tokamak edge pedestals and in the
presence of impurities.
"""
import dataclasses
from typing import Annotated, Literal

import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.formulas import common as common_formulas
from torax._src.neoclassical.formulas import redl as redl_formulas
from torax._src.neoclassical import bootstrap_current
from torax._src.physics import collisions
from torax._src.torax_pydantic import torax_pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(bootstrap_current.runtime_params.RuntimeParams):
  """Runtime params for the Redl model."""


class RedlModel(bootstrap_current.base.BootstrapCurrentModel):
  """Redl model for bootstrap current."""

  def calculate_bootstrap_current(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> bootstrap_current.base.BootstrapCurrent:
    """Calculates bootstrap current using the Redl model."""
    bootstrap_params = runtime_params.neoclassical.bootstrap_current
    assert isinstance(bootstrap_params, RuntimeParams)
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


class RedlModelConfig(bootstrap_current.base.BootstrapCurrentModelConfig):
  """Config for the Redl model implementation of bootstrap current."""

  model_name: Annotated[Literal['redl'], torax_pydantic.JAX_STATIC] = 'redl'

  def build_runtime_params(self) -> RuntimeParams:
    return RuntimeParams(bootstrap_multiplier=self.bootstrap_multiplier)

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
    T_e: cell_variable.CellVariable,
    T_i: cell_variable.CellVariable,
    p_e: cell_variable.CellVariable,
    p_i: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    q_face: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
) -> bootstrap_current.base.BootstrapCurrent:
  """Calculates j_parallel_bootstrap using the Redl model.

  Implements the analytical formulae from Redl et al., PoP 28, 022502 (2021).
  These formulae were derived by fitting the NEO code results using the same
  methodology as Sauter, but with improved accuracy particularly at high
  collisionality and for multi-species plasmas.
  """
  # pylint: disable=invalid-name

  # Effective trapped particle fraction
  f_trap = common_formulas.calculate_f_trap(geo)

  # Collision frequencies
  log_lambda_ei = collisions.calculate_log_lambda_ei(
      T_e.face_value(), n_e.face_value()
  )
  log_lambda_ii = collisions.calculate_log_lambda_ii(
      T_i.face_value(), n_i.face_value(), Z_i_face
  )
  nu_e_star = common_formulas.calculate_nu_e_star(
      q=q_face,
      geo=geo,
      n_e=n_e.face_value(),
      T_e=T_e.face_value(),
      Z_eff=Z_eff_face,
      log_lambda_ei=log_lambda_ei,
  )
  nu_i_star = common_formulas.calculate_nu_i_star(
      q=q_face,
      geo=geo,
      n_i=n_i.face_value(),
      T_i=T_i.face_value(),
      Z_eff=Z_eff_face,
      log_lambda_ii=log_lambda_ii,
  )

  # Calculate terms needed for bootstrap current using Redl formulae
  L31 = redl_formulas.calculate_L31(f_trap, nu_e_star, Z_eff_face)
  L32 = redl_formulas.calculate_L32(f_trap, nu_e_star, Z_eff_face)
  # In Redl model, L34 is set equal to L31 (Eq. 19)
  L34 = L31
  alpha = redl_formulas.calculate_alpha(
      f_trap, nu_i_star, Z_eff_face
  )

  return common_formulas.calculate_analytic_bootstrap_current(
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
