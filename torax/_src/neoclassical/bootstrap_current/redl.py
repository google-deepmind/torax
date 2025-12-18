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
import jax.numpy as jnp

from torax._src import array_typing, state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import formulas
from torax._src.neoclassical.bootstrap_current import base
from torax._src.neoclassical.bootstrap_current import (
  runtime_params as bootstrap_runtime_params,
)
from torax._src.physics import collisions
from torax._src.torax_pydantic import torax_pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(bootstrap_runtime_params.RuntimeParams):
  """Runtime params for the Redl model."""


class RedlModel(base.BootstrapCurrentModel):
  """Redl model for bootstrap current."""

  def calculate_bootstrap_current(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.BootstrapCurrent:
    """Calculates bootstrap current using the Redl model."""
    bootstrap_params = runtime_params.neoclassical.bootstrap_current
    assert isinstance(bootstrap_params, RuntimeParams)
    result = _calculate_bootstrap_current(
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
    return base.BootstrapCurrent(
        j_parallel_bootstrap=result.j_parallel_bootstrap,
        j_parallel_bootstrap_face=result.j_parallel_bootstrap_face,
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class RedlModelConfig(base.BootstrapCurrentModelConfig):
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
) -> base.BootstrapCurrent:
  """Calculates j_parallel_bootstrap using the Redl model.

  Implements the analytical formulae from Redl et al., PoP 28, 022502 (2021).
  These formulae were derived by fitting the NEO code results using the same
  methodology as Sauter, but with improved accuracy particularly at high
  collisionality and for multi-species plasmas.
  """
  # pylint: disable=invalid-name

  # Effective trapped particle fraction
  f_trap = formulas.calculate_f_trap(geo)

  # Collision frequencies
  log_lambda_ei = collisions.calculate_log_lambda_ei(
      T_e.face_value(), n_e.face_value()
  )
  log_lambda_ii = collisions.calculate_log_lambda_ii(
      T_i.face_value(), n_i.face_value(), Z_i_face
  )
  nu_e_star = formulas.calculate_nu_e_star(
      q=q_face,
      geo=geo,
      n_e=n_e.face_value(),
      T_e=T_e.face_value(),
      Z_eff=Z_eff_face,
      log_lambda_ei=log_lambda_ei,
  )
  nu_i_star = formulas.calculate_nu_i_star(
      q=q_face,
      geo=geo,
      n_i=n_i.face_value(),
      T_i=T_i.face_value(),
      Z_eff=Z_eff_face,
      log_lambda_ii=log_lambda_ii,
  )

  # Calculate terms needed for bootstrap current using Redl formulae
  L31 = _calculate_L31(f_trap, nu_e_star, Z_eff_face)
  L32 = _calculate_L32(f_trap, nu_e_star, Z_eff_face)
  # In Redl model, L34 is set equal to L31 (Eq. 19)
  L34 = L31
  alpha = _calculate_alpha(f_trap, nu_i_star, Z_eff_face)

  # Calculate bootstrap current
  prefactor = -geo.F_face * bootstrap_multiplier * 2 * jnp.pi / geo.B_0

  pe = p_e.face_value()
  pi = p_i.face_value()

  dpsi_drnorm = psi.face_grad()
  dlnne_drnorm = n_e.face_grad() / n_e.face_value()
  dlnni_drnorm = n_i.face_grad() / n_i.face_value()
  dlnte_drnorm = T_e.face_grad() / T_e.face_value()
  dlnti_drnorm = T_i.face_grad() / T_i.face_value()

  global_coeff = prefactor[1:] / dpsi_drnorm[1:]
  global_coeff = jnp.concatenate([jnp.zeros(1), global_coeff])

  necoeff = L31 * pe
  nicoeff = L31 * pi
  tecoeff = (L31 + L32) * pe
  ticoeff = (L31 + alpha * L34) * pi

  j_parallel_bootstrap_face = global_coeff * (
      necoeff * dlnne_drnorm
      + nicoeff * dlnni_drnorm
      + tecoeff * dlnte_drnorm
      + ticoeff * dlnti_drnorm
  )
  j_parallel_bootstrap = geometry_lib.face_to_cell(j_parallel_bootstrap_face)

  return base.BootstrapCurrent(
      j_parallel_bootstrap=j_parallel_bootstrap,
      j_parallel_bootstrap_face=j_parallel_bootstrap_face,
  )


def _calculate_L31(
    f_trap: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the L31 coefficient: Redl PoP 2021 Eqs. 10-11."""
  # Equation 11
  f_eff_31 = f_trap / (
      1.0
      + (0.67 * (1 - 0.7 * f_trap) * jnp.sqrt(nu_e_star))
      / (0.56 + 0.44 * Z_eff)
      + ((0.52 + 0.086 * jnp.sqrt(nu_e_star)) * (1 + 0.87 * f_trap) * nu_e_star)
      / (1 + 1.13 * jnp.sqrt(Z_eff - 1))
  )

  # Equation 10
  X31 = f_eff_31
  denom = Z_eff**1.2 - 0.71
  return (
      (1.0 + 0.15 / denom) * X31
      - (0.22 / denom) * X31**2
      + (0.01 / denom) * X31**3
      + (0.06 / denom) * X31**4
  )


def _calculate_L32(
    f_trap: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the L32 coefficient: Redl PoP 2021 Eqs. 12-16.

  L32 is split into electron-electron (F32_ee) and electron-ion (F32_ei)
  contributions.
  """
  # Equation 14
  f_eff_32_ee = f_trap / (
      1.0
      + (
        (0.23 * (1 - 0.96 * f_trap) * jnp.sqrt(nu_e_star))
        / Z_eff**0.5
      )
      + (
        (0.13 * (1 - 0.38 * f_trap) * nu_e_star)
        / Z_eff**2
      )
      * (
        jnp.sqrt(1 + 2 * jnp.sqrt(Z_eff - 1))
        + f_trap**2 * jnp.sqrt((0.075 + 0.25 * (Z_eff - 1) ** 2) * nu_e_star)
      )
  )

  # Equation 13
  X32_e = f_eff_32_ee
  F32_ee = (
      (
        (0.1 + 0.6 * Z_eff)
        / (Z_eff * (0.77 + 0.63 * (1 + (Z_eff - 1) ** 1.1)))
      )
      * (X32_e - X32_e**4)
      + (
        0.7
        / (1 + 0.2 * Z_eff)
      )
      * (X32_e**2 - X32_e**4 - 1.2 * (X32_e**3 - X32_e**4))
      + (
        1.3
        / (1 + 0.5 * Z_eff)
      ) * X32_e**4
  )

  # Equation 16
  f_eff_32_ei = f_trap / (
      1.0
      + (
        (0.87 * (1 + 0.39 * f_trap) * jnp.sqrt(nu_e_star))
        / (1 + 2.95 * (Z_eff - 1) ** 2)
      )
      + (1.53 * (1 - 0.37 * f_trap) * nu_e_star)
      * (2 + 0.375 * (Z_eff - 1))
  )

  # Equation 15
  X32_ei = f_eff_32_ei
  F32_ei = (
      (-(0.4 + 1.93 * Z_eff) / (Z_eff * (0.8 + 0.6 * Z_eff)))
      * (X32_ei - X32_ei**4)
      + (5.5 / (1.5 + 2 * Z_eff))
      * (X32_ei**2 - X32_ei**4 - 0.8 * (X32_ei**3 - X32_ei**4))
      - (1.3 / (1 + 0.5 * Z_eff)) * X32_ei**4
  )

  # Equation 12
  return F32_ee + F32_ei


def _calculate_alpha(
    f_trap: array_typing.FloatVectorFace,
    nu_i_star: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the alpha coefficient: Redl PoP 2021 Eqs. 20-21.

  This coefficient accounts for ion temperature gradient effects and includes
  corrections for ion-electron collisions (unlike the Sauter model).
  """
  # Equation 20
  alpha_0 = -(
      (0.62 + 0.055 * (Z_eff - 1))
      / (0.53 + 0.17 * (Z_eff - 1))
  ) * (
      (1 - f_trap)
      / (1 - (0.31 - 0.065 * (Z_eff - 1)) * f_trap - 0.25 * f_trap**2)
  )

  # Equation 21
  alpha = (
    (
      (
        alpha_0
        + 0.7
        * Z_eff
        * f_trap**0.5
        * jnp.sqrt(nu_i_star)
      )
      / (1 + 0.18 * jnp.sqrt(nu_i_star))
      - 0.002 * nu_i_star**2 * f_trap**6
    )
    / (1 + 0.004 * nu_i_star**2 * f_trap**6)
  )

  return alpha
