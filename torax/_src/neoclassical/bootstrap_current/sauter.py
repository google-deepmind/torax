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

from typing import Literal

import chex
import jax.numpy as jnp

from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.bootstrap_current import base
from torax._src.neoclassical.bootstrap_current import runtime_params


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params.DynamicRuntimeParams):
  """Dynamic runtime params for the Sauter model."""

  bootstrap_multiplier: float


class SauterModel(base.BootstrapCurrentModel):
  """Sauter model for bootstrap current."""

  def calculate_bootstrap_current(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.BootstrapCurrent:
    """Calculates bootstrap current."""
    bootstrap_params = (
        dynamic_runtime_params_slice.neoclassical.bootstrap_current
    )
    assert isinstance(bootstrap_params, DynamicRuntimeParams)
    result = _calculate_bootstrap_current(
        bootstrap_multiplier=bootstrap_params.bootstrap_multiplier,
        Z_eff_face=core_profiles.Z_eff_face,
        Z_i_face=core_profiles.Z_i_face,
        n_e=core_profiles.n_e,
        n_i=core_profiles.n_i,
        T_e=core_profiles.T_e,
        T_i=core_profiles.T_i,
        psi=core_profiles.psi,
        q_face=core_profiles.q_face,
        geo=geometry,
    )
    return base.BootstrapCurrent(
        j_bootstrap=result.j_bootstrap,
        j_bootstrap_face=result.j_bootstrap_face,
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class SauterModelConfig(base.BootstrapCurrentModelConfig):
  """Config for the Sauter model implementation of bootstrap current.

  Attributes:
    bootstrap_multiplier: Multiplication factor for bootstrap current.
  """

  model_name: Literal['sauter'] = 'sauter'
  bootstrap_multiplier: float = 1.0

  def build_dynamic_params(self) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(bootstrap_multiplier=self.bootstrap_multiplier)

  def build_model(self) -> SauterModel:
    return SauterModel()


@jax_utils.jit
def _calculate_bootstrap_current(
    *,
    bootstrap_multiplier: float,
    Z_eff_face: chex.Array,
    Z_i_face: chex.Array,
    n_e: cell_variable.CellVariable,
    n_i: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
    T_i: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    q_face: chex.Array,
    geo: geometry_lib.Geometry,
) -> base.BootstrapCurrent:
  """Calculates j_bootstrap and j_bootstrap_face using the Sauter model."""
  # pylint: disable=invalid-name

  # Formulas from Sauter PoP 1999. Future work can include Redl PoP 2021
  # corrections.

  # local r/R0 on face grid
  epsilon = (geo.R_out_face - geo.R_in_face) / (geo.R_out_face + geo.R_in_face)
  epseff = (
      0.67 * (1.0 - 1.4 * jnp.abs(geo.delta_face) * geo.delta_face) * epsilon
  )
  aa = (1.0 - epsilon) / (1.0 + epsilon)
  ftrap = 1.0 - jnp.sqrt(aa) * (1.0 - epseff) / (1.0 + 2.0 * jnp.sqrt(epseff))

  # Spitzer conductivity
  lnLame = (
      31.3 - 0.5 * jnp.log(n_e.face_value()) + jnp.log(T_e.face_value() * 1e3)
  )
  lnLami = (
      30
      - 3 * jnp.log(Z_i_face)
      - 0.5 * jnp.log(n_i.face_value())
      + 1.5 * jnp.log(T_i.face_value() * 1e3)
  )

  nu_e_star = (
      6.921e-18
      * q_face
      * geo.R_major
      * n_e.face_value()
      * Z_eff_face
      * lnLame
      / (
          ((T_e.face_value() * 1e3) ** 2)
          * (epsilon + constants.CONSTANTS.eps) ** 1.5
      )
  )
  nu_i_star = (
      4.9e-18
      * q_face
      * geo.R_major
      * n_i.face_value()
      * Z_eff_face**4
      * lnLami
      / (
          ((T_i.face_value() * 1e3) ** 2)
          * (epsilon + constants.CONSTANTS.eps) ** 1.5
      )
  )

  # Calculate terms needed for bootstrap current
  denom = (
      1.0
      + (1 - 0.1 * ftrap) * jnp.sqrt(nu_e_star)
      + 0.5 * (1.0 - ftrap) * nu_e_star / Z_eff_face
  )
  ft31 = ftrap / denom
  ft32ee = ftrap / (
      1
      + 0.26 * (1 - ftrap) * jnp.sqrt(nu_e_star)
      + 0.18 * (1 - 0.37 * ftrap) * nu_e_star / jnp.sqrt(Z_eff_face)
  )
  ft32ei = ftrap / (
      1
      + (1 + 0.6 * ftrap) * jnp.sqrt(nu_e_star)
      + 0.85 * (1 - 0.37 * ftrap) * nu_e_star * (1 + Z_eff_face)
  )
  ft34 = ftrap / (
      1.0
      + (1 - 0.1 * ftrap) * jnp.sqrt(nu_e_star)
      + 0.5 * (1.0 - 0.5 * ftrap) * nu_e_star / Z_eff_face
  )

  F32ee = (
      (0.05 + 0.62 * Z_eff_face)
      / (Z_eff_face * (1 + 0.44 * Z_eff_face))
      * (ft32ee - ft32ee**4)
      + 1
      / (1 + 0.22 * Z_eff_face)
      * (ft32ee**2 - ft32ee**4 - 1.2 * (ft32ee**3 - ft32ee**4))
      + 1.2 / (1 + 0.5 * Z_eff_face) * ft32ee**4
  )

  F32ei = (
      -(0.56 + 1.93 * Z_eff_face)
      / (Z_eff_face * (1 + 0.44 * Z_eff_face))
      * (ft32ei - ft32ei**4)
      + 4.95
      / (1 + 2.48 * Z_eff_face)
      * (ft32ei**2 - ft32ei**4 - 0.55 * (ft32ei**3 - ft32ei**4))
      - 1.2 / (1 + 0.5 * Z_eff_face) * ft32ei**4
  )

  term_0 = (1 + 1.4 / (Z_eff_face + 1)) * ft31
  term_1 = -1.9 / (Z_eff_face + 1) * ft31**2
  term_2 = 0.3 / (Z_eff_face + 1) * ft31**3
  term_3 = 0.2 / (Z_eff_face + 1) * ft31**4
  L31 = term_0 + term_1 + term_2 + term_3

  L32 = F32ee + F32ei

  L34 = (
      (1 + 1.4 / (Z_eff_face + 1)) * ft34
      - 1.9 / (Z_eff_face + 1) * ft34**2
      + 0.3 / (Z_eff_face + 1) * ft34**3
      + 0.2 / (Z_eff_face + 1) * ft34**4
  )

  alpha0 = -1.17 * (1 - ftrap) / (1 - 0.22 * ftrap - 0.19 * ftrap**2)
  alpha = (
      (alpha0 + 0.25 * (1 - ftrap**2) * jnp.sqrt(nu_i_star))
      / (1 + 0.5 * jnp.sqrt(nu_i_star))
      + 0.315 * nu_i_star**2 * ftrap**6
  ) / (1 + 0.15 * nu_i_star**2 * ftrap**6)

  # calculate bootstrap current
  prefactor = -geo.F_face * bootstrap_multiplier * 2 * jnp.pi / geo.B_0

  pe = n_e.face_value() * T_e.face_value() * 1e3 * 1.6e-19
  pi = n_i.face_value() * T_i.face_value() * 1e3 * 1.6e-19

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

  j_bootstrap_face = global_coeff * (
      necoeff * dlnne_drnorm
      + nicoeff * dlnni_drnorm
      + tecoeff * dlnte_drnorm
      + ticoeff * dlnti_drnorm
  )
  j_bootstrap = geometry_lib.face_to_cell(j_bootstrap_face)

  return base.BootstrapCurrent(
      j_bootstrap=j_bootstrap,
      j_bootstrap_face=j_bootstrap_face,
  )
