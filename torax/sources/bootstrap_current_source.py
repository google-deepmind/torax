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

"""Sauter model implementation - to be moved to neoclassical."""
import chex
from jax import numpy as jnp
from torax import constants
from torax import jax_utils
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.sources import source_profiles


@jax_utils.jit
def calc_sauter_model(
    *,
    bootstrap_multiplier: float,
    density_reference: float,
    Z_eff_face: chex.Array,
    Z_i_face: chex.Array,
    n_e: cell_variable.CellVariable,
    n_i: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
    T_i: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    q_face: chex.Array,
    geo: geometry.Geometry,
) -> source_profiles.BootstrapCurrentProfile:
  """Calculates sigmaneo, j_bootstrap, and I_bootstrap."""
  # pylint: disable=invalid-name

  # Formulas from Sauter PoP 1999. Future work can include Redl PoP 2021
  # corrections.

  true_n_e_face = n_e.face_value() * density_reference
  true_n_i_face = n_i.face_value() * density_reference

  # # local r/R0 on face grid
  epsilon = (geo.R_out_face - geo.R_in_face) / (geo.R_out_face + geo.R_in_face)
  epseff = (
      0.67 * (1.0 - 1.4 * jnp.abs(geo.delta_face) * geo.delta_face) * epsilon
  )
  aa = (1.0 - epsilon) / (1.0 + epsilon)
  ftrap = 1.0 - jnp.sqrt(aa) * (1.0 - epseff) / (1.0 + 2.0 * jnp.sqrt(epseff))

  # Spitzer conductivity
  NZ = 0.58 + 0.74 / (0.76 + Z_eff_face)
  lnLame = (
      31.3 - 0.5 * jnp.log(true_n_e_face) + jnp.log(T_e.face_value() * 1e3)
  )
  lnLami = (
      30
      - 3 * jnp.log(Z_i_face)
      - 0.5 * jnp.log(true_n_i_face)
      + 1.5 * jnp.log(T_i.face_value() * 1e3)
  )

  sigsptz = (
      1.9012e04 * (T_e.face_value() * 1e3) ** 1.5 / Z_eff_face / NZ / lnLame
  )

  nuestar = (
      6.921e-18
      * q_face
      * geo.R_major
      * true_n_e_face
      * Z_eff_face
      * lnLame
      / (
          ((T_e.face_value() * 1e3) ** 2)
          * (epsilon + constants.CONSTANTS.eps) ** 1.5
      )
  )
  nuistar = (
      4.9e-18
      * q_face
      * geo.R_major
      * true_n_i_face
      * Z_eff_face**4
      * lnLami
      / (
          ((T_i.face_value() * 1e3) ** 2)
          * (epsilon + constants.CONSTANTS.eps) ** 1.5
      )
  )

  # Neoclassical correction to spitzer conductivity
  ft33 = ftrap / (
      1.0
      + (0.55 - 0.1 * ftrap) * jnp.sqrt(nuestar)
      + 0.45 * (1.0 - ftrap) * nuestar / (Z_eff_face**1.5)
  )
  signeo_face = 1.0 - ft33 * (
      1.0
      + 0.36 / Z_eff_face
      - ft33 * (0.59 / Z_eff_face - 0.23 / Z_eff_face * ft33)
  )
  sigmaneo = sigsptz * signeo_face

  # Calculate terms needed for bootstrap current
  denom = (
      1.0
      + (1 - 0.1 * ftrap) * jnp.sqrt(nuestar)
      + 0.5 * (1.0 - ftrap) * nuestar / Z_eff_face
  )
  ft31 = ftrap / denom
  ft32ee = ftrap / (
      1
      + 0.26 * (1 - ftrap) * jnp.sqrt(nuestar)
      + 0.18 * (1 - 0.37 * ftrap) * nuestar / jnp.sqrt(Z_eff_face)
  )
  ft32ei = ftrap / (
      1
      + (1 + 0.6 * ftrap) * jnp.sqrt(nuestar)
      + 0.85 * (1 - 0.37 * ftrap) * nuestar * (1 + Z_eff_face)
  )
  ft34 = ftrap / (
      1.0
      + (1 - 0.1 * ftrap) * jnp.sqrt(nuestar)
      + 0.5 * (1.0 - 0.5 * ftrap) * nuestar / Z_eff_face
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
      alpha0
      + 0.25
      * (1 - ftrap**2)
      * jnp.sqrt(nuistar)
      / (1 + 0.5 * jnp.sqrt(nuistar))
      + 0.315 * nuistar**2 * ftrap**6
  ) / (1 + 0.15 * nuistar**2 * ftrap**6)

  # calculate bootstrap current
  prefactor = -geo.F_face * bootstrap_multiplier * 2 * jnp.pi / geo.B_0

  pe = true_n_e_face * (T_e.face_value()) * 1e3 * 1.6e-19
  pi = true_n_i_face * (T_i.face_value()) * 1e3 * 1.6e-19

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

  #  j_bootstrap_face = jnp.concatenate([jnp.zeros(1), j_bootstrap_face])
  j_bootstrap_face = jnp.array(j_bootstrap_face)
  j_bootstrap = geometry.face_to_cell(j_bootstrap_face)
  sigmaneo_cell = geometry.face_to_cell(sigmaneo)

  return source_profiles.BootstrapCurrentProfile(
      sigma=sigmaneo_cell,
      sigma_face=sigmaneo,
      j_bootstrap=j_bootstrap,
      j_bootstrap_face=j_bootstrap_face,
  )
