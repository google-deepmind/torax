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
"""Common formulas used in neoclassical models."""

import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.geometry import geometry as geometry_lib

# pylint: disable=invalid-name


def calculate_f_trap(
    geo: geometry_lib.Geometry,
) -> array_typing.FloatVectorFace:
  """Calculates the effective trapped particle fraction.

  From O. Sauter, Fusion Engineering and Design 112 (2016) 633-645. Eqs 33+34.

  Args:
    geo: The magnetic geometry.

  Returns:
    The effective trapped particle fraction.
  """

  epsilon_effective = (
      0.67
      * (1.0 - 1.4 * jnp.abs(geo.delta_face) * geo.delta_face)
      * geo.epsilon_face
  )
  aa = (1.0 - geo.epsilon_face) / (1.0 + geo.epsilon_face)
  return 1.0 - jnp.sqrt(aa) * (1.0 - epsilon_effective) / (
      1.0 + 2.0 * jnp.sqrt(epsilon_effective)
  )


def calculate_L31(
    f_trap: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the L31 Sauter coefficient: Sauter PoP 1999 Eqs. 14a + 14b."""
  denom = (
      1.0
      + (1 - 0.1 * f_trap) * jnp.sqrt(nu_e_star)
      + 0.5 * (1.0 - f_trap) * nu_e_star / Z_eff
  )
  ft31 = f_trap / denom
  term_0 = (1 + 1.4 / (Z_eff + 1)) * ft31
  term_1 = -1.9 / (Z_eff + 1) * ft31**2
  term_2 = 0.3 / (Z_eff + 1) * ft31**3
  term_3 = 0.2 / (Z_eff + 1) * ft31**4
  return term_0 + term_1 + term_2 + term_3


def calculate_L32(
    f_trap: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the L32 Sauter coefficient: Sauter PoP 1999 Eqs. 15a-e."""
  ft32ee = f_trap / (
      1
      + 0.26 * (1 - f_trap) * jnp.sqrt(nu_e_star)
      + 0.18 * (1 - 0.37 * f_trap) * nu_e_star / jnp.sqrt(Z_eff)
  )
  ft32ei = f_trap / (
      1
      + (1 + 0.6 * f_trap) * jnp.sqrt(nu_e_star)
      + 0.85 * (1 - 0.37 * f_trap) * nu_e_star * (1 + Z_eff)
  )

  F32ee = (
      (0.05 + 0.62 * Z_eff)
      / (Z_eff * (1 + 0.44 * Z_eff))
      * (ft32ee - ft32ee**4)
      + 1
      / (1 + 0.22 * Z_eff)
      * (ft32ee**2 - ft32ee**4 - 1.2 * (ft32ee**3 - ft32ee**4))
      + 1.2 / (1 + 0.5 * Z_eff) * ft32ee**4
  )

  F32ei = (
      -(0.56 + 1.93 * Z_eff)
      / (Z_eff * (1 + 0.44 * Z_eff))
      * (ft32ei - ft32ei**4)
      + 4.95
      / (1 + 2.48 * Z_eff)
      * (ft32ei**2 - ft32ei**4 - 0.55 * (ft32ei**3 - ft32ei**4))
      - 1.2 / (1 + 0.5 * Z_eff) * ft32ei**4
  )
  return F32ee + F32ei


# TODO(b/428166775): currently we have two very similar implementations for
# nu_e_star. We should refactor this to have a single one in physics/collisions
def calculate_nu_e_star(
    q: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
    n_e: array_typing.FloatVectorFace,
    T_e: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
    log_lambda_ei: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the electron collisionality, nu_e_star.

  This is the electron collisionality, defined as the ratio of the electron
  collision frequency to the bounce frequency. From Sauter PoP 1999 Eq. (18b).

  Args:
    q: Safety factor.
    geo: The geometry of the torus.
    n_e: Electron density [m^-3].
    T_e: Electron temperature [keV]. Converted to eV in the formula.
    Z_eff: Effective charge.
    log_lambda_ei: Electron-ion Coulomb logarithm.

  Returns:
    The electron collisionality.
  """
  # Use local major radius for neoclassical collisionality calculation
  r_major_face = (geo.R_in_face + geo.R_out_face) / 2
  return (
      6.921e-18
      * q
      * r_major_face
      * n_e
      * Z_eff
      * log_lambda_ei
      / (
          ((T_e * 1e3) ** 2)
          * (geo.epsilon_face + constants.CONSTANTS.eps) ** 1.5
      )
  )


def calculate_nu_i_star(
    q: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
    n_i: array_typing.FloatVectorFace,
    T_i: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
    log_lambda_ii: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the ion collisionality, nu_i_star.

  This is the ion collisionality, defined as the ratio of the ion
  collision frequency to the bounce frequency. From Sauter PoP 1999 Eq. (18c).

  Args:
    q: Safety factor.
    geo: The geometry of the torus.
    n_i: Ion density.
    T_i: Ion temperature.
    Z_eff: Effective charge.
    log_lambda_ii: Ion-ion Coulomb logarithm.

  Returns:
    The ion collisionality.
  """
  # Use local major radius for neoclassical collisionality calculation
  r_major_face = (geo.R_in_face + geo.R_out_face) / 2
  return (
      4.9e-18
      * q
      * r_major_face
      * n_i
      * Z_eff**4
      * log_lambda_ii
      / (
          ((T_i * 1e3) ** 2)
          * (geo.epsilon_face + constants.CONSTANTS.eps) ** 1.5
      )
  )
