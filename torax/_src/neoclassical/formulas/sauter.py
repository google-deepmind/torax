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

import jax.numpy as jnp
from torax._src import array_typing

# pylint: disable=invalid-name


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


def calculate_L34(
    f_trap: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the L34 coefficient: Sauter PoP 1999 Eqs. 16a+b."""
  ft34 = f_trap / (
      1.0
      + (1 - 0.1 * f_trap) * jnp.sqrt(nu_e_star)
      + 0.5 * (1.0 - 0.5 * f_trap) * nu_e_star / Z_eff
  )
  return (
      (1 + 1.4 / (Z_eff + 1)) * ft34
      - 1.9 / (Z_eff + 1) * ft34**2
      + 0.3 / (Z_eff + 1) * ft34**3
      + 0.2 / (Z_eff + 1) * ft34**4
  )


def calculate_alpha(
    f_trap: array_typing.FloatVectorFace,
    nu_i_star: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the alpha coefficient: Sauter PoP 1999 Eqs. 17a+b."""
  alpha0 = -1.17 * (1 - f_trap) / (1 - 0.22 * f_trap - 0.19 * f_trap**2)
  alpha = (
      (alpha0 + 0.25 * (1 - f_trap**2) * jnp.sqrt(nu_i_star))
      / (1 + 0.5 * jnp.sqrt(nu_i_star))
      + 0.315 * nu_i_star**2 * f_trap**6
  ) / (1 + 0.15 * nu_i_star**2 * f_trap**6)
  return alpha
