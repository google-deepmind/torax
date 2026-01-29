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
import jax.numpy as jnp

from torax._src import array_typing


def calculate_L31(
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


def calculate_L32(
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
      + ((0.23 * (1 - 0.96 * f_trap) * jnp.sqrt(nu_e_star)) / Z_eff**0.5)
      + ((0.13 * (1 - 0.38 * f_trap) * nu_e_star) / Z_eff**2)
      * (
          jnp.sqrt(1 + 2 * jnp.sqrt(Z_eff - 1))
          + f_trap**2 * jnp.sqrt((0.075 + 0.25 * (Z_eff - 1) ** 2) * nu_e_star)
      )
  )

  # Equation 13
  X32_e = f_eff_32_ee
  F32_ee = (
      ((0.1 + 0.6 * Z_eff) / (Z_eff * (0.77 + 0.63 * (1 + (Z_eff - 1) ** 1.1))))
      * (X32_e - X32_e**4)
      + (0.7 / (1 + 0.2 * Z_eff))
      * (X32_e**2 - X32_e**4 - 1.2 * (X32_e**3 - X32_e**4))
      + (1.3 / (1 + 0.5 * Z_eff)) * X32_e**4
  )

  # Equation 16
  f_eff_32_ei = f_trap / (
      1.0
      + (
          (0.87 * (1 + 0.39 * f_trap) * jnp.sqrt(nu_e_star))
          / (1 + 2.95 * (Z_eff - 1) ** 2)
      )
      + (1.53 * (1 - 0.37 * f_trap) * nu_e_star) * (2 + 0.375 * (Z_eff - 1))
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


def calculate_alpha(
    f_trap: array_typing.FloatVectorFace,
    nu_i_star: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  """Calculates the alpha coefficient: Redl PoP 2021 Eqs. 20-21.

  This coefficient accounts for ion temperature gradient effects and includes
  corrections for ion-electron collisions (unlike the Sauter model).
  """
  # Equation 20
  alpha_0 = -((0.62 + 0.055 * (Z_eff - 1)) / (0.53 + 0.17 * (Z_eff - 1))) * (
      (1 - f_trap)
      / (1 - (0.31 - 0.065 * (Z_eff - 1)) * f_trap - 0.25 * f_trap**2)
  )

  # Equation 21
  alpha = (
      (alpha_0 + 0.7 * Z_eff * f_trap**0.5 * jnp.sqrt(nu_i_star))
      / (1 + 0.18 * jnp.sqrt(nu_i_star))
      - 0.002 * nu_i_star**2 * f_trap**6
  ) / (1 + 0.004 * nu_i_star**2 * f_trap**6)

  return alpha
