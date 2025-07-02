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

import chex
import jax.numpy as jnp
from torax._src import constants
from torax._src.geometry import geometry as geometry_lib

# pylint: disable=invalid-name


def calculate_f_trap(geo: geometry_lib.Geometry) -> chex.Array:
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


# TODO(b/428166775): currently we have two very similar implementations for
# nu_e_star. We should refactor this to have a single one in physics/collisions
def calculate_nu_e_star(
    q: chex.Array,
    geo: geometry_lib.Geometry,
    n_e: chex.Array,
    T_e: chex.Array,
    Z_eff: chex.Array,
    log_lambda_ei: chex.Array,
) -> chex.Array:
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
  return (
      6.921e-18
      * q
      * geo.R_major
      * n_e
      * Z_eff
      * log_lambda_ei
      / (
          ((T_e * 1e3) ** 2)
          * (geo.epsilon_face + constants.CONSTANTS.eps) ** 1.5
      )
  )


def calculate_nu_i_star(
    q: chex.Array,
    geo: geometry_lib.Geometry,
    n_i: chex.Array,
    T_i: chex.Array,
    Z_eff: chex.Array,
    log_lambda_ii: chex.Array,
) -> chex.Array:
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
  return (
      4.9e-18
      * q
      * geo.R_major
      * n_i
      * Z_eff**4
      * log_lambda_ii
      / (
          ((T_i * 1e3) ** 2)
          * (geo.epsilon_face + constants.CONSTANTS.eps) ** 1.5
      )
  )
