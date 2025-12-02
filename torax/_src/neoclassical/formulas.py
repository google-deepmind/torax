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

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.physics import collisions


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
  return (
      6.921e-18
      * q
      * geo.R_major_profile_face
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
  return (
      4.9e-18
      * q
      * geo.R_major_profile_face
      * n_i
      * Z_eff**4
      * log_lambda_ii
      / (
          ((T_i * 1e3) ** 2)
          * (geo.epsilon_face + constants.CONSTANTS.eps) ** 1.5
      )
  )


# Functions to calculate the neoclassical poloidal velocity.
def _calculate_neoclassical_k_neo(
    nu_star: array_typing.FloatScalar, epsilon: array_typing.FloatScalar
):
  """Calculates the neoclassical coefficient k_neo.

  Equation (6.135) from
  Hinton, F. L., & Hazeltine, R. D.,
  "Theory of plasma transport in toroidal confinement systems"
  Rev. Mod. Phys. 48(2), 239–308. (1976)
  https://doi.org/10.1103/RevModPhys.48.239

  Limits:
    - Banana regime (nu_star -> 0): ~1.17
    - Pfirsch-Schluter regime (nu_star -> inf): ~ -2.1

  Args:
    nu_star : The normalized ion collisionality.
    epsilon : The inverse aspect ratio.

  Returns:
    k_neo : The neoclassical coefficient.
  """
  # Calculate the first term (Banana-Plateau transition)
  # (1.17 - 0.35 * sqrt(nu)) / (1 + 0.7 * sqrt(nu))
  sqrt_nu = jnp.sqrt(nu_star)
  term1 = (1.17 - 0.35 * sqrt_nu) / (1.0 + 0.7 * sqrt_nu)

  # Calculate the second term (Pfirsch-Schluter driver)
  # 2.1 * nu^2 * epsilon^3
  ps_factor = (nu_star**2) * (epsilon**3)
  term2 = 2.1 * ps_factor

  # Calculate the final denominator (Switching function)
  # 1 + nu^2 * epsilon^3
  denominator = 1.0 + ps_factor

  return (term1 - term2) / denominator

# TODO(b/381199010): Implement alternative Sauter-based k_neo calculation.
# See Sauter (1999) Eq. 17a-17b


@jax.jit
def calculate_poloidal_velocity(
    T_i: cell_variable.CellVariable,
    n_i: array_typing.FloatVectorFace,
    q: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
    Z_i: array_typing.FloatVectorFace,
    B_tor: array_typing.FloatVectorFace,
    B_total_squared: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
    rotation_multiplier: array_typing.FloatScalar = 1.0,
) -> cell_variable.CellVariable:
  """Computes the neoclassical ion poloidal velocity profile.

  Implementing eq.33 from
  Y. B. Kim , P. H. Diamond , R. J. Groebner.
  "Neoclassical poloidal and toroidal rotation in tokamaks"
  Phys. Fluids B 3, 2050–2060 (1991)
  https://doi.org/10.1063/1.859671

  Eq. 33 can be simplified to the following form in SI units:
  v_pol = k_neo * (dT/dr) * (B_tor / <B^2>) / (Z * e)

  Args:
    T_i: Ion temperature as a cell variable [keV].
    n_i: Ion density on the face grid [m^-3].
    q: Safety factor on the face grid.
    Z_eff: Effective charge on the face grid.
    Z_i: Main ion charge on the face grid.
    B_tor: Toroidal magnetic field on the face grid [T].
    B_total_squared: Total magnetic field (toroidal + poloidal) on the face grid
      [T].
    geo : Geometry
    rotation_multiplier: A multiplier to apply to the poloidal velocity.
  Returns:
    v_pol : Poloidal velocity profile [m/s].
  """
  # Note: all computations are performed on the face grid.

  T_i_face = T_i.face_value()
  epsilon = geo.epsilon_face

  # Calculate Neoclassical Coefficient k_i
  log_lambda_ii = collisions.calculate_log_lambda_ii(
      T_i_face,
      n_i,
      Z_eff,
  )
  nu_i_star = calculate_nu_i_star(
      q=q,
      geo=geo,
      n_i=n_i,
      T_i=T_i_face,
      Z_eff=Z_eff,
      log_lambda_ii=log_lambda_ii,
  )
  k_neo = _calculate_neoclassical_k_neo(nu_i_star, epsilon)

  # Calculate Radial Temperature Gradient (dT/dr)
  grad_Ti = T_i.face_grad(geo.r_mid) * constants.CONSTANTS.keV_to_J  # [J/m]

  # Calculate Poloidal Velocity
  # v_pol = k_i * (dT/dr) * (B_tor / <B^2>) / (Z * e)
  B_total_squared_safe = jnp.maximum(B_total_squared, constants.CONSTANTS.eps)
  v_pol = (
      k_neo
      * grad_Ti
      * (B_tor / B_total_squared_safe)
      / (constants.CONSTANTS.q_e * Z_i)
  )

  v_pol = rotation_multiplier * v_pol

  return cell_variable.CellVariable(
      value=geometry_lib.face_to_cell(v_pol),
      dr=geo.drho_norm,
      right_face_constraint=v_pol[-1],
      right_face_grad_constraint=None,
  )
