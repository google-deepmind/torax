# Copyright 2026 DeepMind Technologies Limited
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

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src import state
from torax._src.geometry import geometry as geometry_lib
from torax._src.physics import collisions


# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AnalyticalCache:
  """Cached intermediate quantities for analytical neoclassical models."""

  f_trap: array_typing.FloatVectorFace
  log_lambda_ei: array_typing.FloatVectorFace
  log_lambda_ii: array_typing.FloatVectorFace
  nu_e_star: array_typing.FloatVectorFace
  nu_i_star: array_typing.FloatVectorFace


def compute_analytical_cache(
    geo: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles,
) -> AnalyticalCache:
  """Computes shared quantities for analytical neoclassical models."""
  f_trap = calculate_f_trap(geo)
  log_lambda_ei = collisions.calculate_log_lambda_ei(
      core_profiles.T_e.face_value(), core_profiles.n_e.face_value()
  )
  log_lambda_ii = collisions.calculate_log_lambda_ii(
      core_profiles.T_i.face_value(),
      core_profiles.n_i.face_value(),
      core_profiles.Z_i_face,
  )
  nu_e_star = calculate_nu_e_star(
      q=core_profiles.q_face,
      geo=geo,
      n_e=core_profiles.n_e.face_value(),
      T_e=core_profiles.T_e.face_value(),
      Z_eff=core_profiles.Z_eff_face,
      log_lambda_ei=log_lambda_ei,
  )
  nu_i_star = calculate_nu_i_star(
      q=core_profiles.q_face,
      geo=geo,
      n_i=core_profiles.n_i.face_value(),
      T_i=core_profiles.T_i.face_value(),
      Z_eff=core_profiles.Z_eff_face,
      log_lambda_ii=log_lambda_ii,
  )
  return AnalyticalCache(
      f_trap=f_trap,
      log_lambda_ei=log_lambda_ei,
      log_lambda_ii=log_lambda_ii,
      nu_e_star=nu_e_star,
      nu_i_star=nu_i_star,
  )


# TODO(b/545148156): Add finite-orbit-width effects.
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
      # On the magnetic axis, epsilon_effective is 0, in order to avoid a NaN
      # gradient we define the gradient at zero to be zero.
      1.0 + 2.0 * math_utils.sqrt_with_zero_gradient_at_zero(epsilon_effective)
  )


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


def calculate_neoclassical_k_neo(
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
    nu_star: The normalized ion collisionality.
    epsilon: The inverse aspect ratio.

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

