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

"""Physics formulas mostly related to CoreProfiles calculations.

Functions:
    - calculate_main_ion_dilution_factor: Calculates the main ion dilution
      factor based on average impurity charge and Z_eff.
    - calculate_pressure: Calculates pressure from density and temperatures.
    - calc_dvar_dpsi: Calculates derivative of a CellVariable with respect to
      poloidal flux.
    - calc_pprime: Calculates total pressure gradient with respect to poloidal
      flux.
    - calc_FFprime: Calculates FF', an output quantity used for equilibrium
      coupling.
    - calculate_stored_thermal_energy: Calculates stored thermal energy from
      pressures.
    - calculate_greenwald_fraction: Calculates the Greenwald fraction from the
      averaged electron density (can be line-averaged or volume-averaged).
    - calculate_beta_volume_avg: Calculates the volume-averaged plasma beta
      based on thermal pressure.
    - calculate_beta_pol_profile: Calculates local poloidal beta profile as a
      CellVariable.
    - calc_beta_pol_prime: Calculates
      beta_pol_prime = -d(beta_pol) / d(psi_norm) on the face grid.
"""
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import psi_calculations


# pylint: disable=invalid-name


# TODO(b/377225415): generalize to arbitrary number of ions.
def calculate_main_ion_dilution_factor(
    Z_i: array_typing.FloatScalar,
    Z_impurity: array_typing.FloatVector,
    Z_eff: array_typing.FloatVector,
) -> array_typing.FloatVector:
  """Calculates the main ion dilution factor based on a single assumed impurity and general main ion charge."""
  return (Z_impurity - Z_eff) / (Z_i * (Z_impurity - Z_i))


def calc_dvar_dpsi(
    var: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    normalized: bool = False,
) -> array_typing.FloatVectorFace:
  r"""Calculates d(var) / d(psi) on the face grid.

  Away from the magnetic axis, computes:
    d(var) / d(psi) = (d(var) / drhon) / (d(psi) / drhon)
  using face gradients with respect to normalized radius rhon. On axis
  (rho=0), uses L'Hôpital's rule with a 2nd order forward difference
  approximation for the second derivative.

  Args:
    var: CellVariable to differentiate.
    psi: Poloidal flux CellVariable.
    normalized: If True, differentiates with respect to normalized poloidal flux
      psi_N in [0, 1].

  Returns:
    Face-grid array of the derivative.
  """
  dvar_drhon = var.face_grad()
  dpsi_drhon = psi.face_grad()
  var_face = var.face_value()
  psi_face = psi.face_value()

  # 2nd order forward difference stencil coefficients [2, -5, 4, -1] for axis
  # second derivative evaluation via L'Hopital's rule.
  coeffs = jnp.array([2.0, -5.0, 4.0, -1.0])
  axis_derivative = jnp.expand_dims(
      jnp.dot(coeffs, var_face[:4]) / jnp.dot(coeffs, psi_face[:4]),
      axis=0,
  )

  dvar_dpsi = jnp.concatenate(
      [axis_derivative, dvar_drhon[1:] / dpsi_drhon[1:]]
  )

  if normalized:
    psi_range = psi.right_face_value - psi.left_face_value
    return dvar_dpsi * psi_range
  return dvar_dpsi


def calc_pprime(
    core_profiles: state.CoreProfiles,
) -> array_typing.FloatVectorFace:
  """Calculates total pressure gradient with respect to poloidal flux."""
  return calc_dvar_dpsi(
      var=core_profiles.pressure_total,
      psi=core_profiles.psi,
      normalized=False,
  )


def calc_FFprime(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.FloatVector:
  r"""Calculates FF', an output quantity used for equilibrium coupling.

  Calculation is based on the following formulation of the magnetic
  equilibrium equation:
  :math:`-j_{tor} = 2\pi (Rp' + \frac{1}{\mu_0 R}FF')`

  And following division by R and flux surface averaging:

  :math:`-\langle \frac{j_{tor}}{R} \rangle = 2\pi (p' +
  \langle\frac{1}{R^2}\rangle\frac{FF'}{\mu_0})`

  Args:
    core_profiles: CoreProfiles object containing information on temperatures
      and densities.
    geo: Magnetic equilibrium.

  Returns:
    FFprime:   F is the toroidal flux function, and F' is its derivative with
      respect to the poloidal flux.
  """

  mu0 = constants.CONSTANTS.mu_0
  pprime = calc_pprime(core_profiles)
  # g3 = <1/R^2>
  # gm9 = <1/R>
  g3 = geo.g3_face
  jtor_over_R = core_profiles.j_total_face * geo.gm9_face

  FFprime_face = -(jtor_over_R / (2 * jnp.pi) + pprime) * mu0 / g3
  return FFprime_face


def calculate_stored_thermal_energy(
    p_el: cell_variable.CellVariable,
    p_ion: cell_variable.CellVariable,
    p_tot: cell_variable.CellVariable,
    geo: geometry.Geometry,
) -> tuple[array_typing.FloatScalar, ...]:
  """Calculates stored thermal energy from pressures.

  Args:
    p_el: Electron pressure [Pa]
    p_ion: Ion pressure [Pa]
    p_tot: Total pressure [Pa]
    geo: Geometry object

  Returns:
    wth_el: Electron thermal stored energy [J]
    wth_ion: Ion thermal stored energy [J]
    wth_tot: Total thermal stored energy [J]
  """
  wth_el = math_utils.volume_integration(1.5 * p_el.value, geo)  # pyrefly: ignore[bad-argument-type]
  wth_ion = math_utils.volume_integration(1.5 * p_ion.value, geo)  # pyrefly: ignore[bad-argument-type]
  wth_tot = math_utils.volume_integration(1.5 * p_tot.value, geo)  # pyrefly: ignore[bad-argument-type]

  return wth_el, wth_ion, wth_tot


def calculate_greenwald_fraction(
    n_e_avg: array_typing.FloatScalar,
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
  """Calculates the Greenwald fraction from the averaged electron density.

  Different averaging can be used, e.g. volume-averaged or line-averaged.

  Args:
    n_e_avg: Averaged electron density [m^-3]
    core_profiles: CoreProfiles object containing information on currents and
      densities.
    geo: Geometry object

  Returns:
    fgw: Greenwald density fraction
  """
  # gw_limit is in units of 10^20 m^-3 when Ip is in MA and a_minor is in m.
  gw_limit = (
      core_profiles.Ip_profile_face[-1] * 1e-6 / (jnp.pi * geo.a_minor**2)
  )
  fgw = n_e_avg / (gw_limit * 1e20)
  return fgw


def calculate_betas(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
  """Calculates the beta_tor, beta_pol, and beta_N plasma beta quantities.

  beta_tor is defined as the ratio of volume-averaged plasma pressure to
  toroidal magnetic pressure on-axis:

  beta_tor = P_total_volume_avg / (B0^2 / (2 * mu0))

  beta_pol is defined as the ratio of volume-averaged plasma pressure to
  averaged poloidal magnetic pressure at the LCFS:

  beta_pol = P_total_volume_avg / (Bpol_lcfs^2 / (2 * mu0))

  Using an approximation for Bpol_lcfs^2 = mu0^2 * Ip^2 / 4*pi^2*a_V^2, where
  a_V is the effective minor radius satisfying the volume calculation
  V = 2 * pi^2 * R_0 * a_V^2, we get:

  beta_pol = 4 * V * P_total_volume_avg / (mu0 * Ip^2 * R_0)

  beta_N is the normalized toroidal plasma beta in percent:

  beta_N = beta_tor * (a * B0 / Ip) * 1e8
  where beta_tor is fractional toroidal beta. Ip is in A, a is minor radius.

  Args:
    core_profiles: CoreProfiles object.
    geo: Geometry object.

  Returns:
    Tuple of beta_tor, beta_pol, and beta_N
  """
  p_total_volume_avg = math_utils.volume_average(
      core_profiles.pressure_total.value, geo  # pyrefly: ignore[bad-argument-type]
  )

  magnetic_pressure_on_axis = geo.B_0**2 / (2 * constants.CONSTANTS.mu_0)
  # Add a division guard though B0 should typically be non-zero.
  beta_tor = math_utils.safe_divide(
      num=p_total_volume_avg, denom=magnetic_pressure_on_axis, eps=1e-7  # pyrefly: ignore[bad-argument-type]
  )

  beta_pol = (
      4.0
      * geo.volume_face[-1]
      * p_total_volume_avg
      / (
          constants.CONSTANTS.mu_0
          * core_profiles.Ip_profile_face[-1] ** 2
          * geo.R_major
          + constants.CONSTANTS.eps
      )
  )

  beta_N = (
      1e8
      * beta_tor
      * (
          geo.a_minor
          * geo.B_0
          / (core_profiles.Ip_profile_face[-1] + constants.CONSTANTS.eps)
      )
  )

  return beta_tor, beta_pol, beta_N  # pyrefly: ignore[bad-return]


def calculate_beta_pol_profile(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Calculates the local poloidal beta profile on the cell grid.

  beta_pol_local(psi) = P_total(psi) / (<Bp^2(psi)> / (2 * mu0))

  Args:
    core_profiles: CoreProfiles object.
    geo: Geometry object.

  Returns:
    beta_pol_profile: CellVariable of local poloidal beta profile.
  """
  bpol2_face = psi_calculations.calc_bpol_squared(geo, core_profiles.psi)
  bpol2_cell = geometry.face_to_cell(bpol2_face)
  denom_cell = (
      bpol2_cell / (2.0 * constants.CONSTANTS.mu_0) + constants.CONSTANTS.eps
  )
  denom_right = (
      bpol2_face[-1] / (2.0 * constants.CONSTANTS.mu_0)
      + constants.CONSTANTS.eps
  )
  right_face_constraint = (
      core_profiles.pressure_total.right_face_constraint / denom_right
      if core_profiles.pressure_total.right_face_constraint is not None
      else None
  )
  return cell_variable.CellVariable(
      value=core_profiles.pressure_total.value / denom_cell,
      face_centers=core_profiles.pressure_total.face_centers,
      right_face_constraint=right_face_constraint,
      right_face_grad_constraint=None,
  )


def calculate_beta_pol_prime(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.FloatVectorFace:
  r"""Calculates beta_pol_prime on the face grid.

  Defined as:
    beta_pol_prime = -d(beta_pol_local) / d(psi_norm)
  where beta_pol_local is the local poloidal beta CellVariable and psi_norm is
  the normalized poloidal flux in [0, 1]. In normal confinement, pressure
  decreases toward the edge, making d(beta_pol)/d(psi_norm) negative, so
  beta_pol_prime represents the positive gradient magnitude.

  Args:
    core_profiles: CoreProfiles object.
    geo: Geometry object.

  Returns:
    beta_pol_prime: Face-grid array of the derivative magnitude [dimensionless].
  """
  beta_pol = calculate_beta_pol_profile(core_profiles, geo)
  return -calc_dvar_dpsi(
      var=beta_pol,
      psi=core_profiles.psi,
      normalized=True,
  )


