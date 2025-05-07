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
    - calc_pprime: Calculates total pressure gradient with respect to poloidal
      flux.
    - calc_FFprime: Calculates FF', an output quantity used for equilibrium
      coupling.
    - calculate_stored_thermal_energy: Calculates stored thermal energy from
      pressures.
    - calculate_greenwald_fraction: Calculates the Greenwald fraction from the
      averaged electron density (can be line-averaged or volume-averaged).
"""
import jax
from jax import numpy as jnp
from torax import array_typing
from torax import constants
from torax import state
from torax.geometry import geometry

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


# TODO(b/377225415): generalize to arbitrary number of ions.
def calculate_main_ion_dilution_factor(
    Zi: array_typing.ScalarFloat,
    Zimp: array_typing.ArrayFloat,
    Z_eff: array_typing.ArrayFloat,
) -> jax.Array:
  """Calculates the main ion dilution factor based on a single assumed impurity and general main ion charge."""
  return (Zimp - Z_eff) / (Zi * (Zimp - Zi))


def calculate_pressure(
    core_profiles: state.CoreProfiles,
) -> tuple[array_typing.ArrayFloat, ...]:
  """Calculates pressure from density and temperatures on the face grid.

  Args:
    core_profiles: CoreProfiles object containing information on temperatures
      and densities.

  Returns:
    pressure_thermal_el_face: Electron thermal pressure [Pa]
    pressure_thermal_ion_face: Ion thermal pressure [Pa]
    pressure_thermal_tot_face: Total thermal pressure [Pa]
  """
  n_e = core_profiles.n_e.face_value()
  ni = core_profiles.ni.face_value()
  nimp = core_profiles.nimp.face_value()
  temp_ion = core_profiles.temp_ion.face_value()
  temp_el = core_profiles.temp_el.face_value()
  prefactor = constants.CONSTANTS.keV2J * core_profiles.density_reference
  pressure_thermal_el_face = n_e * temp_el * prefactor
  pressure_thermal_ion_face = (ni + nimp) * temp_ion * prefactor
  pressure_thermal_tot_face = (
      pressure_thermal_el_face + pressure_thermal_ion_face
  )
  return (
      pressure_thermal_el_face,
      pressure_thermal_ion_face,
      pressure_thermal_tot_face,
  )


def calc_pprime(
    core_profiles: state.CoreProfiles,
) -> array_typing.ArrayFloat:
  r"""Calculates total pressure gradient with respect to poloidal flux.

  Args:
    core_profiles: CoreProfiles object containing information on temperatures
      and densities.

  Returns:
    pprime: Total pressure gradient :math:`\partial p / \partial \psi`
      with respect to the normalized toroidal flux coordinate, on the face grid.
  """

  prefactor = constants.CONSTANTS.keV2J * core_profiles.density_reference

  _, _, p_total = calculate_pressure(core_profiles)
  psi = core_profiles.psi.face_value()
  n_e = core_profiles.n_e.face_value()
  ni = core_profiles.ni.face_value()
  nimp = core_profiles.nimp.face_value()
  temp_ion = core_profiles.temp_ion.face_value()
  temp_el = core_profiles.temp_el.face_value()
  dne_drhon = core_profiles.n_e.face_grad()
  dni_drhon = core_profiles.ni.face_grad()
  dnimp_drhon = core_profiles.nimp.face_grad()
  dti_drhon = core_profiles.temp_ion.face_grad()
  dte_drhon = core_profiles.temp_el.face_grad()
  dpsi_drhon = core_profiles.psi.face_grad()

  dptot_drhon = prefactor * (
      n_e * dte_drhon
      + ni * dti_drhon
      + nimp * dti_drhon
      + dne_drhon * temp_el
      + dni_drhon * temp_ion
      + dnimp_drhon * temp_ion
  )

  # Calculate on-axis value with L'Hôpital's rule using 2nd order forward
  # difference approximation for second derivative at edge.
  pprime_face_axis = jnp.expand_dims(
      (2 * p_total[0] - 5 * p_total[1] + 4 * p_total[2] - p_total[3])
      / (2 * psi[0] - 5 * psi[1] + 4 * psi[2] - psi[3]),
      axis=0,
  )

  # Zero on-axis due to boundary conditions. Avoid division by zero.
  pprime_face = jnp.concatenate(
      [pprime_face_axis, dptot_drhon[1:] / dpsi_drhon[1:]]
  )

  return pprime_face


def calc_FFprime(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.ArrayFloat:
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

  mu0 = constants.CONSTANTS.mu0
  pprime = calc_pprime(core_profiles)
  # g3 = <1/R^2>
  g3 = geo.g3_face
  jtor_over_R = core_profiles.currents.jtot_face / geo.R_major

  FFprime_face = -(jtor_over_R / (2 * jnp.pi) + pprime) * mu0 / g3
  return FFprime_face


def calculate_stored_thermal_energy(
    p_el: array_typing.ArrayFloat,
    p_ion: array_typing.ArrayFloat,
    p_tot: array_typing.ArrayFloat,
    geo: geometry.Geometry,
) -> tuple[array_typing.ScalarFloat, ...]:
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
  wth_el = _trapz(1.5 * p_el * geo.vpr_face, geo.rho_face_norm)
  wth_ion = _trapz(1.5 * p_ion * geo.vpr_face, geo.rho_face_norm)
  wth_tot = _trapz(1.5 * p_tot * geo.vpr_face, geo.rho_face_norm)

  return wth_el, wth_ion, wth_tot


def calculate_greenwald_fraction(
    n_e_avg: array_typing.ScalarFloat,
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.ScalarFloat:
  """Calculates the Greenwald fraction from the averaged electron density.

  Different averaging can be used, e.g. volume-averaged or line-averaged.

  Args:
    n_e_avg: Averaged electron density [density_reference m^-3]
    core_profiles: CoreProfiles object containing information on currents and
      densities.
    geo: Geometry object

  Returns:
    fgw: Greenwald density fraction
  """
  # gw_limit is in units of 10^20 m^-3 when Ip is in MA and Rmin is in m.
  gw_limit = core_profiles.currents.Ip_total * 1e-6 / (jnp.pi * geo.a_minor**2)
  fgw = n_e_avg * core_profiles.density_reference / (gw_limit * 1e20)
  return fgw
