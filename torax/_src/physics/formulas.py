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
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry

# pylint: disable=invalid-name


# TODO(b/377225415): generalize to arbitrary number of ions.
def calculate_main_ion_dilution_factor(
    Z_i: array_typing.ScalarFloat,
    Z_impurity: array_typing.ArrayFloat,
    Z_eff: array_typing.ArrayFloat,
) -> array_typing.ArrayFloat:
  """Calculates the main ion dilution factor based on a single assumed impurity and general main ion charge."""
  return (Z_impurity - Z_eff) / (Z_i * (Z_impurity - Z_i))


def calculate_pressure(
    core_profiles: state.CoreProfiles,
) -> tuple[cell_variable.CellVariable, ...]:
  """Calculates pressure from density and temperatures.

  Args:
    core_profiles: CoreProfiles object containing information on temperatures
      and densities.

  Returns:
    pressure_thermal_el: Electron thermal pressure [Pa]
    pressure_thermal_ion: Ion thermal pressure [Pa]
    pressure_thermal_tot: Total thermal pressure [Pa]
  """

  pressure_thermal_el = cell_variable.CellVariable(
      value=core_profiles.n_e.value
      * core_profiles.T_e.value
      * constants.CONSTANTS.keV2J,
      dr=core_profiles.n_e.dr,
      right_face_constraint=core_profiles.n_e.right_face_constraint
      * core_profiles.T_e.right_face_constraint
      * constants.CONSTANTS.keV2J,
      right_face_grad_constraint=None,
  )

  pressure_thermal_ion = cell_variable.CellVariable(
      value=core_profiles.T_i.value
      * constants.CONSTANTS.keV2J
      * (core_profiles.n_i.value + core_profiles.n_impurity.value),
      dr=core_profiles.n_i.dr,
      right_face_constraint=core_profiles.T_i.right_face_constraint
      * constants.CONSTANTS.keV2J
      * (
          core_profiles.n_i.right_face_constraint
          + core_profiles.n_impurity.right_face_constraint
      ),
      right_face_grad_constraint=None,
  )

  pressure_thermal_tot = cell_variable.CellVariable(
      value=pressure_thermal_el.value + pressure_thermal_ion.value,
      dr=pressure_thermal_el.dr,
      right_face_constraint=pressure_thermal_el.right_face_constraint
      + pressure_thermal_ion.right_face_constraint,
      right_face_grad_constraint=None,
  )

  return (
      pressure_thermal_el,
      pressure_thermal_ion,
      pressure_thermal_tot,
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

  _, _, p_total = calculate_pressure(core_profiles)
  psi = core_profiles.psi.face_value()
  n_e = core_profiles.n_e.face_value()
  n_i = core_profiles.n_i.face_value()
  n_impurity = core_profiles.n_impurity.face_value()
  T_i = core_profiles.T_i.face_value()
  T_e = core_profiles.T_e.face_value()
  dne_drhon = core_profiles.n_e.face_grad()
  dni_drhon = core_profiles.n_i.face_grad()
  dnimp_drhon = core_profiles.n_impurity.face_grad()
  dti_drhon = core_profiles.T_i.face_grad()
  dte_drhon = core_profiles.T_e.face_grad()
  dpsi_drhon = core_profiles.psi.face_grad()

  dptot_drhon = constants.CONSTANTS.keV2J * (
      n_e * dte_drhon
      + n_i * dti_drhon
      + n_impurity * dti_drhon
      + dne_drhon * T_e
      + dni_drhon * T_i
      + dnimp_drhon * T_i
  )

  p_total_face = p_total.face_value()

  # Calculate on-axis value with L'Hôpital's rule using 2nd order forward
  # difference approximation for second derivative at edge.
  pprime_face_axis = jnp.expand_dims(
      (
          2 * p_total_face[0]
          - 5 * p_total_face[1]
          + 4 * p_total_face[2]
          - p_total_face[3]
      )
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
  jtor_over_R = core_profiles.j_total_face / geo.R_major

  FFprime_face = -(jtor_over_R / (2 * jnp.pi) + pprime) * mu0 / g3
  return FFprime_face


def calculate_stored_thermal_energy(
    p_el: cell_variable.CellVariable,
    p_ion: cell_variable.CellVariable,
    p_tot: cell_variable.CellVariable,
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
  wth_el = math_utils.volume_integration(1.5 * p_el.value, geo)
  wth_ion = math_utils.volume_integration(1.5 * p_ion.value, geo)
  wth_tot = math_utils.volume_integration(1.5 * p_tot.value, geo)

  return wth_el, wth_ion, wth_tot


def calculate_greenwald_fraction(
    n_e_avg: array_typing.ScalarFloat,
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.ScalarFloat:
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
