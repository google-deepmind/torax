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

"""Functions for adding post-processed outputs to the simulation state."""

# In torax/post_processing.py

import dataclasses
import jax
from jax import numpy as jnp
from torax import array_typing
from torax import constants
from torax import geometry
from torax import jax_utils
from torax import state

_trapz = jax.scipy.integrate.trapezoid


@jax_utils.jit
def _compute_pressure(
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
  ne = core_profiles.ne.face_value()
  ni = core_profiles.ni.face_value()
  nimp = core_profiles.nimp.face_value()
  temp_ion = core_profiles.temp_ion.face_value()
  temp_el = core_profiles.temp_el.face_value()
  prefactor = constants.CONSTANTS.keV2J * core_profiles.nref
  pressure_thermal_el_face = ne * temp_el * prefactor
  pressure_thermal_ion_face = (ni + nimp) * temp_ion * prefactor
  pressure_thermal_tot_face = (
      pressure_thermal_el_face + pressure_thermal_ion_face
  )
  return (
      pressure_thermal_el_face,
      pressure_thermal_ion_face,
      pressure_thermal_tot_face,
  )


@jax_utils.jit
def _compute_pprime(
    core_profiles: state.CoreProfiles,
) -> array_typing.ArrayFloat:
  r"""Calculates total pressure gradient with respect to poloidal flux.

  Args:
    core_profiles: CoreProfiles object containing information on temperatures
      and densities.

  Returns:
    pprime: Total pressure gradient `\partial p / \partial \psi`
      with respect to the normalized toroidal flux coordinate, on the face grid.
  """

  prefactor = constants.CONSTANTS.keV2J * core_profiles.nref

  ne = core_profiles.ne.face_value()
  ni = core_profiles.ni.face_value()
  nimp = core_profiles.nimp.face_value()
  temp_ion = core_profiles.temp_ion.face_value()
  temp_el = core_profiles.temp_el.face_value()
  dne_drhon = core_profiles.ne.face_grad()
  dni_drhon = core_profiles.ni.face_grad()
  dnimp_drhon = core_profiles.nimp.face_grad()
  dti_drhon = core_profiles.temp_ion.face_grad()
  dte_drhon = core_profiles.temp_el.face_grad()
  dpsi_drhon = core_profiles.psi.face_grad()

  dptot_drhon = prefactor * (
      ne * dte_drhon
      + ni * dti_drhon
      + nimp * dti_drhon
      + dne_drhon * temp_el
      + dni_drhon * temp_ion
      + dnimp_drhon * temp_ion
  )
  # Calculate on-axis value with L'Hôpital's rule.
  pprime_face_axis = jnp.expand_dims(dptot_drhon[1] / dpsi_drhon[1], axis=0)

  # Zero on-axis due to boundary conditions. Avoid division by zero.
  pprime_face = jnp.concatenate(
      [pprime_face_axis, dptot_drhon[1:] / dpsi_drhon[1:]]
  )

  return pprime_face


# pylint: disable=invalid-name
@jax_utils.jit
def _compute_FFprime(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.ArrayFloat:
  r"""Calculates FF', an output quantity used for equilibrium coupling.

  Calculation is based on the following formulation of the magnetic
  equilibrium equation:
  ```
  -j_{tor} = 2\pi (Rp' + \frac{1}{\mu_0 R}FF')

  And following division by R and flux surface averaging:

  -\langle \frac{j_{tor}}{R} \rangle = 2\pi (p' +
  \langle\frac{1}{R^2}\rangle\frac{FF'}{\mu_0})
  ```

  Args:
    core_profiles: CoreProfiles object containing information on temperatures
      and densities.
    geo: Magnetic equilibrium.

  Returns:
    FFprime:   F is the toroidal flux function, and F' is its derivative with
      respect to the poloidal flux.
  """

  mu0 = constants.CONSTANTS.mu0
  pprime = _compute_pprime(core_profiles)
  # g3 = <1/R^2>
  g3 = geo.g3_face
  jtor_over_R = core_profiles.currents.jtot_face / geo.Rmaj

  FFprime_face = -(jtor_over_R / (2 * jnp.pi) + pprime) * mu0 / g3
  return FFprime_face


# pylint: enable=invalid-name


@jax_utils.jit
def _compute_stored_thermal_energy(
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


def make_outputs(
    sim_state: state.ToraxSimState, geo: geometry.Geometry
) -> state.ToraxSimState:
  """Calculates post-processed outputs based on the latest state.

  Called at the beginning and end of each `sim.run_simulation` step.
  Args:
    sim_state: The state to add outputs to.
    geo: Geometry object

  Returns:
    sim_state: A ToraxSimState object, with any updated attributes.
  """
  (
      pressure_thermal_el_face,
      pressure_thermal_ion_face,
      pressure_thermal_tot_face,
  ) = _compute_pressure(sim_state.core_profiles)
  pprime_face = _compute_pprime(sim_state.core_profiles)
  wth_el, wth_ion, wth_tot = _compute_stored_thermal_energy(
      pressure_thermal_el_face,
      pressure_thermal_ion_face,
      pressure_thermal_tot_face,
      geo,
  )
  # pylint: disable=invalid-name
  FFprime_face = _compute_FFprime(sim_state.core_profiles, geo)
  # pylint: enable=invalid-name
  # Calculate normalized poloidal flux.
  psi_face = sim_state.core_profiles.psi.face_value()
  psi_norm_face = (psi_face - psi_face[0]) / (psi_face[-1] - psi_face[0])
  return dataclasses.replace(
      sim_state,
      post_processed_outputs=state.PostProcessedOutputs(
          pressure_thermal_ion_face=pressure_thermal_ion_face,
          pressure_thermal_el_face=pressure_thermal_el_face,
          pressure_thermal_tot_face=pressure_thermal_tot_face,
          pprime_face=pprime_face,
          wth_thermal_ion=wth_ion,
          wth_thermal_el=wth_el,
          wth_thermal_tot=wth_tot,
          FFprime_face=FFprime_face,
          psi_norm_face=psi_norm_face,
          psi_face=sim_state.core_profiles.psi.face_value(),
      ),
  )
