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
from torax import math_utils
from torax import physics
from torax import state
from torax.sources import source_profiles

_trapz = jax.scipy.integrate.trapezoid

# TODO(b/376010694): use the various SOURCE_NAMES for the keys.
ION_EL_HEAT_SOURCE_TRANSFORMATIONS = {
    'generic_ion_el_heat_source': 'P_generic',
    'fusion_heat_source': 'P_alpha',
    'ion_cyclotron_source': 'P_icrh',
}
EL_HEAT_SOURCE_TRANSFORMATIONS = {
    'ohmic_heat_source': 'P_ohmic',
    'bremsstrahlung_heat_sink': 'P_brems',
    'electron_cyclotron_source': 'P_ecrh',
    'impurity_radiation_heat_sink': 'P_rad',
}
EXTERNAL_HEATING_SOURCES = [
    'generic_ion_el_heat_source',
    'electron_cyclotron_source',
    'ohmic_heat_source',
    'ion_cyclotron_source',
]
CURRENT_SOURCE_TRANSFORMATIONS = {
    'generic_current_source': 'I_generic',
    'electron_cyclotron_source': 'I_ecrh',
}


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


@jax_utils.jit
def _calculate_integrated_sources(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    core_sources: source_profiles.SourceProfiles,
) -> dict[str, jax.Array]:
  """Calculates total integrated internal and external source power and current.

  Args:
    geo: Magnetic geometry
    core_profiles: Kinetic profiles such as temperature and density
    core_sources: Internal and external sources

  Returns:
    Dictionary with integrated quantities for all existing sources.
    See state.PostProcessedOutputs for the full list of keys.

    Output dict used to update the `PostProcessedOutputs` object in the
    output `ToraxSimState`. Sources that don't exist do not have integrated
    quantities included in the returned dict. The corresponding
    `PostProcessedOutputs` attributes remain at their initialized zero values.
  """
  integrated = {}

  # Initialize total alpha power to zero. Needed for Q calculation.
  integrated['P_alpha_tot'] = jnp.array(0.0)

  # electron-ion heat exchange always exists, and is not in
  # core_sources.profiles, so we calculate it here.
  qei = core_sources.qei.qei_coef * (
      core_profiles.temp_el.value - core_profiles.temp_ion.value
  )
  integrated['P_ei_exchange_ion'] = math_utils.cell_integration(
      qei * geo.vpr, geo
  )
  integrated['P_ei_exchange_el'] = -integrated['P_ei_exchange_ion']

  # Initialize total electron and ion powers
  # TODO(b/380848256): P_sol is now correct for stationary state. However,
  # for generality need to add dWth/dt to the equation (time dependence of
  # stored energy).
  integrated['P_sol_ion'] = integrated['P_ei_exchange_ion']
  integrated['P_sol_el'] = integrated['P_ei_exchange_el']
  integrated['P_external_ion'] = jnp.array(0.0)
  integrated['P_external_el'] = jnp.array(0.0)

  # Calculate integrated sources with convenient names, transformed from
  # TORAX internal names.
  for key, value in ION_EL_HEAT_SOURCE_TRANSFORMATIONS.items():
    # Only populate integrated dict with sources that exist.
    if key in core_sources.profiles:
      profile_ion, profile_el = core_sources.profiles[key]
      integrated[f'{value}_ion'] = math_utils.cell_integration(
          profile_ion * geo.vpr, geo
      )
      integrated[f'{value}_el'] = math_utils.cell_integration(
          profile_el * geo.vpr, geo
      )
      integrated[f'{value}_tot'] = (
          integrated[f'{value}_ion'] + integrated[f'{value}_el']
      )
      integrated['P_sol_ion'] += integrated[f'{value}_ion']
      integrated['P_sol_el'] += integrated[f'{value}_el']
      if key in EXTERNAL_HEATING_SOURCES:
        integrated['P_external_ion'] += integrated[f'{value}_ion']
        integrated['P_external_el'] += integrated[f'{value}_el']

  for key, value in EL_HEAT_SOURCE_TRANSFORMATIONS.items():
    # Only populate integrated dict with sources that exist.
    if key in core_sources.profiles:
      # TODO(b/376010694): better automation of splitting profiles into
      # separate variables.
      # index 0 corresponds to the electron heating source profile.
      if key == 'electron_cyclotron_source':
        profile = core_sources.profiles[key][0, :]
      else:
        profile = core_sources.profiles[key]
      integrated[f'{value}'] = math_utils.cell_integration(
          profile * geo.vpr, geo
      )
      integrated['P_sol_el'] += integrated[f'{value}']
      if key in EXTERNAL_HEATING_SOURCES:
        integrated['P_external_el'] += integrated[f'{value}']

  for key, value in CURRENT_SOURCE_TRANSFORMATIONS.items():
    # Only populate integrated dict with sources that exist.
    if key in core_sources.profiles:
      # TODO(b/376010694): better automation of splitting profiles into
      # separate variables.
      # index 1 corresponds to the current source profile.
      if key == 'electron_cyclotron_source':
        profile = core_sources.profiles[key][1, :]
      elif key == 'generic_current_source':
        profile = geometry.face_to_cell(core_sources.profiles[key])
      else:
        profile = core_sources.profiles[key]
      integrated[f'{value}'] = math_utils.cell_integration(
          profile * geo.spr_cell, geo
      )

  integrated['P_sol_tot'] = integrated['P_sol_ion'] + integrated['P_sol_el']
  integrated['P_external_tot'] = (
      integrated['P_external_ion'] + integrated['P_external_el']
  )

  return integrated

@jax_utils.jit
def _calculate_q95(
    psi_norm_face: array_typing.ArrayFloat,
    core_profiles: state.CoreProfiles,
) -> tuple[array_typing.ScalarFloat]:
  """Calculates q95 from the q profile and the normalized poloidal flux.

  Args:
    psi_norm_face: normalized poloidal flux
    core_profiles: Kinetic profiles such as temperature and density

  Returns:
    q95: q at 95% of the normalized poloidal flux
  """
  # Calculate q at 95% of the normalized poloidal flux
  q95 = jnp.interp(0.95, psi_norm_face, core_profiles.q_face)

  return q95

def make_outputs(
    sim_state: state.ToraxSimState,
    geo: geometry.Geometry,
    previous_sim_state: state.ToraxSimState | None = None,
) -> state.ToraxSimState:
  """Calculates post-processed outputs based on the latest state.

  Called at the beginning and end of each `sim.run_simulation` step.
  Args:
    sim_state: The state to add outputs to.
    geo: Geometry object
    previous_sim_state: The previous state, used to calculate cumulative
      quantities. Optional input. If None, then cumulative quantities are set at
      the initialized values in sim_state itself. This is used for the first
      time step of a the simulation. The initialized values are zero for a clean
      simulation, or the last value of the previous simulation for a restarted
      simulation.

  Returns:
    sim_state: A ToraxSimState object, with any updated attributes.
  """
  (
      pressure_thermal_el_face,
      pressure_thermal_ion_face,
      pressure_thermal_tot_face,
  ) = _compute_pressure(sim_state.core_profiles)
  pprime_face = _compute_pprime(sim_state.core_profiles)
  # pylint: disable=invalid-name
  W_thermal_el, W_thermal_ion, W_thermal_tot = _compute_stored_thermal_energy(
      pressure_thermal_el_face,
      pressure_thermal_ion_face,
      pressure_thermal_tot_face,
      geo,
  )
  FFprime_face = _compute_FFprime(sim_state.core_profiles, geo)
  # Calculate normalized poloidal flux.
  psi_face = sim_state.core_profiles.psi.face_value()
  psi_norm_face = (psi_face - psi_face[0]) / (psi_face[-1] - psi_face[0])
  integrated_sources = _calculate_integrated_sources(
      geo,
      sim_state.core_profiles,
      sim_state.core_sources,
  )
  # Calculate fusion gain with a zero division guard.
  # Total energy released per reaction is 5 times the alpha particle energy.
  Q_fusion = (
      integrated_sources['P_alpha_tot']
      * 5.0
      / (integrated_sources['P_external_tot'] + constants.CONSTANTS.eps)
  )

  P_LH_hi_dens, P_LH_low_dens, ne_min_P_LH = (
      physics.calculate_plh_scaling_factor(geo, sim_state.core_profiles)
  )

  # Thermal energy confinement time is the stored energy divided by the total
  # input power into the plasma.

  # Ploss term here does not include the reduction of radiated power. Most
  # analysis of confinement times from databases have not included this term.
  # Therefore highly radiative scenarios can lead to skewed results.

  Ploss = (
      integrated_sources['P_alpha_tot'] + integrated_sources['P_external_tot']
  )
  # TODO(b/380848256): include dW/dt term
  tauE = W_thermal_tot / Ploss

  tauH98 = physics.calculate_scaling_law_confinement_time(
      geo, sim_state.core_profiles, Ploss/1e6, 'H98'
  )
  tauH97L = physics.calculate_scaling_law_confinement_time(
      geo, sim_state.core_profiles, Ploss/1e6, 'H97L'
  )
  tauH20 = physics.calculate_scaling_law_confinement_time(
      geo, sim_state.core_profiles, Ploss/1e6, 'H20'
  )

  H98 = tauE / tauH98
  H97L = tauE / tauH97L
  H20 = tauE / tauH20

  # Calculate total external (injected) and fusion (generated) energies based on
  # interval average.
  if previous_sim_state is not None:
    # Factor 5 due to including neutron energy: E_fusion = 5.0 * E_alpha
    E_cumulative_fusion = (
        previous_sim_state.post_processed_outputs.E_cumulative_fusion
        + 5.0
        * sim_state.dt
        * (
            integrated_sources['P_alpha_tot']
            + previous_sim_state.post_processed_outputs.P_alpha_tot
        )
        / 2.0
    )
    E_cumulative_external = (
        previous_sim_state.post_processed_outputs.E_cumulative_external
        + sim_state.dt
        * (
            integrated_sources['P_external_tot']
            + previous_sim_state.post_processed_outputs.P_external_tot
        )
        / 2.0
    )
  else:
    # First step of simulation, so no previous state. We set cumulative
    # quantities to whatever the initial_state was initialized to, which is
    # typically zero for a clean simulation, or the last value of the previous
    # simulation for a restarted simulation.
    E_cumulative_fusion = sim_state.post_processed_outputs.E_cumulative_fusion
    E_cumulative_external = (
        sim_state.post_processed_outputs.E_cumulative_external
    )
  
  # Calculate q at 95% of the normalized poloidal flux
  q95 = _calculate_q95(psi_norm_face, sim_state.core_profiles)

  # Calculate te and ti volume average [keV]
  te_volume_avg = math_utils.cell_integration(sim_state.core_profiles.temp_el.value*geo.vpr, geo) / geo.volume[-1]
  ti_volume_avg = math_utils.cell_integration(sim_state.core_profiles.temp_ion.value*geo.vpr, geo) / geo.volume[-1]

  # Calculate ne and ni (main ion) volume average [nref m^-3]
  ne_volume_avg = math_utils.cell_integration(sim_state.core_profiles.ne.value*geo.vpr, geo) / geo.volume[-1]
  ni_volume_avg = math_utils.cell_integration(sim_state.core_profiles.ni.value*geo.vpr, geo) / geo.volume[-1]

  # pylint: enable=invalid-name
  updated_post_processed_outputs = dataclasses.replace(
      sim_state.post_processed_outputs,
      pressure_thermal_ion_face=pressure_thermal_ion_face,
      pressure_thermal_el_face=pressure_thermal_el_face,
      pressure_thermal_tot_face=pressure_thermal_tot_face,
      pprime_face=pprime_face,
      W_thermal_ion=W_thermal_ion,
      W_thermal_el=W_thermal_el,
      W_thermal_tot=W_thermal_tot,
      tauE=tauE,
      H98=H98,
      H97L=H97L,
      H20=H20,
      FFprime_face=FFprime_face,
      psi_norm_face=psi_norm_face,
      psi_face=sim_state.core_profiles.psi.face_value(),
      **integrated_sources,
      Q_fusion=Q_fusion,
      P_LH_low_dens=P_LH_low_dens,
      P_LH_hi_dens=P_LH_hi_dens,
      ne_min_P_LH=ne_min_P_LH,
      E_cumulative_fusion=E_cumulative_fusion,
      E_cumulative_external=E_cumulative_external,
      te_volume_avg = te_volume_avg,
      ti_volume_avg = ti_volume_avg,
      ne_volume_avg = ne_volume_avg,
      ni_volume_avg = ni_volume_avg,
      q95 = q95,
  )
  # pylint: enable=invalid-name
  return dataclasses.replace(
      sim_state,
      post_processed_outputs=updated_post_processed_outputs,
  )
