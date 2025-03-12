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
from torax import constants
from torax import jax_utils
from torax import math_utils
from torax import state
from torax.geometry import geometry
from torax.physics import formulas
from torax.physics import psi_calculations
from torax.physics import scaling_laws
from torax.sources import source_profiles
from torax.config import runtime_params_slice
from torax.sources import generic_ion_el_heat_source
from torax.sources import ion_cyclotron_source

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
    'cyclotron_radiation_heat_sink': 'P_cycl',
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


def _calculate_integrated_sources(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    core_sources: source_profiles.SourceProfiles,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
) -> dict[str, jax.Array]:
  """Calculates total integrated internal and external source power and current.

  Args:
    geo: Magnetic geometry
    core_profiles: Kinetic profiles such as temperature and density
    core_sources: Internal and external sources
    dynamic_runtime_params_slice: Dynamic runtime parameters, used for power
      calculations.

  Returns:
    Dictionary with integrated quantities for all existing sources.
    See state.PostProcessedOutputs for the full list of keys.

    Output dict used to update the `PostProcessedOutputs` object in the
    output `ToraxSimState`. Sources that don't exist do not have integrated
    quantities included in the returned dict. The corresponding
    `PostProcessedOutputs` attributes remain at their initialized zero values.
  """
  integrated = {}

  # Calculate alpha power.
  if 'alpha_source' in core_sources.temp_ion:
    integrated['P_alpha_ion'] = math_utils.volume_integration(
        core_sources.temp_ion['alpha_source'], geo
    )
    integrated['P_alpha_el'] = math_utils.volume_integration(
        core_sources.temp_el['alpha_source'], geo
    )
    integrated['P_alpha_tot'] = integrated['P_alpha_ion'] + integrated['P_alpha_el']
  else:
    integrated['P_alpha_ion'] = jnp.array(0.0)
    integrated['P_alpha_el'] = jnp.array(0.0)
    integrated['P_alpha_tot'] = jnp.array(0.0)

  # Calculate electron-ion heat exchange.
  if 'ei_exchange' in core_sources.temp_ion:
    integrated['P_ei_exchange_ion'] = math_utils.volume_integration(
        core_sources.temp_ion['ei_exchange'], geo
    )
    integrated['P_ei_exchange_el'] = math_utils.volume_integration(
        core_sources.temp_el['ei_exchange'], geo
    )
  else:
    # For backward compatibility with older code that uses qei
    qei = core_sources.qei.qei_coef * (
        core_profiles.temp_el.value - core_profiles.temp_ion.value
    )
    integrated['P_ei_exchange_ion'] = math_utils.volume_integration(qei, geo)
    integrated['P_ei_exchange_el'] = -integrated['P_ei_exchange_ion']
    
  integrated['P_sol_ion'] = integrated['P_ei_exchange_ion']
  integrated['P_sol_el'] = integrated['P_ei_exchange_el']
  integrated['P_external_ion'] = jnp.array(0.0)
  integrated['P_external_el'] = jnp.array(0.0)
  integrated['P_external_injected'] = jnp.array(0.0)

  # Calculate integrated sources with convenient names, transformed from
  # TORAX internal names.
  for key, value in ION_EL_HEAT_SOURCE_TRANSFORMATIONS.items():
    # Only populate integrated dict with sources that exist.
    ion_profiles = core_sources.temp_ion
    el_profiles = core_sources.temp_el
    if key in ion_profiles and key in el_profiles:
      profile_ion, profile_el = ion_profiles[key], el_profiles[key]
      integrated[f'{value}_ion'] = math_utils.volume_integration(
          profile_ion, geo
      )
      integrated[f'{value}_el'] = math_utils.volume_integration(
          profile_el, geo
      )
      integrated[f'{value}_tot'] = (
          integrated[f'{value}_ion'] + integrated[f'{value}_el']
      )
      integrated['P_sol_ion'] += integrated[f'{value}_ion']
      integrated['P_sol_el'] += integrated[f'{value}_el']
      if key in EXTERNAL_HEATING_SOURCES:
        integrated['P_external_ion'] += integrated[f'{value}_ion']
        integrated['P_external_el'] += integrated[f'{value}_el']
        
        # Track injected power for heating sources that have absorption_fraction
        if key in ['generic_ion_el_heat_source', 'ion_cyclotron_source']:
          # Get the absorption_fraction from dynamic params
          source_params = dynamic_runtime_params_slice.sources.get(key)
          
          if source_params and hasattr(source_params, 'absorption_fraction'):
            # Calculate injected power based on absorption_fraction
            total_absorbed = integrated[f'{value}_ion'] + integrated[f'{value}_el']
            absorption_fraction = source_params.absorption_fraction
            injected_power = total_absorbed / jnp.maximum(absorption_fraction, constants.CONSTANTS.eps)
            integrated['P_external_injected'] += injected_power
          else:
            # If no absorption_fraction is found, injected power equals absorbed power
            integrated['P_external_injected'] += integrated[f'{value}_ion'] + integrated[f'{value}_el']
        else:
          integrated['P_external_injected'] += integrated[f'{value}_ion'] + integrated[f'{value}_el']

  for key, value in EL_HEAT_SOURCE_TRANSFORMATIONS.items():
    # Only populate integrated dict with sources that exist.
    profiles = core_sources.temp_el
    if key in profiles:
      integrated[f'{value}'] = math_utils.volume_integration(profiles[key], geo)
      integrated['P_sol_el'] += integrated[f'{value}']
      if key in EXTERNAL_HEATING_SOURCES:
        integrated['P_external_el'] += integrated[f'{value}']
        integrated['P_external_injected'] += integrated[f'{value}']

  for key, value in CURRENT_SOURCE_TRANSFORMATIONS.items():
    # Only populate integrated dict with sources that exist.
    profiles = core_sources.psi
    if key in profiles:
      integrated[f'{value}'] = math_utils.area_integration(profiles[key], geo)

  integrated['P_sol_tot'] = integrated['P_sol_ion'] + integrated['P_sol_el']
  integrated['P_external_tot'] = (
      integrated['P_external_ion'] + integrated['P_external_el']
  )

  return integrated


@jax_utils.jit
def make_outputs(
    sim_state: state.ToraxSimState,
    geo: geometry.Geometry,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    previous_sim_state: state.ToraxSimState | None = None,
) -> state.ToraxSimState:
  """Calculates post-processed outputs based on the latest state.

  Called at the beginning and end of each `sim.run_simulation` step.
  Args:
    sim_state: The state to add outputs to.
    geo: Geometry object
    dynamic_runtime_params_slice: Dynamic runtime parameters used for power
      calculations.
    previous_sim_state: The previous state, used to calculate cumulative
      quantities. Optional input. If None, then cumulative quantities are set at
      the initialized values in sim_state itself. This is used for the first
      step of a simulation.

  Returns:
    Updated state with post-processed outputs.
  """
  (
      pressure_thermal_el_face,
      pressure_thermal_ion_face,
      pressure_thermal_tot_face,
  ) = formulas.calculate_pressure(sim_state.core_profiles)
  pprime_face = formulas.calc_pprime(sim_state.core_profiles)
  # pylint: disable=invalid-name
  W_thermal_el, W_thermal_ion, W_thermal_tot = (
      formulas.calculate_stored_thermal_energy(
          pressure_thermal_el_face,
          pressure_thermal_ion_face,
          pressure_thermal_tot_face,
          geo,
      )
  )
  FFprime_face = formulas.calc_FFprime(sim_state.core_profiles, geo)
  # Calculate normalized poloidal flux.
  psi_face = sim_state.core_profiles.psi.face_value()
  psi_norm_face = (psi_face - psi_face[0]) / (psi_face[-1] - psi_face[0])
  integrated_sources = _calculate_integrated_sources(
      geo,
      sim_state.core_profiles,
      sim_state.core_sources,
      dynamic_runtime_params_slice,
  )
  # Calculate fusion gain with a zero division guard.
  # Total energy released per reaction is 5 times the alpha particle energy.
  Q_fusion = (
      integrated_sources['P_alpha_tot']
      * 5.0
      / (integrated_sources['P_external_injected'] + constants.CONSTANTS.eps)
  )

  P_LH_hi_dens, P_LH_min, P_LH, ne_min_P_LH = (
      scaling_laws.calculate_plh_scaling_factor(geo, sim_state.core_profiles)
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

  tauH89P = scaling_laws.calculate_scaling_law_confinement_time(
      geo, sim_state.core_profiles, Ploss, 'H89P'
  )
  tauH98 = scaling_laws.calculate_scaling_law_confinement_time(
      geo, sim_state.core_profiles, Ploss, 'H98'
  )
  tauH97L = scaling_laws.calculate_scaling_law_confinement_time(
      geo, sim_state.core_profiles, Ploss, 'H97L'
  )
  tauH20 = scaling_laws.calculate_scaling_law_confinement_time(
      geo, sim_state.core_profiles, Ploss, 'H20'
  )

  H89P = tauE / tauH89P
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
  q95 = psi_calculations.calc_q95(psi_norm_face, sim_state.core_profiles.q_face)

  # Calculate te and ti volume average [keV]
  te_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.temp_el.value, geo
  )
  ti_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.temp_ion.value, geo
  )

  # Calculate ne and ni (main ion) volume and line averages [nref m^-3]
  ne_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.ne.value, geo
  )
  ni_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.ni.value, geo
  )
  ne_line_avg = math_utils.line_average(sim_state.core_profiles.ne.value, geo)
  ni_line_avg = math_utils.line_average(sim_state.core_profiles.ni.value, geo)
  fgw_ne_volume_avg = formulas.calculate_greenwald_fraction(
      ne_volume_avg, sim_state.core_profiles, geo
  )
  fgw_ne_line_avg = formulas.calculate_greenwald_fraction(
      ne_line_avg, sim_state.core_profiles, geo
  )
  Wpol = psi_calculations.calc_Wpol(geo, sim_state.core_profiles.psi)
  li3 = psi_calculations.calc_li3(
      geo.Rmaj, Wpol, sim_state.core_profiles.currents.Ip_profile_face[-1]
  )

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
      H89P=H89P,
      H98=H98,
      H97L=H97L,
      H20=H20,
      FFprime_face=FFprime_face,
      psi_norm_face=psi_norm_face,
      psi_face=sim_state.core_profiles.psi.face_value(),
      **integrated_sources,
      Q_fusion=Q_fusion,
      P_LH=P_LH,
      P_LH_min=P_LH_min,
      P_LH_hi_dens=P_LH_hi_dens,
      ne_min_P_LH=ne_min_P_LH,
      E_cumulative_fusion=E_cumulative_fusion,
      E_cumulative_external=E_cumulative_external,
      te_volume_avg=te_volume_avg,
      ti_volume_avg=ti_volume_avg,
      ne_volume_avg=ne_volume_avg,
      ni_volume_avg=ni_volume_avg,
      ne_line_avg=ne_line_avg,
      ni_line_avg=ni_line_avg,
      fgw_ne_volume_avg=fgw_ne_volume_avg,
      fgw_ne_line_avg=fgw_ne_line_avg,
      q95=q95,
      Wpol=Wpol,
      li3=li3,
  )
  # pylint: enable=invalid-name
  return dataclasses.replace(
      sim_state,
      post_processed_outputs=updated_post_processed_outputs,
  )
