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

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import jax
from jax import numpy as jnp
import numpy as np

from torax import array_typing
from torax import constants
from torax import jax_utils
from torax import math_utils
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.physics import formulas
from torax.physics import psi_calculations
from torax.physics import scaling_laws
from torax.sources import source_profiles
from torax.sources import generic_ion_el_heat_source
from torax.config import runtime_params_slice

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
    core_sources: source_profiles.SourceProfiles,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice | None = None,
) -> dict[str, jax.Array]:
  """Calculates integrated sources based on the latest state.

  Args:
    geo: Magnetic geometry.
    core_sources: Internal and external sources.
    dynamic_runtime_params_slice: Dynamic runtime parameters, used for power
      calculations. If None, certain calculations that require it will be skipped.

  Returns:
    Dictionary with integrated quantities for all existing sources.
  """
  integrated = {}

  # Calculate alpha power.
  if hasattr(core_sources, 'temp_ion') and 'fusion_heat_source' in core_sources.temp_ion:
    integrated['P_alpha_ion'] = math_utils.volume_integration(
        core_sources.temp_ion['fusion_heat_source'], geo
    )
    integrated['P_alpha_el'] = math_utils.volume_integration(
        core_sources.temp_el['fusion_heat_source'], geo
    )
    integrated['P_alpha_tot'] = integrated['P_alpha_ion'] + integrated['P_alpha_el']
  else:
    integrated['P_alpha_ion'] = jnp.array(0.0)
    integrated['P_alpha_el'] = jnp.array(0.0)
    integrated['P_alpha_tot'] = jnp.array(0.0)

  # Initialize values for heat exchange
  integrated['P_ei_exchange_ion'] = jnp.array(0.0)
  integrated['P_ei_exchange_el'] = jnp.array(0.0)

  # Calculate electron-ion heat exchange if available
  if hasattr(core_sources, 'temp_ion') and 'ei_exchange' in core_sources.temp_ion:
    integrated['P_ei_exchange_ion'] = math_utils.volume_integration(
        core_sources.temp_ion['ei_exchange'], geo
    )
    integrated['P_ei_exchange_el'] = math_utils.volume_integration(
        core_sources.temp_el['ei_exchange'], geo
    )

  # Initialize other values
  integrated['P_sol_ion'] = integrated['P_alpha_ion'] + integrated['P_ei_exchange_ion']
  integrated['P_sol_el'] = integrated['P_alpha_el'] + integrated['P_ei_exchange_el']
  integrated['P_sol_tot'] = integrated['P_sol_ion'] + integrated['P_sol_el']
  integrated['P_external_ion'] = jnp.array(0.0)
  integrated['P_external_el'] = jnp.array(0.0)
  integrated['P_external_tot'] = jnp.array(0.0)
  integrated['P_external_injected'] = jnp.array(0.0)
  integrated['P_generic_injected'] = jnp.array(0.0)
  integrated['I_tot'] = jnp.array(0.0)
  integrated['I_ecrh'] = jnp.array(0.0)
  integrated['I_generic'] = jnp.array(0.0)
  
  # Calculate brems power if available
  if hasattr(core_sources, 'temp_el') and 'bremsstrahlung_heat_sink' in core_sources.temp_el:
    integrated['P_brems'] = math_utils.volume_integration(
        core_sources.temp_el['bremsstrahlung_heat_sink'], geo
    )
    integrated['P_sol_el'] += integrated['P_brems']
  else:
    integrated['P_brems'] = jnp.array(0.0)
    
  # Calculate cyclotron power if available
  if hasattr(core_sources, 'temp_el') and 'cyclotron_radiation_heat_sink' in core_sources.temp_el:
    integrated['P_cycl'] = math_utils.volume_integration(
        core_sources.temp_el['cyclotron_radiation_heat_sink'], geo
    )
    integrated['P_sol_el'] += integrated['P_cycl']
  else:
    integrated['P_cycl'] = jnp.array(0.0)
    
  # Calculate ohmic power if available
  if hasattr(core_sources, 'temp_el') and 'ohmic_heat_source' in core_sources.temp_el:
    integrated['P_ohmic'] = math_utils.volume_integration(
        core_sources.temp_el['ohmic_heat_source'], geo
    )
    integrated['P_sol_el'] += integrated['P_ohmic']
    integrated['P_external_el'] += integrated['P_ohmic']
    integrated['P_external_tot'] += integrated['P_ohmic']
  else:
    integrated['P_ohmic'] = jnp.array(0.0)
    
  # Calculate ecrh power if available
  if hasattr(core_sources, 'temp_el') and 'electron_cyclotron_source' in core_sources.temp_el:
    integrated['P_ecrh'] = math_utils.volume_integration(
        core_sources.temp_el['electron_cyclotron_source'], geo
    )
    integrated['P_sol_el'] += integrated['P_ecrh']
    integrated['P_external_el'] += integrated['P_ecrh']
    integrated['P_external_tot'] += integrated['P_ecrh']
    integrated['P_external_injected'] += integrated['P_ecrh']
  else:
    integrated['P_ecrh'] = jnp.array(0.0)
    
  # Calculate generic heat source powers if available
  if hasattr(core_sources, 'temp_ion') and 'generic_ion_el_heat_source' in core_sources.temp_ion:
    integrated['P_generic_ion'] = math_utils.volume_integration(
        core_sources.temp_ion['generic_ion_el_heat_source'], geo
    )
    integrated['P_sol_ion'] += integrated['P_generic_ion']
    integrated['P_external_ion'] += integrated['P_generic_ion']
    integrated['P_external_tot'] += integrated['P_generic_ion']
  else:
    integrated['P_generic_ion'] = jnp.array(0.0)
    
  if hasattr(core_sources, 'temp_el') and 'generic_ion_el_heat_source' in core_sources.temp_el:
    integrated['P_generic_el'] = math_utils.volume_integration(
        core_sources.temp_el['generic_ion_el_heat_source'], geo
    )
    integrated['P_sol_el'] += integrated['P_generic_el']
    integrated['P_external_el'] += integrated['P_generic_el']
    integrated['P_external_tot'] += integrated['P_generic_el']
  else:
    integrated['P_generic_el'] = jnp.array(0.0)
    
  integrated['P_generic_tot'] = integrated['P_generic_ion'] + integrated['P_generic_el']
  
  # Add external powers
  if dynamic_runtime_params_slice is not None and hasattr(dynamic_runtime_params_slice, 'sources'):
    # Calculate injected power from dynamic runtime params if available
    for source_name, source_config in dynamic_runtime_params_slice.sources.items():
      if source_name == generic_ion_el_heat_source.GenericIonElHeatSource.SOURCE_NAME:
        if hasattr(source_config, 'injected_power'):
          integrated['P_generic_injected'] = source_config.injected_power
          integrated['P_external_injected'] += source_config.injected_power
  
  # Calculate currents if available
  if hasattr(core_sources, 'psi'):
    if 'electron_cyclotron_source' in core_sources.psi:
      integrated['I_ecrh'] = math_utils.volume_integration(
          core_sources.psi['electron_cyclotron_source'], geo
      )
      integrated['I_tot'] += integrated['I_ecrh']
      
    if 'generic_current_source' in core_sources.psi:
      integrated['I_generic'] = math_utils.volume_integration(
          core_sources.psi['generic_current_source'], geo


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
      integrated[f'{value}_el'] = math_utils.volume_integration(profile_el, geo)
      integrated[f'{value}_tot'] = (
          integrated[f'{value}_ion'] + integrated[f'{value}_el']

      )
      integrated['I_tot'] += integrated['I_generic']
  
  # Calculate ICRH powers if available
  integrated['P_icrh_ion'] = jnp.array(0.0)
  integrated['P_icrh_el'] = jnp.array(0.0)
  integrated['P_icrh_tot'] = jnp.array(0.0)
  
  if hasattr(core_sources, 'temp_ion') and 'ion_cyclotron_source' in core_sources.temp_ion:
    integrated['P_icrh_ion'] = math_utils.volume_integration(
        core_sources.temp_ion['ion_cyclotron_source'], geo
    )
    integrated['P_sol_ion'] += integrated['P_icrh_ion']
    integrated['P_external_ion'] += integrated['P_icrh_ion']
    integrated['P_external_tot'] += integrated['P_icrh_ion']
    integrated['P_external_injected'] += integrated['P_icrh_ion']
    
  if hasattr(core_sources, 'temp_el') and 'ion_cyclotron_source' in core_sources.temp_el:
    integrated['P_icrh_el'] = math_utils.volume_integration(
        core_sources.temp_el['ion_cyclotron_source'], geo
    )
    integrated['P_sol_el'] += integrated['P_icrh_el']
    integrated['P_external_el'] += integrated['P_icrh_el']
    integrated['P_external_tot'] += integrated['P_icrh_el']
    integrated['P_external_injected'] += integrated['P_icrh_el']
    
  integrated['P_icrh_tot'] = integrated['P_icrh_ion'] + integrated['P_icrh_el']
  
  # Update total sol power
  integrated['P_sol_tot'] = integrated['P_sol_ion'] + integrated['P_sol_el']
  
  return integrated


@jax_utils.jit
def make_outputs(
    sim_state: state.ToraxSimState,
    geo: geometry.Geometry,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice | None = None,
) -> state.ToraxSimState:
  """Calculates integrated sources based on the latest state.

  This function does all the post-processing calculations to make various outputs
  from the simulation that can be useful when analyzing the output later.

  Args:
    sim_state: The current simulation state (after the core has been evolved).
    geo: Geometry of the torus.
    dynamic_runtime_params_slice: Slice of dynamic runtime parameters, which is
      used for fusion power calculation. If None, fusion power calculation is
      skipped.

  Returns:
    Updated state with post-processed outputs.
  """
  # Calculate integrated source powers in the core.
  integrated_sources = _calculate_integrated_sources(
      geo,
      sim_state.core_sources,

      dynamic_runtime_params_slice,

  )
  # Calculate fusion gain with a zero division guard.
  # Total energy released per reaction is 5 times the alpha particle energy.
  Q_fusion = (
      integrated_sources['P_alpha_tot']
      * 5.0
      / (integrated_sources['P_external_tot'] + constants.CONSTANTS.eps)
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

  if previous_sim_state is not None:
    dW_th_dt = (
        W_thermal_tot - previous_sim_state.post_processed_outputs.W_thermal_tot
    ) / sim_state.dt
  else:
    dW_th_dt = 0.0

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

  # Calculate Ohmic energy.
  if hasattr(sim_state.core_profiles, 'j_tot') and hasattr(sim_state.core_profiles, 'e_field_parallel'):
    e_field = sim_state.core_profiles.e_field_parallel  # [V/m]
    j_tot = sim_state.core_profiles.j_tot  # [A/m^2]
    p_ohm = e_field * j_tot  # [W/m^3]

    # Integrate Ohmic power over volume.
    ohmic_power = jnp.trapz(
        p_ohm * geo.vpr, geo.rho
    )  # int[p_ohm * dV] = [W]
  else:
    ohmic_power = 0.0

  # Calculate Q_fusion
  if hasattr(sim_state.core_profiles, 'temp_ion') and dynamic_runtime_params_slice is not None:
    # Alpha energy is 1/5 of the total fusion energy
    # 5 times the alpha energy gives the total energy released per reaction
    # Q_fusion is the ratio of fusion power to input power
    # Note: P_fusion = 5 * P_alpha_heating
    if hasattr(sim_state.post_processed_outputs, 'P_alpha_heating') and hasattr(sim_state.post_processed_outputs, 'P_ext'):
      P_alpha_heating = sim_state.post_processed_outputs.P_alpha_heating
      P_ext = sim_state.post_processed_outputs.P_ext
      Q_fusion = 5.0 * P_alpha_heating / jnp.where(P_ext > 0, P_ext, 1.0)
    else:
      Q_fusion = 0.0
  else:
    Q_fusion = 0.0

  # Create the updated outputs.
  outputs = state.PostProcessedOutputs(
      # Initialize with zeros for fields that might not be calculated
      pressure_thermal_ion_face=sim_state.post_processed_outputs.pressure_thermal_ion_face,
      pressure_thermal_el_face=sim_state.post_processed_outputs.pressure_thermal_el_face,
      pressure_thermal_tot_face=sim_state.post_processed_outputs.pressure_thermal_tot_face,
      pprime_face=sim_state.post_processed_outputs.pprime_face,
      W_thermal_ion=sim_state.post_processed_outputs.W_thermal_ion,
      W_thermal_el=sim_state.post_processed_outputs.W_thermal_el,
      W_thermal_tot=sim_state.post_processed_outputs.W_thermal_tot,
      tauE=sim_state.post_processed_outputs.tauE,
      H89P=sim_state.post_processed_outputs.H89P,
      H98=sim_state.post_processed_outputs.H98,
      H97L=sim_state.post_processed_outputs.H97L,
      H20=sim_state.post_processed_outputs.H20,
      FFprime_face=sim_state.post_processed_outputs.FFprime_face,
      psi_norm_face=sim_state.post_processed_outputs.psi_norm_face,
      psi_face=sim_state.post_processed_outputs.psi_face,
      # Integrated sources
      P_sol_ion=integrated_sources.get('P_sol_ion', jnp.array(0.0)),
      P_sol_el=integrated_sources.get('P_sol_el', jnp.array(0.0)),
      P_sol_tot=integrated_sources.get('P_sol_ion', jnp.array(0.0)) + integrated_sources.get('P_sol_el', jnp.array(0.0)),
      P_external_ion=integrated_sources.get('P_external_ion', jnp.array(0.0)),
      P_external_el=integrated_sources.get('P_external_el', jnp.array(0.0)),
      P_external_tot=integrated_sources.get('P_external_tot', jnp.array(0.0)),
      P_external_injected=integrated_sources.get('P_external_injected', jnp.array(0.0)),
      P_ei_exchange_ion=integrated_sources.get('P_ei_exchange_ion', jnp.array(0.0)),
      P_ei_exchange_el=integrated_sources.get('P_ei_exchange_el', jnp.array(0.0)),
      P_generic_ion=integrated_sources.get('P_generic_ion', jnp.array(0.0)),
      P_generic_el=integrated_sources.get('P_generic_el', jnp.array(0.0)),
      P_generic_tot=integrated_sources.get('P_generic_tot', jnp.array(0.0)),
      P_alpha_ion=integrated_sources.get('P_alpha_ion', jnp.array(0.0)),
      P_alpha_el=integrated_sources.get('P_alpha_el', jnp.array(0.0)),
      P_alpha_tot=integrated_sources.get('P_alpha_tot', jnp.array(0.0)),
      P_ohmic=ohmic_power,
      P_brems=integrated_sources.get('P_brems', jnp.array(0.0)),
      P_cycl=integrated_sources.get('P_cycl', jnp.array(0.0)),
      P_ecrh=integrated_sources.get('P_ecrh', jnp.array(0.0)),
      P_rad=integrated_sources.get('P_rad', jnp.array(0.0)),
      I_ecrh=integrated_sources.get('I_ecrh', jnp.array(0.0)),
      I_generic=integrated_sources.get('I_generic', jnp.array(0.0)),
      Q_fusion=Q_fusion,

      P_icrh_ion=integrated_sources.get('P_icrh_ion', jnp.array(0.0)),
      P_icrh_el=integrated_sources.get('P_icrh_el', jnp.array(0.0)),
      P_icrh_tot=integrated_sources.get('P_icrh_tot', jnp.array(0.0)),
      P_LH_hi_dens=sim_state.post_processed_outputs.P_LH_hi_dens,
      P_LH_min=sim_state.post_processed_outputs.P_LH_min,
      P_LH=sim_state.post_processed_outputs.P_LH,
      ne_min_P_LH=sim_state.post_processed_outputs.ne_min_P_LH,
      E_cumulative_fusion=sim_state.post_processed_outputs.E_cumulative_fusion,
      E_cumulative_external=sim_state.post_processed_outputs.E_cumulative_external,
      te_volume_avg=sim_state.post_processed_outputs.te_volume_avg,
      ti_volume_avg=sim_state.post_processed_outputs.ti_volume_avg,
      ne_volume_avg=sim_state.post_processed_outputs.ne_volume_avg,
      ni_volume_avg=sim_state.post_processed_outputs.ni_volume_avg,
      ne_line_avg=sim_state.post_processed_outputs.ne_line_avg,
      ni_line_avg=sim_state.post_processed_outputs.ni_line_avg,
      fgw_ne_volume_avg=sim_state.post_processed_outputs.fgw_ne_volume_avg,
      fgw_ne_line_avg=sim_state.post_processed_outputs.fgw_ne_line_avg,
      q95=sim_state.post_processed_outputs.q95,
      Wpol=sim_state.post_processed_outputs.Wpol,
      li3=sim_state.post_processed_outputs.li3,
      P_generic_injected=integrated_sources.get('P_generic_injected', jnp.array(0.0)),
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
      dW_th_dt=dW_th_dt,

  )

  return state.ToraxSimState(
      core_profiles=sim_state.core_profiles,
      core_transport=sim_state.core_transport,
      core_sources=sim_state.core_sources,
      post_processed_outputs=outputs,
      t=sim_state.t,
      dt=sim_state.dt,
      time_step_calculator_state=sim_state.time_step_calculator_state,
      stepper_numeric_outputs=sim_state.stepper_numeric_outputs,
      geometry=sim_state.geometry,
  )
