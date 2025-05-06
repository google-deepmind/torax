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

import jax
from jax import numpy as jnp
from torax import constants
from torax import jax_utils
from torax import math_utils
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.output_tools import safety_factor_fit
from torax.physics import formulas
from torax.physics import psi_calculations
from torax.physics import scaling_laws
from torax.sources import source_profiles

_trapz = jax.scipy.integrate.trapezoid

# TODO(b/376010694): use the various SOURCE_NAMES for the keys.
ION_EL_HEAT_SOURCE_TRANSFORMATIONS = {
    'generic_heat': 'P_aux_generic',
    'fusion': 'P_alpha',
    'icrh': 'P_icrh',
}
EL_HEAT_SOURCE_TRANSFORMATIONS = {
    'ohmic': 'P_ohmic_e',
    'bremsstrahlung': 'P_bremsstrahlung_e',
    'cyclotron_radiation': 'P_cyclotron_e',
    'ecrh': 'P_ecrh_e',
    'impurity_radiation': 'P_radiation_e',
}
EXTERNAL_HEATING_SOURCES = [
    'generic_heat',
    'ecrh',
    'ohmic',
    'icrh',
]
CURRENT_SOURCE_TRANSFORMATIONS = {
    'generic_current': 'I_aux_generic',
    'ecrh': 'I_ecrh',
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
    dynamic_runtime_params_slice: Runtime parameters slice for current time step

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
  integrated['P_alpha_total'] = jnp.array(0.0, dtype=jax_utils.get_dtype())

  # electron-ion heat exchange always exists, and is not in
  # core_sources.profiles, so we calculate it here.
  qei = core_sources.qei.qei_coef * (
      core_profiles.temp_el.value - core_profiles.temp_ion.value
  )
  integrated['P_ei_exchange_i'] = math_utils.volume_integration(qei, geo)
  integrated['P_ei_exchange_e'] = -integrated['P_ei_exchange_i']

  # Initialize total electron and ion powers
  # TODO(b/380848256): P_sol is now correct for stationary state. However,
  # for generality need to add dWth/dt to the equation (time dependence of
  # stored energy).
  integrated['P_SOL_i'] = integrated['P_ei_exchange_i']
  integrated['P_SOL_e'] = integrated['P_ei_exchange_e']
  integrated['P_external_ion'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
  integrated['P_external_el'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
  integrated['P_external_injected'] = jnp.array(
      0.0, dtype=jax_utils.get_dtype()
  )

  # Calculate integrated sources with convenient names, transformed from
  # TORAX internal names.
  for key, value in ION_EL_HEAT_SOURCE_TRANSFORMATIONS.items():
    ion_profiles = core_sources.temp_ion
    el_profiles = core_sources.temp_el
    if key in ion_profiles and key in el_profiles:
      profile_ion, profile_el = ion_profiles[key], el_profiles[key]
      integrated[f'{value}_i'] = math_utils.volume_integration(
          profile_ion, geo
      )
      integrated[f'{value}_e'] = math_utils.volume_integration(profile_el, geo)
      integrated[f'{value}_total'] = (
          integrated[f'{value}_i'] + integrated[f'{value}_e']
      )
      integrated['P_SOL_i'] += integrated[f'{value}_i']
      integrated['P_SOL_e'] += integrated[f'{value}_e']
      if key in EXTERNAL_HEATING_SOURCES:
        integrated['P_external_ion'] += integrated[f'{value}_i']
        integrated['P_external_el'] += integrated[f'{value}_e']

        # Track injected power for heating sources that have absorption_fraction
        # These are only for sources like ICRH or NBI that are
        # ion_el_heat_sources.
        source_params = dynamic_runtime_params_slice.sources[key]
        if hasattr(source_params, 'absorption_fraction'):
          total_absorbed = integrated[f'{value}_total']
          injected_power = total_absorbed / source_params.absorption_fraction
          integrated['P_external_injected'] += injected_power
        else:
          integrated['P_external_injected'] += integrated[f'{value}_total']
    else:
      integrated[f'{value}_i'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
      integrated[f'{value}_e'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
      integrated[f'{value}_total'] = jnp.array(0.0, dtype=jax_utils.get_dtype())

  for key, value in EL_HEAT_SOURCE_TRANSFORMATIONS.items():
    profiles = core_sources.temp_el
    if key in profiles:
      integrated[f'{value}'] = math_utils.volume_integration(
          profiles[key], geo
      )
      integrated['P_SOL_e'] += integrated[f'{value}']
      if key in EXTERNAL_HEATING_SOURCES:
        integrated['P_external_el'] += integrated[f'{value}']
        integrated['P_external_injected'] += integrated[f'{value}']
    else:
      integrated[f'{value}'] = jnp.array(0.0, dtype=jax_utils.get_dtype())

  for key, value in CURRENT_SOURCE_TRANSFORMATIONS.items():
    profiles = core_sources.psi
    if key in profiles:
      integrated[f'{value}'] = math_utils.area_integration(profiles[key], geo)
    else:
      integrated[f'{value}'] = jnp.array(0.0, dtype=jax_utils.get_dtype())

  integrated['P_SOL_total'] = integrated['P_SOL_i'] + integrated['P_SOL_e']
  integrated['P_external_tot'] = (
      integrated['P_external_ion'] + integrated['P_external_el']
  )

  return integrated


@jax_utils.jit
def make_post_processed_outputs(
    sim_state: state.ToraxSimState,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    previous_post_processed_outputs: state.PostProcessedOutputs | None = None,
) -> state.PostProcessedOutputs:
  """Calculates post-processed outputs based on the latest state.

  Called at the beginning and end of each `sim.run_simulation` step.
  Args:
    sim_state: The state to add outputs to.
    dynamic_runtime_params_slice: Runtime parameters slice for the current time
      step, needed for calculating integrated power.
    previous_post_processed_outputs: The previous outputs, used to calculate
      cumulative quantities. Optional input. If None, then cumulative quantities
      are set at the initialized values in sim_state itself. This is used for
      the first time step of a the simulation. The initialized values are zero
      for a clean simulation, or the last value of the previous simulation for a
      restarted simulation.

  Returns:
    post_processed_outputs: The post_processed_outputs for the given state.
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
          sim_state.geometry,
      )
  )
  FFprime_face = formulas.calc_FFprime(
      sim_state.core_profiles, sim_state.geometry
  )
  # Calculate normalized poloidal flux.
  psi_face = sim_state.core_profiles.psi.face_value()
  psi_norm_face = (psi_face - psi_face[0]) / (psi_face[-1] - psi_face[0])
  integrated_sources = _calculate_integrated_sources(
      sim_state.geometry,
      sim_state.core_profiles,
      sim_state.core_sources,
      dynamic_runtime_params_slice,
  )
  # Calculate fusion gain with a zero division guard.
  # Total energy released per reaction is 5 times the alpha particle energy.
  Q_fusion = (
      integrated_sources['P_alpha_total']
      * 5.0
      / (integrated_sources['P_external_injected'] + constants.CONSTANTS.eps)
  )

  P_LH_hi_dens, P_LH_min, P_LH, n_e_min_P_LH = (
      scaling_laws.calculate_plh_scaling_factor(
          sim_state.geometry, sim_state.core_profiles
      )
  )

  # Thermal energy confinement time is the stored energy divided by the total
  # input power into the plasma.

  # Ploss term here does not include the reduction of radiated power. Most
  # analysis of confinement times from databases have not included this term.
  # Therefore highly radiative scenarios can lead to skewed results.

  Ploss = (
      integrated_sources['P_alpha_total'] + integrated_sources['P_external_tot']
  )

  if previous_post_processed_outputs is not None:
    dW_th_dt = (
        W_thermal_tot - previous_post_processed_outputs.W_thermal_total
    ) / sim_state.dt
  else:
    dW_th_dt = 0.0

  tauE = W_thermal_tot / Ploss

  tauH89P = scaling_laws.calculate_scaling_law_confinement_time(
      sim_state.geometry, sim_state.core_profiles, Ploss, 'H89P'
  )
  tauH98 = scaling_laws.calculate_scaling_law_confinement_time(
      sim_state.geometry, sim_state.core_profiles, Ploss, 'H98'
  )
  tauH97L = scaling_laws.calculate_scaling_law_confinement_time(
      sim_state.geometry, sim_state.core_profiles, Ploss, 'H97L'
  )
  tauH20 = scaling_laws.calculate_scaling_law_confinement_time(
      sim_state.geometry, sim_state.core_profiles, Ploss, 'H20'
  )

  H89P = tauE / tauH89P
  H98 = tauE / tauH98
  H97L = tauE / tauH97L
  H20 = tauE / tauH20

  # Calculate total external (injected) and fusion (generated) energies based on
  # interval average.
  if previous_post_processed_outputs is not None:
    # Factor 5 due to including neutron energy: E_fusion = 5.0 * E_alpha
    E_cumulative_fusion = (
        previous_post_processed_outputs.E_fusion
        + 5.0
        * sim_state.dt
        * (
            integrated_sources['P_alpha_total']
            + previous_post_processed_outputs.P_alpha_total
        )
        / 2.0
    )
    E_cumulative_external = (
        previous_post_processed_outputs.E_aux
        + sim_state.dt
        * (
            integrated_sources['P_external_tot']
            + previous_post_processed_outputs.P_external_tot
        )
        / 2.0
    )
  else:
    # Used during initiailization. Note for restarted simulations this should
    # be overwritten by the previous_post_processed_outputs.
    E_cumulative_fusion = 0.0
    E_cumulative_external = 0.0

  # Calculate q at 95% of the normalized poloidal flux
  q95 = psi_calculations.calc_q95(psi_norm_face, sim_state.core_profiles.q_face)

  # Calculate te and ti volume average [keV]
  te_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.temp_el.value, sim_state.geometry
  )
  ti_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.temp_ion.value, sim_state.geometry
  )

  # Calculate n_e and ni (main ion) volume and line averages [nref m^-3]
  n_e_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.n_e.value, sim_state.geometry
  )
  ni_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.ni.value, sim_state.geometry
  )
  n_e_line_avg = math_utils.line_average(
      sim_state.core_profiles.n_e.value, sim_state.geometry
  )
  ni_line_avg = math_utils.line_average(
      sim_state.core_profiles.ni.value, sim_state.geometry
  )
  fgw_n_e_volume_avg = formulas.calculate_greenwald_fraction(
      n_e_volume_avg, sim_state.core_profiles, sim_state.geometry
  )
  fgw_n_e_line_avg = formulas.calculate_greenwald_fraction(
      n_e_line_avg, sim_state.core_profiles, sim_state.geometry
  )
  Wpol = psi_calculations.calc_Wpol(
      sim_state.geometry, sim_state.core_profiles.psi
  )
  li3 = psi_calculations.calc_li3(
      sim_state.geometry.R_major,
      Wpol,
      sim_state.core_profiles.currents.Ip_profile_face[-1],
  )

  safety_factor_fit_outputs = (
      safety_factor_fit.find_min_q_and_q_surface_intercepts(
          sim_state.geometry.rho_face_norm,
          sim_state.core_profiles.q_face,
      )
  )

  # pylint: enable=invalid-name
  return state.PostProcessedOutputs(
      pressure_thermal_i=pressure_thermal_ion_face,
      pressure_thermal_e=pressure_thermal_el_face,
      pressure_thermal_total=pressure_thermal_tot_face,
      pprime=pprime_face,
      W_thermal_i=W_thermal_ion,
      W_thermal_e=W_thermal_el,
      W_thermal_total=W_thermal_tot,
      tau_E=tauE,
      H89P=H89P,
      H98=H98,
      H97L=H97L,
      H20=H20,
      FFprime=FFprime_face,
      psi_norm=psi_norm_face,
      **integrated_sources,
      Q_fusion=Q_fusion,
      P_LH=P_LH,
      P_LH_min=P_LH_min,
      P_LH_high_density=P_LH_hi_dens,
      n_e_min_P_LH=n_e_min_P_LH,
      E_fusion=E_cumulative_fusion,
      E_aux=E_cumulative_external,
      T_e_volume_avg=te_volume_avg,
      T_i_volume_avg=ti_volume_avg,
      n_e_volume_avg=n_e_volume_avg,
      n_i_volume_avg=ni_volume_avg,
      n_e_line_avg=n_e_line_avg,
      n_i_line_avg=ni_line_avg,
      fgw_n_e_volume_avg=fgw_n_e_volume_avg,
      fgw_n_e_line_avg=fgw_n_e_line_avg,
      q95=q95,
      W_pol=Wpol,
      li3=li3,
      dW_thermal_dt=dW_th_dt,
      rho_q_min=safety_factor_fit_outputs.rho_q_min,
      q_min=safety_factor_fit_outputs.q_min,
      rho_q_3_2_first=safety_factor_fit_outputs.rho_q_3_2_first,
      rho_q_2_1_first=safety_factor_fit_outputs.rho_q_2_1_first,
      rho_q_3_1_first=safety_factor_fit_outputs.rho_q_3_1_first,
      rho_q_3_2_second=safety_factor_fit_outputs.rho_q_3_2_second,
      rho_q_2_1_second=safety_factor_fit_outputs.rho_q_2_1_second,
      rho_q_3_1_second=safety_factor_fit_outputs.rho_q_3_1_second,
  )
