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

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.output_tools import safety_factor_fit
from torax._src.physics import formulas
from torax._src.physics import psi_calculations
from torax._src.physics import scaling_laws
from torax._src.sources import source_profiles
import typing_extensions


@chex.dataclass(frozen=True, eq=False)
class PostProcessedOutputs:
  """Collection of outputs calculated after each simulation step.

  These variables are not used internally, but are useful as outputs or
  intermediate observations for overarching workflows.

  Attributes:
    pressure_thermal_i: Ion thermal pressure on the face grid [Pa]
    pressure_thermal_e: Electron thermal pressure on the face grid [Pa]
    pressure_thermal_total: Total thermal pressure on the face grid [Pa]
    pprime: Derivative of total pressure with respect to poloidal flux on the
      face grid [Pa/Wb]
    W_thermal_i: Ion thermal stored energy [J]
    W_thermal_e: Electron thermal stored energy [J]
    W_thermal_total: Total thermal stored energy [J]
    tau_E: Thermal energy confinement time [s]
    H89P: L-mode confinement quality factor with respect to the ITER89P scaling
      law derived from the ITER L-mode confinement database
    H98: H-mode confinement quality factor with respect to the ITER98y2 scaling
      law derived from the ITER H-mode confinement database
    H97L: L-mode confinement quality factor with respect to the ITER97L scaling
      law derived from the ITER L-mode confinement database
    H20: H-mode confinement quality factor with respect to the ITER20 scaling
      law derived from the updated (2020) ITER H-mode confinement database
    FFprime: FF' on the face grid, where F is the toroidal flux function
    psi_norm: Normalized poloidal flux on the face grid [Wb]
    P_SOL_i: Total ion heating power exiting the plasma with all sources:
      auxiliary heating + ion-electron exchange + fusion [W]
    P_SOL_e: Total electron heating power exiting the plasma with all sources
      and sinks: auxiliary heating + ion-electron exchange + Ohmic + fusion +
      radiation sinks [W]
    P_SOL_total: Total heating power exiting the plasma with all sources and
      sinks
    P_aux_i: Total auxiliary ion heating power [W]
    P_aux_e: Total auxiliary electron heating power [W]
    P_aux_total: Total auxiliary heating power [W]
    P_external_injected: Total external injected power before absorption [W]
    P_ei_exchange_i: Electron-ion heat exchange power to ions [W]
    P_ei_exchange_e: Electron-ion heat exchange power to electrons [W]
    P_aux_generic_i: Total generic_ion_el_heat_source power to ions [W]
    P_aux_generic_e: Total generic_ion_el_heat_source power to electrons [W]
    P_aux_generic_total: Total generic_ion_el_heat power [W]
    P_alpha_i: Total fusion power to ions [W]
    P_alpha_e: Total fusion power to electrons [W]
    P_alpha_total: Total fusion power to plasma [W]
    P_ohmic_e: Ohmic heating power to electrons [W]
    P_bremsstrahlung_e: Bremsstrahlung electron heat sink [W]
    P_cyclotron_e: Cyclotron radiation electron heat sink [W]
    P_ecrh_e: Total electron cyclotron source power [W]
    P_radiation_e: Impurity radiation heat sink [W]
    I_ecrh: Total electron cyclotron source current [A]
    I_aux_generic: Total generic source current [A]
    Q_fusion: Fusion power gain
    P_icrh_e: Ion cyclotron resonance heating to electrons [W]
    P_icrh_i: Ion cyclotron resonance heating to ions [W]
    P_icrh_total: Total ion cyclotron resonance heating power [W]
    P_LH_high_density: H-mode transition power for high density branch [W]
    P_LH_min: Minimum H-mode transition power for at n_e_min_P_LH [W]
    P_LH: H-mode transition power from maximum of P_LH_high_density and P_LH_min
      [W]
    n_e_min_P_LH: Density corresponding to the P_LH_min [m^-3]
    E_fusion: Total cumulative fusion energy [J]
    E_aux: Total external injected energy (Ohmic + auxiliary heating) [J]
    T_e_volume_avg: Volume average electron temperature [keV]
    T_i_volume_avg: Volume average ion temperature [keV]
    n_e_volume_avg: Volume average electron density [m^-3]
    n_e_volume_avg: Volume average electron density [m^-3]
    n_i_volume_avg: Volume average main ion density [m^-3]
    n_e_line_avg: Line averaged electron density [m^-3]
    n_i_line_avg: Line averaged main ion density [m^-3]
    fgw_n_e_volume_avg: Greenwald fraction from volume-averaged electron density
      [dimensionless]
    fgw_n_e_line_avg: Greenwald fraction from line-averaged electron density
      [dimensionless]
    q95: q at 95% of the normalized poloidal flux
    W_pol: Total magnetic energy [J]
    li3: Normalized plasma internal inductance, ITER convention [dimensionless]
    dW_thermal_dt: Time derivative of the total stored thermal energy [W]
    q_min: Minimum q value
    rho_q_min: rho_norm at the minimum q
    rho_q_3_2_first: First outermost rho_norm value that intercepts the q=3/2
      plane. If no intercept is found, set to -inf.
    rho_q_2_1_first: First outermost rho_norm value that intercepts the q=2/1
      plane. If no intercept is found, set to -inf.
    rho_q_3_1_first: First outermost rho_norm value that intercepts the q=3/1
      plane. If no intercept is found, set to -inf.
    rho_q_3_2_second: Second outermost rho_norm value that intercepts the q=3/2
      plane. If no intercept is found, set to -inf.
    rho_q_2_1_second: Second outermost rho_norm value that intercepts the q=2/1
      plane. If no intercept is found, set to -inf.
    rho_q_3_1_second: Second outermost rho_norm value that intercepts the q=3/1
      plane. If no intercept is found, set to -inf.
    I_bootstrap: Total bootstrap current [A].
    j_external: Total current from psi sources which are external to the
      plasma (aka not bootstrap) [A]
    j_ohmic: Ohmic current density [A]
  """

  pressure_thermal_i: array_typing.ArrayFloat
  pressure_thermal_e: array_typing.ArrayFloat
  pressure_thermal_total: array_typing.ArrayFloat
  pprime: array_typing.ArrayFloat
  # pylint: disable=invalid-name
  W_thermal_i: array_typing.ScalarFloat
  W_thermal_e: array_typing.ScalarFloat
  W_thermal_total: array_typing.ScalarFloat
  tau_E: array_typing.ScalarFloat
  H89P: array_typing.ScalarFloat
  H98: array_typing.ScalarFloat
  H97L: array_typing.ScalarFloat
  H20: array_typing.ScalarFloat
  FFprime: array_typing.ArrayFloat
  psi_norm: array_typing.ArrayFloat
  # Integrated heat sources
  P_SOL_i: array_typing.ScalarFloat
  P_SOL_e: array_typing.ScalarFloat
  P_SOL_total: array_typing.ScalarFloat
  P_aux_i: array_typing.ScalarFloat
  P_aux_e: array_typing.ScalarFloat
  P_aux_total: array_typing.ScalarFloat
  P_external_injected: array_typing.ScalarFloat
  P_ei_exchange_i: array_typing.ScalarFloat
  P_ei_exchange_e: array_typing.ScalarFloat
  P_aux_generic_i: array_typing.ScalarFloat
  P_aux_generic_e: array_typing.ScalarFloat
  P_aux_generic_total: array_typing.ScalarFloat
  P_alpha_i: array_typing.ScalarFloat
  P_alpha_e: array_typing.ScalarFloat
  P_alpha_total: array_typing.ScalarFloat
  P_ohmic_e: array_typing.ScalarFloat
  P_bremsstrahlung_e: array_typing.ScalarFloat
  P_cyclotron_e: array_typing.ScalarFloat
  P_ecrh_e: array_typing.ScalarFloat
  P_radiation_e: array_typing.ScalarFloat
  I_ecrh: array_typing.ScalarFloat
  I_aux_generic: array_typing.ScalarFloat
  Q_fusion: array_typing.ScalarFloat
  P_icrh_e: array_typing.ScalarFloat
  P_icrh_i: array_typing.ScalarFloat
  P_icrh_total: array_typing.ScalarFloat
  P_LH_high_density: array_typing.ScalarFloat
  P_LH_min: array_typing.ScalarFloat
  P_LH: array_typing.ScalarFloat
  n_e_min_P_LH: array_typing.ScalarFloat
  E_fusion: array_typing.ScalarFloat
  E_aux: array_typing.ScalarFloat
  T_e_volume_avg: array_typing.ScalarFloat
  T_i_volume_avg: array_typing.ScalarFloat
  n_e_volume_avg: array_typing.ScalarFloat
  n_i_volume_avg: array_typing.ScalarFloat
  n_e_line_avg: array_typing.ScalarFloat
  n_i_line_avg: array_typing.ScalarFloat
  fgw_n_e_volume_avg: array_typing.ScalarFloat
  fgw_n_e_line_avg: array_typing.ScalarFloat
  q95: array_typing.ScalarFloat
  W_pol: array_typing.ScalarFloat
  li3: array_typing.ScalarFloat
  dW_thermal_dt: array_typing.ScalarFloat
  rho_q_min: array_typing.ScalarFloat
  q_min: array_typing.ScalarFloat
  rho_q_3_2_first: array_typing.ScalarFloat
  rho_q_3_2_second: array_typing.ScalarFloat
  rho_q_2_1_first: array_typing.ScalarFloat
  rho_q_2_1_second: array_typing.ScalarFloat
  rho_q_3_1_first: array_typing.ScalarFloat
  rho_q_3_1_second: array_typing.ScalarFloat
  I_bootstrap: array_typing.ScalarFloat
  j_external: array_typing.ArrayFloat
  j_ohmic: array_typing.ArrayFloat
  # pylint: enable=invalid-name

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> typing_extensions.Self:
    """Returns a PostProcessedOutputs with all zeros, used for initializing."""
    return cls(
        pressure_thermal_i=jnp.zeros(geo.rho_face.shape),
        pressure_thermal_e=jnp.zeros(geo.rho_face.shape),
        pressure_thermal_total=jnp.zeros(geo.rho_face.shape),
        pprime=jnp.zeros(geo.rho_face.shape),
        W_thermal_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        W_thermal_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        W_thermal_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        tau_E=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H89P=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H98=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H97L=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H20=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        FFprime=jnp.zeros(geo.rho_face.shape),
        psi_norm=jnp.zeros(geo.rho_face.shape),
        P_SOL_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_SOL_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_SOL_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_aux_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_aux_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_aux_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_external_injected=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ei_exchange_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ei_exchange_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_aux_generic_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_aux_generic_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_aux_generic_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ohmic_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_bremsstrahlung_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_cyclotron_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ecrh_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_radiation_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_ecrh=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_aux_generic=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        Q_fusion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_high_density=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_e_min_P_LH=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        E_fusion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        E_aux=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        T_e_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        T_i_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_e_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_i_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_e_line_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_i_line_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        fgw_n_e_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        fgw_n_e_line_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        q95=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        W_pol=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        li3=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        dW_thermal_dt=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        q_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_2_first=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_2_1_first=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_1_first=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_2_second=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_2_1_second=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_1_second=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_bootstrap=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        j_external=jnp.zeros(geo.rho_face.shape),
        j_ohmic=jnp.zeros(geo.rho_face.shape),
    )

  def check_for_errors(self):
    if any([jnp.any(jnp.isnan(x)) for x in jax.tree.leaves(self)]):
      return state.SimError.NAN_DETECTED
    else:
      return state.SimError.NO_ERROR


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
    See PostProcessedOutputs for the full list of keys.

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
      core_profiles.T_e.value - core_profiles.T_i.value
  )
  integrated['P_ei_exchange_i'] = math_utils.volume_integration(qei, geo)
  integrated['P_ei_exchange_e'] = -integrated['P_ei_exchange_i']

  # Initialize total electron and ion powers
  # TODO(b/380848256): P_sol is now correct for stationary state. However,
  # for generality need to add dWth/dt to the equation (time dependence of
  # stored energy).
  integrated['P_SOL_i'] = integrated['P_ei_exchange_i']
  integrated['P_SOL_e'] = integrated['P_ei_exchange_e']
  integrated['P_aux_i'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
  integrated['P_aux_e'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
  integrated['P_external_injected'] = jnp.array(
      0.0, dtype=jax_utils.get_dtype()
  )

  # Calculate integrated sources with convenient names, transformed from
  # TORAX internal names.
  for key, value in ION_EL_HEAT_SOURCE_TRANSFORMATIONS.items():
    ion_profiles = core_sources.T_i
    el_profiles = core_sources.T_e
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
        integrated['P_aux_i'] += integrated[f'{value}_i']
        integrated['P_aux_e'] += integrated[f'{value}_e']

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
    profiles = core_sources.T_e
    if key in profiles:
      integrated[f'{value}'] = math_utils.volume_integration(
          profiles[key], geo
      )
      integrated['P_SOL_e'] += integrated[f'{value}']
      if key in EXTERNAL_HEATING_SOURCES:
        integrated['P_aux_e'] += integrated[f'{value}']
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
  integrated['P_aux_total'] = (
      integrated['P_aux_i'] + integrated['P_aux_e']
  )

  return integrated


@jax_utils.jit
def make_post_processed_outputs(
    sim_state: sim_state_lib.ToraxSimState,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    previous_post_processed_outputs: PostProcessedOutputs | None = None,
) -> PostProcessedOutputs:
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
      / (
          integrated_sources['P_external_injected']
          + integrated_sources['P_ohmic_e']
          + constants.CONSTANTS.eps
      )
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
      integrated_sources['P_alpha_total']
      + integrated_sources['P_aux_total']
      + integrated_sources['P_ohmic_e']
      + constants.CONSTANTS.eps  # Division guard.
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
            integrated_sources['P_aux_total']
            + integrated_sources['P_ohmic_e']
            + previous_post_processed_outputs.P_aux_total
            + previous_post_processed_outputs.P_ohmic_e
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
      sim_state.core_profiles.T_e.value, sim_state.geometry
  )
  ti_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.T_i.value, sim_state.geometry
  )

  # Calculate n_e and n_i (main ion) volume and line averages in m^-3
  n_e_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.n_e.value, sim_state.geometry
  )
  n_i_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.n_i.value, sim_state.geometry
  )
  n_e_line_avg = math_utils.line_average(
      sim_state.core_profiles.n_e.value, sim_state.geometry
  )
  n_i_line_avg = math_utils.line_average(
      sim_state.core_profiles.n_i.value, sim_state.geometry
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
      sim_state.core_profiles.Ip_profile_face[-1],
  )

  safety_factor_fit_outputs = (
      safety_factor_fit.find_min_q_and_q_surface_intercepts(
          sim_state.geometry.rho_face_norm,
          sim_state.core_profiles.q_face,
      )
  )

  I_bootstrap = math_utils.area_integration(
      sim_state.core_sources.bootstrap_current.j_bootstrap, sim_state.geometry
  )

  j_external = sum(sim_state.core_sources.psi.values())
  psi_current = (
      j_external + sim_state.core_sources.bootstrap_current.j_bootstrap
  )
  j_ohmic = sim_state.core_profiles.j_total - psi_current

  # pylint: enable=invalid-name
  return PostProcessedOutputs(
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
      n_i_volume_avg=n_i_volume_avg,
      n_e_line_avg=n_e_line_avg,
      n_i_line_avg=n_i_line_avg,
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
      I_bootstrap=I_bootstrap,
      j_external=j_external,
      j_ohmic=j_ohmic,
  )
