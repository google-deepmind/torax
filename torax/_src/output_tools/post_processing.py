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

import dataclasses
from typing import Callable

from absl import logging
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.output_tools import impurity_radiation
from torax._src.output_tools import safety_factor_fit
from torax._src.physics import formulas
from torax._src.physics import psi_calculations
from torax._src.physics import rotation
from torax._src.physics import scaling_laws
from torax._src.sources import source_profiles
import typing_extensions


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PostProcessedOutputs:
  """Collection of outputs calculated after each simulation step.

  These variables are not used internally, but are useful as outputs or
  intermediate observations for overarching workflows.

  Attributes:
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
    P_heat_i: Total ion heating power: all sources - sinks. i.e. auxiliary
      heating + ion-electron exchange + fusion + (negative) radiation sinks [W].
    P_heat_e: Total electron heating power: all sources - sinks. i.e. auxiliary
      heating + ion-electron exchange + Ohmic + fusion + (negative) radiation
      sinks [W].
    P_heat_total: Total heating power: all sources - sinks. i.e. auxiliary
      heating + fusion + (negative) radiation sinks [W].
    P_SOL_i: Total ion heating power exiting the plasma: P_heat_i -
      dW_thermal_i/dt. The dW/dt term is smoothed.
    P_SOL_e: Total electron heating power exiting the plasma: P_heat_e -
      dW_thermal_e/dt. The dW/dt term is smoothed.
    P_SOL_total: Total heating power exiting the plasma: P_heat_total -
      dW_thermal_total/dt. The dW/dt term is smoothed.
    P_aux_i: Total auxiliary ion heating power [W]
    P_aux_e: Total auxiliary electron heating power [W]
    P_aux_total: Total auxiliary heating power [W]
    P_external_injected: Total external injected power before absorption [W]
    P_external_total: Total external power before absorption (
      P_external_injected + P_ohmic_e) [W]
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
    P_fusion: Generated fusion power (5*P_alpha_total) [W]
    I_ecrh: Total electron cyclotron source current [A]
    I_aux_generic: Total generic source current [A]
    Q_fusion: Fusion power gain (P_fusion / P_external_total) [dimensionless]
    P_icrh_e: Ion cyclotron resonance heating to electrons [W]
    P_icrh_i: Ion cyclotron resonance heating to ions [W]
    P_icrh_total: Total ion cyclotron resonance heating power [W]
    P_LH_high_density: H-mode transition power for high density branch [W]
    P_LH_min: Minimum H-mode transition power for at n_e_min_P_LH [W]
    P_LH: H-mode transition power from maximum of P_LH_high_density and P_LH_min
      [W]
    n_e_min_P_LH: Density corresponding to the P_LH_min [m^-3]
    E_fusion: Total cumulative fusion energy [J]
    E_aux_total: Total auxiliary heating energy absorbed by the plasma ( time
      integral of P_aux_total) [J].
    E_ohmic_e: Total Ohmic heating energy to electrons (time integral of
      P_ohmic_e) [J].
    E_external_injected: Total external injected energy before absorption ( time
      integral of P_external_injected) [J].
    E_external_total: Total external energy absorbed by the plasma ( time
      integral of P_external_total) [J].
    T_e_volume_avg: Volume average electron temperature [keV]
    T_i_volume_avg: Volume average ion temperature [keV]
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
    dW_thermal_dt: Time derivative of the total stored thermal energy [W], raw
      unsmoothed value.
    dW_thermal_dt_smoothed: Smoothed time derivative of total stored thermal
      energy [W].
    dW_thermal_i_dt_smoothed: Smoothed time derivative of ion stored thermal
      energy [W].
    dW_thermal_e_dt_smoothed: Smoothed time derivative of electron stored
      thermal energy [W].
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
    j_total: Total toroidal current density [Am^-2]
    j_parallel_total: Total parallel current density [Am^-2]
    j_bootstrap: Toroidal bootstrap current density [Am^-2]
    j_bootstrap_face: Toroidal bootstrap current density on face grid [Am^-2]
    j_external: Toroidal current density from external psi sources (i.e.,
      excluding bootstrap) [A m^-2]
    j_parallel_external: Parallel current density from external psi sources
      (i.e., excluding bootstrap) [A m^-2]
    j_ohmic: Toroidal ohmic current density [Am^-2]
    j_parallel_ohmic: Parallel ohmic current density [Am^-2]
    j_generic_current: Toroidal current density from generic current source
      [Am^-2]
    j_parallel_generic_current: Parallel current density from generic current
      source [Am^-2]
    j_ecrh: Toroidal current density from electron cyclotron heating and current
      source [Am^-2]
    j_parallel_ecrh: Toroidal current density from electron cyclotron heating
      and current source [Am^-2]
    j_non_inductive: Total toroidal non-inductive current density [Am^-2]
    j_parallel_non_inductive: Total parallel non-inductive current density
      [Am^-2]
    I_external: Total external current [A]
    I_non_inductive: Total non-inductive current [A]
    f_non_inductive: Non-inductive current fraction of the total current
      [dimensionless]
    f_bootstrap: Bootstrap current fraction of the total current [dimensionless]
    S_gas_puff: Integrated gas puff source [s^-1]
    S_pellet: Integrated pellet source [s^-1]
    S_generic_particle: Integrated generic particle source [s^-1]
    S_total: Total integrated particle sources [s^-1]
    beta_tor: Volume-averaged toroidal plasma beta (thermal) [dimensionless]
    beta_pol: Volume-averaged poloidal plasma beta (thermal) [dimensionless]
    beta_N: Normalized toroidal plasma beta (thermal) [dimensionless].
    impurity_species: Dictionary of outputs for each impurity species.
    poloidal_velocity: Poloidal velocity [m/s]
    radial_electric_field: Radial electric field [V/m]
    first_step: Whether the outputs are from the first step of the simulation.
  """

  pprime: array_typing.FloatVector
  # pylint: disable=invalid-name
  W_thermal_i: array_typing.FloatScalar
  W_thermal_e: array_typing.FloatScalar
  W_thermal_total: array_typing.FloatScalar
  tau_E: array_typing.FloatScalar
  H89P: array_typing.FloatScalar
  H98: array_typing.FloatScalar
  H97L: array_typing.FloatScalar
  H20: array_typing.FloatScalar
  FFprime: array_typing.FloatVector
  psi_norm: array_typing.FloatVector
  # Integrated heat sources
  P_SOL_i: array_typing.FloatScalar
  P_SOL_e: array_typing.FloatScalar
  P_SOL_total: array_typing.FloatScalar
  P_aux_i: array_typing.FloatScalar
  P_aux_e: array_typing.FloatScalar
  P_aux_total: array_typing.FloatScalar
  P_external_injected: array_typing.FloatScalar
  P_external_total: array_typing.FloatScalar
  P_heat_i: array_typing.FloatScalar
  P_heat_e: array_typing.FloatScalar
  P_heat_total: array_typing.FloatScalar
  P_ei_exchange_i: array_typing.FloatScalar
  P_ei_exchange_e: array_typing.FloatScalar
  P_aux_generic_i: array_typing.FloatScalar
  P_aux_generic_e: array_typing.FloatScalar
  P_aux_generic_total: array_typing.FloatScalar
  P_alpha_i: array_typing.FloatScalar
  P_alpha_e: array_typing.FloatScalar
  P_alpha_total: array_typing.FloatScalar
  P_ohmic_e: array_typing.FloatScalar
  P_bremsstrahlung_e: array_typing.FloatScalar
  P_cyclotron_e: array_typing.FloatScalar
  P_ecrh_e: array_typing.FloatScalar
  P_radiation_e: array_typing.FloatScalar
  I_ecrh: array_typing.FloatScalar
  I_aux_generic: array_typing.FloatScalar
  P_fusion: array_typing.FloatScalar
  Q_fusion: array_typing.FloatScalar
  P_icrh_e: array_typing.FloatScalar
  P_icrh_i: array_typing.FloatScalar
  P_icrh_total: array_typing.FloatScalar
  P_LH_high_density: array_typing.FloatScalar
  P_LH_min: array_typing.FloatScalar
  P_LH: array_typing.FloatScalar
  n_e_min_P_LH: array_typing.FloatScalar
  E_fusion: array_typing.FloatScalar
  E_aux_total: array_typing.FloatScalar
  E_ohmic_e: array_typing.FloatScalar
  E_external_injected: array_typing.FloatScalar
  E_external_total: array_typing.FloatScalar
  T_e_volume_avg: array_typing.FloatScalar
  T_i_volume_avg: array_typing.FloatScalar
  n_e_volume_avg: array_typing.FloatScalar
  n_i_volume_avg: array_typing.FloatScalar
  n_e_line_avg: array_typing.FloatScalar
  n_i_line_avg: array_typing.FloatScalar
  fgw_n_e_volume_avg: array_typing.FloatScalar
  fgw_n_e_line_avg: array_typing.FloatScalar
  q95: array_typing.FloatScalar
  W_pol: array_typing.FloatScalar
  li3: array_typing.FloatScalar
  dW_thermal_dt: array_typing.FloatScalar
  dW_thermal_dt_smoothed: array_typing.FloatScalar
  dW_thermal_i_dt_smoothed: array_typing.FloatScalar
  dW_thermal_e_dt_smoothed: array_typing.FloatScalar
  rho_q_min: array_typing.FloatScalar
  q_min: array_typing.FloatScalar
  rho_q_3_2_first: array_typing.FloatScalar
  rho_q_3_2_second: array_typing.FloatScalar
  rho_q_2_1_first: array_typing.FloatScalar
  rho_q_2_1_second: array_typing.FloatScalar
  rho_q_3_1_first: array_typing.FloatScalar
  rho_q_3_1_second: array_typing.FloatScalar
  I_bootstrap: array_typing.FloatScalar
  # Note: The default j profiles (j_total, j_parallel_bootstrap, j_parallel_*
  # for * in sources) are saved elsewhere
  # TODO(b/434175938): rename j_* to j_toroidal_* for clarity
  j_parallel_total: array_typing.FloatVector
  j_external: array_typing.FloatVector
  j_parallel_external: array_typing.FloatVector
  j_ohmic: array_typing.FloatVector
  j_parallel_ohmic: array_typing.FloatVector
  j_bootstrap: array_typing.FloatVector
  j_bootstrap_face: array_typing.FloatVector
  j_generic_current: array_typing.FloatVector
  j_ecrh: array_typing.FloatVector
  j_non_inductive: array_typing.FloatVector
  j_parallel_non_inductive: array_typing.FloatVector
  I_external: array_typing.FloatScalar
  I_non_inductive: array_typing.FloatScalar
  f_non_inductive: array_typing.FloatScalar
  f_bootstrap: array_typing.FloatScalar
  S_gas_puff: array_typing.FloatScalar
  S_pellet: array_typing.FloatScalar
  S_generic_particle: array_typing.FloatScalar
  beta_tor: array_typing.FloatScalar
  beta_pol: array_typing.FloatScalar
  beta_N: array_typing.FloatScalar
  S_total: array_typing.FloatScalar
  impurity_species: dict[str, impurity_radiation.ImpuritySpeciesOutput]
  poloidal_velocity: array_typing.FloatVector
  radial_electric_field: array_typing.FloatVector
  first_step: array_typing.BoolScalar
  # pylint: enable=invalid-name

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> typing_extensions.Self:
    """Returns a PostProcessedOutputs with all zeros, used for initializing."""
    return cls(
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
        P_external_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_heat_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_heat_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_heat_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
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
        P_fusion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        Q_fusion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_high_density=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_e_min_P_LH=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        E_fusion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        E_aux_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        E_ohmic_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        E_external_injected=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        E_external_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
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
        dW_thermal_dt_smoothed=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        dW_thermal_i_dt_smoothed=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        dW_thermal_e_dt_smoothed=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        q_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_2_first=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_2_1_first=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_1_first=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_2_second=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_2_1_second=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_1_second=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_bootstrap=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        # TODO(b/434175938): rename j_* to j_toroidal_* for clarity
        j_parallel_total=jnp.zeros(geo.rho_face.shape),
        j_bootstrap=jnp.zeros(geo.rho.shape),
        j_bootstrap_face=jnp.zeros(geo.rho_face.shape),
        j_ohmic=jnp.zeros(geo.rho_face.shape),
        j_parallel_ohmic=jnp.zeros(geo.rho_face.shape),
        j_external=jnp.zeros(geo.rho_face.shape),
        j_generic_current=jnp.zeros(geo.rho_face.shape),
        j_ecrh=jnp.zeros(geo.rho_face.shape),
        j_non_inductive=jnp.zeros(geo.rho_face.shape),
        j_parallel_external=jnp.zeros(geo.rho_face.shape),
        j_parallel_non_inductive=jnp.zeros(geo.rho_face.shape),
        I_external=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_non_inductive=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        f_non_inductive=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        f_bootstrap=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        S_gas_puff=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        S_pellet=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        S_generic_particle=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        beta_tor=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        beta_pol=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        beta_N=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        S_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        impurity_species={},
        poloidal_velocity=jnp.zeros(geo.rho_face.shape),
        radial_electric_field=jnp.zeros(geo.rho_face.shape),
        first_step=jnp.array(True),
    )

  def check_for_errors(self):
    if any([np.any(np.isnan(x)) for x in jax.tree.leaves(self)]):
      path_vals, _ = jax.tree.flatten_with_path(self)
      for path, value in path_vals:
        if np.any(np.isnan(value)):
          logging.info(
              'Found NaNs in post_processed_outputs%s',
              jax.tree_util.keystr(path),
          )
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
PARTICLE_SOURCE_TRANSFORMATIONS = {
    'gas_puff': 'S_gas_puff',
    'pellet': 'S_pellet',
    'generic_particle': 'S_generic_particle',
}


def _get_integrated_source_value(
    source_profiles_dict: dict[str, array_typing.FloatVector],
    internal_source_name: str,
    geo: geometry.Geometry,
    integration_fn: Callable[
        [array_typing.FloatVector, geometry.Geometry], jax.Array
    ],
) -> jax.Array:
  """Integrates a source profile if it exists, otherwise returns 0.0."""
  if internal_source_name in source_profiles_dict:
    return integration_fn(source_profiles_dict[internal_source_name], geo)
  else:
    return jnp.array(0.0, dtype=jax_utils.get_dtype())


def _calculate_integrated_sources(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    core_sources: source_profiles.SourceProfiles,
    runtime_params: runtime_params_lib.RuntimeParams,
) -> dict[str, jax.Array]:
  """Calculates total integrated internal and external source power and current.

  Args:
    geo: Magnetic geometry
    core_profiles: Kinetic profiles such as temperature and density
    core_sources: Internal and external sources
    runtime_params: Runtime parameters slice for the current time step

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

  # Initialize total particle sources to zero.
  integrated['S_total'] = jnp.array(0.0, dtype=jax_utils.get_dtype())

  # electron-ion heat exchange always exists, and is not in
  # core_sources.profiles, so we calculate it here.
  qei = core_sources.qei.qei_coef * (
      core_profiles.T_e.value - core_profiles.T_i.value
  )
  integrated['P_ei_exchange_i'] = math_utils.volume_integration(qei, geo)
  integrated['P_ei_exchange_e'] = -integrated['P_ei_exchange_i']

  # Initialize total electron and ion auxiliary powers.
  integrated['P_aux_i'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
  integrated['P_aux_e'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
  integrated['P_external_injected'] = jnp.array(
      0.0, dtype=jax_utils.get_dtype()
  )

  # Calculate integrated sources with convenient names, transformed from
  # TORAX internal names.
  for key, value in ION_EL_HEAT_SOURCE_TRANSFORMATIONS.items():
    is_in_T_i = key in core_sources.T_i
    is_in_T_e = key in core_sources.T_e
    if is_in_T_i != is_in_T_e:
      raise ValueError(
          f"Source '{key}' is expected to be defined for both ion and electron"
          ' channels (in core_sources.T_i and core_sources.T_e respectively).'
          f' Found in T_i: {is_in_T_i}, Found in T_e: {is_in_T_e}.'
      )
    integrated[f'{value}_i'] = _get_integrated_source_value(
        core_sources.T_i, key, geo, math_utils.volume_integration
    )
    integrated[f'{value}_e'] = _get_integrated_source_value(
        core_sources.T_e, key, geo, math_utils.volume_integration
    )
    integrated[f'{value}_total'] = (
        integrated[f'{value}_i'] + integrated[f'{value}_e']
    )

    if key in EXTERNAL_HEATING_SOURCES:
      integrated['P_aux_i'] += integrated[f'{value}_i']
      integrated['P_aux_e'] += integrated[f'{value}_e']

      # Track injected power for heating sources that have absorption_fraction
      # These are only for sources like ICRH or NBI that are
      # ion_el_heat_sources.
      source_params = runtime_params.sources.get(key)
      if source_params is not None and hasattr(
          source_params, 'absorption_fraction'
      ):
        total_absorbed = integrated[f'{value}_total']
        injected_power = total_absorbed / source_params.absorption_fraction
        integrated['P_external_injected'] += injected_power
      else:
        integrated['P_external_injected'] += integrated[f'{value}_total']

  for key, value in EL_HEAT_SOURCE_TRANSFORMATIONS.items():
    if key in core_sources.T_e and key in core_sources.T_i:
      raise ValueError(
          f"Source '{key}' was expected to only be an electron heat source (in"
          ' core_sources.T_e), but it was also found in ion heat sources'
          ' (core_sources.T_i).'
      )
    integrated[f'{value}'] = _get_integrated_source_value(
        core_sources.T_e, key, geo, math_utils.volume_integration
    )
    if key in EXTERNAL_HEATING_SOURCES:
      integrated['P_aux_e'] += integrated[f'{value}']
      integrated['P_external_injected'] += integrated[f'{value}']

  for key, value in CURRENT_SOURCE_TRANSFORMATIONS.items():
    integrated[value] = _get_integrated_source_value(
        core_sources.psi,
        key,
        geo,
        # Convert current sources to toroidal current before integrating
        lambda x, geo: math_utils.area_integration(
            psi_calculations.j_parallel_to_j_toroidal(
                x, geo, runtime_params.numerics.min_rho_norm
            ),
            geo,
        ),
    )

  for key, value in PARTICLE_SOURCE_TRANSFORMATIONS.items():
    integrated[f'{value}'] = _get_integrated_source_value(
        core_sources.n_e, key, geo, math_utils.volume_integration
    )
    integrated['S_total'] += integrated[f'{value}']

  integrated['P_aux_total'] = integrated['P_aux_i'] + integrated['P_aux_e']
  integrated['P_fusion'] = 5 * integrated['P_alpha_total']
  integrated['P_external_total'] = (
      integrated['P_external_injected'] + integrated['P_ohmic_e']
  )
  integrated['P_heat_e'] = (
      integrated['P_aux_e']
      + integrated['P_alpha_e']
      + integrated['P_ohmic_e']
      + integrated['P_ei_exchange_e']
      + integrated['P_cyclotron_e']
      + integrated['P_bremsstrahlung_e']
      + integrated['P_radiation_e']
  )

  integrated['P_heat_i'] = (
      integrated['P_aux_i']
      + integrated['P_alpha_i']
      + integrated['P_ei_exchange_i']
  )

  integrated['P_heat_total'] = integrated['P_heat_i'] + integrated['P_heat_e']

  return integrated


@jax.jit
def make_post_processed_outputs(
    sim_state: sim_state_lib.SimState,
    runtime_params: runtime_params_lib.RuntimeParams,
    previous_post_processed_outputs: PostProcessedOutputs,
) -> PostProcessedOutputs:
  """Calculates post-processed outputs based on the latest state.

  Args:
    sim_state: The state to add outputs to.
    runtime_params: Runtime parameters slice for the current time step, needed
      for calculating integrated power.
    previous_post_processed_outputs: The previous outputs, used to calculate
      cumulative quantities. If no previous outputs exist, then the
      `PostProcessedOutputs.zeros()` method can be used to create an object that
      can be used. In this case cumulative quantities will be set to zero. This
      is used for the first time step of a simulation.

  Returns:
    post_processed_outputs: The post_processed_outputs for the given state.
  """
  # TODO(b/444380540): Remove once aux outputs from sources are exposed.
  impurity_radiation_outputs = (
      impurity_radiation.calculate_impurity_species_output(
          sim_state, runtime_params
      )
  )

  pprime_face = formulas.calc_pprime(sim_state.core_profiles)
  # pylint: disable=invalid-name
  W_thermal_el, W_thermal_ion, W_thermal_tot = (
      formulas.calculate_stored_thermal_energy(
          sim_state.core_profiles.pressure_thermal_e,
          sim_state.core_profiles.pressure_thermal_i,
          sim_state.core_profiles.pressure_thermal_total,
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
      runtime_params,
  )
  # Calculate fusion gain with a zero division guard.
  Q_fusion = integrated_sources['P_fusion'] / (
      integrated_sources['P_external_total'] + constants.CONSTANTS.eps
  )

  P_LH_hi_dens, P_LH_min, P_LH, n_e_min_P_LH = (
      scaling_laws.calculate_plh_scaling_factor(
          sim_state.geometry, sim_state.core_profiles
      )
  )

  # Calculate dW/dt.
  # We perform raw calculation and smoothing inside a conditional block to
  # prevent division by zero on the first step (when dt=0) and to avoid
  # large transients (since previous W is initialized to 0).
  def _calculate_dW_dt_terms():
    # Raw values
    dW_i_dt_raw = (
        W_thermal_ion - previous_post_processed_outputs.W_thermal_i
    ) / sim_state.dt
    dW_e_dt_raw = (
        W_thermal_el - previous_post_processed_outputs.W_thermal_e
    ) / sim_state.dt
    dW_total_dt_raw = dW_i_dt_raw + dW_e_dt_raw

    # Calculate smoothing parameter
    alpha = jax.lax.cond(
        runtime_params.numerics.dW_dt_smoothing_time_scale > 0.0,
        lambda: jnp.array(1.0, dtype=jax_utils.get_dtype())
        - jnp.exp(
            -sim_state.dt / runtime_params.numerics.dW_dt_smoothing_time_scale
        ),
        lambda: jnp.array(1.0, dtype=jax_utils.get_dtype()),
    )

    dW_i_dt_smoothed = _exponential_smoothing(
        dW_i_dt_raw,
        previous_post_processed_outputs.dW_thermal_i_dt_smoothed,
        alpha,
    )
    dW_e_dt_smoothed = _exponential_smoothing(
        dW_e_dt_raw,
        previous_post_processed_outputs.dW_thermal_e_dt_smoothed,
        alpha,
    )
    dW_total_dt_smoothed = dW_i_dt_smoothed + dW_e_dt_smoothed

    return (
        dW_total_dt_raw,
        dW_total_dt_smoothed,
        dW_i_dt_smoothed,
        dW_e_dt_smoothed,
    )

  (
      dW_thermal_total_dt_raw,
      dW_thermal_total_dt_smoothed,
      dW_thermal_i_dt_smoothed,
      dW_thermal_e_dt_smoothed,
  ) = jax.lax.cond(
      previous_post_processed_outputs.first_step,
      lambda: (0.0, 0.0, 0.0, 0.0),
      _calculate_dW_dt_terms,
  )

  # Calculate P_SOL (Power crossing separatrix) = P_sources - P_sinks - dW/dt
  integrated_sources['P_SOL_i'] = (
      integrated_sources['P_heat_i'] - dW_thermal_i_dt_smoothed
  )

  integrated_sources['P_SOL_e'] = (
      integrated_sources['P_heat_e'] - dW_thermal_e_dt_smoothed
  )

  integrated_sources['P_SOL_total'] = (
      integrated_sources['P_SOL_i'] + integrated_sources['P_SOL_e']
  )

  # Calculate P_loss term used for confinement time calculations.
  # As per standard definitions, P_loss does not include radiation terms.
  # Therefore highly radiative scenarios can lead to skewed results.

  P_loss = (
      integrated_sources['P_alpha_total']
      + integrated_sources['P_aux_total']
      + integrated_sources['P_ohmic_e']
      - dW_thermal_total_dt_smoothed
      + constants.CONSTANTS.eps  # Division guard.
  )

  def cumulative_values():

    E_fusion = (
        previous_post_processed_outputs.E_fusion
        + sim_state.dt
        * (
            integrated_sources['P_fusion']
            + previous_post_processed_outputs.P_fusion
        )
        / 2.0
    )
    E_aux_total = (
        previous_post_processed_outputs.E_aux_total
        + sim_state.dt
        * (
            integrated_sources['P_aux_total']
            + previous_post_processed_outputs.P_aux_total
        )
        / 2.0
    )
    E_ohmic_e = (
        previous_post_processed_outputs.E_ohmic_e
        + sim_state.dt
        * (
            integrated_sources['P_ohmic_e']
            + previous_post_processed_outputs.P_ohmic_e
        )
        / 2.0
    )
    E_external_injected = (
        previous_post_processed_outputs.E_external_injected
        + sim_state.dt
        * (
            integrated_sources['P_external_injected']
            + previous_post_processed_outputs.P_external_injected
        )
        / 2.0
    )
    E_external_total = (
        previous_post_processed_outputs.E_external_total
        + sim_state.dt
        * (
            integrated_sources['P_external_total']
            + previous_post_processed_outputs.P_external_total
        )
        / 2.0
    )
    return (
        E_fusion,
        E_aux_total,
        E_ohmic_e,
        E_external_injected,
        E_external_total,
    )

  (
      E_fusion,
      E_aux_total,
      E_ohmic_e,
      E_external_injected,
      E_external_total,
  ) = jax.lax.cond(
      previous_post_processed_outputs.first_step,
      lambda: (0.0,) * 5,
      cumulative_values,
  )

  tau_E = W_thermal_tot / P_loss

  tauH89P = scaling_laws.calculate_scaling_law_confinement_time(
      sim_state.geometry, sim_state.core_profiles, P_loss, 'H89P'
  )
  tauH98 = scaling_laws.calculate_scaling_law_confinement_time(
      sim_state.geometry, sim_state.core_profiles, P_loss, 'H98'
  )
  tauH97L = scaling_laws.calculate_scaling_law_confinement_time(
      sim_state.geometry, sim_state.core_profiles, P_loss, 'H97L'
  )
  tauH20 = scaling_laws.calculate_scaling_law_confinement_time(
      sim_state.geometry, sim_state.core_profiles, P_loss, 'H20'
  )

  H89P = tau_E / tauH89P
  H98 = tau_E / tauH98
  H97L = tau_E / tauH97L
  H20 = tau_E / tauH20

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
  W_pol = psi_calculations.calc_Wpol(
      sim_state.geometry, sim_state.core_profiles.psi
  )
  li3 = psi_calculations.calc_li3(
      sim_state.geometry.R_major,
      W_pol,
      sim_state.core_profiles.Ip_profile_face[-1],
  )

  safety_factor_fit_outputs = (
      safety_factor_fit.find_min_q_and_q_surface_intercepts(
          sim_state.geometry.rho_face_norm,
          sim_state.core_profiles.q_face,
      )
  )

  # Parallel current densities
  # j_total is toroidal by default (see psi_calculations.calc_j_total)
  # Core sources psi are all <j.B>/B0
  j_parallel_total = psi_calculations.j_toroidal_to_j_parallel(
      sim_state.core_profiles.j_total,
      sim_state.geometry,
      runtime_params.numerics.min_rho_norm,
  )
  j_parallel_bootstrap = (
      sim_state.core_sources.bootstrap_current.j_parallel_bootstrap
  )
  j_parallel_external = sum(sim_state.core_sources.psi.values())
  j_parallel_ohmic = (
      j_parallel_total - j_parallel_external - j_parallel_bootstrap
  )

  # Toroidal current densities
  # j_total is toroidal by default (see psi_calculations.calc_j_total)
  # Core sources psi are all <j.B>/B0
  j_toroidal_bootstrap = psi_calculations.j_parallel_to_j_toroidal(
      j_parallel_bootstrap,
      sim_state.geometry,
      runtime_params.numerics.min_rho_norm,
  )

  # j_parallel_to_j_toroidal method cannot be used on face grid. Convert with
  # custom function for this.
  j_toroidal_bootstrap_face = _convert_j_parallel_face_to_j_toroidal_face(
      sim_state.core_sources.bootstrap_current.j_parallel_bootstrap_face,
      j_parallel_bootstrap,
      j_toroidal_bootstrap,
      sim_state.geometry,
  )

  j_toroidal_ohmic = psi_calculations.j_parallel_to_j_toroidal(
      j_parallel_ohmic,
      sim_state.geometry,
      runtime_params.numerics.min_rho_norm,
  )
  j_toroidal_external = psi_calculations.j_parallel_to_j_toroidal(
      j_parallel_external,
      sim_state.geometry,
      runtime_params.numerics.min_rho_norm,
  )
  j_toroidal_sources = {}
  for source_name in ['ecrh', 'generic_current']:
    if source_name in sim_state.core_sources.psi.keys():
      # TODO(b/434175938): rename j_* to j_toroidal_* for clarity
      j_toroidal_sources[f'j_{source_name}'] = (
          psi_calculations.j_parallel_to_j_toroidal(
              sim_state.core_sources.psi[source_name],
              sim_state.geometry,
              runtime_params.numerics.min_rho_norm,
          )
      )
    else:
      j_toroidal_sources[f'j_{source_name}'] = jnp.zeros_like(
          sim_state.geometry.rho
      )

  I_bootstrap = math_utils.area_integration(
      j_toroidal_bootstrap, sim_state.geometry
  )
  I_external = math_utils.area_integration(
      j_toroidal_external, sim_state.geometry
  )
  I_non_inductive = I_bootstrap + I_external

  beta_tor, beta_pol, beta_N = formulas.calculate_betas(
      sim_state.core_profiles, sim_state.geometry
  )

  _, radial_electric_field, poloidal_velocity = rotation.calculate_rotation(
      T_i=sim_state.core_profiles.T_i,
      psi=sim_state.core_profiles.psi,
      n_i=sim_state.core_profiles.n_i,
      q_face=sim_state.core_profiles.q_face,
      Z_eff_face=sim_state.core_profiles.Z_eff_face,
      Z_i_face=sim_state.core_profiles.Z_i_face,
      toroidal_angular_velocity=sim_state.core_profiles.toroidal_angular_velocity,
      pressure_thermal_i=sim_state.core_profiles.pressure_thermal_i,
      geo=sim_state.geometry,
      poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier,
  )

  return PostProcessedOutputs(
      pprime=pprime_face,
      W_thermal_i=W_thermal_ion,
      W_thermal_e=W_thermal_el,
      W_thermal_total=W_thermal_tot,
      tau_E=tau_E,
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
      E_fusion=E_fusion,
      E_aux_total=E_aux_total,
      E_ohmic_e=E_ohmic_e,
      E_external_injected=E_external_injected,
      E_external_total=E_external_total,
      T_e_volume_avg=te_volume_avg,
      T_i_volume_avg=ti_volume_avg,
      n_e_volume_avg=n_e_volume_avg,
      n_i_volume_avg=n_i_volume_avg,
      n_e_line_avg=n_e_line_avg,
      n_i_line_avg=n_i_line_avg,
      fgw_n_e_volume_avg=fgw_n_e_volume_avg,
      fgw_n_e_line_avg=fgw_n_e_line_avg,
      q95=q95,
      W_pol=W_pol,
      li3=li3,
      dW_thermal_dt=dW_thermal_total_dt_raw,
      dW_thermal_dt_smoothed=dW_thermal_total_dt_smoothed,
      dW_thermal_i_dt_smoothed=dW_thermal_i_dt_smoothed,
      dW_thermal_e_dt_smoothed=dW_thermal_e_dt_smoothed,
      rho_q_min=safety_factor_fit_outputs.rho_q_min,
      q_min=safety_factor_fit_outputs.q_min,
      rho_q_3_2_first=safety_factor_fit_outputs.rho_q_3_2_first,
      rho_q_2_1_first=safety_factor_fit_outputs.rho_q_2_1_first,
      rho_q_3_1_first=safety_factor_fit_outputs.rho_q_3_1_first,
      rho_q_3_2_second=safety_factor_fit_outputs.rho_q_3_2_second,
      rho_q_2_1_second=safety_factor_fit_outputs.rho_q_2_1_second,
      rho_q_3_1_second=safety_factor_fit_outputs.rho_q_3_1_second,
      I_bootstrap=I_bootstrap,
      j_parallel_total=j_parallel_total,
      j_parallel_ohmic=j_parallel_ohmic,
      j_ohmic=j_toroidal_ohmic,
      j_bootstrap=j_toroidal_bootstrap,
      j_bootstrap_face=j_toroidal_bootstrap_face,
      j_external=j_toroidal_external,
      j_ecrh=j_toroidal_sources['j_ecrh'],
      j_generic_current=j_toroidal_sources['j_generic_current'],
      j_non_inductive=j_toroidal_bootstrap + j_toroidal_external,
      j_parallel_external=j_parallel_external,
      j_parallel_non_inductive=j_parallel_bootstrap + j_parallel_external,
      I_external=I_external,
      I_non_inductive=I_non_inductive,
      f_non_inductive=math_utils.safe_divide(
          I_non_inductive, sim_state.core_profiles.Ip_profile_face[-1]
      ),
      f_bootstrap=math_utils.safe_divide(
          I_bootstrap, sim_state.core_profiles.Ip_profile_face[-1]
      ),
      beta_tor=beta_tor,
      beta_pol=beta_pol,
      beta_N=beta_N,
      impurity_species=impurity_radiation_outputs,
      poloidal_velocity=poloidal_velocity.face_value(),
      radial_electric_field=radial_electric_field.face_value(),
      first_step=jnp.array(False),
  )


def _convert_j_parallel_face_to_j_toroidal_face(
    j_parallel_face: array_typing.FloatVectorFace,
    j_parallel_cell: array_typing.FloatVectorCell,
    j_toroidal_cell: array_typing.FloatVectorCell,
    geo: geometry.Geometry,
) -> array_typing.FloatVectorFace:
  """Converts j_parallel on the face grid to j_toroidal on the face grid."""

  safe_denominator = jnp.where(
      jnp.abs(j_parallel_cell) > 1e-10,
      j_parallel_cell,
      1.0,
  )
  j_parallel_to_j_toroidal_factor_cell = jnp.where(
      jnp.abs(j_parallel_cell) > 1e-10,
      j_toroidal_cell / safe_denominator,
      1.0,
  )
  # Interpolate conversion factor to face grid with constant extrapolation.
  # Introduces a small and acceptable error on the face grid boundaries.
  j_parallel_to_j_toroidal_factor_face = jnp.interp(
      geo.rho_face_norm,
      geo.rho_norm,
      j_parallel_to_j_toroidal_factor_cell,
  )
  return j_parallel_to_j_toroidal_factor_face * j_parallel_face


def _exponential_smoothing(new_raw, old_smoothed, alpha):
  """Exponential moving average (EMA)."""
  return (1.0 - alpha) * old_smoothed + alpha * new_raw
