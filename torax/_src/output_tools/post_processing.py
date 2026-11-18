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

from collections.abc import Sequence
import dataclasses
from typing import Callable, Final, Self

from absl import logging
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.orchestration import sim_error
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.output_tools import impurity_radiation
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.output_tools import safety_factor_fit
from torax._src.physics import collisions
from torax._src.physics import formulas
from torax._src.physics import psi_calculations
from torax._src.physics import rotation
from torax._src.physics import scaling_laws
from torax._src.sources import core_sources as core_sources_lib


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
    tau_ei: Electron-ion collision time on the cell grid [s]
    H89P: L-mode confinement quality factor with respect to the ITER89P scaling
      law derived from the ITER L-mode confinement database
    H98: H-mode confinement quality factor with respect to the ITER98y2 scaling
      law derived from the ITER H-mode confinement database
    H97L: L-mode confinement quality factor with respect to the ITER97L scaling
      law derived from the ITER L-mode confinement database
    H20: H-mode confinement quality factor with respect to the ITER20 scaling
      law derived from the updated (2020) ITER H-mode confinement database
    FFprime: FF' on the face grid, where F is the toroidal flux function
    psi_norm: Normalized poloidal flux on the face grid [dimensionless]
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
    P_generic_heat_i: Total generic_heat source power to ions [W]
    P_generic_heat_e: Total generic_heat source power to electrons [W]
    P_generic_heat_total: Total generic_heat source power [W]
    P_alpha_i: Total fusion alpha heating power to ions [W]
    P_alpha_e: Total fusion alpha heating power to electrons [W]
    P_alpha_total: Total fusion alpha heating power to plasma [W]
    P_ohmic_e: Ohmic heating power to electrons [W]
    P_bremsstrahlung_e: Bremsstrahlung electron heat sink [W]
    P_cyclotron_radiation_e: Cyclotron radiation electron heat sink [W]
    P_ecrh_e: Total electron cyclotron source power [W]
    P_impurity_radiation_e: Impurity radiation electron heat sink [W]
    P_fusion: Generated fusion power (5*P_alpha_total) [W]
    I_ecrh: Total electron cyclotron source current [A]
    I_generic_current: Total generic_current source current [A]
    I_nbi: Total neutral beam injection source current [A]
    Q_fusion: Fusion power gain (P_fusion / P_external_total) [dimensionless]
    P_icrh_e: Ion cyclotron resonance heating to electrons [W]
    P_icrh_i: Ion cyclotron resonance heating to ions [W]
    P_icrh_total: Total ion cyclotron resonance heating power [W]
    P_nbi_e: Neutral beam injection heating power to electrons [W]
    P_nbi_i: Neutral beam injection heating power to ions [W]
    P_nbi_total: Total neutral beam injection heating power [W]
    P_LH_martin_high_density: H-mode transition power from the Martin scaling
      law for high density branch [W]
    P_LH_martin_min: Minimum H-mode transition power from the Martin scaling law
      at n_e_min_P_LH [W]
    P_LH_martin: H-mode transition power from the Martin scaling law, taken as
      the maximum of P_LH_martin_high_density and P_LH_martin_min [W]
    P_LH_martin_low_density: H-mode transition power from a heuristic extension
      of the Martin scaling law for low density branch [W]
    n_e_min_P_LH: Density corresponding to the P_LH_min, from the Ryter scaling
      law [m^-3]
    P_LH_delabie_high_density: H-mode transition power from the Delabie scaling
      law for high density branch [W]
    P_LH_delabie_min: Minimum H-mode transition power from the Delabie scaling
      law at n_e_min_P_LH [W]
    P_LH_delabie_low_density: H-mode transition power from a heuristic extension
      of the Delabie scaling law for low density branch [W]
    P_LH_delabie: H-mode transition power from the Delabie scaling law, taken as
      the maximum of P_LH_delabie_high_density and P_LH_delabie_min [W]
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
    f_GW_n_e_volume_avg: Greenwald fraction from volume-averaged electron
      density [dimensionless]
    f_GW_n_e_line_avg: Greenwald fraction from line-averaged electron density
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
    j_parallel_total: Total parallel current density [Am^-2]
    j_toroidal_bootstrap: Toroidal bootstrap current density [Am^-2]
    j_toroidal_bootstrap_face: Toroidal bootstrap current density on face grid
      [Am^-2]
    j_toroidal_external: Toroidal current density from external psi sources
      (i.e., excluding bootstrap) [A m^-2]
    j_parallel_external: Parallel current density from external psi sources
      (i.e., excluding bootstrap) [A m^-2]
    j_toroidal_ohmic: Toroidal ohmic current density [Am^-2]
    j_parallel_ohmic: Parallel ohmic current density [Am^-2]
    j_toroidal_generic_current: Toroidal current density from generic current
      source [Am^-2]
    j_toroidal_ecrh: Toroidal current density from electron cyclotron heating
      and current source [Am^-2]
    j_toroidal_nbi: Toroidal current density from neutral beam injection source
      [Am^-2]
    j_toroidal_non_inductive: Total toroidal non-inductive current density
      [Am^-2]
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
    S_nbi: Integrated neutral beam injection particle source [s^-1]
    S_total: Total integrated particle sources [s^-1]
    beta_tor: Volume-averaged toroidal plasma beta (thermal) [dimensionless]
    beta_pol: Volume-averaged poloidal plasma beta (thermal) [dimensionless]
    beta_N: Normalized toroidal plasma beta (thermal) [dimensionless].
    beta_pol_profile: Local poloidal beta profile on the face grid
      [dimensionless]
    beta_pol_prime: Derivative of local poloidal beta with respect to normalized
      poloidal flux on the face grid: -d(beta_pol_local) / d(psi_norm)
      [dimensionless]
    alpha_mhd: s-alpha (shifted-circle) MHD ballooning parameter on the face
      grid (Connor, Hastie & Taylor, Phys. Rev. Lett. 40, 396, 1978): -(2 * mu_0
        * R_major * q^2 / B_0^2) * d(p_total) / d(r_mid) [dimensionless]
    alpha_mhd_miller: General-geometry (Miller) MHD ballooning parameter on the
      face grid (Miller et al., Phys. Plasmas 5, 973, 1998): -(2 * dV/dpsi_rad)
      / (2 * pi)^2 * sqrt(V / (2 * pi^2 * R_0)) * mu_0 * dp_total/dpsi_rad
      [dimensionless]
    impurity_species: Dictionary of outputs for each impurity species.
    poloidal_velocity: Poloidal velocity [m/s]
    radial_electric_field: Radial electric field [V/m]
    first_step: Whether the outputs are from the first step of the simulation.
  """

  pprime: array_typing.FloatVector
  # pylint: disable=invalid-name
  # TODO(b/434175938) Remove W_thermal terms in favor of terms from energy_state
  W_thermal_i: array_typing.FloatScalar
  W_thermal_e: array_typing.FloatScalar
  W_thermal_total: array_typing.FloatScalar
  tau_E: array_typing.FloatScalar
  tau_ei: array_typing.FloatVector
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
  P_generic_heat_i: array_typing.FloatScalar
  P_generic_heat_e: array_typing.FloatScalar
  P_generic_heat_total: array_typing.FloatScalar
  P_alpha_i: array_typing.FloatScalar
  P_alpha_e: array_typing.FloatScalar
  P_alpha_total: array_typing.FloatScalar
  P_ohmic_e: array_typing.FloatScalar
  P_bremsstrahlung_e: array_typing.FloatScalar
  P_cyclotron_radiation_e: array_typing.FloatScalar
  P_ecrh_e: array_typing.FloatScalar
  P_impurity_radiation_e: array_typing.FloatScalar
  I_ecrh: array_typing.FloatScalar
  I_generic_current: array_typing.FloatScalar
  I_nbi: array_typing.FloatScalar
  P_fusion: array_typing.FloatScalar
  Q_fusion: array_typing.FloatScalar
  P_icrh_e: array_typing.FloatScalar
  P_icrh_i: array_typing.FloatScalar
  P_icrh_total: array_typing.FloatScalar
  P_nbi_e: array_typing.FloatScalar
  P_nbi_i: array_typing.FloatScalar
  P_nbi_total: array_typing.FloatScalar
  P_LH_martin_high_density: array_typing.FloatScalar
  P_LH_martin_min: array_typing.FloatScalar
  P_LH_martin_low_density: array_typing.FloatScalar
  P_LH_martin: array_typing.FloatScalar
  n_e_min_P_LH: array_typing.FloatScalar
  P_LH_delabie_high_density: array_typing.FloatScalar
  P_LH_delabie_min: array_typing.FloatScalar
  P_LH_delabie_low_density: array_typing.FloatScalar
  P_LH_delabie: array_typing.FloatScalar
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
  f_GW_n_e_volume_avg: array_typing.FloatScalar
  f_GW_n_e_line_avg: array_typing.FloatScalar
  q95: array_typing.FloatScalar
  W_pol: array_typing.FloatScalar
  li3: array_typing.FloatScalar
  # TODO(b/434175938) Remove dW_dt terms in favor of terms from energy_state.
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
  # Note: The default j profiles (j_toroidal_total, j_parallel_bootstrap,
  # j_parallel_* for * in sources) are saved elsewhere
  j_parallel_total: array_typing.FloatVector
  j_toroidal_external: array_typing.FloatVector
  j_parallel_external: array_typing.FloatVector
  j_toroidal_ohmic: array_typing.FloatVector
  j_parallel_ohmic: array_typing.FloatVector
  j_toroidal_bootstrap: array_typing.FloatVector
  j_toroidal_bootstrap_face: array_typing.FloatVector
  j_toroidal_generic_current: array_typing.FloatVector
  j_toroidal_ecrh: array_typing.FloatVector
  j_toroidal_nbi: array_typing.FloatVector
  j_toroidal_non_inductive: array_typing.FloatVector
  j_parallel_non_inductive: array_typing.FloatVector
  I_external: array_typing.FloatScalar
  I_non_inductive: array_typing.FloatScalar
  f_non_inductive: array_typing.FloatScalar
  f_bootstrap: array_typing.FloatScalar
  S_gas_puff: array_typing.FloatScalar
  S_pellet: array_typing.FloatScalar
  S_generic_particle: array_typing.FloatScalar
  S_nbi: array_typing.FloatScalar
  beta_tor: array_typing.FloatScalar
  beta_pol: array_typing.FloatScalar
  beta_N: array_typing.FloatScalar
  beta_pol_profile: array_typing.FloatVector
  beta_pol_prime: array_typing.FloatVector
  alpha_mhd: array_typing.FloatVector
  alpha_mhd_miller: array_typing.FloatVector
  S_total: array_typing.FloatScalar
  impurity_species: dict[str, impurity_radiation.ImpuritySpeciesOutput]
  poloidal_velocity: array_typing.FloatVector
  radial_electric_field: array_typing.FloatVector
  first_step: array_typing.BoolScalar
  # pylint: enable=invalid-name

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> Self:
    """Returns a PostProcessedOutputs with all zeros, used for initializing."""
    return cls(
        pprime=jnp.zeros(geo.rho_face.shape),
        W_thermal_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        W_thermal_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        W_thermal_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        tau_E=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        tau_ei=jnp.zeros(geo.rho.shape),
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
        P_generic_heat_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_generic_heat_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_generic_heat_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ohmic_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_bremsstrahlung_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_cyclotron_radiation_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ecrh_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_impurity_radiation_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_ecrh=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_generic_current=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_nbi=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_fusion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        Q_fusion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_nbi_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_nbi_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_nbi_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_martin_high_density=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_martin_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_martin_low_density=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_martin=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_e_min_P_LH=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_delabie_high_density=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_delabie_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_delabie_low_density=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_delabie=jnp.array(0.0, dtype=jax_utils.get_dtype()),
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
        f_GW_n_e_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        f_GW_n_e_line_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
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
        j_parallel_total=jnp.zeros(geo.rho.shape),
        j_toroidal_bootstrap=jnp.zeros(geo.rho.shape),
        j_toroidal_bootstrap_face=jnp.zeros(geo.rho_face.shape),
        j_toroidal_ohmic=jnp.zeros(geo.rho.shape),
        j_parallel_ohmic=jnp.zeros(geo.rho.shape),
        j_toroidal_external=jnp.zeros(geo.rho.shape),
        j_toroidal_generic_current=jnp.zeros(geo.rho.shape),
        j_toroidal_ecrh=jnp.zeros(geo.rho.shape),
        j_toroidal_nbi=jnp.zeros(geo.rho.shape),
        j_toroidal_non_inductive=jnp.zeros(geo.rho.shape),
        j_parallel_external=jnp.zeros(geo.rho.shape),
        j_parallel_non_inductive=jnp.zeros(geo.rho.shape),
        I_external=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_non_inductive=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        f_non_inductive=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        f_bootstrap=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        S_gas_puff=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        S_pellet=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        S_generic_particle=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        S_nbi=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        beta_tor=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        beta_pol=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        beta_N=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        beta_pol_profile=jnp.zeros(geo.rho_face.shape),
        beta_pol_prime=jnp.zeros(geo.rho_face.shape),
        alpha_mhd=jnp.zeros(geo.rho_face.shape),
        alpha_mhd_miller=jnp.zeros(geo.rho_face.shape),
        S_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        impurity_species={},
        poloidal_velocity=jnp.zeros(geo.rho_face.shape),
        radial_electric_field=jnp.zeros(geo.rho_face.shape),
        first_step=jnp.array(True),
    )

  @jax.jit
  def has_nan(self) -> jax.Array:
    """Checks if any leaf in PostProcessedOutputs contains NaNs."""
    return jnp.any(
        jnp.stack([jnp.any(jnp.isnan(x)) for x in jax.tree.leaves(self)])
    )

  def check_for_errors(self) -> sim_error.SimError:
    """Checks for NaNs in the post-processed outputs.

    Returns:
      SimError.NAN_DETECTED if any NaNs are present, otherwise
      SimError.NO_ERROR.
    """
    if self.has_nan():
      path_vals, _ = jax.tree.flatten_with_path(self)
      for path, value in path_vals:
        if np.any(np.isnan(value)):
          logging.info(
              'Found NaNs in post_processed_outputs%s',
              jax.tree_util.keystr(path),
          )
      return sim_error.SimError.NAN_DETECTED
    else:
      return sim_error.SimError.NO_ERROR

  def to_output_dict(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Packs time-stacked post-processed outputs into xarray-ready variables.

    Args:
      context: Grid context providing time and radial coordinates for shape
        validation and packing. Also registers `impurity` species coordinates
        in-place when `impurity_species` is non-empty.

    Returns:
      Mapping from xarray output variable names to `(dims, data, attrs)`
      `OutputVar` tuples:
        - Face profiles (`pprime`, `FFprime`, `psi_norm`, `poloidal_velocity`,
          `E_r`, beta/alpha profiles): `('time', 'rho_face_norm')`, shape
          `(n_time, n_rho_face)`.
        - Cell profiles (`j_*`, `tau_ei`): `('time', 'rho_cell_norm')`, shape
          `(n_time, n_rho_cell)`.
        - Extended bootstrap toroidal current (`j_toroidal_bootstrap`):
          `('time', 'rho_norm')`, shape `(n_time, n_rho_cell + 2)`.
        - Scalar quantities (stored energies, powers, averages, currents,
          beta, `q` diagnostics): `('time',)`, shape `(n_time,)`.
        - Per-species impurity radiation/density/charge (if present):
          `('impurity', 'time', 'rho_cell_norm')`, shape
          `(n_impurities, n_time, n_rho_cell)`.
    """
    out_dict: dict[str, output_grid_context.OutputVar] = {}

    # 1. Profiles on the face grid (FACE)
    face_vars = {
        output_keys.PPRIME: self.pprime,
        output_keys.FFPRIME: self.FFprime,
        output_keys.PSI_NORM: self.psi_norm,
        output_keys.POLOIDAL_VELOCITY: self.poloidal_velocity,
        output_keys.RADIAL_ELECTRIC_FIELD: self.radial_electric_field,
        output_keys.BETA_POL_PROFILE: self.beta_pol_profile,
        output_keys.BETA_POL_PRIME: self.beta_pol_prime,
        output_keys.ALPHA_MHD: self.alpha_mhd,
        output_keys.ALPHA_MHD_MILLER: self.alpha_mhd_miller,
    }
    for key, face_data in face_vars.items():
      out_dict[key] = context.pack(key, face_data)

    # 2. Profiles on the cell grid (CELL)
    cell_vars = {
        output_keys.TAU_EI: self.tau_ei,
        output_keys.J_PARALLEL_TOTAL: self.j_parallel_total,
        output_keys.J_TOROIDAL_EXTERNAL: self.j_toroidal_external,
        output_keys.J_PARALLEL_EXTERNAL: self.j_parallel_external,
        output_keys.J_TOROIDAL_OHMIC: self.j_toroidal_ohmic,
        output_keys.J_PARALLEL_OHMIC: self.j_parallel_ohmic,
        output_keys.J_TOROIDAL_GENERIC_CURRENT: self.j_toroidal_generic_current,
        output_keys.J_TOROIDAL_ECRH: self.j_toroidal_ecrh,
        output_keys.J_TOROIDAL_NBI: self.j_toroidal_nbi,
        output_keys.J_TOROIDAL_NON_INDUCTIVE: self.j_toroidal_non_inductive,
        output_keys.J_PARALLEL_NON_INDUCTIVE: self.j_parallel_non_inductive,
    }
    for key, cell_data in cell_vars.items():
      out_dict[key] = context.pack(key, cell_data)

    # 3. Profiles extended across boundaries (CELL_PLUS_BOUNDARIES)
    extended_j_bootstrap = output_grid_context.extend_cell_grid_to_boundaries(
        self.j_toroidal_bootstrap, self.j_toroidal_bootstrap_face
    )
    out_dict[output_keys.J_TOROIDAL_BOOTSTRAP] = context.pack(
        output_keys.J_TOROIDAL_BOOTSTRAP, extended_j_bootstrap
    )

    # 4. Scalars (SCALAR)
    scalar_vars = {
        # Stored energy and confinement
        output_keys.W_THERMAL_I: self.W_thermal_i,
        output_keys.W_THERMAL_E: self.W_thermal_e,
        output_keys.W_THERMAL_TOTAL: self.W_thermal_total,
        output_keys.TAU_E: self.tau_E,
        output_keys.H89P: self.H89P,
        output_keys.H98: self.H98,
        output_keys.H97L: self.H97L,
        output_keys.H20: self.H20,
        output_keys.W_POL: self.W_pol,
        output_keys.LI3: self.li3,
        output_keys.DW_THERMAL_DT: self.dW_thermal_dt,
        output_keys.DW_THERMAL_DT_SMOOTHED: self.dW_thermal_dt_smoothed,
        output_keys.DW_THERMAL_I_DT_SMOOTHED: self.dW_thermal_i_dt_smoothed,
        output_keys.DW_THERMAL_E_DT_SMOOTHED: self.dW_thermal_e_dt_smoothed,
        # Integrated powers
        output_keys.P_SOL_I: self.P_SOL_i,
        output_keys.P_SOL_E: self.P_SOL_e,
        output_keys.P_SOL_TOTAL: self.P_SOL_total,
        output_keys.P_AUX_I: self.P_aux_i,
        output_keys.P_AUX_E: self.P_aux_e,
        output_keys.P_AUX_TOTAL: self.P_aux_total,
        output_keys.P_EXTERNAL_INJECTED: self.P_external_injected,
        output_keys.P_EXTERNAL_TOTAL: self.P_external_total,
        output_keys.P_HEAT_I: self.P_heat_i,
        output_keys.P_HEAT_E: self.P_heat_e,
        output_keys.P_HEAT_TOTAL: self.P_heat_total,
        output_keys.P_EI_EXCHANGE_I: self.P_ei_exchange_i,
        output_keys.P_EI_EXCHANGE_E: self.P_ei_exchange_e,
        output_keys.P_GENERIC_HEAT_I: self.P_generic_heat_i,
        output_keys.P_GENERIC_HEAT_E: self.P_generic_heat_e,
        output_keys.P_GENERIC_HEAT_TOTAL: self.P_generic_heat_total,
        output_keys.P_ALPHA_I: self.P_alpha_i,
        output_keys.P_ALPHA_E: self.P_alpha_e,
        output_keys.P_ALPHA_TOTAL: self.P_alpha_total,
        output_keys.P_OHMIC_E: self.P_ohmic_e,
        output_keys.P_BREMSSTRAHLUNG_E: self.P_bremsstrahlung_e,
        output_keys.P_CYCLOTRON_RADIATION_E: self.P_cyclotron_radiation_e,
        output_keys.P_ECRH_E: self.P_ecrh_e,
        output_keys.P_IMPURITY_RADIATION_E: self.P_impurity_radiation_e,
        output_keys.P_FUSION: self.P_fusion,
        output_keys.P_ICRH_E: self.P_icrh_e,
        output_keys.P_ICRH_I: self.P_icrh_i,
        output_keys.P_ICRH_TOTAL: self.P_icrh_total,
        output_keys.P_NBI_E: self.P_nbi_e,
        output_keys.P_NBI_I: self.P_nbi_i,
        output_keys.P_NBI_TOTAL: self.P_nbi_total,
        # L-H transition thresholds
        output_keys.P_LH_MARTIN_HIGH_DENSITY: self.P_LH_martin_high_density,
        output_keys.P_LH_MARTIN_MIN: self.P_LH_martin_min,
        output_keys.P_LH_MARTIN_LOW_DENSITY: self.P_LH_martin_low_density,
        output_keys.P_LH_MARTIN: self.P_LH_martin,
        output_keys.N_E_MIN_P_LH: self.n_e_min_P_LH,
        output_keys.P_LH_DELABIE_HIGH_DENSITY: self.P_LH_delabie_high_density,
        output_keys.P_LH_DELABIE_MIN: self.P_LH_delabie_min,
        output_keys.P_LH_DELABIE_LOW_DENSITY: self.P_LH_delabie_low_density,
        output_keys.P_LH_DELABIE: self.P_LH_delabie,
        # Integrated energies
        output_keys.E_FUSION: self.E_fusion,
        output_keys.E_AUX_TOTAL: self.E_aux_total,
        output_keys.E_OHMIC_E: self.E_ohmic_e,
        output_keys.E_EXTERNAL_INJECTED: self.E_external_injected,
        output_keys.E_EXTERNAL_TOTAL: self.E_external_total,
        # Volume and line averages
        output_keys.T_E_VOLUME_AVG: self.T_e_volume_avg,
        output_keys.T_I_VOLUME_AVG: self.T_i_volume_avg,
        output_keys.N_E_VOLUME_AVG: self.n_e_volume_avg,
        output_keys.N_I_VOLUME_AVG: self.n_i_volume_avg,
        output_keys.N_E_LINE_AVG: self.n_e_line_avg,
        output_keys.N_I_LINE_AVG: self.n_i_line_avg,
        output_keys.F_GW_N_E_VOLUME_AVG: self.f_GW_n_e_volume_avg,
        output_keys.F_GW_N_E_LINE_AVG: self.f_GW_n_e_line_avg,
        # q-profile derived scalars
        output_keys.Q_FUSION: self.Q_fusion,
        output_keys.Q95: self.q95,
        output_keys.Q_MIN: self.q_min,
        output_keys.RHO_Q_MIN: self.rho_q_min,
        output_keys.RHO_Q_3_2_FIRST: self.rho_q_3_2_first,
        output_keys.RHO_Q_2_1_FIRST: self.rho_q_2_1_first,
        output_keys.RHO_Q_3_1_FIRST: self.rho_q_3_1_first,
        output_keys.RHO_Q_3_2_SECOND: self.rho_q_3_2_second,
        output_keys.RHO_Q_2_1_SECOND: self.rho_q_2_1_second,
        output_keys.RHO_Q_3_1_SECOND: self.rho_q_3_1_second,
        # Integrated currents and fractions
        output_keys.I_BOOTSTRAP: self.I_bootstrap,
        output_keys.I_EXTERNAL: self.I_external,
        output_keys.I_ECRH: self.I_ecrh,
        output_keys.I_GENERIC_CURRENT: self.I_generic_current,
        output_keys.I_NBI: self.I_nbi,
        output_keys.I_NON_INDUCTIVE: self.I_non_inductive,
        output_keys.F_NON_INDUCTIVE: self.f_non_inductive,
        output_keys.F_BOOTSTRAP: self.f_bootstrap,
        # Integrated particle sources
        output_keys.S_GAS_PUFF: self.S_gas_puff,
        output_keys.S_PELLET: self.S_pellet,
        output_keys.S_GENERIC_PARTICLE: self.S_generic_particle,
        output_keys.S_NBI: self.S_nbi,
        output_keys.S_TOTAL: self.S_total,
        # Plasma beta
        output_keys.BETA_TOR: self.beta_tor,
        output_keys.BETA_POL: self.beta_pol,
        output_keys.BETA_N: self.beta_N,
    }
    for key, scalar_data in scalar_vars.items():
      out_dict[key] = context.pack(key, scalar_data)

    # 5. Multi-dimensional impurity radiation outputs
    if self.impurity_species:
      context.register_coord(
          impurity_radiation.IMPURITY_DIM,
          np.asarray(list(self.impurity_species.keys())),
      )
      radiation_data = []
      n_impurity_data = []
      z_impurity_data = []
      for symbol in self.impurity_species:
        species_output = self.impurity_species[symbol]
        radiation_data.append(species_output.radiation)
        n_impurity_data.append(species_output.n_impurity)
        z_impurity_data.append(species_output.Z_impurity)
      rad_dims = (
          impurity_radiation.IMPURITY_DIM,
          output_keys.TIME,
          output_keys.RHO_CELL_NORM,
      )
      out_dict[output_keys.RADIATION_IMPURITY_SPECIES] = (
          rad_dims,
          np.stack(radiation_data, axis=0),
          output_keys.get_units(output_keys.RADIATION_IMPURITY_SPECIES),
      )
      out_dict[output_keys.N_IMPURITY_SPECIES] = (
          rad_dims,
          np.stack(n_impurity_data, axis=0),
          output_keys.get_units(output_keys.N_IMPURITY_SPECIES),
      )
      out_dict[output_keys.Z_IMPURITY_SPECIES] = (
          rad_dims,
          np.stack(z_impurity_data, axis=0),
          output_keys.get_units(output_keys.Z_IMPURITY_SPECIES),
      )

    return out_dict


# Integrated source scalar names follow the source's config / SOURCE_ID name
# directly via output_keys.SOURCE_NAME_RENAMES. The sole exception is 'fusion',
# whose thermal plasma heating is carried by alpha particles ('alpha'; total D-T
# fusion power P_fusion is 5 * P_alpha_total, so naming ion/electron thermal
# heating P_fusion_i / P_fusion_e would be misleading).
ION_EL_HEAT_SOURCES: Final[Sequence[str]] = (
    'generic_heat',
    'fusion',
    'icrh',
    'nbi',
)
EL_HEAT_SOURCES: Final[Sequence[str]] = (
    'ohmic',
    'bremsstrahlung',
    'cyclotron_radiation',
    'ecrh',
    'impurity_radiation',
)
EXTERNAL_HEATING_SOURCES: Final[Sequence[str]] = (
    'generic_heat',
    'ecrh',
    'icrh',
    'nbi',
)
CURRENT_SOURCES: Final[Sequence[str]] = (
    'generic_current',
    'ecrh',
    'nbi',
)
PARTICLE_SOURCES: Final[Sequence[str]] = (
    'gas_puff',
    'pellet',
    'generic_particle',
    'nbi',
)


def _get_integrated_source_value(
    profile: array_typing.FloatVector | None,
    geo: geometry.Geometry,
    integration_fn: Callable[
        [array_typing.FloatVector, geometry.Geometry], array_typing.FloatScalar
    ],
) -> jax.Array:
  """Integrates a source profile if it is not None, otherwise returns 0.0."""
  if profile is not None:
    return jnp.asarray(integration_fn(profile, geo))
  else:
    return jnp.array(0.0, dtype=jax_utils.get_dtype())


def _calculate_integrated_sources(
    geo: geometry.Geometry,
    core_sources: core_sources_lib.CoreSources,
    runtime_params: runtime_params_lib.RuntimeParams,
) -> dict[str, jax.Array]:
  """Calculates total integrated internal and external source power and current.

  Args:
    geo: Magnetic geometry
    core_sources: Internal and external sources
    runtime_params: Runtime parameters slice for the current time step

  Returns:
    Dictionary with integrated quantities for all existing sources.
    See PostProcessedOutputs for the full list of keys.

    Output dict used to update the `PostProcessedOutputs` object in the
    output `SimState`. Sources that don't exist do not have integrated
    quantities included in the returned dict. The corresponding
    `PostProcessedOutputs` attributes remain at their initialized zero values.
  """
  integrated = {}

  # Initialize total alpha power to zero. Needed for Q calculation.
  integrated['P_alpha_total'] = jnp.array(0.0, dtype=jax_utils.get_dtype())

  # Initialize total particle sources to zero.
  integrated['S_total'] = jnp.array(0.0, dtype=jax_utils.get_dtype())

  integrated['P_ei_exchange_i'] = math_utils.volume_integration(
      core_sources.qei.p_ei, geo
  )
  integrated['P_ei_exchange_e'] = -integrated['P_ei_exchange_i']

  # Initialize total electron and ion auxiliary powers.
  integrated['P_aux_i'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
  integrated['P_aux_e'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
  integrated['P_external_injected'] = jnp.array(
      0.0, dtype=jax_utils.get_dtype()
  )

  # Calculate integrated sources using unified output key names.
  for key in ION_EL_HEAT_SOURCES:
    source_output = core_sources.sources.get(key)
    power_density_i = (
        source_output.power_density_i if source_output is not None else None
    )
    power_density_e = (
        source_output.power_density_e if source_output is not None else None
    )
    if (power_density_i is not None) != (power_density_e is not None):
      raise ValueError(
          f"Source '{key}' is expected to be defined for both ion and electron"
          ' channels (power_density_i and power_density_e).'
          f' Found power_density_i: {power_density_i is not None},'
          f' Found power_density_e: {power_density_e is not None}.'
      )
    output_name = output_keys.SOURCE_NAME_RENAMES.get(key, key)
    integrated[f'P_{output_name}_i'] = _get_integrated_source_value(
        power_density_i, geo, math_utils.volume_integration
    )
    integrated[f'P_{output_name}_e'] = _get_integrated_source_value(
        power_density_e, geo, math_utils.volume_integration
    )
    integrated[f'P_{output_name}_total'] = (
        integrated[f'P_{output_name}_i'] + integrated[f'P_{output_name}_e']
    )

    if key in EXTERNAL_HEATING_SOURCES:
      integrated['P_aux_i'] += integrated[f'P_{output_name}_i']
      integrated['P_aux_e'] += integrated[f'P_{output_name}_e']

      # Track injected power for heating sources that have absorption_fraction
      # These are only for sources like ICRH or NBI that are
      # ion_el_heat_sources.
      source_params = runtime_params.sources.get(key)
      if source_params is not None and hasattr(
          source_params, 'absorption_fraction'
      ):
        total_absorbed = integrated[f'P_{output_name}_total']
        injected_power = total_absorbed / source_params.absorption_fraction
        integrated['P_external_injected'] += injected_power
      else:
        integrated['P_external_injected'] += integrated[
            f'P_{output_name}_total'
        ]

  for key in EL_HEAT_SOURCES:
    source_output = core_sources.sources.get(key)
    power_density_e = (
        source_output.power_density_e if source_output is not None else None
    )
    if (
        source_output is not None
        and power_density_e is not None
        and source_output.power_density_i is not None
    ):
      raise ValueError(
          f"Source '{key}' was expected to only be an electron heat source"
          ' (power_density_e), but it was also found in ion heat sources'
          ' (power_density_i).'
      )
    output_name = output_keys.SOURCE_NAME_RENAMES.get(key, key)
    scalar_key = f'P_{output_name}_e'
    integrated[scalar_key] = _get_integrated_source_value(
        power_density_e, geo, math_utils.volume_integration
    )
    if key in EXTERNAL_HEATING_SOURCES:
      integrated['P_aux_e'] += integrated[scalar_key]
      integrated['P_external_injected'] += integrated[scalar_key]

  for key in CURRENT_SOURCES:
    source_output = core_sources.sources.get(key)
    j_parallel = source_output.j_parallel if source_output is not None else None
    output_name = output_keys.SOURCE_NAME_RENAMES.get(key, key)
    integrated[f'I_{output_name}'] = _get_integrated_source_value(
        j_parallel,
        geo,
        # Convert current sources to toroidal current before integrating
        lambda x, geo: math_utils.area_integration(  # pyrefly: ignore[bad-argument-type]
            psi_calculations.j_parallel_to_j_toroidal(
                x, geo, runtime_params.numerics.min_rho_norm
            ),
            geo,
        ),
    )

  for key in PARTICLE_SOURCES:
    source_output = core_sources.sources.get(key)
    particle_source = (
        source_output.particle_source_e if source_output is not None else None
    )
    output_name = output_keys.SOURCE_NAME_RENAMES.get(key, key)
    scalar_key = f'S_{output_name}'
    integrated[scalar_key] = _get_integrated_source_value(
        particle_source, geo, math_utils.volume_integration
    )
    integrated['S_total'] += integrated[scalar_key]

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
      + integrated['P_cyclotron_radiation_e']
      + integrated['P_bremsstrahlung_e']
      + integrated['P_impurity_radiation_e']
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
  W_thermal_e, W_thermal_i, W_thermal_total = (
      formulas.calculate_stored_thermal_energy(
          sim_state.core_profiles.pressure_thermal_e,
          sim_state.core_profiles.pressure_thermal_i,
          sim_state.core_profiles.pressure_thermal_total,
          sim_state.geometry,
      )
  )

  def _first_step_energy():
    zero = jnp.array(0.0, dtype=jax_utils.get_dtype())
    return zero, zero, zero, zero

  def _subsequent_step_energy():
    smoothing_time = runtime_params.numerics.dW_dt_smoothing_time_scale
    dW_i_dt_raw, dW_i_smoothed = (
        formulas.calculate_time_derivative_with_exponential_smoothing(
            current_value=W_thermal_i,
            previous_value=previous_post_processed_outputs.W_thermal_i,
            previous_smoothed_derivative=(
                previous_post_processed_outputs.dW_thermal_i_dt_smoothed
            ),
            dt=sim_state.dt,
            smoothing_time_scale=smoothing_time,
        )
    )
    dW_e_dt_raw, dW_e_smoothed = (
        formulas.calculate_time_derivative_with_exponential_smoothing(
            current_value=W_thermal_e,
            previous_value=previous_post_processed_outputs.W_thermal_e,
            previous_smoothed_derivative=(
                previous_post_processed_outputs.dW_thermal_e_dt_smoothed
            ),
            dt=sim_state.dt,
            smoothing_time_scale=smoothing_time,
        )
    )
    dW_dt = dW_i_dt_raw + dW_e_dt_raw
    dW_smoothed = dW_i_smoothed + dW_e_smoothed
    return dW_dt, dW_smoothed, dW_i_smoothed, dW_e_smoothed

  (
      dW_thermal_dt,
      dW_thermal_dt_smoothed,
      dW_thermal_i_dt_smoothed,
      dW_thermal_e_dt_smoothed,
  ) = jax.lax.cond(
      previous_post_processed_outputs.first_step,
      _first_step_energy,
      _subsequent_step_energy,
  )
  # TODO(b/444380540): Remove once aux outputs from sources are exposed.
  impurity_radiation_outputs = (
      impurity_radiation.calculate_impurity_species_output(
          sim_state, runtime_params
      )
  )

  pprime_face = formulas.calc_pprime(sim_state.core_profiles)
  FFprime_face = formulas.calc_FFprime(
      sim_state.core_profiles, sim_state.geometry
  )
  # Calculate normalized poloidal flux.
  psi_norm_face = sim_state.core_profiles.psi_norm_face
  integrated_sources = _calculate_integrated_sources(
      sim_state.geometry,
      sim_state.core_sources,
      runtime_params,
  )
  # Calculate fusion gain with a zero division guard.
  Q_fusion = integrated_sources['P_fusion'] / (
      integrated_sources['P_external_total'] + constants.CONSTANTS.eps
  )

  P_LH_martin, P_LH_martin_aux_data = scaling_laws.calculate_P_LH(
      sim_state.geometry,
      sim_state.core_profiles,
      scaling_law=scaling_laws.PLHScalingLaw.MARTIN,
  )
  P_LH_delabie, P_LH_delabie_aux_data = scaling_laws.calculate_P_LH(
      sim_state.geometry,
      sim_state.core_profiles,
      scaling_law=scaling_laws.PLHScalingLaw.DELABIE,
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
      - dW_thermal_dt_smoothed
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

  tau_E = W_thermal_total / P_loss
  log_lambda_ei = collisions.calculate_log_lambda_ei(
      sim_state.core_profiles.kinetic.T_e.value,
      sim_state.core_profiles.kinetic.n_e.value,
  )
  tau_ei = collisions.calculate_tau_e(
      sim_state.core_profiles.kinetic.T_e.value,
      sim_state.core_profiles.kinetic.n_e.value,
      sim_state.core_profiles.ions.Z_eff,
      log_lambda_ei,
  )

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
  q95 = psi_calculations.calc_q95(
      psi_norm_face, sim_state.core_profiles.poloidal_flux.q_face
  )

  # Calculate te and ti volume average [keV]
  te_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.kinetic.T_e.value, sim_state.geometry
  )
  ti_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.kinetic.T_i.value, sim_state.geometry
  )

  # Calculate n_e and n_i (main ion) volume and line averages in m^-3
  n_e_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.kinetic.n_e.value, sim_state.geometry
  )
  n_i_volume_avg = math_utils.volume_average(
      sim_state.core_profiles.kinetic.n_i.value, sim_state.geometry
  )
  n_e_line_avg = math_utils.line_average(
      sim_state.core_profiles.kinetic.n_e.value, sim_state.geometry
  )
  n_i_line_avg = math_utils.line_average(
      sim_state.core_profiles.kinetic.n_i.value, sim_state.geometry
  )
  f_GW_n_e_volume_avg = formulas.calculate_greenwald_fraction(
      n_e_volume_avg, sim_state.core_profiles, sim_state.geometry
  )
  f_GW_n_e_line_avg = formulas.calculate_greenwald_fraction(
      n_e_line_avg, sim_state.core_profiles, sim_state.geometry
  )
  W_pol = psi_calculations.calc_Wpol(
      sim_state.geometry, sim_state.core_profiles.poloidal_flux.psi
  )
  li3 = psi_calculations.calc_li3(
      sim_state.geometry.R_major,
      W_pol,
      sim_state.core_profiles.poloidal_flux.I_p_profile_face[-1],
  )

  safety_factor_fit_outputs = (
      safety_factor_fit.find_min_q_and_q_surface_intercepts(
          sim_state.geometry.rho_face_norm,
          sim_state.core_profiles.poloidal_flux.q_face,
      )
  )

  # Parallel current densities
  # j_toroidal_total is toroidal by default (see
  # psi_calculations.calc_j_toroidal_total)
  # Core sources psi are all <j.B>/B0
  j_parallel_total = psi_calculations.j_toroidal_to_j_parallel(
      sim_state.core_profiles.poloidal_flux.j_toroidal_total,
      sim_state.geometry,
      runtime_params.numerics.min_rho_norm,
  )
  j_parallel_bootstrap = (
      sim_state.core_sources.bootstrap_current.j_parallel_bootstrap
  )
  j_parallel_external = sim_state.core_sources.j_parallel(sim_state.geometry)
  j_parallel_ohmic = (
      j_parallel_total - j_parallel_external - j_parallel_bootstrap
  )

  # Toroidal current densities
  # j_toroidal_total is toroidal by default (see
  # psi_calculations.calc_j_toroidal_total)
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
  for source_name in ['ecrh', 'generic_current', 'nbi']:
    source_output = sim_state.core_sources.sources.get(source_name)
    if source_output is not None and source_output.j_parallel is not None:
      j_toroidal_sources[f'j_toroidal_{source_name}'] = (
          psi_calculations.j_parallel_to_j_toroidal(
              source_output.j_parallel,
              sim_state.geometry,
              runtime_params.numerics.min_rho_norm,
          )
      )
    else:
      j_toroidal_sources[f'j_toroidal_{source_name}'] = jnp.zeros_like(
          sim_state.geometry.rho
      )

  I_bootstrap = math_utils.area_integration(
      j_toroidal_bootstrap, sim_state.geometry
  )
  I_external = math_utils.area_integration(
      j_toroidal_external, sim_state.geometry
  )
  I_non_inductive = I_bootstrap + I_external

  beta_tor, beta_pol, beta_N = formulas.calculate_betas(  # pyrefly: ignore[not-iterable]
      sim_state.core_profiles, sim_state.geometry
  )
  beta_pol_profile = formulas.calculate_beta_pol_profile(
      sim_state.core_profiles, sim_state.geometry
  )
  beta_pol_prime = formulas.calculate_beta_pol_prime(
      sim_state.core_profiles, sim_state.geometry
  )
  alpha_mhd = formulas.calculate_alpha_mhd(
      sim_state.core_profiles, sim_state.geometry
  )
  alpha_mhd_miller = formulas.calculate_alpha_mhd_miller(
      sim_state.core_profiles, sim_state.geometry
  )

  rotation_output = rotation.calculate_rotation(
      psi=sim_state.core_profiles.poloidal_flux.psi,
      n_i=sim_state.core_profiles.kinetic.n_i,
      Z_i_face=sim_state.core_profiles.ions.Z_i_face,
      toroidal_angular_velocity=(
          sim_state.core_profiles.kinetic.toroidal_angular_velocity
      ),
      poloidal_velocity=sim_state.core_profiles.poloidal_velocity,
      pressure_total_i=sim_state.core_profiles.pressure_total_i,
      geo=sim_state.geometry,
  )

  return PostProcessedOutputs(
      pprime=pprime_face,
      W_thermal_i=W_thermal_i,
      W_thermal_e=W_thermal_e,
      W_thermal_total=W_thermal_total,
      tau_E=tau_E,
      tau_ei=tau_ei,
      H89P=H89P,
      H98=H98,
      H97L=H97L,
      H20=H20,
      FFprime=FFprime_face,
      psi_norm=psi_norm_face,
      **integrated_sources,
      Q_fusion=Q_fusion,
      P_LH_martin=P_LH_martin,
      P_LH_martin_min=P_LH_martin_aux_data.P_LH_min,
      P_LH_martin_high_density=P_LH_martin_aux_data.P_LH_high_density,
      P_LH_martin_low_density=P_LH_martin_aux_data.P_LH_low_density,
      n_e_min_P_LH=P_LH_martin_aux_data.line_average_n_e_at_P_LH_min,
      P_LH_delabie=P_LH_delabie,
      P_LH_delabie_min=P_LH_delabie_aux_data.P_LH_min,
      P_LH_delabie_high_density=P_LH_delabie_aux_data.P_LH_high_density,
      P_LH_delabie_low_density=P_LH_delabie_aux_data.P_LH_low_density,
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
      f_GW_n_e_volume_avg=f_GW_n_e_volume_avg,
      f_GW_n_e_line_avg=f_GW_n_e_line_avg,
      q95=q95,
      W_pol=W_pol,
      li3=li3,
      dW_thermal_dt=dW_thermal_dt,
      dW_thermal_dt_smoothed=dW_thermal_dt_smoothed,
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
      j_toroidal_ohmic=j_toroidal_ohmic,
      j_toroidal_bootstrap=j_toroidal_bootstrap,
      j_toroidal_bootstrap_face=j_toroidal_bootstrap_face,
      j_toroidal_external=j_toroidal_external,
      j_toroidal_ecrh=j_toroidal_sources['j_toroidal_ecrh'],
      j_toroidal_generic_current=j_toroidal_sources[
          'j_toroidal_generic_current'
      ],
      j_toroidal_nbi=j_toroidal_sources['j_toroidal_nbi'],
      j_toroidal_non_inductive=j_toroidal_bootstrap + j_toroidal_external,
      j_parallel_external=j_parallel_external,
      j_parallel_non_inductive=j_parallel_bootstrap + j_parallel_external,
      I_external=I_external,
      I_non_inductive=I_non_inductive,
      f_non_inductive=math_utils.safe_divide(
          num=I_non_inductive,
          denom=sim_state.core_profiles.poloidal_flux.I_p_profile_face[-1],
          eps=1e-7,
      ),
      f_bootstrap=math_utils.safe_divide(
          num=I_bootstrap,
          denom=sim_state.core_profiles.poloidal_flux.I_p_profile_face[-1],
          eps=1e-7,
      ),
      beta_tor=beta_tor,
      beta_pol=beta_pol,
      beta_N=beta_N,
      beta_pol_profile=beta_pol_profile.face_value(),
      beta_pol_prime=beta_pol_prime,
      alpha_mhd=alpha_mhd,
      alpha_mhd_miller=alpha_mhd_miller,
      impurity_species=impurity_radiation_outputs,
      poloidal_velocity=sim_state.core_profiles.poloidal_velocity.face_value(),
      radial_electric_field=rotation_output.Er,
      first_step=jnp.array(False),
  )


@jax.jit
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


@jax.jit
def _exponential_smoothing(new_raw, old_smoothed, alpha):
  """Exponential moving average (EMA)."""
  return (1.0 - alpha) * old_smoothed + alpha * new_raw
