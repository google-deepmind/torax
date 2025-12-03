# Copyright 2025 DeepMind Technologies Limited
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

"""Defines the data structures and core logic for the 1D divertor/SOL model.

This module contains:
- `ExtendedLengyelParameters`: Dataclass for static physics and geometry inputs.
- `ExtendedLengyelState`: Dataclass for the dynamic state variables (unknowns)
  that are evolved by the solver.
- `DivertorSOL1D`: A computational object that combines parameters and state to
  provide on-demand calculation of derived physical quantities through its
  properties.
- State update functions (`calc_q_parallel`, `calc_alpha_t`, etc.) used in the
  iterative solver loops.

This module represents the structure and interconnectedness of the physics
model, while `extended_lengyel_formulas.py` provides more fundamental,
self-contained physics formulas that are used as building blocks here.
"""

import dataclasses
from typing import Mapping
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_formulas

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ExtendedLengyelParameters:
  """Static inputs and physics parameters fixed for a model run."""

  # --- Geometric and Magnetic Parameters ---
  major_radius: array_typing.FloatScalar  # [m]
  minor_radius: array_typing.FloatScalar  # [m]
  connection_length_divertor: array_typing.FloatScalar  # [m]
  connection_length_target: array_typing.FloatScalar  # [m]
  separatrix_average_poloidal_field: array_typing.FloatScalar  # [T]
  ratio_bpol_omp_to_bpol_avg: array_typing.FloatScalar  # [dimensionless]
  fieldline_pitch_at_omp: array_typing.FloatScalar  # [dimensionless]
  cylindrical_safety_factor: array_typing.FloatScalar  # [dimensionless]
  toroidal_flux_expansion: array_typing.FloatScalar  # [dimensionless]
  angle_of_incidence_target: array_typing.FloatScalar  # [degrees]

  # --- Plasma and Physics Config ---
  power_crossing_separatrix: array_typing.FloatScalar  # [W]
  separatrix_electron_density: array_typing.FloatScalar  # [m^-3]
  main_ion_charge: array_typing.FloatScalar  # [dimensionless]
  average_ion_mass: array_typing.FloatScalar  # [amu]
  mean_ion_charge_state: array_typing.FloatScalar  # [dimensionless]
  ne_tau: array_typing.FloatScalar  # [s m^-3]
  fraction_of_P_SOL_to_divertor: array_typing.FloatScalar  # [dimensionless]
  SOL_conduction_fraction: array_typing.FloatScalar  # [dimensionless]
  divertor_broadening_factor: array_typing.FloatScalar  # [dimensionless]
  sheath_heat_transmission_factor: array_typing.FloatScalar  # [dimensionless]
  ratio_of_molecular_to_ion_mass: array_typing.FloatScalar  # [dimensionless]
  T_wall: array_typing.FloatScalar  # [K]

  # --- Boundary Condition Data: all dimensionless ---
  mach_separatrix: array_typing.FloatScalar
  T_i_T_e_ratio_separatrix: array_typing.FloatScalar
  n_e_n_i_ratio_separatrix: array_typing.FloatScalar
  mach_target: array_typing.FloatScalar
  T_i_T_e_ratio_target: array_typing.FloatScalar
  n_e_n_i_ratio_target: array_typing.FloatScalar

  # --- Impurity Config ---
  seed_impurity_weights: Mapping[
      str, array_typing.FloatScalar
  ]  # [dimensionless]
  fixed_impurity_concentrations: Mapping[
      str, array_typing.FloatScalar
  ]  # [dimensionless] (n_e_ratio)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ExtendedLengyelState:
  """Varying state variables (unknowns) evolved by the solvers."""

  q_parallel: array_typing.FloatScalar  # [W/m^2]
  alpha_t: array_typing.FloatScalar  # [dimensionless]
  kappa_e: array_typing.FloatScalar  # [W/(m*eV^3.5)]
  T_e_target: array_typing.FloatScalar  # [eV]
  c_z_prefactor: array_typing.FloatScalar  # [m^-3]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DivertorSOL1D:
  """Compute object providing on-demand physics properties.

  Combines static parameters and dynamic state to calculate derived
  quantities via @property methods.
  """

  params: ExtendedLengyelParameters
  state: ExtendedLengyelState

  @property
  def electron_temp_at_cc_interface(self) -> jax.Array:
    """Calculates electron temperature at the convection/conduction interface.

    This function determines the electron temperature at the boundary between
    the convection-dominated sheath/divertor region and the upstream conduction
    layer. It uses empirical fit functions for momentum and density loss within
    the convection layer, which depend on the electron temperature at the
    divertor target. The formula relates the interface temperature to the target
    temperature modified by these loss factors.

    See section 4 of T. Body et al 2025 Nucl. Fusion 65 086002 for details.
    https://doi.org/10.1088/1741-4326/ade4d9
    """
    momentum_loss = (
        extended_lengyel_formulas.calc_momentum_loss_in_convection_layer(
            self.state.T_e_target
        )
    )
    density_ratio = (
        extended_lengyel_formulas.calc_density_ratio_in_convection_layer(
            self.state.T_e_target
        )
    )
    return self.state.T_e_target / (
        (1.0 - momentum_loss) / (2.0 * density_ratio)
    )

  @property
  def divertor_entrance_electron_temp(self) -> jax.Array:
    """Electron temperature at the divertor entrance [eV].

    This formula is derived from the heat conduction equation integrated
    along the scrape-off layer.

    Eq 44, Body et al. 2025. https://doi.org/10.1088/1741-4326/ade4d9
    """
    return (
        self.electron_temp_at_cc_interface**3.5
        + 3.5
        * self.params.SOL_conduction_fraction
        * self.state.q_parallel
        / self.params.divertor_broadening_factor
        * self.params.connection_length_divertor
        / self.state.kappa_e
    ) ** (2.0 / 7.0)

  @property
  def T_e_separatrix(self) -> jax.Array:
    """Electron temperature at the separatrix [eV].

    This formula is derived from the heat conduction equation integrated
    along the scrape-off layer.

    Eq 45, Body et al. 2025. https://doi.org/10.1088/1741-4326/ade4d9
    """
    return (
        self.divertor_entrance_electron_temp**3.5
        + 3.5
        * self.params.SOL_conduction_fraction
        * self.state.q_parallel
        * (
            self.params.connection_length_target
            - self.params.connection_length_divertor
        )
        / self.state.kappa_e
    ) ** (2.0 / 7.0)

  @property
  def separatrix_total_pressure(self) -> jax.Array:
    """Total pressure at the separatrix [Pa].

    This is the definition of total pressure (static + dynamic) at the
    separatrix, including both electron and ion contributions.
    """
    return (
        (1.0 + self.params.mach_separatrix**2)
        * self.params.separatrix_electron_density
        * self.T_e_separatrix
        * constants.CONSTANTS.eV_to_J
        * (
            1.0
            + self.params.T_i_T_e_ratio_separatrix
            / self.params.n_e_n_i_ratio_separatrix
        )
    )

  @property
  def required_power_loss(self) -> jax.Array:
    """Required power loss fraction from the two-point model.

    Calculate momentum loss in the convection layer using an empirical fit.

    Eq 46, Body et al. 2025. https://doi.org/10.1088/1741-4326/ade4d9
    """
    momentum_loss = (
        extended_lengyel_formulas.calc_momentum_loss_in_convection_layer(
            self.state.T_e_target
        )
    )

    log_T_e_target_basic = (
        jnp.log(8.0)
        + jnp.log(self.params.average_ion_mass)
        + jnp.log(constants.CONSTANTS.m_amu)
        - 2.0 * jnp.log(self.params.sheath_heat_transmission_factor)
        + 2.0 * jnp.log(self.state.q_parallel)
        - 2.0 * jnp.log(self.separatrix_total_pressure)
        - jnp.log(constants.CONSTANTS.q_e)
    )

    T_e_target_basic = jnp.exp(log_T_e_target_basic)

    f_other_T_e_target = (
        (
            (1.0 + self.params.T_i_T_e_ratio_target)
            / (self.params.n_e_n_i_ratio_target * 2.0)
        )
        * (
            (1.0 + self.params.mach_target**2) ** 2
            / (4.0 * self.params.mach_target**2)
        )
        * self.params.toroidal_flux_expansion**-2
    )

    return 1.0 - jnp.sqrt(
        self.state.T_e_target
        / T_e_target_basic
        * (1.0 - momentum_loss) ** 2
        / f_other_T_e_target
    )

  @property
  def parallel_heat_flux_at_target(self) -> jax.Array:
    """Parallel heat flux at the divertor target [W/m^2]."""
    return self.state.q_parallel * (1.0 - self.required_power_loss)

  @property
  def parallel_heat_flux_at_cc_interface(self) -> jax.Array:
    """Parallel heat flux at the convection-conduction interface [W/m^2].

    Eq 29, Body et al. 2025. https://doi.org/10.1088/1741-4326/ade4d9
    """
    power_loss_conv_layer = (
        extended_lengyel_formulas.calc_power_loss_in_convection_layer(
            self.state.T_e_target
        )
    )
    return self.parallel_heat_flux_at_target / (1.0 - power_loss_conv_layer)

  @property
  def divertor_Z_eff(self) -> jax.Array:
    return extended_lengyel_formulas.calc_Z_eff(
        c_z=self.state.c_z_prefactor,
        T_e=self.divertor_entrance_electron_temp / 1e3,  # to keV
        Z_i=self.params.main_ion_charge,
        ne_tau=self.params.ne_tau,
        seed_impurity_weights=self.params.seed_impurity_weights,
        fixed_impurity_concentrations=self.params.fixed_impurity_concentrations,
    )

  @property
  def Z_eff_separatrix(self) -> jax.Array:
    return extended_lengyel_formulas.calc_Z_eff(
        c_z=self.state.c_z_prefactor,
        T_e=self.T_e_separatrix / 1e3,  # to keV
        Z_i=self.params.main_ion_charge,
        ne_tau=self.params.ne_tau,
        seed_impurity_weights=self.params.seed_impurity_weights,
        fixed_impurity_concentrations=self.params.fixed_impurity_concentrations,
    )

  @property
  def seed_impurity_concentrations(
      self,
  ) -> Mapping[str, array_typing.FloatScalar]:
    return {
        key: value * self.state.c_z_prefactor
        for key, value in self.params.seed_impurity_weights.items()
    }


def calc_q_parallel(
    *,
    params: ExtendedLengyelParameters,
    T_e_separatrix: array_typing.FloatScalar,
    alpha_t: array_typing.FloatScalar,
) -> jax.Array:
  """Calculates the parallel heat flux density.

  For the flux-tube assumed in the extended Lengyel model.
  See T. Body et al 2025 Nucl. Fusion 65 086002 for details.
  https://doi.org/10.1088/1741-4326/ade4d9

  Args:
    params: fixed physics parameters in the divertor/SOL region.
    T_e_separatrix: Electron temperature at the separatrix [eV].
    alpha_t: Turbulence broadening parameter alpha_t.

  Returns:
    q_parallel: Parallel heat flux density [W/m^2].
  """

  # Body NF 2025 Eq 53.
  separatrix_average_rho_s_pol = (
      jnp.sqrt(
          T_e_separatrix
          * params.average_ion_mass
          * constants.CONSTANTS.m_amu
          / constants.CONSTANTS.q_e
      )
      / params.separatrix_average_poloidal_field
  )

  # Body NF 2025 Eq 49.
  separatrix_average_lambda_q = (
      0.6 * (1.0 + 2.1 * alpha_t**1.7) * separatrix_average_rho_s_pol
  )

  # Scaling lambda_q by the scalings of the average and upstream toroidal and
  # poloidal fields. Body NF 2025 Eq 50.
  ratio_of_upstream_to_average_lambda_q = (
      params.ratio_bpol_omp_to_bpol_avg
      * (params.major_radius + params.minor_radius)
      / params.major_radius
  )
  lambda_q_outboard_midplane = (
      separatrix_average_lambda_q / ratio_of_upstream_to_average_lambda_q
  )

  # Power reduction for the fraction of power inside one e-folding length
  # (lambda_q).
  fraction_of_power_entering_flux_tube = (
      1.0 - 1.0 / jnp.e
  ) * params.fraction_of_P_SOL_to_divertor

  # Parallel heat flux at the target.
  # Body NF 2025 Eq 48.

  q_parallel = (
      params.power_crossing_separatrix
      * fraction_of_power_entering_flux_tube
      / (
          2.0
          * jnp.pi
          * (params.major_radius + params.minor_radius)
          * lambda_q_outboard_midplane
      )
      * params.fieldline_pitch_at_omp
  )

  return q_parallel


def calc_alpha_t(
    params: ExtendedLengyelParameters,
    T_e_separatrix: array_typing.FloatScalar,
    Z_eff_separatrix: array_typing.FloatScalar,
) -> array_typing.FloatScalar:
  """Calculate the turbulence broadening parameter alpha_t.

  Equation 9 from T. Eich et al. Nuclear Fusion, 60(5), 056016. (2020),
  with an additional factor of an ion_to_electron_temp_ratio.
  https://doi.org/10.1088/1741-4326/ab7a66

  Args:
    params: fixed physics parameters in the divertor/SOL region.
    T_e_separatrix: electron temperature at the separatrix [eV].
    Z_eff_separatrix: effective ion charge at the separatrix [dimensionless].

  Returns:
    alpha_t: the turbulence parameter alpha_t.
  """
  average_ion_mass_kg = params.average_ion_mass * constants.CONSTANTS.m_amu

  # Variant from Verdoolaege et al., 2021 Nucl. Fusion 61 076006.
  # Differs from Wesson 3rd edition p727 by a small absolute value of 0.1.
  coulomb_logarithm = (
      30.9
      - 0.5 * jnp.log(params.separatrix_electron_density)
      + jnp.log(T_e_separatrix)
  )

  # Plasma ion sound speed. Differs from that stated in Eich 2020 by the
  # inclusion of the mean ion charge state.
  ion_sound_speed = jnp.sqrt(
      params.mean_ion_charge_state
      * T_e_separatrix
      * constants.CONSTANTS.eV_to_J
      / average_ion_mass_kg
  )

  # electron-electron collision frequency. Equation B1 from Eich 2020.
  # In log space to avoid over/underflows in fp32.
  log_nu_ee = (
      jnp.log(4.0 / 3.0)
      + 0.5 * jnp.log(2.0 * jnp.pi)
      + jnp.log(params.separatrix_electron_density)
      + 4 * jnp.log(constants.CONSTANTS.q_e)
      + jnp.log(coulomb_logarithm)
      - 2 * jnp.log(4.0 * jnp.pi * constants.CONSTANTS.epsilon_0)
      - 0.5 * jnp.log(constants.CONSTANTS.m_e)
      - 1.5 * jnp.log(T_e_separatrix * constants.CONSTANTS.eV_to_J)
  )

  nu_ee = jnp.exp(log_nu_ee)

  # Z_eff correction to transform electron-electron collisions to ion-electron
  # collisions. Equation B2 in Eich 2020
  Z_eff_correction = (1.0 - 0.569) * jnp.exp(
      -(((Z_eff_separatrix - 1.0) / 3.25) ** 0.85)
  ) + 0.569

  nu_ei = nu_ee * Z_eff_correction * Z_eff_separatrix

  # Equation 9 from Eich 2020, with an additional factor of an
  # ion_to_electron_temp_ratio.
  alpha_t = (
      1.02
      * nu_ei
      / ion_sound_speed
      * (1.0 * constants.CONSTANTS.m_e / average_ion_mass_kg)
      * params.cylindrical_safety_factor**2
      * params.major_radius
      * (1.0 + params.T_i_T_e_ratio_separatrix / params.mean_ion_charge_state)
  )

  return alpha_t


def calc_T_e_target(
    sol_model: DivertorSOL1D,
    parallel_heat_flux_at_cc_interface: array_typing.FloatScalar,
) -> jax.Array:
  """Calculate the target electron temp from the two-point model.

  Args:
    sol_model: The model object for the extended Lengyel model.
    parallel_heat_flux_at_cc_interface: The parallel heat flux at the
      convection-conduction interface pre-calculated with forward model [W/m^2].

  Returns:
    The target electron temperature [eV].

  Eq 46, Body et al. 2025. https://doi.org/10.1088/1741-4326/ade4d9
  """
  # Calculate momentum loss in the convection layer using an empirical fit.
  # See Section 4 of Body et al. 2025.
  momentum_loss = (
      extended_lengyel_formulas.calc_momentum_loss_in_convection_layer(
          sol_model.state.T_e_target
      )
  )

  # Calculate power loss in the convection layer using an empirical fit.
  # See Section 4 of Body et al. 2025.
  power_loss_conv_layer = (
      extended_lengyel_formulas.calc_power_loss_in_convection_layer(
          sol_model.state.T_e_target
      )
  )

  # Calculate the basic target electron temperature from the two-point model,
  # which assumes no power or momentum loss.
  # In log space to avoid over/underflows in fp32.
  log_T_e_target_basic = (
      jnp.log(8.0)
      + jnp.log(sol_model.params.average_ion_mass)
      + jnp.log(constants.CONSTANTS.m_amu)
      - 2.0 * jnp.log(sol_model.params.sheath_heat_transmission_factor)
      + 2.0 * jnp.log(sol_model.state.q_parallel)
      - 2.0 * jnp.log(sol_model.separatrix_total_pressure)
      - jnp.log(constants.CONSTANTS.q_e)
  )

  T_e_target_basic = jnp.exp(log_T_e_target_basic)
  # Correction factor for additional physics at the target, including
  # ion temperature, mach number, and flux expansion.
  f_other_T_e_target = (
      (
          (1.0 + sol_model.params.T_i_T_e_ratio_target)
          / (sol_model.params.n_e_n_i_ratio_target * 2.0)
      )
      * (
          (1.0 + sol_model.params.mach_target**2) ** 2
          / (4.0 * sol_model.params.mach_target**2)
      )
      * sol_model.params.toroidal_flux_expansion**-2
  )

  parallel_heat_flux_at_target = (
      1.0 - power_loss_conv_layer
  ) * parallel_heat_flux_at_cc_interface
  SOL_power_loss_fraction = (
      1.0 - parallel_heat_flux_at_target / sol_model.state.q_parallel
  )

  f_vol_loss = (1.0 - SOL_power_loss_fraction) ** 2 / (1.0 - momentum_loss) ** 2

  return T_e_target_basic * f_vol_loss * f_other_T_e_target


def calc_kappa_e(Z_eff: array_typing.FloatScalar) -> jax.Array:
  """Corrected parallel electron heat conductivity prefactor.

  Eq 9, Body NF 2025.
  https://doi.org/10.1088/1741-4326/ade4d9

  Eq 10, A.P. Brown A.O. and R.J. Goldston 2021 Nucl. Mater. Energy 27 101002
  https://doi.org/10.1016/j.nme.2021.101002

  Args:
    Z_eff: Effective ion charge.

  Returns:
    Corrected parallel electron heat conductivity prefactor [W/(m*eV^3.5)].
  """
  kappa_z = 0.672 + 0.076 * jnp.sqrt(Z_eff) + 0.252 * Z_eff
  return extended_lengyel_defaults.KAPPA_E_0 / kappa_z
