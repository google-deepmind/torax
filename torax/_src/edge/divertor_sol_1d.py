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

"""State and Parameter definitions for the extended Lengyel model."""

import dataclasses
from typing import Mapping
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.edge import extended_lengyel_formulas

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ExtendedLengyelParameters:
  """Static inputs and physics parameters fixed for a model run."""

  # --- Geometric and Magnetic Parameters ---
  major_radius: array_typing.FloatScalar  # [m]
  minor_radius: array_typing.FloatScalar  # [m]
  divertor_parallel_length: array_typing.FloatScalar  # [m]
  parallel_connection_length: array_typing.FloatScalar  # [m]
  separatrix_average_poloidal_field: array_typing.FloatScalar  # [T]
  ratio_of_upstream_to_average_poloidal_field: (
      array_typing.FloatScalar
  )  # [dimensionless]
  fieldline_pitch_at_omp: array_typing.FloatScalar  # [dimensionless]
  cylindrical_safety_factor: array_typing.FloatScalar  # [dimensionless]
  toroidal_flux_expansion: array_typing.FloatScalar  # [dimensionless]
  target_angle_of_incidence: array_typing.FloatScalar  # [degrees]

  # --- Plasma and Physics Config ---
  power_crossing_separatrix: array_typing.FloatScalar  # [W]
  separatrix_electron_density: array_typing.FloatScalar  # [m^-3]
  main_ion_charge: array_typing.FloatScalar  # [dimensionless]
  average_ion_mass: array_typing.FloatScalar  # [amu]
  ne_tau: array_typing.FloatScalar  # [s m^-3]
  fraction_of_P_SOL_to_divertor: array_typing.FloatScalar  # [dimensionless]
  SOL_conduction_fraction: array_typing.FloatScalar  # [dimensionless]
  divertor_broadening_factor: array_typing.FloatScalar  # [dimensionless]
  sheath_heat_transmission_factor: array_typing.FloatScalar  # [dimensionless]
  ratio_of_molecular_to_ion_mass: array_typing.FloatScalar  # [dimensionless]
  wall_temperature: array_typing.FloatScalar  # [K]

  # --- Boundary Condition Data: all dimensionless ---
  separatrix_mach_number: array_typing.FloatScalar
  separatrix_ratio_of_ion_to_electron_temp: array_typing.FloatScalar
  separatrix_ratio_of_electron_to_ion_density: array_typing.FloatScalar
  target_mach_number: array_typing.FloatScalar
  target_ratio_of_ion_to_electron_temp: array_typing.FloatScalar
  target_ratio_of_electron_to_ion_density: array_typing.FloatScalar

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
  target_electron_temp: array_typing.FloatScalar  # [eV]
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
            self.state.target_electron_temp
        )
    )
    density_ratio = (
        extended_lengyel_formulas.calc_density_ratio_in_convection_layer(
            self.state.target_electron_temp
        )
    )
    return self.state.target_electron_temp / (
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
        * self.params.divertor_parallel_length
        / self.state.kappa_e
    ) ** (2.0 / 7.0)

  @property
  def separatrix_electron_temp(self) -> jax.Array:
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
            self.params.parallel_connection_length
            - self.params.divertor_parallel_length
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
        (1.0 + self.params.separatrix_mach_number**2)
        * self.params.separatrix_electron_density
        * self.separatrix_electron_temp
        * constants.CONSTANTS.eV_to_J
        * (
            1.0
            + self.params.separatrix_ratio_of_ion_to_electron_temp
            / self.params.separatrix_ratio_of_electron_to_ion_density
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
            self.state.target_electron_temp
        )
    )

    log_target_electron_temp_basic = (
        jnp.log(8.0)
        + jnp.log(self.params.average_ion_mass)
        + jnp.log(constants.CONSTANTS.m_amu)
        - 2.0 * jnp.log(self.params.sheath_heat_transmission_factor)
        + 2.0 * jnp.log(self.state.q_parallel)
        - 2.0 * jnp.log(self.separatrix_total_pressure)
        - jnp.log(constants.CONSTANTS.q_e)
    )

    target_electron_temp_basic = jnp.exp(log_target_electron_temp_basic)

    f_other_target_electron_temp = (
        (
            (1.0 + self.params.target_ratio_of_ion_to_electron_temp)
            / (self.params.target_ratio_of_electron_to_ion_density * 2.0)
        )
        * (
            (1.0 + self.params.target_mach_number**2) ** 2
            / (4.0 * self.params.target_mach_number**2)
        )
        * self.params.toroidal_flux_expansion**-2
    )

    return 1.0 - jnp.sqrt(
        self.state.target_electron_temp
        / target_electron_temp_basic
        * (1.0 - momentum_loss) ** 2
        / f_other_target_electron_temp
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
            self.state.target_electron_temp
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
  def separatrix_Z_eff(self) -> jax.Array:
    return extended_lengyel_formulas.calc_Z_eff(
        c_z=self.state.c_z_prefactor,
        T_e=self.separatrix_electron_temp / 1e3,  # to keV
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


def calc_target_electron_temp(
    sol_model: DivertorSOL1D,
    parallel_heat_flux_at_cc_interface: array_typing.FloatScalar,
    previous_target_electron_temp: array_typing.FloatScalar,
) -> jax.Array:
  """Calculate the target electron temp from the two-point model.

  Args:
    sol_model: The model object for the extended Lengyel model.
    parallel_heat_flux_at_cc_interface: The parallel heat flux at the
      convection-conduction interface pre-calculated with forward model [W/m^2].
    previous_target_electron_temp: The target electron temperature from the
      previous iteration [eV].

  Returns:
    The target electron temperature [eV].

  Eq 46, Body et al. 2025. https://doi.org/10.1088/1741-4326/ade4d9
  """
  # Calculate momentum loss in the convection layer using an empirical fit.
  # See Section 4 of Body et al. 2025.
  momentum_loss = (
      extended_lengyel_formulas.calc_momentum_loss_in_convection_layer(
          previous_target_electron_temp
      )
  )

  # Calculate power loss in the convection layer using an empirical fit.
  # See Section 4 of Body et al. 2025.
  power_loss_conv_layer = (
      extended_lengyel_formulas.calc_power_loss_in_convection_layer(
          previous_target_electron_temp
      )
  )

  # Calculate the basic target electron temperature from the two-point model,
  # which assumes no power or momentum loss.
  # In log space to avoid over/underflows in fp32.
  log_target_electron_temp_basic = (
      jnp.log(8.0)
      + jnp.log(sol_model.params.average_ion_mass)
      + jnp.log(constants.CONSTANTS.m_amu)
      - 2.0 * jnp.log(sol_model.params.sheath_heat_transmission_factor)
      + 2.0 * jnp.log(sol_model.state.q_parallel)
      - 2.0 * jnp.log(sol_model.separatrix_total_pressure)
      - jnp.log(constants.CONSTANTS.q_e)
  )

  target_electron_temp_basic = jnp.exp(log_target_electron_temp_basic)
  # Correction factor for additional physics at the target, including
  # ion temperature, mach number, and flux expansion.
  f_other_target_electron_temp = (
      (
          (1.0 + sol_model.params.target_ratio_of_ion_to_electron_temp)
          / (sol_model.params.target_ratio_of_electron_to_ion_density * 2.0)
      )
      * (
          (1.0 + sol_model.params.target_mach_number**2) ** 2
          / (4.0 * sol_model.params.target_mach_number**2)
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

  return target_electron_temp_basic * f_vol_loss * f_other_target_electron_temp
