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

"""State object for the extended Lengyel model inner loop calculations."""

import dataclasses
from typing import Mapping
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.edge import extended_lengyel_formulas

# pylint: disable=invalid-name


# TODO(b/446608829) - investigate separating the structure to dataclasses with
# fixed variables and variables that are modified during the workflows.
@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DivertorSOL1D:
  """Represents the state of a 1D divertor scrape-off layer (SOL) model.

  This object encapsulates the parameters and intermediate calculations for a
  single iteration within the extended Lengyel model's inner loop. It computes
  derived quantities as properties.

  See T. Body et al 2025 Nucl. Fusion 65 086002 for further details.
  """

  # Inputs which may vary within extended-lengyel model iterations, depending on
  # whether the model is being run in inverse or forward mode.
  q_parallel: array_typing.FloatScalar
  alpha_t: array_typing.FloatScalar
  target_electron_temp: array_typing.FloatScalar
  c_z_prefactor: array_typing.FloatScalar
  kappa_e: array_typing.FloatScalar

  # Input physics parameters which are fixed for a given run of the model.
  main_ion_charge: array_typing.FloatScalar
  seed_impurity_weights: Mapping[str, array_typing.FloatScalar]
  fixed_impurity_concentrations: Mapping[str, array_typing.FloatScalar]
  ne_tau: array_typing.FloatScalar
  SOL_conduction_fraction: array_typing.FloatScalar
  divertor_broadening_factor: array_typing.FloatScalar
  divertor_parallel_length: array_typing.FloatScalar
  parallel_connection_length: array_typing.FloatScalar
  separatrix_mach_number: array_typing.FloatScalar
  separatrix_electron_density: array_typing.FloatScalar
  separatrix_ratio_of_ion_to_electron_temp: array_typing.FloatScalar
  separatrix_ratio_of_electron_to_ion_density: array_typing.FloatScalar
  average_ion_mass: array_typing.FloatScalar
  sheath_heat_transmission_factor: array_typing.FloatScalar
  target_mach_number: array_typing.FloatScalar
  target_ratio_of_ion_to_electron_temp: array_typing.FloatScalar
  target_ratio_of_electron_to_ion_density: array_typing.FloatScalar
  toroidal_flux_expansion: array_typing.FloatScalar

  @property
  def electron_temp_at_cc_interface(self) -> jax.Array:
    """Electron temperature at the convection/conduction interface [eV].

    Section 4, Body et al. 2025.
    """

    momentum_loss = (
        extended_lengyel_formulas.calc_momentum_loss_in_convection_layer(
            self.target_electron_temp
        )
    )
    density_ratio = (
        extended_lengyel_formulas.calc_density_ratio_in_convection_layer(
            self.target_electron_temp
        )
    )
    return self.target_electron_temp / (
        (1.0 - momentum_loss) / (2.0 * density_ratio)
    )

  @property
  def divertor_entrance_electron_temp(self) -> jax.Array:
    """Electron temperature at the divertor entrance [eV].

    This formula is derived from the heat conduction equation integrated
    along the scrape-off layer.

    Eq 44, Body et al. 2025.
    """
    return (
        self.electron_temp_at_cc_interface**3.5
        + 3.5
        * self.SOL_conduction_fraction
        * self.q_parallel
        / self.divertor_broadening_factor
        * self.divertor_parallel_length
        / self.kappa_e
    ) ** (2.0 / 7.0)

  @property
  def separatrix_electron_temp(self) -> jax.Array:
    """Electron temperature at the separatrix [eV].

    This formula is derived from the heat conduction equation integrated
    along the scrape-off layer.

    Eq 45, Body et al. 2025.
    """
    return (
        self.divertor_entrance_electron_temp**3.5
        + 3.5
        * self.SOL_conduction_fraction
        * self.q_parallel
        * (self.parallel_connection_length - self.divertor_parallel_length)
        / self.kappa_e
    ) ** (2.0 / 7.0)

  @property
  def separatrix_total_pressure(self) -> jax.Array:
    """Total pressure at the separatrix [Pa].

    This is the definition of total pressure (static + dynamic) at the
    separatrix, including both electron and ion contributions.
    """
    return (
        (1.0 + self.separatrix_mach_number**2)
        * self.separatrix_electron_density
        * self.separatrix_electron_temp
        * constants.CONSTANTS.eV_to_J
        * (
            1.0
            + self.separatrix_ratio_of_ion_to_electron_temp
            / self.separatrix_ratio_of_electron_to_ion_density
        )
    )

  @property
  def required_power_loss(self) -> jax.Array:
    """Required power loss fraction from the two-point model.

    Eq 46, Body et al. 2025.
    """
    # Calculate momentum loss in the convection layer using an empirical fit.
    # See Section 4 of Body et al. 2025.
    momentum_loss = (
        extended_lengyel_formulas.calc_momentum_loss_in_convection_layer(
            self.target_electron_temp
        )
    )

    # Calculate the basic target electron temperature from the two-point model,
    # which assumes no power or momentum loss.
    target_electron_temp_basic = (
        (
            8.0
            * self.average_ion_mass
            * constants.CONSTANTS.m_amu
            / self.sheath_heat_transmission_factor**2
        )
        * (self.q_parallel**2 / self.separatrix_total_pressure**2)
    ) / constants.CONSTANTS.q_e

    # Correction factor for additional physics at the target, including
    # ion temperature, mach number, and flux expansion.
    f_other_target_electron_temp = (
        (
            (1.0 + self.target_ratio_of_ion_to_electron_temp)
            / (self.target_ratio_of_electron_to_ion_density * 2.0)
        )
        * (
            (1.0 + self.target_mach_number**2) ** 2
            / (4.0 * self.target_mach_number**2)
        )
        * self.toroidal_flux_expansion**-2
    )

    # Final calculation for the required power loss, incorporating the momentum
    # loss and other target physics corrections.
    return 1.0 - jnp.sqrt(
        self.target_electron_temp
        / target_electron_temp_basic
        * (1.0 - momentum_loss) ** 2
        / f_other_target_electron_temp
    )

  @property
  def parallel_heat_flux_at_target(self) -> jax.Array:
    """Parallel heat flux at the divertor target [W/m^2]."""
    return self.q_parallel * (1.0 - self.required_power_loss)

  @property
  def parallel_heat_flux_at_cc_interface(self) -> jax.Array:
    """Parallel heat flux at the convection-conduction interface [W/m^2].

    Eq 29, Body et al. 2025.
    """
    power_loss_conv_layer = (
        extended_lengyel_formulas.calc_power_loss_in_convection_layer(
            self.target_electron_temp
        )
    )
    return self.parallel_heat_flux_at_target / (1.0 - power_loss_conv_layer)

  @property
  def divertor_Z_eff(self) -> array_typing.FloatScalar:
    return extended_lengyel_formulas.calc_Z_eff(
        c_z=self.c_z_prefactor,
        T_e=self.divertor_entrance_electron_temp / 1e3,  # to keV
        Z_i=self.main_ion_charge,
        ne_tau=self.ne_tau,
        seed_impurity_weights=self.seed_impurity_weights,
        fixed_impurity_concentrations=self.fixed_impurity_concentrations,
    )

  @property
  def separatrix_Z_eff(self) -> array_typing.FloatScalar:
    return extended_lengyel_formulas.calc_Z_eff(
        c_z=self.c_z_prefactor,
        T_e=self.separatrix_electron_temp / 1e3,  # to keV
        Z_i=self.main_ion_charge,
        ne_tau=self.ne_tau,
        seed_impurity_weights=self.seed_impurity_weights,
        fixed_impurity_concentrations=self.fixed_impurity_concentrations,
    )

  @property
  def impurity_concentrations(self) -> Mapping[str, array_typing.FloatScalar]:
    return {
        key: value * self.c_z_prefactor
        for key, value in self.seed_impurity_weights.items()
    }
