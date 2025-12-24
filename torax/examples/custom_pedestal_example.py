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

"""Example: Custom Pedestal Model using Registration API.

This example demonstrates how to create and register a custom pedestal model
with TORAX. The model implements EPED-like scaling laws for pedestal temperature
and density, similar to what might be used for STEP or other machines.

The key steps are:
1. Define a custom JAX pedestal model class (inherits from PedestalModel)
2. Define a Pydantic configuration class (inherits from BasePedestal)
3. Register the model using register_pedestal_model()
4. Use it in your configuration
"""

import dataclasses
from typing import Annotated, Literal

from torax import CoreProfiles
from torax import Geometry
from torax import JAX_STATIC
from torax import pedestal
import jax.numpy as jnp


# =============================================================================
# STEP 1: Define the JAX Pedestal Model
# =============================================================================
@dataclasses.dataclass(frozen=True)
class EPEDLikePedestalModel(pedestal.PedestalModel):
  """EPED-like pedestal model with power-law scaling.

  This model implements a simplified EPED-like scaling:
  - T_e_ped ∝ Ip^a * B0^b * (other parameters)
  - T_i_ped = ratio * T_e_ped
  - n_e_ped can be specified as Greenwald fraction or absolute density
  """

  def _call_implementation(
      self,
      runtime_params: 'RuntimeParams',
      geo: Geometry,
      core_profiles: CoreProfiles,
  ) -> pedestal.PedestalModelOutput:
    """Compute pedestal values using EPED-like scaling."""

    # Extract plasma parameters
    Ip_MA = runtime_params.profile_conditions.Ip / 1e6  # Plasma current in MA
    B0 = geo.B0  # Toroidal field in T
    a = geo.Rmin  # Minor radius in m
    epsilon = geo.epsilon  # Inverse aspect ratio

    # EPED-like T_e scaling (simplified)
    # Real EPED would include more physics (triangularity, beta, etc.)
    T_e_ped = (
        runtime_params.T_e_scaling_factor
        * (Ip_MA ** runtime_params.Ip_exponent)
        * (B0 ** runtime_params.B0_exponent)
        * (epsilon ** runtime_params.epsilon_exponent)
    )

    # T_i from T_e using ratio
    T_i_ped = runtime_params.T_i_T_e_ratio * T_e_ped

    # Density (either Greenwald fraction or absolute)
    if runtime_params.n_e_ped_is_fGW:
      # Convert Greenwald fraction to absolute density
      # nGW = Ip / (π * a^2) in 10^20 m^-3
      n_GW = Ip_MA / (jnp.pi * a**2)  # in 10^20 m^-3
      n_e_ped = runtime_params.n_e_ped_value * n_GW * 1e20  # Convert to m^-3
    else:
      n_e_ped = runtime_params.n_e_ped_value

    # Pedestal location
    rho_norm_ped_top = runtime_params.rho_norm_ped_top

    # Find the index in the mesh
    rho_norm_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_norm - rho_norm_ped_top)
    )

    return pedestal.PedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        rho_norm_ped_top_idx=rho_norm_ped_top_idx,
        T_i_ped=T_i_ped,
        T_e_ped=T_e_ped,
        n_e_ped=n_e_ped,
    )


@dataclasses.dataclass(frozen=True)
class RuntimeParams(pedestal.RuntimeParams):
  """Runtime parameters for EPED-like pedestal model.

  Attributes:
    T_e_scaling_factor: Overall scaling factor for T_e [keV].
    Ip_exponent: Power law exponent for plasma current.
    B0_exponent: Power law exponent for toroidal field.
    epsilon_exponent: Power law exponent for inverse aspect ratio.
    T_i_T_e_ratio: Ratio of ion to electron temperature at pedestal.
    n_e_ped_value: Electron density value (either Greenwald fraction or m^-3).
    n_e_ped_is_fGW: Whether n_e_ped_value is Greenwald fraction.
    rho_norm_ped_top: Location of pedestal top in normalized radius.
  """
  T_e_scaling_factor: float = 5.0
  Ip_exponent: float = 0.2
  B0_exponent: float = 0.8
  epsilon_exponent: float = 0.3
  T_i_T_e_ratio: float = 1.0
  n_e_ped_value: float = 0.7
  n_e_ped_is_fGW: bool = True
  rho_norm_ped_top: float = 0.91


# =============================================================================
# STEP 2: Define the Pydantic Configuration Class
# =============================================================================
class EPEDLikePedestal(pedestal.BasePedestal):
  """Pydantic configuration for EPED-like pedestal model.

  This class defines the user-facing configuration interface for the
  EPED-like pedestal model. Users can set the scaling parameters in their
  config files.

  Attributes:
    model_name: The model identifier. Must be 'eped_like'.
    T_e_scaling_factor: Overall scaling factor for T_e [keV].
    Ip_exponent: Power law exponent for plasma current.
    B0_exponent: Power law exponent for toroidal field.
    epsilon_exponent: Power law exponent for inverse aspect ratio.
    T_i_T_e_ratio: Ratio of ion to electron temperature at pedestal.
    n_e_ped: Electron density value (either Greenwald fraction or m^-3).
    n_e_ped_is_fGW: Whether n_e_ped is Greenwald fraction.
    rho_norm_ped_top: Location of pedestal top in normalized radius.
  """

  model_name: Annotated[Literal['eped_like'], JAX_STATIC] = 'eped_like'

  # Scaling parameters
  T_e_scaling_factor: float = 5.0
  Ip_exponent: float = 0.2
  B0_exponent: float = 0.8
  epsilon_exponent: float = 0.3
  T_i_T_e_ratio: float = 1.0

  # Pedestal values
  n_e_ped: float = 0.7
  n_e_ped_is_fGW: bool = True
  rho_norm_ped_top: float = 0.91

  def build_pedestal_model(self) -> EPEDLikePedestalModel:
    """Build the JAX pedestal model."""
    return EPEDLikePedestalModel()

  def build_runtime_params(self, t) -> RuntimeParams:
    """Build runtime parameters for the given time."""
    return RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        T_e_scaling_factor=self.T_e_scaling_factor,
        Ip_exponent=self.Ip_exponent,
        B0_exponent=self.B0_exponent,
        epsilon_exponent=self.epsilon_exponent,
        T_i_T_e_ratio=self.T_i_T_e_ratio,
        n_e_ped_value=self.n_e_ped,
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        rho_norm_ped_top=self.rho_norm_ped_top,
    )


# =============================================================================
# STEP 3: Register the Model
# =============================================================================
# This makes the model available to TORAX's configuration system
pedestal.register_pedestal_model(EPEDLikePedestal)


# =============================================================================
# STEP 4: Use in Configuration
# =============================================================================
# Minimal example configuration using the registered model
CONFIG = {
    'pedestal': {
        'model_name': 'eped_like',
        'set_pedestal': True,
        # Custom parameters for EPED-like scaling
        'T_e_scaling_factor': 5.0,
        'Ip_exponent': 0.2,
        'B0_exponent': 0.8,
        'epsilon_exponent': 0.3,
        'T_i_T_e_ratio': 1.0,
        'n_e_ped': 0.7,  # Greenwald fraction
        'n_e_ped_is_fGW': True,
        'rho_norm_ped_top': 0.91,
    },
}
