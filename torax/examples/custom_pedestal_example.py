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

from typing import Annotated, Literal

import chex
from torax._src import geometry
from torax._src import state
from torax._src.pedestal_model import pedestal_model as pm
from torax._src.pedestal_model import pydantic_model
from torax._src.pedestal_model import register_model
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params
from torax._src.torax_pydantic import torax_pydantic
import jax.numpy as jnp


# =============================================================================
# STEP 1: Define the JAX Pedestal Model
# =============================================================================
@chex.dataclass(frozen=True)
class EPEDLikePedestalModel(pm.PedestalModel):
  """EPED-like pedestal model with power-law scaling.

  This model implements a simplified EPED-like scaling:
  - T_e_ped ∝ Ip^a * B0^b * (other parameters)
  - T_i_ped = ratio * T_e_ped
  - n_e_ped can be specified as Greenwald fraction or absolute density
  - Pedestal width can be dynamic based on poloidal beta
  """

  def _call_implementation(
      self,
      runtime_params: 'RuntimeParams',
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pm.PedestalModelOutput:
    """Compute pedestal values using EPED-like scaling."""

    # Extract plasma parameters
    Ip_MA = runtime_params.profile_conditions.Ip / 1e6  # Plasma current in MA
    B0 = geo.B0  # Toroidal field in T
    a = geo.Rmin  # Minor radius in m
    epsilon = geo.epsilon  # Inverse aspect ratio
    kappa = runtime_params.profile_conditions.kappa  # Elongation

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

    # Pedestal location (can be dynamic based on beta_p if desired)
    # For now, use the configured value
    rho_norm_ped_top = runtime_params.rho_norm_ped_top

    # Find the index in the mesh
    rho_norm_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_norm - rho_norm_ped_top)
    )

    return pm.PedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        rho_norm_ped_top_idx=rho_norm_ped_top_idx,
        T_i_ped=T_i_ped,
        T_e_ped=T_e_ped,
        n_e_ped=n_e_ped,
    )


@chex.dataclass(frozen=True)
class RuntimeParams(pedestal_runtime_params.RuntimeParams):
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
class EPEDLikePedestal(pydantic_model.BasePedestal):
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

  model_name: Annotated[Literal['eped_like'], torax_pydantic.JAX_STATIC] = (
      'eped_like'
  )

  # Scaling parameters (can be time-varying if needed)
  T_e_scaling_factor: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5.0)
  )
  Ip_exponent: float = 0.2
  B0_exponent: float = 0.8
  epsilon_exponent: float = 0.3
  T_i_T_e_ratio: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )

  # Pedestal values
  n_e_ped: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.7)
  )
  n_e_ped_is_fGW: bool = True
  rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_pedestal_model(self) -> EPEDLikePedestalModel:
    """Build the JAX pedestal model."""
    return EPEDLikePedestalModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> RuntimeParams:
    """Build runtime parameters for the given time."""
    return RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        T_e_scaling_factor=self.T_e_scaling_factor.get_value(t),
        Ip_exponent=self.Ip_exponent,
        B0_exponent=self.B0_exponent,
        epsilon_exponent=self.epsilon_exponent,
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
        n_e_ped_value=self.n_e_ped.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
    )


# =============================================================================
# STEP 3: Register the Model
# =============================================================================
# This makes the model available to TORAX's configuration system
register_model.register_pedestal_model(EPEDLikePedestal)


# =============================================================================
# STEP 4: Use in Configuration
# =============================================================================
# Example configuration using the registered model
CONFIG = {
    'profile_conditions': {
        'Ip': 15e6,  # 15 MA plasma current
    },
    'plasma_composition': {},
    'numerics': {},
    'geometry': {
        'geometry_type': 'circular',
        'B0': 5.3,  # 5.3 T toroidal field
        'a_minor': 1.65,  # 1.65 m minor radius
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'sources': {
        'generic_current': {},
        'generic_particle': {},
        'gas_puff': {},
        'pellet': {},
        'generic_heat': {},
        'fusion': {},
        'ei_exchange': {},
        'ohmic': {},
    },
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
    'transport': {
        'model_name': 'constant',
    },
    'solver': {
        'solver_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}


# =============================================================================
# Example 2: Simple Constant Pedestal Model
# =============================================================================
# For comparison, here's a simpler example with constant values

@chex.dataclass(frozen=True)
class SimplePedestalModel(pm.PedestalModel):
  """Simple pedestal model with constant values."""

  def _call_implementation(
      self,
      runtime_params: 'SimpleRuntimeParams',
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pm.PedestalModelOutput:
    """Return constant pedestal values."""
    rho_norm_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_norm - runtime_params.rho_norm_ped_top)
    )

    return pm.PedestalModelOutput(
        rho_norm_ped_top=runtime_params.rho_norm_ped_top,
        rho_norm_ped_top_idx=rho_norm_ped_top_idx,
        T_i_ped=runtime_params.T_i_ped,
        T_e_ped=runtime_params.T_e_ped,
        n_e_ped=runtime_params.n_e_ped,
    )


@chex.dataclass(frozen=True)
class SimpleRuntimeParams(pedestal_runtime_params.RuntimeParams):
  """Runtime parameters for simple pedestal model."""
  T_i_ped: float = 5.0
  T_e_ped: float = 5.0
  n_e_ped: float = 0.7e20
  rho_norm_ped_top: float = 0.91


class SimplePedestal(pydantic_model.BasePedestal):
  """Pydantic config for simple constant pedestal."""

  model_name: Annotated[Literal['simple'], torax_pydantic.JAX_STATIC] = (
      'simple'
  )

  T_i_ped: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5.0)
  )
  T_e_ped: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5.0)
  )
  n_e_ped: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.7e20)
  )
  rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_pedestal_model(self) -> SimplePedestalModel:
    return SimplePedestalModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> SimpleRuntimeParams:
    return SimpleRuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        T_i_ped=self.T_i_ped.get_value(t),
        T_e_ped=self.T_e_ped.get_value(t),
        n_e_ped=self.n_e_ped.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
    )


# Register the simple model too
register_model.register_pedestal_model(SimplePedestal)

# Simple config example
SIMPLE_CONFIG = {
    'profile_conditions': {},
    'plasma_composition': {},
    'numerics': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'sources': {
        'generic_current': {},
        'generic_particle': {},
        'gas_puff': {},
        'pellet': {},
        'generic_heat': {},
        'fusion': {},
        'ei_exchange': {},
        'ohmic': {},
    },
    'pedestal': {
        'model_name': 'simple',
        'set_pedestal': True,
        'T_i_ped': 5.0,
        'T_e_ped': 5.0,
        'n_e_ped': 0.7e20,
        'rho_norm_ped_top': 0.91,
    },
    'transport': {
        'model_name': 'constant',
    },
    'solver': {
        'solver_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
