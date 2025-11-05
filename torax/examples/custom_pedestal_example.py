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

"""Example configuration demonstrating the custom pedestal model API.

This example shows how to use the custom pedestal model to implement
machine-specific scaling laws without modifying the TORAX source code.

The example implements:
1. A simple EPED-like scaling for electron temperature
2. A T_i/T_e ratio model for ion temperature
3. A Greenwald fraction model for density
4. A dynamic pedestal width model based on poloidal beta

This pattern enables users to couple custom pedestal models for machines like
STEP that use power-law fits to Europed data and modified EPED scaling.
"""

import jax.numpy as jnp


# Define custom pedestal scaling functions
def custom_T_e_ped(runtime_params, geo, core_profiles):
  """Compute electron temperature at pedestal using EPED-like scaling.

  This is a simplified example. Real implementations might use more complex
  fits to experimental data or physics-based models.

  Args:
    runtime_params: Runtime parameters containing plasma current, etc.
    geo: Geometry object containing magnetic field, minor radius, etc.
    core_profiles: Core plasma profiles.

  Returns:
    Electron temperature at pedestal in keV.
  """
  # Extract relevant parameters
  Ip_MA = runtime_params.profile_conditions.Ip / 1e6  # Convert to MA
  B_T = geo.B0  # Toroidal magnetic field

  # Simple EPED-like scaling: T_e ~ Ip^0.2 * B^0.8
  # In reality, this might include more dependencies on geometry, shape, etc.
  T_e_ped = 0.8 * (Ip_MA**0.25) * (B_T**0.75)

  return T_e_ped


def custom_T_i_ped(runtime_params, geo, core_profiles):
  """Compute ion temperature at pedestal.

  Uses a simple ratio model: T_i = ratio * T_e.
  More sophisticated models might vary the ratio based on heating power,
  collisionality, etc.

  Args:
    runtime_params: Runtime parameters.
    geo: Geometry object.
    core_profiles: Core plasma profiles.

  Returns:
    Ion temperature at pedestal in keV.
  """
  T_e_ped = custom_T_e_ped(runtime_params, geo, core_profiles)

  # T_i/T_e ratio (could be made more sophisticated)
  ratio = 1.2

  return ratio * T_e_ped


def custom_n_e_ped(runtime_params, geo, core_profiles):
  """Compute electron density at pedestal as Greenwald fraction.

  This example returns a fixed Greenwald fraction, but real implementations
  might include dependencies on heating power, gas puffing rate, etc.

  Args:
    runtime_params: Runtime parameters.
    geo: Geometry object.
    core_profiles: Core plasma profiles.

  Returns:
    Electron density as Greenwald fraction (dimensionless).
  """
  # Return as Greenwald fraction
  # The framework will automatically convert this to absolute density
  f_GW = 0.7  # 70% of Greenwald density

  return f_GW


def custom_rho_norm_ped_top(runtime_params, geo, core_profiles):
  """Compute pedestal top location dynamically.

  This example demonstrates how the pedestal width can be made to depend
  on plasma parameters. Here we use a simple model based on poloidal beta,
  but real implementations might use more sophisticated models.

  Args:
    runtime_params: Runtime parameters.
    geo: Geometry object.
    core_profiles: Core plasma profiles.

  Returns:
    Normalized poloidal flux at pedestal top (dimensionless).
  """
  # Simple model: pedestal width depends on poloidal beta
  # In reality, you might access core_profiles or geo for beta_p
  # For this example, we'll use a simplified placeholder

  # Get plasma current as a proxy for beta_p dependencies
  Ip_MA = runtime_params.profile_conditions.Ip / 1e6

  # Simple scaling: higher current -> slightly narrower pedestal
  # Real models might use: rho_ped = f(beta_p, collisionality, etc.)
  base_rho = 0.92
  current_correction = -0.005 * (Ip_MA - 15.0)  # Normalized to 15 MA

  rho_norm_ped_top = jnp.clip(base_rho + current_correction, 0.85, 0.95)

  return rho_norm_ped_top


# Configuration dictionary
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
    # Custom pedestal configuration
    'pedestal': {
        'model_name': 'custom',  # Use the custom pedestal model
        'set_pedestal': True,  # Enable pedestal
        # Provide the custom functions
        'T_i_ped_fn': custom_T_i_ped,
        'T_e_ped_fn': custom_T_e_ped,
        'n_e_ped_fn': custom_n_e_ped,
        'rho_norm_ped_top_fn': custom_rho_norm_ped_top,  # Optional
        # If rho_norm_ped_top_fn is not provided, this fallback value is used:
        # 'rho_norm_ped_top': 0.91,
        'n_e_ped_is_fGW': True,  # n_e_ped_fn returns Greenwald fraction
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


# Alternative example: Simple constant pedestal values
# This shows how you can also use the custom pedestal API for simple cases
def simple_T_e_ped(runtime_params, geo, core_profiles):
  return 5.0  # 5 keV


def simple_T_i_ped(runtime_params, geo, core_profiles):
  return 6.0  # 6 keV


def simple_n_e_ped(runtime_params, geo, core_profiles):
  return 0.8e20  # 0.8e20 m^-3


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
        'model_name': 'custom',
        'set_pedestal': True,
        'T_i_ped_fn': simple_T_i_ped,
        'T_e_ped_fn': simple_T_e_ped,
        'n_e_ped_fn': simple_n_e_ped,
        'rho_norm_ped_top': 0.91,  # Use fixed value
        'n_e_ped_is_fGW': False,  # Use absolute density units
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
