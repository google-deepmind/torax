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

"""Functions to load a config from config file or directory."""

import importlib
import logging

from torax.torax_pydantic import model_config

# Tracks all the modules imported so far. Maps the name to the module object.
_ALL_MODULES = {}


def build_torax_config_from_config_module(
    config_module_str: str,
    config_package: str | None = None,
) -> model_config.ToraxConfig:
  """Returns a Sim and RuntimeParams from the config module.

  Args:
    config_module_str: Python package path to config module. E.g.
      torax.examples.iterhybrid_predictor_corrector.
    config_package: Optional, base package config is imported from. See
      config_package flag docs.
  """
  config_module = import_module(config_module_str, config_package)
  if hasattr(config_module, 'CONFIG'):
    # The module likely uses the "basic" config setup which has a single CONFIG
    # dictionary defining the full simulation.
    config = config_module.CONFIG
    torax_config = model_config.ToraxConfig.from_dict(config)
    # Perform additional sanity checks on the configuration
    perform_config_sanity_checks(torax_config)
  else:
    raise ValueError(
        f'Config module {config_module_str} must define a CONFIG dictionary.'
    )
  return torax_config


def perform_config_sanity_checks(torax_config: model_config.ToraxConfig) -> None:
  """Performs sanity checks on the ToraxConfig.
  
  Validates that the configuration is physically and numerically sensible.
  Raises ValueError with detailed error messages when issues are found.
  
  Args:
    torax_config: The ToraxConfig object to validate.
    
  Raises:
    ValueError: If any sanity check fails.
  """
  # Check numerics parameters
  numerics = torax_config.runtime_params.numerics
  if numerics.t_final <= numerics.t_initial:
    raise ValueError(
        f'Final time {numerics.t_final} must be greater than initial time '
        f'{numerics.t_initial}.'
    )
  
  if numerics.mindt <= 0:
    raise ValueError(f'Minimum time step (mindt) must be positive, got {numerics.mindt}.')

  if numerics.dt_reduction_factor <= 1.0:
    raise ValueError(
        f'Time step reduction factor must be greater than 1.0, got '
        f'{numerics.dt_reduction_factor}.'
    )
  
  # Check if at least one equation is being solved
  if not (numerics.ion_heat_eq or numerics.el_heat_eq or 
          numerics.current_eq or numerics.dens_eq):
    raise ValueError(
        'At least one equation must be enabled. Set at least one of ion_heat_eq, '
        'el_heat_eq, current_eq, or dens_eq to True.'
    )
  
  # Check transport parameters
  transport = torax_config.transport
  if transport.chimin >= transport.chimax:
    raise ValueError(
        f'Minimum chi ({transport.chimin}) must be less than maximum chi '
        f'({transport.chimax}).'
    )
  
  if transport.Demin >= transport.Demax:
    raise ValueError(
        f'Minimum electron diffusivity (Demin: {transport.Demin}) must be less '
        f'than maximum (Demax: {transport.Demax}).'
    )
  
  if transport.Vemin >= transport.Vemax:
    raise ValueError(
        f'Minimum electron convection (Vemin: {transport.Vemin}) must be less '
        f'than maximum (Vemax: {transport.Vemax}).'
    )
  
  # Check for potential geometry issues
  geometry = torax_config.geometry
  if hasattr(geometry, 'n_rho') and geometry.n_rho < 10:
    raise ValueError(
        f'Number of radial grid points (n_rho: {geometry.n_rho}) is too small. '
        f'Recommended minimum is 10.'
    )
    
  # Check for consistency between models
  if (numerics.dens_eq and 
      torax_config.pedestal.set_pedestal and 
      not torax_config.pedestal.neped_is_fGW):
    logging.warning(
        'Density equation is enabled with pedestal model, but neped_is_fGW is False. '
        'Consider setting neped_is_fGW to True for better consistency.'
    )

  # Check for potentially problematic combinations of stepper and transport model
  stepper_type = torax_config.stepper.stepper_type
  transport_model = transport.transport_model
  if transport_model in ['qlknn', 'CGM'] and stepper_type == 'linear':
    logging.warning(
        f'Using advanced transport model ({transport_model}) with linear stepper. '
        f'This may lead to numerical instabilities. Consider using newton_raphson or '
        f'optimizer stepper type instead.'
    )


def import_module(module_name: str, config_package: str | None = None):
  """Imports a module."""
  try:
    if module_name in _ALL_MODULES:
      return importlib.reload(_ALL_MODULES[module_name])
    else:
      module = importlib.import_module(module_name, config_package)
      _ALL_MODULES[module_name] = module
      return module
  except Exception as e:
    logging.info('Exception raised: %s', e)
    raise ValueError('Exception while importing.') from e
