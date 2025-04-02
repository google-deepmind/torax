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
  else:
    raise ValueError(
        f'Config module {config_module_str} must define a CONFIG dictionary.'
    )
  return torax_config


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
