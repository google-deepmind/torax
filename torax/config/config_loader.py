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

from torax import sim
from torax.config import build_sim
from torax.config import runtime_params

# Tracks all the modules imported so far. Maps the name to the module object.
_ALL_MODULES = {}


def build_sim_and_runtime_params_from_config_module(
    config_module_str: str,
    config_package: str | None = None,
) -> tuple[sim.Sim, runtime_params.GeneralRuntimeParams]:
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
    new_runtime_params = runtime_params.GeneralRuntimeParams.from_dict(
        config['runtime_params']
    )
    simulator = build_sim.build_sim_from_config(config)
  elif hasattr(config_module, 'get_runtime_params') and hasattr(
      config_module, 'get_sim'
  ):
    # The module is likely using the "advances", more Python-forward
    # configuration setup.
    new_runtime_params = config_module.get_runtime_params()
    simulator = config_module.get_sim()
  else:
    raise ValueError(
        f'Config module {config_module_str} must either define a get_sim() '
        'method or a CONFIG dictionary.'
    )
  return simulator, new_runtime_params


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
