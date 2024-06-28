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
from typing import Any
import torax
from torax.config import build_sim

# Tracks all the modules imported so far. Maps the name to the module object.
_ALL_MODULES = {}


def build_sim_and_runtime_params_from_config_module(
    config_module_str: str,
    qlknn_model_path: str | None,
    config_package: str | None = None,
) -> tuple[torax.Sim, torax.GeneralRuntimeParams]:
  """Returns a Sim and RuntimeParams from the config module.

  Args:
    config_module_str: Python package path to config module. E.g.
      torax.examples.iterhybrid_predictor_corrector.
    qlknn_model_path: QLKNN model path set by flag. See qlknn_model_path flag
      docs.
    config_package: Optional, base package config is imported from. See
      config_package flag docs.
  """
  config_module = import_module(config_module_str, config_package)
  if hasattr(config_module, 'CONFIG'):
    # The module likely uses the "basic" config setup which has a single CONFIG
    # dictionary defining the full simulation.
    config = config_module.CONFIG
    maybe_update_config_with_qlknn_model_path(config, qlknn_model_path)
    new_runtime_params = build_sim.build_runtime_params_from_config(
        config['runtime_params']
    )
    sim = build_sim.build_sim_from_config(config)
  elif hasattr(config_module, 'get_runtime_params') and hasattr(
      config_module, 'get_sim'
  ):
    # The module is likely using the "advances", more Python-forward
    # configuration setup.
    if qlknn_model_path is not None:
      logging.warning('Cannot override qlknn model for this type of config.')
    new_runtime_params = config_module.get_runtime_params()
    sim = config_module.get_sim()
  else:
    raise ValueError(
        f'Config module {config_module_str} must either define a get_sim() '
        'method or a CONFIG dictionary.'
    )
  return sim, new_runtime_params


def maybe_update_config_with_qlknn_model_path(
    config: dict[str, Any], qlknn_model_path: str | None
) -> None:
  """Sets the qlknn_model_path in the config if needed."""
  if qlknn_model_path is None:
    return
  if (
      'transport' not in config
      or 'transport_model' not in config['transport']
      or config['transport']['transport_model'] != 'qlknn'
  ):
    return
  qlknn_params = config['transport'].get('qlknn_params', {})
  config_model_path = qlknn_params.get('model_path', '')
  if config_model_path:
    logging.info(
        'Overriding QLKNN model path from "%s" to "%s"',
        config_model_path,
        qlknn_model_path,
    )
  else:
    logging.info('Setting QLKNN model path to "%s".', qlknn_model_path)
  qlknn_params['model_path'] = qlknn_model_path
  config['transport']['qlknn_params'] = qlknn_params


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
