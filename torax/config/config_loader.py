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
import pathlib
import sys
import types
import typing
from typing import Any, Literal, TypeAlias
from torax.torax_pydantic import model_config

# Tracks all the modules imported so far. Maps the name to the module object.
_ALL_MODULES = {}

ExampleConfig: TypeAlias = Literal[
    'basic_config', 'iterhybrid_predictor_corrector', 'iterhybrid_rampup'
]


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


def torax_path() -> pathlib.Path:
  """Returns the absolute path to the Torax directory."""

  path = pathlib.Path(__file__).parent.parent
  assert path.is_dir(), f'Path {path} is not a directory.'
  assert path.parts[-1] == 'torax', f'Path {path} is not a Torax directory.'
  return path


def example_config_paths() -> dict[ExampleConfig, pathlib.Path]:
  """Returns a tuple of example config paths."""

  example_dir = torax_path().joinpath('examples')
  assert example_dir.is_dir()

  def _get_path(path):
    path = example_dir.joinpath(path + '.py')
    assert path.is_file(), f'Path {path} to the example config does not exist.'
    return path

  return {path: _get_path(path) for path in typing.get_args(ExampleConfig)}


# Taken from
# https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
def _import_from_path(module_name: str, file_path: str) -> types.ModuleType:

  spec = importlib.util.spec_from_file_location(module_name, file_path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  if module is None:
    raise ValueError(f'No loader found for module {module_name}.')
  else:
    spec.loader.exec_module(module)  # pytype: disable=attribute-error
  return module


def import_config_dict(path: str | pathlib.Path) -> dict[str, Any]:
  """Import a Torax config dictionary from a file.

  Args:
    path: The path to the config file. The path can be represented as a string
      or a `pathlib.Path` object.

  Returns:
    The config dictionary.
  """

  path = pathlib.Path(path) if isinstance(path, str) else path
  if not path.is_file():
    raise ValueError(f'Path {path} is not a file.')

  arbitrary_module_name = '_torax_temp_config_import'
  module = _import_from_path(arbitrary_module_name, path)
  if not hasattr(module, 'CONFIG'):
    raise ValueError(
        f'The file {str(path)} is an invalid Torax config file, as it does not'
        ' have a `CONFIG` variable defined.'
    )
  return module.CONFIG
