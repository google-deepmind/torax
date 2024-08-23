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
"""Helper script to run and regenerate all benchmarks under test_data."""

from collections.abc import Sequence
import importlib
import os

from absl import app
from absl import flags
from absl import logging
from torax import simulation_app
from torax.config import build_sim
from torax.tests.test_lib import paths
from torax.tests.test_lib import sim_test_case


_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', '/tmp/torax_sim_outputs', 'Where to save sim outputs.'
)


def _get_config_module(
    test_data_dir: str,
    config_name: str,
):
  """Returns an input config from the name given."""
  test_config_path = os.path.join(test_data_dir, config_name)
  assert os.path.exists(test_config_path), test_config_path

  # Load config structure with test-case-specific values.
  assert config_name.endswith('.py'), config_name
  config_name_no_py = config_name[:-3]
  python_config_module = sim_test_case.PYTHON_MODULE_PREFIX + config_name_no_py
  return importlib.import_module(
      python_config_module, sim_test_case.PYTHON_CONFIG_PACKAGE
  )


def _run_sim(test_data_dir: str, config_name: str):
  """Run simulation for given config."""
  logging.info('Running %s', config_name)
  config_module = _get_config_module(test_data_dir, config_name + '.py')
  if hasattr(config_module, 'get_sim'):
    # The config module likely uses the "advanced" configuration setup with
    # python functions defining all the Sim object attributes.
    sim = config_module.get_sim()
  elif hasattr(config_module, 'CONFIG'):
    # The config module is using the "basic" configuration setup with a single
    # CONFIG dictionary defining everything.
    # This CONFIG needs to be built into an actual Sim object.
    sim = build_sim.build_sim_from_config(config_module.CONFIG)
  else:
    raise ValueError(
        f'Config module {config_name} must either define a get_sim() method'
        ' or a CONFIG dictionary.'
    )
  simulation_app.main(
      lambda: sim,
      output_dir=os.path.join(_OUTPUT_DIR.value, config_name),
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  test_data_dir = paths.test_data_dir()
  for path in os.listdir(test_data_dir):
    if path.endswith('.nc'):
      basename = os.path.basename(path)
      config_name, _ = basename.split('.')
      try:
        _run_sim(test_data_dir, config_name)
      except Exception as e:  # pylint: disable=broad-except
        logging.exception('Failed to run %s: %s', config_name, e)


if __name__ == '__main__':
  app.run(main)
