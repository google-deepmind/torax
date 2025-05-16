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
import functools
import os
import time

from absl import app
from absl import flags
from torax._src import simulation_app
from torax._src.config import config_loader
from torax._src.orchestration import run_simulation
from torax._src.torax_pydantic import model_config
from torax.tests.test_lib import paths

import shutil

import multiprocessing


_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', '/tmp/torax_sim_outputs', 'Where to save sim outputs.'
)
_NUM_PROCESSES = flags.DEFINE_integer(
    'num_proc', 16, 'Number of processes to use.'
)


def _run_sim(config_name: str, test_data_dir: str, output_dir: str):
  """Run simulation for given config."""
  flags.FLAGS.mark_as_parsed()
  print(f'Running {config_name}')

  try:
    path = os.path.join(test_data_dir, config_name + '.py')
    config_module = config_loader.import_module(path)
  except ValueError:
    print(f'Failed to import config module {config_name}, skipping.')
    return

  if 'get_sim' in config_module:
    # The config module likely uses the "advanced" configuration setup with
    # python functions defining all the Sim object attributes.
    raise NotImplementedError(
        'get_sim() is not supported in this script. Please use the "basic"'
        ' configuration setup with a single CONFIG dictionary defining'
        ' everything.'
    )
  elif 'CONFIG' in config_module:
    # The config module is using the "basic" configuration setup with a single
    # CONFIG dictionary defining everything.
    # This CONFIG needs to be built into an actual Sim object.
    torax_config = model_config.ToraxConfig.from_dict(config_module['CONFIG'])
  else:
    raise ValueError(
        f'Config module {config_name} must either define a get_sim() method'
        ' or a CONFIG dictionary.'
    )
  try:
    simulation_xr, _ = run_simulation.run_simulation(
        torax_config, progress_bar=False
    )
    output_file = os.path.join(output_dir, f'{config_name}.nc')
    simulation_app.write_output_to_file(
        output_file, simulation_xr
    )
    print(f'Finished running {config_name}, output saved to {output_file}')
  except Exception as e:  # pylint: disable=broad-except
    print(f'Failed to run {config_name}: {e}')


def main(argv: Sequence[str]) -> None:
  start_time_s = time.time()
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  configs = []
  test_data_dir = paths.test_data_dir()
  for path in os.listdir(test_data_dir):
    # avoid rerunning qualikiz tests, which are more expensive and are not
    # part of the standard sim tests.
    if path.endswith('.nc') and 'qualikiz' not in path:
      basename = os.path.basename(path)
      config_name, _ = basename.split('.')
      configs.append(config_name)
  print(f'Found {len(configs)} config experiments to run.')
  output_dir = _OUTPUT_DIR.value
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  run_sim = functools.partial(
      _run_sim,
      test_data_dir=test_data_dir,
      output_dir=_OUTPUT_DIR.value,
  )
  # Important to use 'spawn' over 'forkserver' as JAX is not fork-safe.
  mp_context = multiprocessing.get_context('spawn')
  with mp_context.Pool(processes=_NUM_PROCESSES.value) as pool:
    pool.map(run_sim, configs)
    pool.close()
    pool.join()
  print(f'Running took {time.time() - start_time_s}s')


if __name__ == '__main__':
  app.run(main)
