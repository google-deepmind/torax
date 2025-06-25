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

"""A module providing a runnable main for transport simulation.

You can use the `main()` function in this module as an
entry point for your simulation (see the example below).

You need to provide a method that returns a valid Sim object. Then
this module handles the rest: runs the simulation and writes the output to file
if directed to.

See run_simulation_main.py for a ready-made entrypoint for running simulations.
You can also implement your own main with a different implementation as
long as you pass in your sim-getter to this module. This is shown below using
the existing TORAX sim-getter and an example TORAX config. The example below
also enables optional output logging.
Note that `app.run` cannot be directly run on the `main()` function in this
module, due to the tuple return type.

.. code-block:: python

  # In my_runnable_sim.py:

  from absl import app
  from absl import logging
  from torax._src import simulation_app
  from torax.examples import basic_config
  from torax._src.config import build_sim

  def run(_):
    sim_builder = lambda: build_sim.build_sim_from_config(basic_config.CONFIG)
    simulation_app.main(
      sim_builder,
      log_sim_progress=True,
      )

  if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(run)
"""
import datetime
import enum
import os
import shutil
import sys
from collections.abc import Sequence
from typing import Callable
from typing import Final

import jax
import numpy as np
import xarray as xr
from absl import logging

from torax._src import state
from torax._src.orchestration import run_simulation
from torax._src.torax_pydantic import model_config

# String printed before printing the output file path
WRITE_PREFIX: Final[str] = 'Wrote simulation output to '


# For logging.
# ANSI color codes for pretty-printing
@enum.unique
class AnsiColors(enum.Enum):
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'


_ANSI_END = '\033[0m'

_DEFAULT_OUTPUT_DIR: Final[str] = '/tmp/torax_results'
_STATE_HISTORY_FILENAME: Final[str] = 'state_history'


def log_to_stdout(
    log_output: str,
    color: AnsiColors | None = None,
    exc_info: bool = False,
) -> None:
  if not color or not sys.stderr.isatty():
    logging.info(log_output, exc_info=exc_info)
  else:
    logging.info(
        '%s%s%s', color.value, log_output, _ANSI_END, exc_info=exc_info
    )


def _write_simulation_output_to_dir(
    output_dir: str, data_tree: xr.DataTree
) -> str:
  """Writes the state history and some geometry information to a NetCDF file."""
  filename = f'{_STATE_HISTORY_FILENAME}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.nc'  # pylint: disable=g-inconsistent-quotes
  output_file = os.path.join(output_dir, filename)
  return write_output_to_file(output_file, data_tree)


def write_output_to_file(path: str, data_tree: xr.DataTree):
  directory = '/'.join(path.split('/')[:-1])
  if not os.path.exists(directory):
    os.makedirs(directory)
  data_tree.to_netcdf(path)
  log_to_stdout(f'{WRITE_PREFIX}{path}', AnsiColors.GREEN)

  return path


def _log_single_state(
    core_profiles: state.CoreProfiles,
    t: float | jax.Array,
) -> None:
  log_to_stdout('At time t = %.4f\n' % float(t), color=AnsiColors.GREEN)
  logging.info('T_i: %s', core_profiles.T_i.value)
  logging.info('T_e: %s', core_profiles.T_e.value)
  logging.info('psi: %s', core_profiles.psi.value)
  logging.info('n_e: %s', core_profiles.n_e.value)
  logging.info('n_i: %s', core_profiles.n_i.value)
  logging.info('q: %s', core_profiles.q_face)
  logging.info('magnetic_shear: %s', core_profiles.s_face)


def log_simulation_output_to_stdout(
    core_profile_history: Sequence[state.CoreProfiles],
    t: np.ndarray,
) -> None:
  _log_single_state(core_profile_history[0], t[0])
  logging.info('\n')
  _log_single_state(core_profile_history[-1], t[-1])


def can_plot() -> bool:
  # TODO(b/335596567): Find way to detect displays that works on all OS's.
  return True


def main(
    get_config: Callable[[], model_config.ToraxConfig],
    *,
    output_dir: str | None = None,
    log_sim_progress: bool = False,
    log_sim_output: bool = False,
    plot_sim_progress: bool = False,
    log_sim_progress_bar: bool = True,
) -> str:
  """Runs a simulation obtained via `get_config`.

  This function will always write files to a directory containing the
  simulation output and the input config. Results will be stored as a file
  `state_history_[timestamp].nc`. If the output directory does not exist it will
  be created.

  Args:
    get_config: Callable that returns a ToraxConfig.
    output_dir: Path to an output directory. If not provided, then results will
      be written to `/tmp/torax_results`
    log_sim_progress: If True, then the times for each step of the simulation
      are written out as they execute. The logging might be deferred or
      asynchronous depending on whether JAX compilation is enabled. If False,
      nothing extra is logged.
    log_sim_output: If True, then the simulation state output is logged at the
      end of the run. If False, nothing happens.
    plot_sim_progress: If True, then a plotting spectator will be attached to
      the sim.
    log_sim_progress_bar: If True, then a progress bar will be logged.

  Returns:
    The output state file path.
  """

  torax_config = get_config()

  log_to_stdout('Starting simulation.', color=AnsiColors.GREEN)
  data_tree, state_history = run_simulation.run_simulation(
      torax_config,
      log_sim_progress,
      progress_bar=log_sim_progress_bar,
  )
  log_to_stdout('Finished running simulation.', color=AnsiColors.GREEN)

  if plot_sim_progress:
    raise NotImplementedError('Plotting progress is temporarily disabled.')

  output_dir = output_dir if output_dir else _DEFAULT_OUTPUT_DIR
  output_file = _write_simulation_output_to_dir(output_dir, data_tree)

  if log_sim_output:
    log_simulation_output_to_stdout(
        state_history.core_profiles,
        data_tree.time.values,
    )

  return output_file
