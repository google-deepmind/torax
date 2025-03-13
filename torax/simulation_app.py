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
  from torax import simulation_app
  from torax.examples import basic_config
  from torax.config import build_sim

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
import sys
from typing import Callable, Final

from absl import logging
import jax
from torax import output
from torax import sim as sim_lib
from torax import state
from torax.config import build_runtime_params
from torax.config import runtime_params as runtime_params_lib
from torax.geometry import geometry
from torax.geometry import geometry_provider
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as sources_pydantic_model
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.transport_model import runtime_params as transport_runtime_params_lib
import xarray as xr

import shutil


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

_DEFAULT_OUTPUT_DIR_PREFIX = '/tmp/torax_results_'
_STATE_HISTORY_FILENAME = 'state_history.nc'


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


def write_simulation_output_to_file(
    output_dir: str, data_tree: xr.DataTree
) -> str:
  """Writes the state history and some geometry information to a NetCDF file."""

  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)

  output_file = os.path.join(output_dir, _STATE_HISTORY_FILENAME)
  data_tree.to_netcdf(output_file)
  log_to_stdout(f'{WRITE_PREFIX}{output_file}', AnsiColors.GREEN)

  return output_file


def _log_single_state(
    core_profiles: state.CoreProfiles,
    t: float | jax.Array,
) -> None:
  log_to_stdout('At time t = %.4f\n' % float(t), color=AnsiColors.GREEN)
  logging.info('temp_ion: %s', core_profiles.temp_ion.value)
  logging.info('temp_el: %s', core_profiles.temp_el.value)
  logging.info('psi: %s', core_profiles.psi.value)
  logging.info('ne: %s', core_profiles.ne.value)
  logging.info('ni: %s', core_profiles.ni.value)
  logging.info('q_face: %s', core_profiles.q_face)
  logging.info('s_face: %s', core_profiles.s_face)


def log_simulation_output_to_stdout(
    core_profile_history: state.CoreProfiles,
    geo: geometry.Geometry,
    t: jax.Array,
) -> None:
  del geo
  _log_single_state(core_profile_history.index(0), t[0])
  logging.info('\n')
  _log_single_state(core_profile_history.index(-1), t[-1])


def _get_output_dir(
    output_dir: str | None,
) -> str:
  if output_dir:
    return output_dir
  return _DEFAULT_OUTPUT_DIR_PREFIX + datetime.datetime.now().strftime(
      '%Y%m%d_%H%M%S'
  )


def update_sim(
    sim: sim_lib.Sim,
    runtime_params: runtime_params_lib.GeneralRuntimeParams,
    geo_provider: geometry_provider.GeometryProvider,
    transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
    sources: sources_pydantic_model.Sources,
    stepper: stepper_pydantic_model.Stepper,
    pedestal: pedestal_pydantic_model.Pedestal,
) -> None:
  """Updates the sim with a new set of runtime params and geometry."""
  # NOTE: This function will NOT update any of the following:
  #  - stepper (for the mesh state)
  #  - transport model object (runtime params are updated)
  #  - spectator
  #  - time step calculator
  #  - source objects (runtime params are updated)
  static_runtime_params_slice = (
      build_runtime_params.build_static_runtime_params_slice(
          runtime_params=runtime_params,
          sources=sources,
          torax_mesh=geo_provider.torax_mesh,
          stepper=stepper,
      )
  )
  dynamic_runtime_params_slice_provider = (
      build_runtime_params.DynamicRuntimeParamsSliceProvider(
          runtime_params=runtime_params,
          transport=transport_runtime_params,
          sources=sources,
          stepper=stepper,
          pedestal=pedestal,
          torax_mesh=geo_provider.torax_mesh,
      )
  )

  sim.update_base_components(
      allow_recompilation=True,
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
      geometry_provider=geo_provider,
  )


def can_plot() -> bool:
  # TODO(b/335596567): Find way to detect displays that works on all OS's.
  return True


def main(
    get_sim: Callable[[], sim_lib.Sim],
    *,
    output_dir: str | None = None,
    log_sim_progress: bool = False,
    log_sim_output: bool = False,
    plot_sim_progress: bool = False,
) -> str:
  """Runs a simulation obtained via `get_sim`.

  This function will always write files to a directory containing the
  simulation output and the input config. If the output directory exists, that
  folder will be deleted before writing new results. If it does not exist, then
  a new folder will be created.

  Args:
    get_sim: Callable that returns a Sim.
    output_dir: Path to an output directory. If not provided, then a folder in
      /tmp is created with a timestamp of this run. If the chosen directory
      already exists, then it will be deleted before writing new results to
      file. If not provided here, this function will try to use the value from
      the flag.
    log_sim_progress: If True, then the times for each step of the simulation
      are written out as they execute. The logging might be deferred or
      asynchronous depending on whether JAX compilation is enabled. If False,
      nothing extra is logged.
    log_sim_output: If True, then the simulation state output is logged at the
      end of the run. If False, nothing happens.
    plot_sim_progress: If True, then a plotting spectator will be attached to
      the sim.

  Returns:
    The output state file path.
  """
  output_dir = _get_output_dir(output_dir)

  sim = get_sim()
  geo = sim.geometry_provider(sim.initial_state.t)

  log_to_stdout('Starting simulation.', color=AnsiColors.GREEN)
  sim_outputs = sim.run(
      log_timestep_info=log_sim_progress,
  )
  log_to_stdout('Finished running simulation.', color=AnsiColors.GREEN)
  state_history = output.StateHistory(sim_outputs, sim.source_models)

  if plot_sim_progress:
    raise NotImplementedError('Plotting progress is temporarily disabled.')

  data_tree = state_history.simulation_output_to_xr(sim.file_restart)

  output_file = write_simulation_output_to_file(output_dir, data_tree)

  if log_sim_output:
    log_simulation_output_to_stdout(
        state_history.core_profiles, geo, state_history.times
    )

  return output_file
