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
from matplotlib import pyplot as plt
import torax
from torax import geometry_provider
from torax import output
from torax import sim as sim_lib
from torax.config import runtime_params_slice
from torax.sources import runtime_params as source_runtime_params_lib
from torax.spectators import plotting
from torax.stepper import runtime_params as stepper_runtime_params_lib
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


def write_simulation_output_to_file(output_dir: str, ds: xr.Dataset) -> str:
  """Writes the state history and some geometry information to a NetCDF file."""

  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)

  output_file = os.path.join(output_dir, _STATE_HISTORY_FILENAME)
  ds.to_netcdf(output_file)
  log_to_stdout(f'{WRITE_PREFIX}{output_file}', AnsiColors.GREEN)

  return output_file


def _log_single_state(
    core_profiles: torax.CoreProfiles,
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
    core_profile_history: torax.CoreProfiles,
    geo: torax.Geometry,
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
    runtime_params: torax.GeneralRuntimeParams,
    geo_provider: geometry_provider.GeometryProvider,
    transport_runtime_params_getter: Callable[
        [], transport_runtime_params_lib.RuntimeParams
    ],
    source_runtime_params: dict[str, source_runtime_params_lib.RuntimeParams],
    stepper_runtime_params_getter: Callable[
        [], stepper_runtime_params_lib.RuntimeParams
    ],
) -> sim_lib.Sim:
  """Updates the sim with a new set of runtime params and geometry."""
  # NOTE: This function will NOT update any of the following:
  #  - stepper (for the mesh state)
  #  - transport model object (runtime params are updated)
  #  - spectator
  #  - time step calculator
  #  - source objects (runtime params are updated)
  # TODO(b/323504363): change this to take a geometry provider instead of a
  # geometry object.

  _update_source_params(sim, source_runtime_params)
  static_runtime_params_slice = (
      runtime_params_slice.build_static_runtime_params_slice(
          runtime_params,
          stepper=stepper_runtime_params_getter(),
      )
  )
  dynamic_runtime_params_slice_provider = (
      runtime_params_slice.DynamicRuntimeParamsSliceProvider(
          runtime_params=runtime_params,
          transport_getter=transport_runtime_params_getter,
          sources_getter=lambda: sim.source_models_builder.runtime_params,
          stepper_getter=stepper_runtime_params_getter,
      )
  )

  geo = geo_provider(runtime_params.numerics.t_initial)
  dynamic_runtime_params_slice = dynamic_runtime_params_slice_provider(
      t=runtime_params.numerics.t_initial,
      geo=geo,
  )
  dynamic_runtime_params_slice, geo = runtime_params_slice.make_ip_consistent(
      dynamic_runtime_params_slice, geo
  )
  initial_state = sim_lib.get_initial_state(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      time_step_calculator=sim.time_step_calculator,
      source_models=sim.source_models,
  )
  return sim_lib.Sim(
      time_step_calculator=sim.time_step_calculator,
      initial_state=initial_state,
      geometry_provider=geo_provider,
      dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
      static_runtime_params_slice=static_runtime_params_slice,
      step_fn=sim.step_fn,
      source_models_builder=sim.source_models_builder,
  )


def _update_source_params(
    sim: sim_lib.Sim,
    source_runtime_params: dict[str, source_runtime_params_lib.RuntimeParams],
) -> None:
  for source_name, source_runtime_params in source_runtime_params.items():
    if source_name not in sim.source_models.sources:
      raise ValueError(f'Source {source_name} not found in sim.')
    sim.source_models_builder.source_builders[source_name].runtime_params = (
        source_runtime_params
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
) -> tuple[xr.Dataset, str]:
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
    An xarray Dataset containing the simulated output and the state file path.
  """
  output_dir = _get_output_dir(output_dir)

  sim = get_sim()
  geo = sim.geometry_provider(sim.initial_state.t)

  spectator = None
  if plot_sim_progress:
    if can_plot():
      plt.ion()
      spectator = plotting.PlotSpectator(
          plots=plotting.get_default_plot_config(geo=geo),
          pyplot_figure_kwargs=dict(
              figsize=(12, 6),
          ),
      )
      plt.show()
    else:
      logging.warning(
          'plotting requested, but there is no display connected to show the '
          'plot.'
      )

  log_to_stdout('Starting simulation.', color=AnsiColors.GREEN)
  torax_outputs = sim.run(
      log_timestep_info=log_sim_progress,
      spectator=spectator,
  )
  log_to_stdout('Finished running simulation.', color=AnsiColors.GREEN)
  state_history = output.StateHistory(torax_outputs)

  ds = state_history.simulation_output_to_xr(geo)

  output_file = write_simulation_output_to_file(output_dir, ds)

  if log_sim_output:
    history = output.StateHistory(torax_outputs)
    log_simulation_output_to_stdout(history.core_profiles, geo, history.times)

  return ds, output_file
