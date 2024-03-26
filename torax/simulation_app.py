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

r"""A module providing a runnable main for transport simulation.

You can use the `main()` function in this module as an
entry point for your simulation (see the example below).

You need to provide a method that returns a valid Sim object. Then
this module handles the rest: runs the simulation and writes the output to file
if directed to.

See run_simulation_main.py for a ready-made entrypoint for running simulations.
You can also implement your own main with a different implementation as
long as you pass in your sim-getter to this module, as shown below.

```
# In my_runnable_sim.py:

from absl import app
import torax
from torax import simulation_app

def get_sim():
  return torax.Sim(<your args here>)

if __name__ == '__main__':
  app.run(simulation_app.main(get_sim()))
```
"""

import datetime
import enum
import os
import sys
from typing import Callable

from absl import logging
import chex
import h5py
from jax import numpy as jnp
from matplotlib import pyplot as plt
import torax
from torax import config_slice
from torax import geometry
from torax import sim as sim_lib
from torax import state as state_lib
from torax.spectators import plotting

import shutil


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
_STATE_HISTORY_FILENAME = 'state_history.h5'


def log_to_stdout(output: str, color: AnsiColors | None = None) -> None:
  if not color or not sys.stderr.isatty():
    logging.info(output)
  else:
    logging.info('%s%s%s', color.value, output, _ANSI_END)


def write_simulation_output_to_file(
    output_dir: str,
    state_history: torax.CoreProfiles,
    aux_history: torax.AuxOutput,
    geo: torax.Geometry,
    t: jnp.ndarray,
) -> None:
  """Writes the state history and some geometry information to an HDF5 file."""
  output_file = os.path.join(output_dir, _STATE_HISTORY_FILENAME)
  with h5py.File(output_file, 'w') as h5_file:
    h5_file.create_dataset('t', data=t.tolist())
    h5_file.create_dataset('r_cell', data=geo.r.tolist())
    h5_file.create_dataset('r_face', data=geo.r_face.tolist())
    h5_file.create_dataset('r_cell_norm', data=geo.r_norm.tolist())
    h5_file.create_dataset('r_face_norm', data=geo.r_face_norm.tolist())
    h5_file.create_dataset(
        'temp_ion', data=state_history.temp_ion.value.tolist()
    )
    h5_file.create_dataset('temp_el', data=state_history.temp_el.value.tolist())
    h5_file.create_dataset('psi', data=state_history.psi.value.tolist())
    h5_file.create_dataset('ne', data=state_history.ne.value.tolist())
    h5_file.create_dataset('ni', data=state_history.ni.value.tolist())
    h5_file.create_dataset('q_face', data=state_history.q_face.tolist())
    h5_file.create_dataset('s_face', data=state_history.s_face.tolist())
    h5_file.create_dataset('jtot', data=state_history.currents.jtot.tolist())
    h5_file.create_dataset('johm', data=state_history.currents.johm.tolist())
    h5_file.create_dataset('jext', data=state_history.currents.jext.tolist())
    h5_file.create_dataset(
        'j_bootstrap', data=state_history.currents.j_bootstrap.tolist()
    )
    h5_file.create_dataset('sigma', data=state_history.currents.sigma.tolist())
    h5_file.create_dataset(
        'chi_face_ion', data=aux_history.chi_face_ion.tolist()
    )
    h5_file.create_dataset('chi_face_el', data=aux_history.chi_face_el.tolist())
    h5_file.create_dataset('source_ion', data=aux_history.source_ion.tolist())
    h5_file.create_dataset('source_el', data=aux_history.source_el.tolist())
    h5_file.create_dataset('Pfus_i', data=aux_history.Pfus_i.tolist())
    h5_file.create_dataset('Pfus_e', data=aux_history.Pfus_e.tolist())
    h5_file.create_dataset('Pohm', data=aux_history.Pohm.tolist())
    h5_file.create_dataset('Qei', data=aux_history.Qei.tolist())
  log_to_stdout(f'Wrote simulation output to {output_file}', AnsiColors.GREEN)


def _log_single_state(
    core_profiles: torax.CoreProfiles,
    t: float | jnp.ndarray,
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
    state_history: torax.CoreProfiles,
    geo: torax.Geometry,
    t: jnp.ndarray,
) -> None:
  del geo
  _log_single_state(state_history.index(0), t[0])
  logging.info('\n')
  _log_single_state(state_history.index(-1), t[-1])


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
    config: torax.Config,
    geo: geometry.Geometry,
) -> sim_lib.Sim:
  """Updates the sim with a new config and geometry."""
  # NOTE: This function will NOT update any of the following in the config:
  #  - stepper (for the mesh state)
  #  - transport model
  #  - spectator
  #  - time step calculator
  # TODO(b/323504363): Add checks to make sure the SimulationStepFn can be reused
  # correctly given the new config. If any of the attributes above change, then
  # ether raise an error or build a new SimulationStepFn (and notify the user).
  # TODO(b/323504363): If the static slice is updated, add checks or logs
  # notifying the user that using this new config will result in recompiling
  # the SimulationStepFn.
  static_config_slice = config_slice.build_static_config_slice(config)
  initial_state = sim_lib.get_initial_state(
      config=config,
      geo=geo,
      time_step_calculator=sim.time_step_calculator,
      sources=sim.sources,
  )
  return sim_lib.Sim(
      time_step_calculator=sim.time_step_calculator,
      initial_state=initial_state,
      geometry_provider=sim_lib.ConstantGeometryProvider(geo),
      dynamic_config_slice_provider=(
          config_slice.TimeDependentDynamicConfigSliceProvider(config)
      ),
      static_config_slice=static_config_slice,
      step_fn=sim.step_fn,
  )


def can_plot() -> bool:
  # TODO(b/323504363): Find way to detect displays that works on all OS's.
  return True


def main(
    get_sim: Callable[[], sim_lib.Sim],
    *,
    output_dir: str | None = None,
    log_sim_progress: bool = False,
    log_sim_output: bool = False,
    plot_sim_progress: bool = False,
) -> None:
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
  """
  output_dir = _get_output_dir(output_dir)

  sim = get_sim()
  geo = sim.geometry_provider(sim.initial_state)

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
  state_history, aux_history = state_lib.build_history_from_outputs(
      torax_outputs
  )
  t = state_lib.build_time_history_from_outputs(torax_outputs)
  log_to_stdout('Finished running simulation.', color=AnsiColors.GREEN)

  chex.assert_rank(t, 1)

  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)
  write_simulation_output_to_file(
      output_dir, state_history, aux_history, geo, t
  )
  # TODO(b/323504363): Add back functionality to write configs to file after
  # running to help with keeping track of simulation runs. This may need to
  # happen after we move to Fiddle.

  if log_sim_output:
    log_simulation_output_to_stdout(state_history, geo, t)
