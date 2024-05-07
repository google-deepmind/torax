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
long as you pass in your sim-getter to this module, as shown below.

.. code-block:: python

  # In my_runnable_sim.py:

  from absl import app
  import torax
  from torax import simulation_app

  def get_sim():
    return torax.Sim(<your args here>)

  if __name__ == '__main__':
    app.run(simulation_app.main(get_sim()))
"""

import datetime
import enum
import os
import sys
from typing import Callable

from absl import logging
import chex
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
import torax
from torax import geometry
from torax import sim as sim_lib
from torax import state as state_lib
from torax.config import runtime_params_slice
from torax.sources import runtime_params as source_runtime_params_lib
from torax.spectators import plotting
from torax.stepper import runtime_params as stepper_runtime_params_lib
from torax.transport_model import runtime_params as transport_runtime_params_lib
import xarray as xr
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
_STATE_HISTORY_FILENAME = 'state_history.nc'


def log_to_stdout(output: str, color: AnsiColors | None = None) -> None:
  if not color or not sys.stderr.isatty():
    logging.info(output)
  else:
    logging.info('%s%s%s', color.value, output, _ANSI_END)


def simulation_output_to_xr(
    torax_outputs: tuple[state_lib.ToraxSimState, ...],
    geo: torax.Geometry,
) -> xr.Dataset:
  """Build an xr.Dataset of the simulation output."""

  # TODO(b/338033916). Extend outputs with:
  # Post-processed integrals, more geo outputs.
  # Cleanup structure by excluding QeiInfo from core_sources altogether.
  # Add attribute to dataset variables with explanation of contents + units.

  #  Exclude uninteresting variables from output DataSet
  exclude_set = {
      'explicit_e',
      'explicit_i',
      'implicit_ee',
      'implicit_ii',
      'implicit_ei',
      'implicit_ie',
      'qei_coef',
  }

  name_map = {
      'fusion_heat_source_el': 'Qfus_e',
      'fusion_heat_source_ion': 'Qfus_i',
      'generic_ion_el_heat_source_el': 'Qext_e',
      'generic_ion_el_heat_source_ion': 'Qext_i',
      'ohmic_heat_source': 'Qohm',
      'qei_source': 'Qei',
      'gas_puff_source': 's_puff',
      'nbi_particle_source': 's_nbi',
      'pellet_source': 's_pellet',
  }

  def name_mapper(name):
    if name in name_map:
      return name_map[name]
    return name

  core_profile_history, core_transport_history, core_sources_history = (
      state_lib.build_history_from_states(torax_outputs)
  )
  t = state_lib.build_time_history_from_states(torax_outputs)
  chex.assert_rank(t, 1)

  # Get the coordinate variables for dimensions ("time", "rho_face", "rho_cell")
  time = xr.DataArray(t, dims=['time'], name='time')
  r_face_norm = xr.DataArray(
      geo.r_face_norm, dims=['rho_face'], name='r_face_norm'
  )
  r_cell_norm = xr.DataArray(geo.r_norm, dims=['rho_cell'], name='r_cell_norm')
  r_face = xr.DataArray(geo.r_face, dims=['rho_face'], name='r_face')
  r_cell = xr.DataArray(geo.r, dims=['rho_cell'], name='r_cell')

  # Build a PyTree of variables we will want to log.
  tree = (core_profile_history, core_transport_history, core_sources_history)

  # Only try to log arrays.
  leaves_with_path = jax.tree_util.tree_leaves_with_path(
      tree, is_leaf=lambda x: isinstance(x, jax.Array)
  )

  # Functions to check if a leaf is a face or cell variable
  # Assume that all arrays with shape (time, rho_face) are face variables
  # and all arrays with shape (time, rho_cell) are cell variables
  is_face_var = lambda x: x.ndim == 2 and x.shape == (
      len(time),
      len(geo.r_face),
  )
  is_cell_var = lambda x: x.ndim == 2 and x.shape == (len(time), len(geo.r))

  is_scalar = lambda x: x.ndim == 1 and x.shape == (len(time),)

  def translate_leaf_with_path(path, leaf):
    # Assume name is the last part of the path, unless the name is "value"
    # in which case we use the second to last part of the path.
    if isinstance(path[-1], jax.tree_util.DictKey):
      name = path[-1].key
    else:
      name = path[-1].name if path[-1].name != 'value' else path[-2].name
    name = name_mapper(name)
    if is_face_var(leaf):
      return name, xr.DataArray(leaf, dims=['time', 'rho_face'], name=name)
    elif is_cell_var(leaf):
      return name, xr.DataArray(leaf, dims=['time', 'rho_cell'], name=name)
    elif is_scalar(leaf):
      return name, xr.DataArray(leaf, dims=['time'], name=name)
    else:
      return name, None

  # Initialize dict with desired geometry and reference variables
  xr_dict = {
      'vpr': xr.DataArray(geo.vpr, dims=['rho_cell'], name='vpr'),
      'spr': xr.DataArray(geo.spr_cell, dims=['rho_cell'], name='spr'),
      'vpr_face': xr.DataArray(
          geo.vpr_face, dims=['rho_face'], name='vpr_face'
      ),
      'spr_face': xr.DataArray(
          geo.spr_face, dims=['rho_face'], name='spr_face'
      ),
  }

  # Extend with desired core_profiles, core_sources, core_transport variables
  for path, leaf in leaves_with_path:
    name, da = translate_leaf_with_path(path, leaf)
    if da is not None and name not in exclude_set:
      xr_dict[name] = da
  ds = xr.Dataset(
      xr_dict,
      coords={
          'time': time,
          'r_face_norm': r_face_norm,
          'r_cell_norm': r_cell_norm,
          'r_face': r_face,
          'r_cell': r_cell,
      },
  )
  return ds


def write_simulation_output_to_file(output_dir: str, ds: xr.Dataset) -> None:
  """Writes the state history and some geometry information to a NetCDF file."""
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)
  output_file = os.path.join(output_dir, _STATE_HISTORY_FILENAME)
  ds.to_netcdf(output_file)
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
    core_profile_history: torax.CoreProfiles,
    geo: torax.Geometry,
    t: jnp.ndarray,
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
    geo: geometry.Geometry,
    transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
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
  sim.transport_model.runtime_params = transport_runtime_params
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
          transport_getter=lambda: sim.transport_model.runtime_params,
          sources_getter=lambda: sim.source_models.runtime_params,
          stepper_getter=stepper_runtime_params_getter,
      )
  )
  initial_state = sim_lib.get_initial_state(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice_provider(
          t=runtime_params.numerics.t_initial
      ),
      geo=geo,
      time_step_calculator=sim.time_step_calculator,
      source_models=sim.source_models,
  )
  return sim_lib.Sim(
      time_step_calculator=sim.time_step_calculator,
      initial_state=initial_state,
      geometry_provider=sim_lib.ConstantGeometryProvider(geo),
      dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
      static_runtime_params_slice=static_runtime_params_slice,
      step_fn=sim.step_fn,
  )


def _update_source_params(
    sim: sim_lib.Sim,
    source_runtime_params: dict[str, source_runtime_params_lib.RuntimeParams],
) -> None:
  for source_name, source_runtime_params in source_runtime_params.items():
    if source_name not in sim.source_models.sources:
      raise ValueError(f'Source {source_name} not found in sim.')
    sim.source_models.sources[source_name].runtime_params = (
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
) -> xr.Dataset:
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
    An xarray Dataset containing the simulated output.
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
  log_to_stdout('Finished running simulation.', color=AnsiColors.GREEN)

  ds = simulation_output_to_xr(torax_outputs, geo)

  write_simulation_output_to_file(output_dir, ds)

  if log_sim_output:
    core_profile_history, _, _ = state_lib.build_history_from_states(
        torax_outputs
    )
    t = state_lib.build_time_history_from_states(torax_outputs)
    log_simulation_output_to_stdout(core_profile_history, geo, t)

  return ds
