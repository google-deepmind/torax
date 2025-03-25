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

"""Functionality for running simulations.

This includes the `run_simulation` main loop, logging functionality,
and functionality for translating between our particular physics
simulation and generic fluid dynamics PDE solvers.

Use the TORAX_COMPILATION_ENABLED environment variable to turn
jax compilation off and on. Compilation is on by default. Turning
compilation off can sometimes help with debugging (e.g. by making
it easier to print error messages in context).
"""

import dataclasses
import time

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from torax import output
from torax import post_processing
from torax import state
from torax.config import build_runtime_params
from torax.config import runtime_params_slice
from torax.core_profiles import initialization
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.orchestration import step_function
from torax.sources import source_profile_builders
import tqdm
import xarray as xr


def get_initial_state(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    step_fn: step_function.SimulationStepFn,
) -> state.ToraxSimState:
  """Returns the initial state to be used by run_simulation()."""
  initial_core_profiles = initialization.initial_core_profiles(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      step_fn.stepper.source_models,
  )
  # Populate the starting state with source profiles from the implicit sources
  # before starting the run-loop. The explicit source profiles will be computed
  # inside the loop and will be merged with these implicit source profiles.
  initial_core_sources = source_profile_builders.get_initial_source_profiles(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=initial_core_profiles,
      source_models=step_fn.stepper.source_models,
  )

  return state.ToraxSimState(
      t=jnp.array(dynamic_runtime_params_slice.numerics.t_initial),
      dt=jnp.zeros(()),
      core_profiles=initial_core_profiles,
      # This will be overridden within run_simulation().
      core_sources=initial_core_sources,
      core_transport=state.CoreTransport.zeros(geo),
      post_processed_outputs=state.PostProcessedOutputs.zeros(geo),
      time_step_calculator_state=step_fn.time_step_calculator.initial_state(),
      stepper_numeric_outputs=state.StepperNumericOutputs(
          stepper_error_state=0,
          outer_stepper_iterations=0,
          inner_solver_iterations=0,
      ),
      geometry=geo,
  )


def _override_initial_runtime_params_from_file(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    t_restart: float,
    ds: xr.Dataset,
) -> tuple[
    runtime_params_slice.DynamicRuntimeParamsSlice,
    geometry.Geometry,
]:
  """Override parts of runtime params slice from state in a file."""
  # pylint: disable=invalid-name
  dynamic_runtime_params_slice.numerics.t_initial = t_restart
  dynamic_runtime_params_slice.profile_conditions.Ip_tot = ds.data_vars[
      output.IP_PROFILE_FACE
  ].to_numpy()[-1]/1e6  # Convert from A to MA.
  dynamic_runtime_params_slice.profile_conditions.Te = ds.data_vars[
      output.TEMP_EL
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.Te_bound_right = ds.data_vars[
      output.TEMP_EL_RIGHT_BC
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.Ti = ds.data_vars[
      output.TEMP_ION
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.Ti_bound_right = ds.data_vars[
      output.TEMP_ION_RIGHT_BC
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.ne = ds.data_vars[
      output.NE
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.ne_bound_right = ds.data_vars[
      output.NE_RIGHT_BC
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.psi = ds.data_vars[
      output.PSI
  ].to_numpy()
  # When loading from file we want ne not to have transformations.
  # Both ne and the boundary condition are given in absolute values (not fGW).
  dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_fGW = False
  dynamic_runtime_params_slice.profile_conditions.ne_is_fGW = False
  dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_absolute = (
      True
  )
  # Additionally we want to avoid normalizing to nbar.
  dynamic_runtime_params_slice.profile_conditions.normalize_to_nbar = False
  # pylint: enable=invalid-name

  dynamic_runtime_params_slice, geo = runtime_params_slice.make_ip_consistent(
      dynamic_runtime_params_slice, geo
  )

  return dynamic_runtime_params_slice, geo


def _override_initial_state_post_processed_outputs_from_file(
    geo: geometry.Geometry,
    ds: xr.Dataset,
) -> state.PostProcessedOutputs:
  """Override parts of initial state post processed outputs from file."""
  post_processed_outputs = state.PostProcessedOutputs.zeros(geo)
  post_processed_outputs = dataclasses.replace(
      post_processed_outputs,
      E_cumulative_fusion=ds.data_vars['E_cumulative_fusion'].to_numpy(),
      E_cumulative_external=ds.data_vars['E_cumulative_external'].to_numpy(),
  )
  return post_processed_outputs


def _run_simulation(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_provider: build_runtime_params.DynamicRuntimeParamsSliceProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    initial_state: state.ToraxSimState,
    restart_case: bool,
    step_fn: step_function.SimulationStepFn,
    log_timestep_info: bool = False,
    progress_bar: bool = True,
) -> output.ToraxSimOutputs:
  """Runs the transport simulation over a prescribed time interval.

  This is the main entrypoint for running a TORAX simulation.

  This function runs a variable number of time steps until the
  time_step_calculator determines the sim is done, using a Python while loop.

  Args:
    static_runtime_params_slice: A static set of arguments to provide to the
      step_fn. If internal functions in step_fn are JAX-compiled, then these
      params are "compile-time constant" meaning that they are considered static
      to the compiled functions. If they change (i.e. the same step_fn is called
      again with a different static_runtime_params_slice), then internal
      functions will be recompiled. JAX determines if recompilation is necessary
      via the hash of static_runtime_params_slice.
    dynamic_runtime_params_slice_provider: Provides a DynamicRuntimeParamsSlice
      to use as input for each time step. See static_runtime_params_slice and
      the runtime_params_slice module docstring for runtime_params_slice to
      understand why we need the dynamic and static config slices and what they
      control.
    geometry_provider: Provides the magnetic geometry for each time step based
      on the ToraxSimState at the start of the time step. The geometry may
      change from time step to time step, so the sim needs a function to provide
      which geometry to use for a given time step. A GeometryProvider is any
      callable (class or function) which takes the ToraxSimState at the start of
      a time step and returns the Geometry for that time step. For most use
      cases, only the time will be relevant from the ToraxSimState (in order to
      support time-dependent geometries).
    initial_state: The starting state of the simulation. This includes both the
      state variables which the stepper.Stepper will evolve (like ion temp, psi,
      etc.) as well as other states that need to be be tracked, like time.
    restart_case: If True, the simulation is being restarted from a saved state.
    step_fn: Callable which takes in ToraxSimState and outputs the ToraxSimState
      after one timestep. Note that step_fn determines dt (how long the timestep
      is). The state_history that run_simulation() outputs comes from these
      ToraxSimState objects.
    log_timestep_info: If True, logs basic timestep info, like time, dt, on
      every step.
    progress_bar: If True, displays a progress bar.

  Returns:
    ToraxSimOutputs, containing information on the sim error state, and the
    simulation history, consisting of a tuple of ToraxSimState objects, one for
    each time step. There are N+1 objects returned, where N is the number of
    simulation steps taken. The first object in the tuple is for the initial
    state. If the sim error state is 1, then a trunctated simulation history is
    returned up until the last valid timestep.
  """

  # Provide logging information on precision setting
  if jax.config.read('jax_enable_x64'):
    logging.info('Precision is set at float64')
  else:
    logging.info('Precision is set at float32')

  logging.info('Starting simulation.')
  # Python while loop implementation.
  # Not efficient for grad, jit of grad.
  # Uses time_step_calculator.not_done to decide when to stop.
  # Note: can't use a jax while loop due to appending to history.

  running_main_loop_start_time = time.time()
  wall_clock_step_times = []

  dynamic_runtime_params_slice, geo = (
      build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
          t=initial_state.t,
          dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
          geometry_provider=geometry_provider,
      )
  )

  sim_state = initial_state
  sim_history = []
  sim_state = post_processing.make_outputs(
      sim_state=sim_state,
      geo=geo,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
  )
  sim_history.append(sim_state)

  # Set the sim_error to NO_ERROR. If we encounter an error, we will set it to
  # the appropriate error code.
  sim_error = state.SimError.NO_ERROR

  with tqdm.tqdm(
      total=100,  # This makes it so that the progress bar measures a percentage
      desc='Simulating',
      disable=not progress_bar,
      leave=True,
  ) as pbar:
    # Advance the simulation until the time_step_calculator tells us we are done
    first_step = True if not restart_case else False
    while step_fn.time_step_calculator.not_done(
        sim_state.t,
        dynamic_runtime_params_slice.numerics.t_final,
        sim_state.time_step_calculator_state,
    ):
      # Measure how long in wall clock time each simulation step takes.
      step_start_time = time.time()
      if log_timestep_info:
        _log_timestep(sim_state)

      sim_state, sim_error = step_fn(
          static_runtime_params_slice,
          dynamic_runtime_params_slice_provider,
          geometry_provider,
          sim_state,
      )

      wall_clock_step_times.append(time.time() - step_start_time)

      # Checks if sim_state is valid. If not, exit simulation early.
      # We don't raise an Exception because we want to return the truncated
      # simulation history to the user for inspection.
      if sim_error != state.SimError.NO_ERROR:
        sim_error.log_error()
        break
      else:
        if first_step:
          first_step = False
          if not static_runtime_params_slice.use_vloop_lcfs_boundary_condition:
            # For the Ip BC case, set vloop_lcfs[0] to the same value as
            # vloop_lcfs[1] due the vloop_lcfs timeseries being underconstrained
            sim_history[0].core_profiles = dataclasses.replace(
                sim_history[0].core_profiles,
                vloop_lcfs=sim_state.core_profiles.vloop_lcfs,
            )
        sim_history.append(sim_state)
        # Calculate progress ratio and update pbar.n
        progress_ratio = (
            float(sim_state.t) - dynamic_runtime_params_slice.numerics.t_initial
        ) / (
            dynamic_runtime_params_slice.numerics.t_final
            - dynamic_runtime_params_slice.numerics.t_initial
        )
        pbar.n = int(progress_ratio * pbar.total)
        pbar.set_description(f'Simulating (t={sim_state.t:.5f})')
        pbar.refresh()

  # Log final timestep
  if log_timestep_info and sim_error == state.SimError.NO_ERROR:
    # The "sim_state" here has been updated by the loop above.
    _log_timestep(sim_state)

  # If the first step of the simulation was very long, call it out. It might
  # have to do with tracing the jitted step_fn.
  std_devs = 2  # Check if the first step is more than 2 std devs longer.
  if wall_clock_step_times and wall_clock_step_times[0] > (
      np.mean(wall_clock_step_times) + std_devs * np.std(wall_clock_step_times)
  ):
    long_first_step = True
    logging.info(
        'The first step took more than %.1f std devs longer than other steps. '
        'It likely was tracing and compiling the step_fn. It took %.2fs '
        'of wall clock time.',
        std_devs,
        wall_clock_step_times[0],
    )
  else:
    long_first_step = False

  wall_clock_time_elapsed = time.time() - running_main_loop_start_time
  simulation_time = sim_history[-1].t - sim_history[0].t
  if long_first_step:
    # Don't include the long first step in the total time logged.
    wall_clock_time_elapsed -= wall_clock_step_times[0]
  logging.info(
      'Simulated %.2fs of physics in %.2fs of wall clock time.',
      simulation_time,
      wall_clock_time_elapsed,
  )
  return output.ToraxSimOutputs(
      sim_error=sim_error, sim_history=tuple(sim_history)
  )


def _log_timestep(
    sim_state: state.ToraxSimState,
) -> None:
  """Logs basic timestep info."""
  log_str = (
      f'Simulation time: {sim_state.t:.5f}, previous dt: {sim_state.dt:.6f},'
      ' previous stepper iterations:'
      f' {sim_state.stepper_numeric_outputs.outer_stepper_iterations}'
  )
  # TODO(b/330172917): once tol and coarse_tol are configurable in the
  # runtime_params, also log the value of tol and coarse_tol below
  match sim_state.stepper_numeric_outputs.stepper_error_state:
    case 0:
      pass
    case 1:
      log_str += 'Solver did not converge in previous step.'
    case 2:
      log_str += (
          'Solver converged only within coarse tolerance in previous step.'
      )
  tqdm.tqdm.write(log_str)
