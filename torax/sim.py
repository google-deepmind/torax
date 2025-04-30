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
import numpy as np
from torax import state
from torax.config import build_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.orchestration import step_function
import tqdm


def _run_simulation(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_provider: build_runtime_params.DynamicRuntimeParamsSliceProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    initial_state: state.ToraxSimState,
    initial_post_processed_outputs: state.PostProcessedOutputs,
    restart_case: bool,
    step_fn: step_function.SimulationStepFn,
    log_timestep_info: bool = False,
    progress_bar: bool = True,
) -> tuple[
    tuple[state.ToraxSimState, ...],
    tuple[state.PostProcessedOutputs, ...],
    state.SimError,
]:
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
    initial_post_processed_outputs: The post-processed outputs at the start of
      the simulation. This is used to calculate cumulative quantities.
    restart_case: If True, the simulation is being restarted from a saved state.
    step_fn: Callable which takes in ToraxSimState and outputs the ToraxSimState
      after one timestep. Note that step_fn determines dt (how long the timestep
      is). The state_history that run_simulation() outputs comes from these
      ToraxSimState objects.
    log_timestep_info: If True, logs basic timestep info, like time, dt, on
      every step.
    progress_bar: If True, displays a progress bar.

  Returns:
    A tuple of:
      - the simulation history, consisting of a tuple of ToraxSimState objects,
        one for each time step. There are N+1 objects returned, where N is the
        number of simulation steps taken. The first object in the tuple is for
        the initial state. If the sim error state is 1, then a trunctated
        simulation history is returned up until the last valid timestep.
      - the post-processed outputs history, consisting of a tuple of
        PostProcessedOutputs objects, one for each time step. There are N+1
        objects returned, where N is the number of simulation steps taken. The
        first object in the tuple is for the initial state. If the sim error
        state is 1, then a trunctated simulation history is returned up until
        the last valid timestep.
      - The sim error state.
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

  dynamic_runtime_params_slice, _ = (
      build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
          t=initial_state.t,
          dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
          geometry_provider=geometry_provider,
      )
  )

  sim_state = initial_state
  state_history = [sim_state]
  post_processing_history = [initial_post_processed_outputs]

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
    ):
      # Measure how long in wall clock time each simulation step takes.
      step_start_time = time.time()
      if log_timestep_info:
        _log_timestep(sim_state)

      sim_state, post_processed_outputs, sim_error = step_fn(
          static_runtime_params_slice,
          dynamic_runtime_params_slice_provider,
          geometry_provider,
          sim_state,
          post_processing_history[-1],
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
            state_history[0].core_profiles = dataclasses.replace(
                state_history[0].core_profiles,
                vloop_lcfs=sim_state.core_profiles.vloop_lcfs,
            )
        state_history.append(sim_state)
        post_processing_history.append(post_processed_outputs)
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
  simulation_time = state_history[-1].t - state_history[0].t
  if long_first_step:
    # Don't include the long first step in the total time logged.
    wall_clock_time_elapsed -= wall_clock_step_times[0]
  logging.info(
      'Simulated %.2fs of physics in %.2fs of wall clock time.',
      simulation_time,
      wall_clock_time_elapsed,
  )
  return tuple(state_history), tuple(post_processing_history), sim_error


def _log_timestep(
    sim_state: state.ToraxSimState,
) -> None:
  """Logs basic timestep info."""
  log_str = (
      f'Simulation time: {sim_state.t:.5f}, previous dt: {sim_state.dt:.6f},'
      ' previous stepper iterations:'
      f' {sim_state.solver_numeric_outputs.outer_solver_iterations}'
  )
  # TODO(b/330172917): once tol and coarse_tol are configurable in the
  # runtime_params, also log the value of tol and coarse_tol below
  match sim_state.solver_numeric_outputs.solver_error_state:
    case 0:
      pass
    case 1:
      log_str += ' Solver did not converge in previous step.'
    case 2:
      log_str += (
          ' Solver converged only within coarse tolerance in previous step.'
      )
  tqdm.tqdm.write(log_str)
