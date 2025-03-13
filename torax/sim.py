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

from __future__ import annotations

import dataclasses
import time
from typing import Optional

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from torax import output
from torax import post_processing
from torax import state
from torax.config import build_runtime_params
from torax.config import config_args
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.core_profiles import initialization
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.orchestration import step_function
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.stepper import stepper as stepper_lib
from torax.time_step_calculator import chi_time_step_calculator
from torax.time_step_calculator import time_step_calculator as ts
from torax.transport_model import transport_model as transport_model_lib
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


class Sim:
  """A lightweight object holding all components of a simulation.

  Use of this object is optional, it is also fine to hold these objects
  in local variables of a script and call `run_simulation` directly.

  The main purpose of the Sim object is to enable configuration via
  constructor arguments. Components are reused in subsequent simulation runs, so
  if a component is compiled, it will be reused for the next `Sim.run()` call
  and will not be recompiled unless a static argument or shape changes.
  """

  def __init__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_provider: build_runtime_params.DynamicRuntimeParamsSliceProvider,
      geometry_provider: geometry_provider_lib.GeometryProvider,
      initial_state: state.ToraxSimState,
      step_fn: step_function.SimulationStepFn,
      file_restart: general_runtime_params.FileRestart | None = None,
  ):
    self._static_runtime_params_slice = static_runtime_params_slice
    self._dynamic_runtime_params_slice_provider = (
        dynamic_runtime_params_slice_provider
    )
    self._geometry_provider = geometry_provider
    self._initial_state = initial_state
    self._step_fn = step_fn
    self._file_restart = file_restart

  @property
  def file_restart(self) -> general_runtime_params.FileRestart | None:
    return self._file_restart

  @property
  def time_step_calculator(self) -> ts.TimeStepCalculator:
    return self._step_fn.time_step_calculator

  @property
  def initial_state(self) -> state.ToraxSimState:
    return self._initial_state

  @property
  def geometry_provider(self) -> geometry_provider_lib.GeometryProvider:
    return self._geometry_provider

  @property
  def dynamic_runtime_params_slice_provider(
      self,
  ) -> build_runtime_params.DynamicRuntimeParamsSliceProvider:
    return self._dynamic_runtime_params_slice_provider

  @property
  def static_runtime_params_slice(
      self,
  ) -> runtime_params_slice.StaticRuntimeParamsSlice:
    return self._static_runtime_params_slice

  @property
  def step_fn(self) -> step_function.SimulationStepFn:
    return self._step_fn

  @property
  def stepper(self) -> stepper_lib.Stepper:
    return self._step_fn.stepper

  @property
  def transport_model(self) -> transport_model_lib.TransportModel:
    return self.stepper.transport_model

  @property
  def pedestal_model(self) -> pedestal_model_lib.PedestalModel:
    return self.stepper.pedestal_model

  @property
  def source_models(self) -> source_models_lib.SourceModels:
    return self.stepper.source_models

  def update_base_components(
      self,
      *,
      allow_recompilation: bool = False,
      static_runtime_params_slice: (
          runtime_params_slice.StaticRuntimeParamsSlice | None
      ) = None,
      dynamic_runtime_params_slice_provider: (
          build_runtime_params.DynamicRuntimeParamsSliceProvider | None
      ) = None,
      geometry_provider: geometry_provider_lib.GeometryProvider | None = None,
  ):
    """Updates the Sim object with components that have already been updated.

    Currently this only supports updating the geometry provider and the dynamic
    runtime params slice provider, both of which can be updated without
    recompilation.

    Args:
      allow_recompilation: Whether recompilation is allowed. If True, the static
        runtime params slice can be updated. NOTE: recompilaton may still occur
        if the mesh is updated or if the shapes returned in the dynamic runtime
        params slice provider change even if this is False.
      static_runtime_params_slice: The new static runtime params slice. If None,
        the existing one is kept.
      dynamic_runtime_params_slice_provider: The new dynamic runtime params
        slice provider. This should already have been updated with modifications
        to the various components. If None, the existing one is kept.
      geometry_provider: The new geometry provider. If None, the existing one is
        kept.

    Raises:
      ValueError: If the Sim object has a file restart or if the geometry
        provider has a different mesh than the existing one.
    """
    if self._file_restart is not None:
      # TODO(b/384767453): Add support for updating a Sim object with a file
      # restart.
      raise ValueError('Cannot update a Sim object with a file restart.')
    if not allow_recompilation and static_runtime_params_slice is not None:
      raise ValueError(
          'Cannot update a Sim object with a static runtime params slice if '
          'recompilation is not allowed.'
      )

    if static_runtime_params_slice is not None:
      assert isinstance(  # Avoid pytype error.
          self._static_runtime_params_slice,
          runtime_params_slice.StaticRuntimeParamsSlice,
      )
      self._static_runtime_params_slice.validate_new(
          static_runtime_params_slice
      )
      self._static_runtime_params_slice = static_runtime_params_slice
    if dynamic_runtime_params_slice_provider is not None:
      assert isinstance(  # Avoid pytype error.
          self._dynamic_runtime_params_slice_provider,
          build_runtime_params.DynamicRuntimeParamsSliceProvider,
      )
      self._dynamic_runtime_params_slice_provider.validate_new(
          dynamic_runtime_params_slice_provider
      )
      self._dynamic_runtime_params_slice_provider = (
          dynamic_runtime_params_slice_provider
      )
    if geometry_provider is not None:
      if geometry_provider.torax_mesh != self._geometry_provider.torax_mesh:
        raise ValueError(
            'Cannot update a Sim object with a geometry provider with a '
            'different mesh.'
        )
      self._geometry_provider = geometry_provider

    dynamic_runtime_params_slice_for_init, geo_for_init = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=self._dynamic_runtime_params_slice_provider.runtime_params_provider.numerics.runtime_params_config.t_initial,
            dynamic_runtime_params_slice_provider=self._dynamic_runtime_params_slice_provider,
            geometry_provider=self._geometry_provider,
        )
    )
    self._initial_state = get_initial_state(
        static_runtime_params_slice=self._static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_for_init,
        geo=geo_for_init,
        step_fn=self._step_fn,
    )

  def run(
      self,
      log_timestep_info: bool = False,
  ) -> output.ToraxSimOutputs:
    """Runs the transport simulation over a prescribed time interval.

    See `run_simulation` for details.

    Args:
      log_timestep_info: See `run_simulation()`.

    Returns:
      Tuple of all ToraxSimStates, one per time step and an additional one at
      the beginning for the starting state.
    """
    return _run_simulation(
        static_runtime_params_slice=self.static_runtime_params_slice,
        dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
        geometry_provider=self.geometry_provider,
        initial_state=self.initial_state,
        step_fn=self.step_fn,
        log_timestep_info=log_timestep_info,
    )

  @classmethod
  def create(
      cls,
      *,
      runtime_params: general_runtime_params.GeneralRuntimeParams,
      geometry_provider: geometry_provider_lib.GeometryProvider,
      stepper: stepper_pydantic_model.Stepper,
      transport_model_builder: transport_model_lib.TransportModelBuilder,
      source_models_builder: source_models_lib.SourceModelsBuilder,
      pedestal: pedestal_pydantic_model.Pedestal,
      time_step_calculator: Optional[ts.TimeStepCalculator] = None,
      file_restart: Optional[general_runtime_params.FileRestart] = None,
  ) -> Sim:
    """Builds a Sim object from the input runtime params and sim components.

    Args:
      runtime_params: The input runtime params used throughout the simulation
        run.
      geometry_provider: The geometry used throughout the simulation run.
      stepper: The stepper config that can be used to build the stepper.
      transport_model_builder: A callable to build the transport model.
      source_models_builder: Builds the SourceModels and holds its
        runtime_params.
      pedestal: The pedestal config that can be used to build the pedestal.
      time_step_calculator: The time_step_calculator, if built, otherwise a
        ChiTimeStepCalculator will be built by default.
      file_restart: If provided we will reconstruct the initial state from the
        provided file at the given time step. This state from the file will only
        be used for constructing the initial state (as well as the config) and
        for all subsequent steps, the evolved state and runtime parameters from
        config are used.

    Returns:
      sim: The built Sim instance.
    """

    transport_model = transport_model_builder()
    pedestal_model = pedestal.build_pedestal_model()

    # TODO(b/385788907): Document all changes that lead to recompilations.
    static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            source_runtime_params=source_models_builder.runtime_params,
            torax_mesh=geometry_provider.torax_mesh,
            stepper=stepper,
        )
    )
    dynamic_runtime_params_slice_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            transport=transport_model_builder.runtime_params,
            sources=source_models_builder.runtime_params,
            stepper=stepper,
            torax_mesh=geometry_provider.torax_mesh,
            pedestal=pedestal,
        )
    )
    source_models = source_models_builder()
    stepper_model = stepper.build_stepper_model(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )

    if time_step_calculator is None:
      time_step_calculator = chi_time_step_calculator.ChiTimeStepCalculator()

    # Build dynamic_runtime_params_slice at t_initial for initial conditions.
    dynamic_runtime_params_slice_for_init, geo_for_init = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=runtime_params.numerics.t_initial,
            dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
            geometry_provider=geometry_provider,
        )
    )
    if file_restart is not None and file_restart.do_restart:
      data_tree = output.load_state_file(file_restart.filename)
      # Find the closest time in the given dataset.
      data_tree = data_tree.sel(time=file_restart.time, method='nearest')
      t_restart = data_tree.time.item()
      core_profiles_dataset = data_tree.children[output.CORE_PROFILES].dataset
      # Remap coordinates in saved file to be consistent with expectations of
      # how config_args parses xarrays.
      core_profiles_dataset = core_profiles_dataset.rename(
          {output.RHO_CELL_NORM: config_args.RHO_NORM}
      )
      core_profiles_dataset = core_profiles_dataset.squeeze()
      if t_restart != runtime_params.numerics.t_initial:
        logging.warning(
            'Requested restart time %f not exactly available in state file %s.'
            ' Restarting from closest available time %f instead.',
            file_restart.time,
            file_restart.filename,
            t_restart,
        )
      # Override some of dynamic runtime params slice from t=t_initial.
      dynamic_runtime_params_slice_for_init, geo_for_init = (
          _override_initial_runtime_params_from_file(
              dynamic_runtime_params_slice_for_init,
              geo_for_init,
              t_restart,
              core_profiles_dataset,
          )
      )
      post_processed_dataset = data_tree.children[
          output.POST_PROCESSED_OUTPUTS
      ].dataset
      post_processed_dataset = post_processed_dataset.rename(
          {output.RHO_CELL_NORM: config_args.RHO_NORM}
      )
      post_processed_dataset = post_processed_dataset.squeeze()
      post_processed_outputs = (
          _override_initial_state_post_processed_outputs_from_file(
              geo_for_init,
              post_processed_dataset,
          )
      )

    step_fn = step_function.SimulationStepFn(
        stepper=stepper_model,
        time_step_calculator=time_step_calculator,
        transport_model=transport_model,
        pedestal_model=pedestal_model,
    )

    initial_state = get_initial_state(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_for_init,
        geo=geo_for_init,
        step_fn=step_fn,
    )

    # If we are restarting from a file, we need to override the initial state
    # post processed outputs such that cumulative outputs remain correct.
    if file_restart is not None and file_restart.do_restart:
      initial_state = dataclasses.replace(
          initial_state,
          post_processed_outputs=post_processed_outputs,  # pylint: disable=undefined-variable
      )

    return cls(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
        geometry_provider=geometry_provider,
        initial_state=initial_state,
        step_fn=step_fn,
        file_restart=file_restart,
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
  ].to_numpy()[-1]
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
      dynamic_runtime_params_slice=dynamic_runtime_params_slice
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
