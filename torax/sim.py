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
from typing import Any, Optional

from absl import logging
import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax import core_profile_setters
from torax import jax_utils
from torax import output
from torax import physics
from torax import post_processing
from torax import state
from torax.config import config_args
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.sources import ohmic_heat_source
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib
from torax.stepper import stepper as stepper_lib
from torax.time_step_calculator import chi_time_step_calculator
from torax.time_step_calculator import time_step_calculator as ts
from torax.transport_model import transport_model as transport_model_lib
import xarray as xr


def get_consistent_dynamic_runtime_params_slice_and_geometry(
    t: chex.Numeric,
    dynamic_runtime_params_slice_provider: runtime_params_slice.DynamicRuntimeParamsSliceProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
) -> tuple[
    runtime_params_slice.DynamicRuntimeParamsSlice,
    geometry.Geometry,
]:
  """Returns the dynamic runtime params and geometry for a given time."""
  geo = geometry_provider(t)
  dynamic_runtime_params_slice = dynamic_runtime_params_slice_provider(
      t=t,
  )
  dynamic_runtime_params_slice, geo = runtime_params_slice.make_ip_consistent(
      dynamic_runtime_params_slice, geo
  )
  return dynamic_runtime_params_slice, geo


class SimulationStepFn:
  """Advances the TORAX simulation one time step.

  Unlike the Stepper class, which updates certain parts of the state, a
  SimulationStepFn takes in the ToraxSimState and outputs the updated
  ToraxSimState, which contains not only the CoreProfiles but also extra
  simulation state useful for stepping as well as extra outputs useful for
  inspection inside the main run loop in `run_simulation()`. It wraps calls to
  Stepper with useful features to increase robustness for convergence, like
  dt-backtracking.
  """

  def __init__(
      self,
      stepper: stepper_lib.Stepper,
      time_step_calculator: ts.TimeStepCalculator,
      transport_model: transport_model_lib.TransportModel,
      pedestal_model: pedestal_model_lib.PedestalModel,
  ):
    """Initializes the SimulationStepFn.

    If you wish to run a simulation with new versions of any of these arguments
    (i.e. want to change to a new stepper), then you will need to build a new
    SimulationStepFn. These arguments are fixed for the lifetime
    of the SimulationStepFn and cannot change even with JAX recompiles.

    Args:
      stepper: Evolves the core profiles.
      time_step_calculator: Calculates the dt for each time step.
      transport_model: Calculates diffusion and convection coefficients.
      pedestal_model: Calculates pedestal coefficients.
    """
    self._stepper_fn = stepper
    self._time_step_calculator = time_step_calculator
    self._transport_model = transport_model
    self._pedestal_model = pedestal_model
    self._jitted_transport_model = jax_utils.jit(
        transport_model.__call__,
    )

  @property
  def pedestal_model(self) -> pedestal_model_lib.PedestalModel:
    return self._pedestal_model

  @property
  def stepper(self) -> stepper_lib.Stepper:
    return self._stepper_fn

  @property
  def transport_model(self) -> transport_model_lib.TransportModel:
    return self._transport_model

  @property
  def time_step_calculator(self) -> ts.TimeStepCalculator:
    return self._time_step_calculator

  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_provider: runtime_params_slice.DynamicRuntimeParamsSliceProvider,
      geometry_provider: geometry_provider_lib.GeometryProvider,
      input_state: state.ToraxSimState,
  ) -> tuple[state.ToraxSimState, state.SimError]:
    """Advances the simulation state one time step.

    Args:
      static_runtime_params_slice: Static parameters that, if they change,
        should trigger a recompilation of the SimulationStepFn.
      dynamic_runtime_params_slice_provider: Object that returns a set of
        runtime parameters which may change from time step to time step or
        simulation run to run. If these runtime parameters change, it does NOT
        trigger a JAX recompilation.
      geometry_provider: Provides the magnetic geometry for each time step based
        on the ToraxSimState at the start of the time step. The geometry may
        change from time step to time step, so the sim needs a function to
        provide which geometry to use for a given time step. A GeometryProvider
        is any callable (class or function) which takes the ToraxSimState at the
        start of a time step and returns the Geometry for that time step. For
        most use cases, only the time will be relevant from the ToraxSimState
        (in order to support time-dependent geometries).
      input_state: State at the start of the time step, including the core
        profiles which are being evolved.

    Returns:
      ToraxSimState containing:
        - the core profiles at the end of the time step.
        - time and time step calculator state info.
        - core_sources and core_transport at the end of the time step.
        - stepper_numeric_outputs. This contains the number of iterations
          performed in the stepper and the error state. The error states are:
            0 if solver converged with fine tolerance for this step
            1 if solver did not converge for this step (was above coarse tol)
            2 if solver converged within coarse tolerance. Allowed to pass with
              a warning. Occasional error=2 has low impact on final sim state.
      SimError indicating if an error has occurred during simulation.
    """
    dynamic_runtime_params_slice_t, geo_t = (
        get_consistent_dynamic_runtime_params_slice_and_geometry(
            input_state.t,
            dynamic_runtime_params_slice_provider,
            geometry_provider,
        )
    )

    # This only computes sources set to explicit in the
    # DynamicSourceConfigSlice. All implicit sources will have their profiles
    # set to 0.
    explicit_source_profiles = source_models_lib.build_source_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_t,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo_t,
        core_profiles=input_state.core_profiles,
        source_models=self.stepper.source_models,
        explicit=True,
    )

    # The previous time step's state has an incomplete set of source profiles
    # which was computed based on the previous time step's "guess" of the core
    # profiles at this time step's t. We can merge those "implicit" source
    # profiles with the explicit ones computed here.
    input_state.core_sources = merge_source_profiles(
        explicit_source_profiles=explicit_source_profiles,
        implicit_source_profiles=input_state.core_sources,
    )

    dt, time_step_calculator_state = self.init_time_step_calculator(
        dynamic_runtime_params_slice_t,
        geo_t,
        input_state,
    )

    # The stepper needs the geo and dynamic_runtime_params_slice at time t + dt
    # for implicit computations in the solver. Once geo_t_plus_dt is calculated
    # we can use it to calculate Phibdot for both geo_t and geo_t_plus_dt, which
    # then update the initialized Phibdot=0 in the geo instances.
    dynamic_runtime_params_slice_t_plus_dt, geo_t, geo_t_plus_dt = (
        _get_geo_and_dynamic_runtime_params_at_t_plus_dt_and_phibdot(
            input_state.t,
            dt,
            dynamic_runtime_params_slice_provider,
            geo_t,
            geometry_provider,
        )
    )

    output_state = self.step(
        dt,
        time_step_calculator_state,
        static_runtime_params_slice,
        dynamic_runtime_params_slice_t,
        dynamic_runtime_params_slice_t_plus_dt,
        geo_t,
        geo_t_plus_dt,
        input_state,
        explicit_source_profiles,
    )

    if static_runtime_params_slice.adaptive_dt:
      # This is a no-op if
      # output_state.stepper_numeric_outputs.stepper_error_state == 0.
      (
          dynamic_runtime_params_slice_t_plus_dt,
          geo_t_plus_dt,
          output_state,
      ) = self.adaptive_step(
          output_state,
          static_runtime_params_slice,
          dynamic_runtime_params_slice_t,
          dynamic_runtime_params_slice_provider,
          geo_t,
          geometry_provider,
          input_state,
          explicit_source_profiles,
      )

    sim_state = self.finalize_output(
        input_state,
        output_state,
        dynamic_runtime_params_slice_t_plus_dt,
        static_runtime_params_slice,
        geo_t_plus_dt,
    )
    return sim_state, sim_state.check_for_errors()

  def init_time_step_calculator(
      self,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t: geometry.Geometry,
      input_state: state.ToraxSimState,
  ) -> tuple[jnp.ndarray, Any]:
    """First phase: Initialize the stepper state.

    Args:
      dynamic_runtime_params_slice_t: Runtime parameters at time t.
      geo_t: The geometry of the torus during this time step of the simulation.
        While the geometry may change, any changes to the grid size can trigger
        recompilation of the stepper (if it is jitted) or an error (assuming it
        is JAX-compiled and lowered).
      input_state: State at the start of the time step, including the core
        profiles which are being evolved.

    Returns:
      Tuple containing:
        - time step duration (dt)
        - internal time stepper state
    """
    # TODO(b/335598388): We call the transport model both here and in the the
    # Stepper / CoeffsCallback. This isn't a problem *so long as all of those
    # calls fall within the same jit scope* because can use
    # functools.lru_cache to avoid building duplicate expressions for the same
    # transport coeffs. We should still refactor the design to more explicitly
    # calculate transport coeffs at delta_t = 0 in only one place, so that we
    # have some flexibility in where to place the jit boundaries.
    pedestal_model_output = self._pedestal_model(
        dynamic_runtime_params_slice_t, geo_t, input_state.core_profiles
    )
    transport_coeffs = self._jitted_transport_model(
        dynamic_runtime_params_slice_t,
        geo_t,
        input_state.core_profiles,
        pedestal_model_output,
    )

    # initialize new dt and reset stepper iterations.
    dt, time_step_calculator_state = self._time_step_calculator.next_dt(
        dynamic_runtime_params_slice_t,
        geo_t,
        input_state.core_profiles,
        input_state.time_step_calculator_state,
        transport_coeffs,
    )

    crosses_t_final = (
        input_state.t < dynamic_runtime_params_slice_t.numerics.t_final
    ) * (
        input_state.t + input_state.dt
        > dynamic_runtime_params_slice_t.numerics.t_final
    )
    dt = jnp.where(
        jnp.logical_and(
            dynamic_runtime_params_slice_t.numerics.exact_t_final,
            crosses_t_final,
        ),
        dynamic_runtime_params_slice_t.numerics.t_final - input_state.t,
        dt,
    )
    if jnp.any(jnp.isnan(dt)):
      raise ValueError('dt is NaN.')

    return (dt, time_step_calculator_state)

  def step(
      self,
      dt: jnp.ndarray,
      time_step_calculator_state: Any,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      input_state: state.ToraxSimState,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
  ) -> state.ToraxSimState:
    """Performs a simulation step with given dt.

    Stepper may fail to converge in which case adaptive_step() can be used to
    try smaller time step durations.

    Args:
      dt: Time step duration.
      time_step_calculator_state: Internal time stepper state.
      static_runtime_params_slice: Static parameters that, if they change,
        should trigger a recompilation of the SimulationStepFn.
      dynamic_runtime_params_slice_t: Runtime parameters at time t.
      dynamic_runtime_params_slice_t_plus_dt: Runtime parameters at time t + dt.
      geo_t: The geometry of the torus during this time step of the simulation.
      geo_t_plus_dt: The geometry of the torus during the next time step of the
        simulation.
      input_state: State at the start of the time step, including the core
        profiles which are being evolved.
      explicit_source_profiles: Explicit source profiles computed based on the
        core profiles at the start of the time step.

    Returns:
      ToraxSimState after the step.
    """

    core_profiles_t = input_state.core_profiles

    # Construct the CoreProfiles object for time t+dt with evolving boundary
    # conditions and time-dependent prescribed profiles not directly solved by
    # PDE system.
    core_profiles_t_plus_dt = provide_core_profiles_t_plus_dt(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
    )

    # Initial trial for stepper. If did not converge (can happen for nonlinear
    # step with large dt) we apply the adaptive time step routine if requested.
    core_profiles, core_sources, core_transport, stepper_numeric_outputs = (
        self._stepper_fn(
            dt=dt,
            static_runtime_params_slice=static_runtime_params_slice,
            dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
            dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
            geo_t=geo_t,
            geo_t_plus_dt=geo_t_plus_dt,
            core_profiles_t=core_profiles_t,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            explicit_source_profiles=explicit_source_profiles,
        )
    )
    stepper_numeric_outputs.outer_stepper_iterations = 1

    # post_processed_outputs set to zero since post-processing is done at the
    # end of the simulation step following recalculation of explicit
    # core_sources to be consistent with the final core_profiles.
    return state.ToraxSimState(
        t=input_state.t + dt,
        dt=dt,
        core_profiles=core_profiles,
        core_transport=core_transport,
        core_sources=core_sources,
        post_processed_outputs=state.PostProcessedOutputs.zeros(geo_t_plus_dt),
        time_step_calculator_state=time_step_calculator_state,
        stepper_numeric_outputs=stepper_numeric_outputs,
    )

  def adaptive_step(
      self,
      output_state: state.ToraxSimState,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_provider: runtime_params_slice.DynamicRuntimeParamsSliceProvider,
      geo_t: geometry.Geometry,
      geometry_provider: geometry_provider_lib.GeometryProvider,
      input_state: state.ToraxSimState,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
  ) -> tuple[
      runtime_params_slice.DynamicRuntimeParamsSlice,
      geometry.Geometry,
      state.ToraxSimState,
  ]:
    """Performs adaptive time stepping until stepper converges.

    If the initial step has converged (i.e.
    output_state.stepper_numeric_outputs.stepper_error_state == 0), this
    function is a no-op.

    Args:
      output_state: State after a full step.
      static_runtime_params_slice: Static parameters that, if they change,
        should trigger a recompilation of the SimulationStepFn.
      dynamic_runtime_params_slice_t: Runtime parameters at time t.
      dynamic_runtime_params_slice_provider: Runtime parameters slice provider.
      geo_t: The geometry of the torus during this time step of the simulation.
      geometry_provider: Provides geometry during the next time step of the
        simulation.
      input_state: State at the start of the time step, including the core
        profiles which are being evolved.
      explicit_source_profiles: Explicit source profiles computed based on the
        core profiles at the start of the time step.

    Returns:
      A tuple containing:
        - Runtime parameters at time t + dt, where dt is the actual time step
          used.
        - Geometry at time t + dt, where dt is the actual time step used.
        - ToraxSimState after adaptive time stepping.
    """
    core_profiles_t = input_state.core_profiles

    # Check if stepper converged. If not, proceed to body_fun
    def cond_fun(updated_output: state.ToraxSimState) -> bool:
      if updated_output.stepper_numeric_outputs.stepper_error_state == 1:
        do_dt_backtrack = True
      else:
        do_dt_backtrack = False
      return do_dt_backtrack

    # Make a new step with a smaller dt, starting with the original core
    # profiles.
    # Exit if dt < mindt
    def body_fun(
        updated_output: state.ToraxSimState,
    ) -> state.ToraxSimState:

      dt = (
          updated_output.dt
          / dynamic_runtime_params_slice_t.numerics.dt_reduction_factor
      )
      if jnp.any(jnp.isnan(dt)):
        raise ValueError('dt is NaN.')
      if dt < dynamic_runtime_params_slice_t.numerics.mindt:
        raise ValueError('dt below minimum timestep following adaptation')

      # Calculate dynamic_runtime_params and geo at t + dt.
      # Update geos with phibdot.
      # The updated geo_t is renamed to geo_t_with_phibdot due to name shadowing
      (
          dynamic_runtime_params_slice_t_plus_dt,
          geo_t_with_phibdot,
          geo_t_plus_dt,
      ) = _get_geo_and_dynamic_runtime_params_at_t_plus_dt_and_phibdot(
          input_state.t,
          dt,
          dynamic_runtime_params_slice_provider,
          geo_t,
          geometry_provider,
      )

      core_profiles_t_plus_dt = provide_core_profiles_t_plus_dt(
          core_profiles_t=core_profiles_t,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
          static_runtime_params_slice=static_runtime_params_slice,
          geo_t_plus_dt=geo_t_plus_dt,
      )
      core_profiles, core_sources, core_transport, stepper_numeric_outputs = (
          self._stepper_fn(
              dt=dt,
              static_runtime_params_slice=static_runtime_params_slice,
              dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
              dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
              geo_t=geo_t_with_phibdot,
              geo_t_plus_dt=geo_t_plus_dt,
              core_profiles_t=core_profiles_t,
              core_profiles_t_plus_dt=core_profiles_t_plus_dt,
              explicit_source_profiles=explicit_source_profiles,
          )
      )
      stepper_numeric_outputs.outer_stepper_iterations = (
          updated_output.stepper_numeric_outputs.outer_stepper_iterations + 1
      )

      stepper_numeric_outputs.inner_solver_iterations += (
          updated_output.stepper_numeric_outputs.inner_solver_iterations
      )
      return dataclasses.replace(
          updated_output,
          t=input_state.t + dt,
          dt=dt,
          core_profiles=core_profiles,
          core_transport=core_transport,
          core_sources=core_sources,
          stepper_numeric_outputs=stepper_numeric_outputs,
      )

    output_state = jax_utils.py_while(cond_fun, body_fun, output_state)

    # Calculate dynamic_runtime_params and geo at t + dt.
    # Update geos with phibdot.
    (
        dynamic_runtime_params_slice_t_plus_dt,
        geo_t,
        geo_t_plus_dt,
    ) = _get_geo_and_dynamic_runtime_params_at_t_plus_dt_and_phibdot(
        input_state.t,
        output_state.dt,
        dynamic_runtime_params_slice_provider,
        geo_t,
        geometry_provider,
    )

    return (
        dynamic_runtime_params_slice_t_plus_dt,
        geo_t_plus_dt,
        output_state,
    )

  def finalize_output(
      self,
      input_state: state.ToraxSimState,
      output_state: state.ToraxSimState,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      geo_t_plus_dt: geometry.Geometry,
  ) -> state.ToraxSimState:
    """Finalizes given output state at the end of the simulation step.

    Args:
      input_state: Previous sim state.
      output_state: State to be finalized.
      dynamic_runtime_params_slice_t_plus_dt: Runtime parameters at time t + dt.
      static_runtime_params_slice: Static runtime parameters.
      geo_t_plus_dt: The geometry of the torus during the next time step of the
        simulation.

    Returns:
      Finalized ToraxSimState.
    """

    # Update total current, q, and s profiles based on new psi
    q_corr = dynamic_runtime_params_slice_t_plus_dt.numerics.q_correction_factor
    output_state.core_profiles = physics.update_jtot_q_face_s_face(
        geo=geo_t_plus_dt,
        core_profiles=output_state.core_profiles,
        q_correction_factor=q_corr,
    )

    # Update ohmic and bootstrap current based on the new core profiles.
    output_state.core_profiles = update_current_distribution(
        source_models=self._stepper_fn.source_models,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_dt,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo_t_plus_dt,
        core_profiles=output_state.core_profiles,
    )

    # Update psidot based on the new core profiles.
    # Will include the phibdot calculation since geo=geo_t_plus_dt.
    output_state.core_profiles = _update_psidot(
        source_models=self._stepper_fn.source_models,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_dt,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo_t_plus_dt,
        core_profiles=output_state.core_profiles,
    )
    output_state = post_processing.make_outputs(
        sim_state=output_state,
        geo=geo_t_plus_dt,
        previous_sim_state=input_state,
    )

    return output_state


def get_initial_state(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    step_fn: SimulationStepFn,
) -> state.ToraxSimState:
  """Returns the initial state to be used by run_simulation()."""
  initial_core_profiles = core_profile_setters.initial_core_profiles(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      step_fn.stepper.source_models,
  )
  # Populate the starting state with source profiles from the implicit sources
  # before starting the run-loop. The explicit source profiles will be computed
  # inside the loop and will be merged with these implicit source profiles.
  initial_core_sources = get_initial_source_profiles(
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
      dynamic_runtime_params_slice_provider: runtime_params_slice.DynamicRuntimeParamsSliceProvider,
      geometry_provider: geometry_provider_lib.GeometryProvider,
      initial_state: state.ToraxSimState,
      step_fn: SimulationStepFn,
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
  ) -> runtime_params_slice.DynamicRuntimeParamsSliceProvider:
    return self._dynamic_runtime_params_slice_provider

  @property
  def static_runtime_params_slice(
      self,
  ) -> runtime_params_slice.StaticRuntimeParamsSlice:
    return self._static_runtime_params_slice

  @property
  def step_fn(self) -> SimulationStepFn:
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
          runtime_params_slice.DynamicRuntimeParamsSliceProvider | None
      ) = None,
      geometry_provider: geometry_provider_lib.GeometryProvider | None = None,
  ):
    """Updates the Sim object with components that have already been updated.

    Currently this only supports updating the geometry provider and the dynamic
    runtime params slice provider, both of which can be updated without
    recompilation.

    Args:
      allow_recompilation: Whether recompilation is allowed. If True, the
        static runtime params slice can be updated. NOTE: recompilaton may
        still occur if the mesh is updated or if the shapes returned in the
        dynamic runtime params slice provider change even if this is False.
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
      self._static_runtime_params_slice.validate_new(
          static_runtime_params_slice
      )
      self._static_runtime_params_slice = static_runtime_params_slice
    if dynamic_runtime_params_slice_provider is not None:
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
        get_consistent_dynamic_runtime_params_slice_and_geometry(
            self._dynamic_runtime_params_slice_provider.runtime_params_provider.numerics.runtime_params_config.t_initial,
            self._dynamic_runtime_params_slice_provider,
            self._geometry_provider,
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


def build_sim_object(
    runtime_params: general_runtime_params.GeneralRuntimeParams,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    stepper_builder: stepper_lib.StepperBuilder,
    transport_model_builder: transport_model_lib.TransportModelBuilder,
    source_models_builder: source_models_lib.SourceModelsBuilder,
    pedestal_model_builder: pedestal_model_lib.PedestalModelBuilder,
    time_step_calculator: Optional[ts.TimeStepCalculator] = None,
    file_restart: Optional[general_runtime_params.FileRestart] = None,
) -> Sim:
  """Builds a Sim object from the input runtime params and sim components.

  The Sim object provides a container for all the components that go into a
  single TORAX simulation run. It gives a way to reuse components without having
  to rebuild or recompile them if JAX shapes or static arguments do not change.

  Read more about the Sim object in its class docstring. The use of it is
  optional, and users may call `sim.run_simulation()` directly as well.

  Args:
    runtime_params: The input runtime params used throughout the simulation run.
    geometry_provider: The geometry used throughout the simulation run.
    stepper_builder: A callable to build the stepper. The stepper has already
      been factored out of the config.
    transport_model_builder: A callable to build the transport model.
    source_models_builder: Builds the SourceModels and holds its runtime_params.
    pedestal_model_builder: A callable to build the pedestal model.
    time_step_calculator: The time_step_calculator, if built, otherwise a
      ChiTimeStepCalculator will be built by default.
    file_restart: If provided we will reconstruct the initial state from the
      provided file at the given time step. This state from the file will only
      be used for constructing the initial state (as well as the config) and for
      all subsequent steps, the evolved state and runtime parameters from config
      are used.

  Returns:
    sim: The built Sim instance.
  """

  transport_model = transport_model_builder()
  pedestal_model = pedestal_model_builder()

  # TODO(b/385788907): Clearly document all changes that lead to recompilations.
  static_runtime_params_slice = (
      runtime_params_slice.build_static_runtime_params_slice(
          runtime_params=runtime_params,
          source_runtime_params=source_models_builder.runtime_params,
          torax_mesh=geometry_provider.torax_mesh,
          stepper=stepper_builder.runtime_params,
      )
  )
  dynamic_runtime_params_slice_provider = (
      runtime_params_slice.DynamicRuntimeParamsSliceProvider(
          runtime_params=runtime_params,
          transport=transport_model_builder.runtime_params,
          sources=source_models_builder.runtime_params,
          stepper=stepper_builder.runtime_params,
          torax_mesh=geometry_provider.torax_mesh,
          pedestal=pedestal_model_builder.runtime_params,
      )
  )
  source_models = source_models_builder()
  stepper = stepper_builder(transport_model, source_models, pedestal_model)

  if time_step_calculator is None:
    time_step_calculator = chi_time_step_calculator.ChiTimeStepCalculator()

  # Build dynamic_runtime_params_slice at t_initial for initial conditions.
  dynamic_runtime_params_slice_for_init, geo_for_init = (
      get_consistent_dynamic_runtime_params_slice_and_geometry(
          runtime_params.numerics.t_initial,
          dynamic_runtime_params_slice_provider,
          geometry_provider,
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

  step_fn = SimulationStepFn(
      stepper=stepper,
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

  return Sim(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
      geometry_provider=geometry_provider,
      initial_state=initial_state,
      step_fn=step_fn,
      file_restart=file_restart,
  )


def _run_simulation(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_provider: runtime_params_slice.DynamicRuntimeParamsSliceProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    initial_state: state.ToraxSimState,
    step_fn: SimulationStepFn,
    log_timestep_info: bool = False,
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
      get_consistent_dynamic_runtime_params_slice_and_geometry(
          initial_state.t,
          dynamic_runtime_params_slice_provider,
          geometry_provider,
      )
  )

  sim_state = initial_state
  sim_history = []
  sim_state = post_processing.make_outputs(sim_state=sim_state, geo=geo)
  sim_history.append(sim_state)

  # Set the sim_error to NO_ERROR. If we encounter an error, we will set it to
  # the appropriate error code.
  sim_error = state.SimError.NO_ERROR

  # Advance the simulation until the time_step_calculator tells us we are done.
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

  # Log final timestep
  if log_timestep_info and sim_error == state.SimError.NO_ERROR:
    # The "sim_state" here has been updated by the loop above.
    _log_timestep(sim_state)

  # Update the final time step's source profiles based on the explicit source
  # profiles computed based on the final state.
  logging.info("Updating last step's source profiles.")
  dynamic_runtime_params_slice, geo = (
      get_consistent_dynamic_runtime_params_slice_and_geometry(
          sim_state.t,
          dynamic_runtime_params_slice_provider,
          geometry_provider,
      )
  )
  explicit_source_profiles = source_models_lib.build_source_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=sim_state.core_profiles,
      source_models=step_fn.stepper.source_models,
      explicit=True,
  )
  sim_state.core_sources = merge_source_profiles(
      explicit_source_profiles=explicit_source_profiles,
      implicit_source_profiles=sim_state.core_sources,
  )

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


def _get_geo_and_dynamic_runtime_params_at_t_plus_dt_and_phibdot(
    t: jnp.ndarray,
    dt: jnp.ndarray,
    dynamic_runtime_params_slice_provider: runtime_params_slice.DynamicRuntimeParamsSliceProvider,
    geo_t: geometry.Geometry,
    geometry_provider: geometry_provider_lib.GeometryProvider,
) -> tuple[
    runtime_params_slice.DynamicRuntimeParamsSlice,
    geometry.Geometry,
    geometry.Geometry,
]:
  """Returns the geos including Phibdot, and dynamic runtime params at t + dt.

  Args:
    t: Time at which the simulation is currently at.
    dt: Time step duration.
    dynamic_runtime_params_slice_provider: Object that returns a set of runtime
      parameters which may change from time step to time step or simulation run
      to run. If these runtime parameters change, it does NOT trigger a JAX
      recompilation.
    geo_t: The geometry of the torus during this time step of the simulation.
    geometry_provider: Provides the magnetic geometry for each time step based
      on the ToraxSimState at the start of the time step.

  Returns:
    Tuple containing:
      - The dynamic runtime params at time t + dt.
      - The geometry of the torus during this time step of the simulation.
      - The geometry of the torus during the next time step of the simulation.
  """
  dynamic_runtime_params_slice_t_plus_dt, geo_t_plus_dt = (
      get_consistent_dynamic_runtime_params_slice_and_geometry(
          t + dt,
          dynamic_runtime_params_slice_provider,
          geometry_provider,
      )
  )
  geo_t, geo_t_plus_dt = _add_Phibdot(
      dt, dynamic_runtime_params_slice_t_plus_dt, geo_t, geo_t_plus_dt
  )

  return (
      dynamic_runtime_params_slice_t_plus_dt,
      geo_t,
      geo_t_plus_dt,
  )


# pylint: disable=invalid-name
def _add_Phibdot(
    dt: jnp.ndarray,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t: geometry.Geometry,
    geo_t_plus_dt: geometry.Geometry,
) -> tuple[geometry.Geometry, geometry.Geometry]:
  """Update Phibdot in the geometry dataclasses used in the time interval.

  Phibdot is used in calc_coeffs to calcuate terms related to time-dependent
  geometry. It should be set to be the same for geo_t and geo_t_plus_dt for
  each given time interval. This means that geo_t_plus_dt.Phibdot will not
  necessarily be the same as the geo_t.Phibdot at the next time step.

  Args:
    dt: Time step duration.
    dynamic_runtime_params_slice: Runtime parameters which may change from time
      step to time step without triggering recompilations.
    geo_t: The geometry of the torus during this time step of the simulation.
    geo_t_plus_dt: The geometry of the torus during the next time step of the
      simulation.

  Returns:
    Tuple containing:
      - The geometry of the torus during this time step of the simulation.
      - The geometry of the torus during the next time step of the simulation.
  """

  # Calculate Phibdot for the time interval.
  # If numerics.calcphibdot is False, set Phibdot to be 0 (useful for testing
  # purposes)
  Phibdot = jnp.where(
      dynamic_runtime_params_slice.numerics.calcphibdot,
      (geo_t_plus_dt.Phib - geo_t.Phib) / dt,
      0.0,
  )

  geo_t = dataclasses.replace(
      geo_t,
      Phibdot=Phibdot,
  )
  geo_t_plus_dt = dataclasses.replace(
      geo_t_plus_dt,
      Phibdot=Phibdot,
  )
  return geo_t, geo_t_plus_dt


# pylint: enable=invalid-name


def update_current_distribution(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> state.CoreProfiles:
  """Update bootstrap current based on the new core_profiles."""

  bootstrap_profile = source_models.j_bootstrap.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
  )

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid
  external_current = source_models.external_current_source(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
  )

  johm = (
      core_profiles.currents.jtot
      - bootstrap_profile.j_bootstrap
      - external_current
  )

  currents = dataclasses.replace(
      core_profiles.currents,
      j_bootstrap=bootstrap_profile.j_bootstrap,
      j_bootstrap_face=bootstrap_profile.j_bootstrap_face,
      I_bootstrap=bootstrap_profile.I_bootstrap,
      sigma=bootstrap_profile.sigma,
      johm=johm,
      external_current_source=external_current,
  )
  new_core_profiles = dataclasses.replace(
      core_profiles,
      currents=currents,
  )
  return new_core_profiles


def _update_psidot(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> state.CoreProfiles:
  """Update psidot based on new core_profiles."""

  psidot = dataclasses.replace(
      core_profiles.psidot,
      value=ohmic_heat_source.calc_psidot(
          static_runtime_params_slice,
          dynamic_runtime_params_slice,
          geo,
          core_profiles,
          source_models,
      ),
  )

  new_core_profiles = dataclasses.replace(
      core_profiles,
      psidot=psidot,
  )
  return new_core_profiles


def provide_core_profiles_t_plus_dt(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t_plus_dt: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> state.CoreProfiles:
  """Provides state at t_plus_dt with new boundary conditions and prescribed profiles."""
  updated_boundary_conditions = (
      core_profile_setters.compute_boundary_conditions(
          dynamic_runtime_params_slice_t_plus_dt,
          geo_t_plus_dt,
      )
  )
  updated_values = core_profile_setters.updated_prescribed_core_profiles(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_dt,
      geo=geo_t_plus_dt,
      core_profiles=core_profiles_t,
  )
  temp_ion = dataclasses.replace(
      core_profiles_t.temp_ion,
      value=updated_values['temp_ion'],
      **updated_boundary_conditions['temp_ion'],
  )
  temp_el = dataclasses.replace(
      core_profiles_t.temp_el,
      value=updated_values['temp_el'],
      **updated_boundary_conditions['temp_el'],
  )
  psi = dataclasses.replace(
      core_profiles_t.psi, **updated_boundary_conditions['psi']
  )
  ne = dataclasses.replace(
      core_profiles_t.ne,
      value=updated_values['ne'],
      **updated_boundary_conditions['ne'],
  )
  ni = dataclasses.replace(
      core_profiles_t.ni,
      value=updated_values['ni'],
      **updated_boundary_conditions['ni'],
  )
  nimp = dataclasses.replace(
      core_profiles_t.nimp,
      value=updated_values['nimp'],
      **updated_boundary_conditions['nimp'],
  )
  core_profiles_t_plus_dt = dataclasses.replace(
      core_profiles_t,
      temp_ion=temp_ion,
      temp_el=temp_el,
      psi=psi,
      ne=ne,
      ni=ni,
      nimp=nimp,
      Zimp=dynamic_runtime_params_slice_t_plus_dt.plasma_composition.Zimp,
  )
  return core_profiles_t_plus_dt


def get_initial_source_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> source_profiles_lib.SourceProfiles:
  """Returns the source profiles for the initial state in run_simulation().

  Args:
    static_runtime_params_slice: Runtime parameters which, when they change,
      trigger recompilations. They should not change within a single run of the
      sim.
    dynamic_runtime_params_slice: Runtime parameters which may change from time
      step to time step without triggering recompilations.
    geo: The geometry of the torus during this time step of the simulation.
    core_profiles: Core profiles that may evolve throughout the course of a
      simulation. These values here are, of course, only the original states.
    source_models: Source models used to compute core source profiles.

  Returns:
    Implicit and explicit SourceProfiles from source models based on the core
    profiles from the starting state.
  """
  implicit_profiles = source_models_lib.build_source_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      explicit=False,
  )
  qei = source_models.qei_source.get_qei(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
  )
  implicit_profiles = dataclasses.replace(implicit_profiles, qei=qei)
  # Also add in the explicit sources to the initial sources.
  explicit_source_profiles = source_models_lib.build_source_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      explicit=True,
  )
  initial_profiles = merge_source_profiles(
      explicit_source_profiles=explicit_source_profiles,
      implicit_source_profiles=implicit_profiles,
  )
  return initial_profiles


# This function can be jitted if source_models is a static argument. However,
# in our tests, jitting this function actually slightly slows down runs, so this
# is left as pure python.
def merge_source_profiles(
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    implicit_source_profiles: source_profiles_lib.SourceProfiles,
) -> source_profiles_lib.SourceProfiles:
  """Returns a SourceProfiles that merges the input profiles.

  Sources can either be explicit or implicit. The explicit_source_profiles
  contain the profiles for all source models that are set to explicit, and it
  contains profiles with all zeros for any implicit source. The opposite holds
  for the implicit_source_profiles.

  This function adds the two dictionaries of profiles and returns a single
  SourceProfiles that includes both.

  Args:
    explicit_source_profiles: Profiles from explicit source models. This
      SourceProfiles dict will include keys for both the explicit and implicit
      sources, but only the explicit sources will have non-zero profiles. See
      source.py and runtime_params.py for more info on explicit vs. implicit.
    implicit_source_profiles: Profiles from implicit source models. This
      SourceProfiles dict will include keys for both the explicit and implicit
      sources, but only the implicit sources will have non-zero profiles. See
      source.py and runtime_params.py for more info on explicit vs. implicit.

  Returns:
    A SourceProfiles with non-zero profiles for all sources, both explicit and
    implicit (assuming the source model outputted a non-zero profile).
  """
  sum_profiles = lambda a, b: a + b
  summed_bootstrap_profile = jax.tree_util.tree_map(
      sum_profiles,
      explicit_source_profiles.j_bootstrap,
      implicit_source_profiles.j_bootstrap,
  )
  summed_qei_info = jax.tree_util.tree_map(
      sum_profiles, explicit_source_profiles.qei, implicit_source_profiles.qei
  )
  summed_other_profiles = jax.tree_util.tree_map(
      sum_profiles,
      explicit_source_profiles.profiles,
      implicit_source_profiles.profiles,
  )
  return source_profiles_lib.SourceProfiles(
      profiles=summed_other_profiles,
      j_bootstrap=summed_bootstrap_profile,
      qei=summed_qei_info,
  )


def _log_timestep(
    sim_state: state.ToraxSimState,
) -> None:
  """Logs basic timestep info."""
  logging.info(
      '\nSimulation time: %.5f, previous dt: %.6f, previous stepper'
      ' iterations: %d',
      sim_state.t,
      sim_state.dt,
      sim_state.stepper_numeric_outputs.outer_stepper_iterations,
  )
  # TODO(b/330172917): once tol and coarse_tol are configurable in the
  # runtime_params, also log the value of tol and coarse_tol below
  match sim_state.stepper_numeric_outputs.stepper_error_state:
    case 0:
      pass
    case 1:
      logging.info('Solver did not converge in previous step.')
    case 2:
      logging.info(
          'Solver converged only within coarse tolerance in previous step.'
      )
