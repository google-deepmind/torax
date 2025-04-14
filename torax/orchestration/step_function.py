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

"""Logic which controls the stepping over time of the simulation."""

import dataclasses
import functools
import jax
import jax.numpy as jnp
from torax import jax_utils
from torax import post_processing
from torax import state
from torax.config import build_runtime_params
from torax.config import runtime_params_slice
from torax.core_profiles import updaters
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.mhd import base as mhd_base
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.sources import source_profile_builders
from torax.sources import source_profiles as source_profiles_lib
from torax.stepper import stepper as stepper_lib
from torax.time_step_calculator import time_step_calculator as ts
from torax.transport_model import transport_model as transport_model_lib


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
      mhd_models: mhd_base.MHDModels,
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
      mhd_models: Collection of MHD models applied, e.g. sawtooth
    """
    self._stepper_fn = stepper
    self._time_step_calculator = time_step_calculator
    self._transport_model = transport_model
    self._pedestal_model = pedestal_model
    self._mhd_models = mhd_models

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
  def mhd_models(self) -> mhd_base.MHDModels:
    return self._mhd_models

  @property
  def time_step_calculator(self) -> ts.TimeStepCalculator:
    return self._time_step_calculator

  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_provider: build_runtime_params.DynamicRuntimeParamsSliceProvider,
      geometry_provider: geometry_provider_lib.GeometryProvider,
      input_state: state.ToraxSimState,
      previous_post_processed_outputs: state.PostProcessedOutputs,
  ) -> tuple[state.ToraxSimState, state.PostProcessedOutputs, state.SimError]:
    """Advances the simulation state one time step.

      If a sawtooth model is provided, it will be checked to see if a sawtooth
    should trigger. If it does, the sawtooth model will be applied and instead
    of a full PDE solve, the step_fn will return early with a state following
    sawtooth redistribution, at a t+dt set by the sawtooth model.

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
      previous_post_processed_outputs: Post-processed outputs from the previous
        time step.

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
      PostProcessedOutputs containing:
        - post-processed outputs at the end of the time step.
        - cumulative quantities.
      SimError indicating if an error has occurred during simulation.
    """
    dynamic_runtime_params_slice_t, geo_t = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=input_state.t,
            dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
            geometry_provider=geometry_provider,
        )
    )

    # Check for sawtooth model and see if sawtooth should trigger.
    # Triggered sawtooth will trigger a redistribution step with early return.
    # Consecutive sawtooth crashes are not allowed since standard PDE steps
    # may then not take place.
    # If sawtooth_crash is True, then a normal PDE step will be carried
    # out. where a ToraxSimState will be returned with the default
    # sawtooth_crash value of False.
    if (
        (self.mhd_models.sawtooth is not None)
        and (not input_state.sawtooth_crash)
    ):
      sawtooth_triggered, output_state, post_processed_outputs = (
          self.mhd_models.sawtooth(
              static_runtime_params_slice,
              dynamic_runtime_params_slice_t,
              input_state,
              previous_post_processed_outputs,
          )
      )
      output_state = dataclasses.replace(
          output_state,
          sawtooth_crash=sawtooth_triggered,
      )
      if sawtooth_triggered:
        return (
            output_state,
            post_processed_outputs,
            state.check_for_errors(output_state, post_processed_outputs),
        )

    # This only computes sources set to explicit in the
    # DynamicSourceConfigSlice. All implicit sources will have their profiles
    # set to 0.
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_t,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo_t,
        core_profiles=input_state.core_profiles,
        source_models=self.stepper.source_models,
        explicit=True,
    )

    dt = self.init_time_step_calculator(
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
      dynamic_runtime_params_slice_t_plus_dt, output_state = self.adaptive_step(
          output_state,
          static_runtime_params_slice,
          dynamic_runtime_params_slice_t,
          dynamic_runtime_params_slice_provider,
          geo_t,
          geometry_provider,
          input_state,
          explicit_source_profiles,
      )

    post_processed_outputs = post_processing.make_post_processed_outputs(
        sim_state=output_state,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_dt,
        previous_post_processed_outputs=previous_post_processed_outputs,
    )

    return (
        output_state,
        post_processed_outputs,
        state.check_for_errors(output_state, post_processed_outputs),
    )

  def init_time_step_calculator(
      self,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t: geometry.Geometry,
      input_state: state.ToraxSimState,
  ) -> jnp.ndarray:
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
      Time step duration (dt)
    """
    # TODO(b/335598388): We call the transport model both here and in the the
    # Stepper / CoeffsCallback. We should still refactor the design to more
    # explicitly calculate transport coeffs at delta_t = 0 in only one place,
    # so that we have some flexibility in where to place the jit boundaries.
    transport_coeffs = _calculate_transport_coeffs(
        self.pedestal_model,
        self.transport_model,
        dynamic_runtime_params_slice_t,
        geo_t,
        input_state.core_profiles,
    )

    # initialize new dt and reset stepper iterations.
    dt = self._time_step_calculator.next_dt(
        dynamic_runtime_params_slice_t,
        geo_t,
        input_state.core_profiles,
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

    return dt

  def step(
      self,
      dt: jnp.ndarray,
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
    core_profiles_t_plus_dt = _provide_core_profiles_t_plus_dt(
        dt=dt,
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
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
        stepper_numeric_outputs=stepper_numeric_outputs,
        geometry=geo_t_plus_dt,
    )

  def adaptive_step(
      self,
      output_state: state.ToraxSimState,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_provider: build_runtime_params.DynamicRuntimeParamsSliceProvider,
      geo_t: geometry.Geometry,
      geometry_provider: geometry_provider_lib.GeometryProvider,
      input_state: state.ToraxSimState,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
  ) -> tuple[
      runtime_params_slice.DynamicRuntimeParamsSlice,
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
        - Dynamic runtime params at time t + dt, where dt is the actual time
          step used.
        - ToraxSimState after adaptive time stepping.
    """
    core_profiles_t = input_state.core_profiles

    # Check if stepper converged. If not, proceed to body_fun
    def cond_fun(
        updated_output: tuple[
            state.ToraxSimState, runtime_params_slice.DynamicRuntimeParamsSlice
        ],
    ) -> bool:
      if updated_output[0].stepper_numeric_outputs.stepper_error_state == 1:
        do_dt_backtrack = True
      else:
        do_dt_backtrack = False
      return do_dt_backtrack

    # Make a new step with a smaller dt, starting with the original core
    # profiles.
    # Exit if dt < mindt
    def body_fun(
        updated_output: tuple[
            state.ToraxSimState, runtime_params_slice.DynamicRuntimeParamsSlice
        ],
    ) -> tuple[
        state.ToraxSimState, runtime_params_slice.DynamicRuntimeParamsSlice
    ]:
      old_state, old_slice = updated_output
      numerics = old_slice.numerics

      dt = old_state.dt / numerics.dt_reduction_factor
      if jnp.any(jnp.isnan(dt)):
        raise ValueError('dt is NaN.')
      if dt < numerics.mindt:
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

      core_profiles_t_plus_dt = _provide_core_profiles_t_plus_dt(
          dt=dt,
          static_runtime_params_slice=static_runtime_params_slice,
          dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
          geo_t_plus_dt=geo_t_plus_dt,
          core_profiles_t=core_profiles_t,
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
          old_state.stepper_numeric_outputs.outer_stepper_iterations + 1
      )

      stepper_numeric_outputs.inner_solver_iterations += (
          old_state.stepper_numeric_outputs.inner_solver_iterations
      )
      return (
          state.ToraxSimState(
              t=input_state.t + dt,
              dt=dt,
              core_profiles=core_profiles,
              core_transport=core_transport,
              core_sources=core_sources,
              stepper_numeric_outputs=stepper_numeric_outputs,
              geometry=geo_t_plus_dt,
          ),
          dynamic_runtime_params_slice_t_plus_dt,
      )

    # Note that the initial state provided here uses the dynamic slice at time t
    # whereas the return value of this function uses the dynamic slice at time
    # t + dt. The input should technically be the dynamic state at time t + dt,
    # but we don't have that value here and we also don't use any of the dynamic
    # state in the cond_fun or body_fun.
    output_state, dynamic_runtime_params_slice_t_plus_dt = jax_utils.py_while(
        cond_fun, body_fun, (output_state, dynamic_runtime_params_slice_t)
    )

    return dynamic_runtime_params_slice_t_plus_dt, output_state


@functools.partial(jax_utils.jit, static_argnums=(0, 1))
def _calculate_transport_coeffs(
    pedestal_model: pedestal_model_lib.PedestalModel,
    transport_model: transport_model_lib.TransportModel,
    dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> state.CoreTransport:
  """Calculates the transport coefficients."""
  pedestal_model_output = pedestal_model(
      dynamic_runtime_params_slice_t, geo_t, core_profiles_t
  )
  return transport_model(
      dynamic_runtime_params_slice_t,
      geo_t,
      core_profiles_t,
      pedestal_model_output,
  )


def _get_geo_and_dynamic_runtime_params_at_t_plus_dt_and_phibdot(
    t: jnp.ndarray,
    dt: jnp.ndarray,
    dynamic_runtime_params_slice_provider: build_runtime_params.DynamicRuntimeParamsSliceProvider,
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
      build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
          t=t + dt,
          dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
          geometry_provider=geometry_provider,
      )
  )
  if dynamic_runtime_params_slice_t_plus_dt.numerics.calcphibdot:
    geo_t, geo_t_plus_dt = geometry.update_geometries_with_Phibdot(
        dt=dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
    )

  return (
      dynamic_runtime_params_slice_t_plus_dt,
      geo_t,
      geo_t_plus_dt,
  )


def _provide_core_profiles_t_plus_dt(
    dt: jax.Array,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t_plus_dt: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> state.CoreProfiles:
  """Provides state at t_plus_dt with new boundary conditions and prescribed profiles."""
  updated_boundary_conditions = updaters.compute_boundary_conditions_for_t_plus_dt(
      dt=dt,
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
      dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      core_profiles_t=core_profiles_t,
  )
  updated_values = updaters.get_prescribed_core_profile_values(
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

  # pylint: disable=invalid-name
  # Update Z_face with boundary condition Z, needed for cases where temp_el
  # is evolving and updated_prescribed_core_profiles is a no-op.
  Zi_face = jnp.concatenate(
      [
          updated_values['Zi_face'][:-1],
          jnp.array([updated_boundary_conditions['Zi_edge']]),
      ],
  )
  Zimp_face = jnp.concatenate(
      [
          updated_values['Zimp_face'][:-1],
          jnp.array([updated_boundary_conditions['Zimp_edge']]),
      ],
  )
  # pylint: enable=invalid-name
  core_profiles_t_plus_dt = dataclasses.replace(
      core_profiles_t,
      temp_ion=temp_ion,
      temp_el=temp_el,
      psi=psi,
      ne=ne,
      ni=ni,
      nimp=nimp,
      Zi=updated_values['Zi'],
      Zi_face=Zi_face,
      Zimp=updated_values['Zimp'],
      Zimp_face=Zimp_face,
  )
  return core_profiles_t_plus_dt
