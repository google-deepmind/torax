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

import functools

import chex
import jax
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import updaters
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.mhd.sawtooth import sawtooth_solver as sawtooth_solver_lib
from torax._src.orchestration import adaptive_step
from torax._src.orchestration import sawtooth_step
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function_processing
from torax._src.orchestration import whilei_loop
from torax._src.output_tools import post_processing
from torax._src.solver import solver as solver_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.time_step_calculator import time_step_calculator as ts


# pylint: disable=invalid-name


def check_for_errors(
    numerics: numerics_lib.Numerics,
    output_state: sim_state.ToraxSimState,
    post_processed_outputs: post_processing.PostProcessedOutputs,
) -> state.SimError:
  """Checks for errors in the simulation state."""
  if numerics.adaptive_dt:
    if output_state.solver_numeric_outputs.solver_error_state == 1:
      # Only check for min dt if the solver did not converge. Else we may have
      # converged at a dt > min_dt just before we reach min_dt.
      if output_state.dt / numerics.dt_reduction_factor < numerics.min_dt:
        return state.SimError.REACHED_MIN_DT
  state_error = output_state.check_for_errors()
  if state_error != state.SimError.NO_ERROR:
    return state_error
  else:
    return post_processed_outputs.check_for_errors()


@jax.tree_util.register_pytree_node_class
class SimulationStepFn:
  """Advances the TORAX simulation one time step.

  Unlike the Solver class, which updates certain parts of the state, a
  SimulationStepFn takes in the ToraxSimState and outputs the updated
  ToraxSimState, which contains not only the CoreProfiles but also extra
  simulation state useful for stepping as well as extra outputs useful for
  inspection inside the main run loop in `run_simulation()`. It wraps calls to
  Solver with useful features to increase robustness for convergence, like
  dt-backtracking.
  """

  def __init__(
      self,
      solver: solver_lib.Solver,
      time_step_calculator: ts.TimeStepCalculator,
      runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
      geometry_provider: geometry_provider_lib.GeometryProvider,
  ):
    """Initializes the SimulationStepFn.

    Args:
      solver: Evolves the core profiles.
      time_step_calculator: Calculates the dt for each time step.
      runtime_params_provider: Object that returns a set of runtime parameters
        which may change from time step to time step or simulation run to run.
      geometry_provider: Provides the magnetic geometry for each time step based
        on the ToraxSimState at the start of the time step. The geometry may
        change from time step to time step, so the sim needs a function to
        provide which geometry to use for a given time step. A GeometryProvider
        is any callable (class or function) which takes the ToraxSimState at the
        start of a time step and returns the Geometry for that time step. For
        most use cases, only the time will be relevant from the ToraxSimState
        (in order to support time-dependent geometries).
    """
    self._solver = solver
    if self._solver.physics_models.mhd_models.sawtooth_models is not None:
      self._sawtooth_solver = sawtooth_solver_lib.SawtoothSolver(
          physics_models=self._solver.physics_models,
      )
    else:
      self._sawtooth_solver = None
    self._time_step_calculator = time_step_calculator
    self._geometry_provider = geometry_provider
    self._runtime_params_provider = runtime_params_provider

  @property
  def runtime_params_provider(
      self,
  ) -> build_runtime_params.RuntimeParamsProvider:
    return self._runtime_params_provider

  def tree_flatten(self):
    children = (
        self._runtime_params_provider,
        self._geometry_provider,
    )
    aux_data = (
        self._solver,
        self._time_step_calculator,
    )
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(
        solver=aux_data[0],
        time_step_calculator=aux_data[1],
        runtime_params_provider=children[0],
        geometry_provider=children[1],
    )

  @property
  def geometry_provider(self) -> geometry_provider_lib.GeometryProvider:
    return self._geometry_provider

  @property
  def solver(self) -> solver_lib.Solver:
    return self._solver

  @property
  def time_step_calculator(self) -> ts.TimeStepCalculator:
    return self._time_step_calculator

  @jax.jit
  def __call__(
      self,
      input_state: sim_state.ToraxSimState,
      previous_post_processed_outputs: post_processing.PostProcessedOutputs,
      max_dt: chex.Numeric = jnp.inf,
  ) -> tuple[
      sim_state.ToraxSimState,
      post_processing.PostProcessedOutputs,
  ]:
    """Advances the simulation state one time step.

      If a sawtooth model is provided, it will be checked to see if a sawtooth
    should trigger. If it does, the sawtooth model will be applied and instead
    of a full PDE solve, the step_fn will return early with a state following
    sawtooth redistribution, at a t+dt set by the sawtooth model.

    Args:
      input_state: State at the start of the time step, including the core
        profiles which are being evolved.
      previous_post_processed_outputs: Post-processed outputs from the previous
        time step.
      max_dt: The maximum time step duration to allow the simulation to take. If
        set this will override the properties set in the numerics config. If
        this is infinite, the time step duration will be chosen based on the
        values set in the numerics config.

    Returns:
      ToraxSimState containing:
        - the core profiles at the end of the time step.
        - time and time step calculator state info.
        - core_sources and core_transport at the end of the time step.
        - solver_numeric_outputs. This contains the number of iterations
          performed in the solver and the error state. The error states are:
            0 if solver converged with fine tolerance for this step
            1 if solver did not converge for this step (was above coarse tol)
            2 if solver converged within coarse tolerance. Allowed to pass with
              a warning. Occasional error=2 has low impact on final sim state.
      PostProcessedOutputs containing:
        - post-processed outputs at the end of the time step.
        - cumulative quantities.
      SimError indicating if an error has occurred during simulation.
    """
    runtime_params_t, geo_t = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=input_state.t,
            runtime_params_provider=self._runtime_params_provider,
            geometry_provider=self._geometry_provider,
        )
    )

    # This only computes sources set to explicit in the
    # SourceConfig. All implicit sources will have their profiles
    # set to 0.
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params_t,
        geo=geo_t,
        core_profiles=input_state.core_profiles,
        source_models=self._solver.physics_models.source_models,
        neoclassical_models=self._solver.physics_models.neoclassical_models,
        explicit=True,
    )

    def _step():
      """Take either the adaptive or fixed step, depending on the config."""
      step_args = (
          max_dt,
          runtime_params_t,
          geo_t,
          explicit_source_profiles,
          input_state,
          previous_post_processed_outputs,
      )
      # If adaptive dt is enabled, take the adaptive step if the max_dt is
      # greater than the min_dt, otherwise take the fixed step.
      if runtime_params_t.numerics.adaptive_dt:
        return jax.lax.cond(
            max_dt > runtime_params_t.numerics.min_dt,
            self._adaptive_step,
            self._fixed_step,
            *step_args,
        )
      else:
        return self._fixed_step(
            *step_args
        )

    if self._sawtooth_solver is not None:
      # If a sawtooth model is provided, there was no previous sawtooth crash
      # and the max_dt is greater than the crash step duration, it will be
      # checked to see if a sawtooth should trigger. If it does, the sawtooth
      # model will be applied and instead of a full PDE solve, the step_fn will
      # return early with a state following sawtooth redistribution, at a t+dt
      # set by the sawtooth model configuration.
      sawtooth_params = runtime_params_t.mhd.sawtooth
      # Sawtooth params should always be provided if a sawtooth model is
      # provided.
      assert sawtooth_params is not None, 'Sawtooth params are None.'
      output_state, post_processed_outputs = jax.lax.cond(
          jnp.logical_and(
              input_state.solver_numeric_outputs.sawtooth_crash,
              max_dt > sawtooth_params.crash_step_duration,
          ),
          lambda *args: (input_state, previous_post_processed_outputs),
          self._sawtooth_step,
          runtime_params_t,
          geo_t,
          explicit_source_profiles,
          input_state,
          previous_post_processed_outputs,
      )

      output_state, post_processed_outputs = jax.lax.cond(
          # If the current state is a sawtooth and the previous state was not,
          # then we triggered a sawtooth crash and exit early.
          output_state.solver_numeric_outputs.sawtooth_crash
          & ~input_state.solver_numeric_outputs.sawtooth_crash,
          lambda: (output_state, post_processed_outputs),
          _step,
      )
    else:
      # If no sawtooth model is provided, take a normal step.
      output_state, post_processed_outputs = _step()

    return output_state, post_processed_outputs

  def fixed_time_step(
      self,
      dt: jax.Array,
      input_state: sim_state.ToraxSimState,
      previous_post_processed_outputs: post_processing.PostProcessedOutputs,
  ) -> tuple[
      sim_state.ToraxSimState,
      post_processing.PostProcessedOutputs,
  ]:
    """Runs the simulation until it has advanced by dt."""
    remaining_dt = dt

    def cond(args):
      remaining_dt, _, _ = args
      return remaining_dt > 0.0

    def body(args):
      remaining_dt, prev_state, prev_post_processed = args
      output_state, post_processed_outputs = self(
          prev_state,
          prev_post_processed,
          max_dt=remaining_dt,
      )
      remaining_dt -= output_state.dt
      return remaining_dt, output_state, post_processed_outputs

    _, output_state, post_processed_outputs = jax.lax.while_loop(
        cond,
        body,
        (remaining_dt, input_state, previous_post_processed_outputs),
    )
    # TODO(b/456188184): Add a return value for the number of steps, sawtooth
    # crashes, and solver error states etc.
    return output_state, post_processed_outputs

  def _sawtooth_step(
      self,
      runtime_params_t: runtime_params_slice.RuntimeParams,
      geo_t: geometry.Geometry,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      input_state: sim_state.ToraxSimState,
      previous_post_processed_outputs: post_processing.PostProcessedOutputs,
  ) -> tuple[
      sim_state.ToraxSimState,
      post_processing.PostProcessedOutputs,
  ]:
    """Performs a simulation step if a sawtooth crash is triggered."""
    assert runtime_params_t.mhd.sawtooth is not None
    dt_crash = runtime_params_t.mhd.sawtooth.crash_step_duration

    runtime_params_t_plus_crash_dt, geo_t_plus_crash_dt = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=input_state.t + dt_crash,
            runtime_params_provider=self._runtime_params_provider,
            geometry_provider=self._geometry_provider,
        )
    )

    # If no sawtooth crash is triggered, output_state and
    # post_processed_outputs will be the same as the input state and
    # previous_post_processed_outputs.
    output_state, post_processed_outputs = sawtooth_step.sawtooth_step(
        sawtooth_solver=self._sawtooth_solver,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_crash_dt=runtime_params_t_plus_crash_dt,
        geo_t=geo_t,
        geo_t_plus_crash_dt=geo_t_plus_crash_dt,
        explicit_source_profiles=explicit_source_profiles,
        input_state=input_state,
        input_post_processed_outputs=previous_post_processed_outputs,
    )
    return output_state, post_processed_outputs

  def step(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_slice.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      input_state: sim_state.ToraxSimState,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Performs a simulation step with given dt.

    Solver may fail to converge in which case _adaptive_step() can be used to
    try smaller time step durations.

    Args:
      dt: Time step duration.
      runtime_params_t: Runtime parameters at time t.
      runtime_params_t_plus_dt: Runtime parameters at time t + dt.
      geo_t: The geometry of the torus during this time step of the simulation.
      geo_t_plus_dt: The geometry of the torus during the next time step of the
        simulation.
      input_state: State at the start of the time step, including the core
        profiles which are being evolved.
      explicit_source_profiles: Explicit source profiles computed based on the
        core profiles at the start of the time step.

    Returns:
      tuple:
        tuple of CellVariables corresponding to the evolved state variables
        SolverNumericOutputs containing error state and other solver-specific
        outputs.
    """

    core_profiles_t = input_state.core_profiles

    # Construct the CoreProfiles object for time t+dt with evolving boundary
    # conditions and time-dependent prescribed profiles not directly solved by
    # PDE system.
    core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
    )

    # Initial trial for solver. If did not converge (can happen for nonlinear
    # step with large dt) we apply the adaptive time step routine if requested.
    return self._solver(
        t=input_state.t,
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles,
    )

  def _adaptive_step(
      self,
      max_dt: chex.Numeric,
      runtime_params_t: runtime_params_slice.RuntimeParams,
      geo_t: geometry.Geometry,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      input_state: sim_state.ToraxSimState,
      previous_post_processed_outputs: post_processing.PostProcessedOutputs,
  ) -> tuple[
      sim_state.ToraxSimState,
      post_processing.PostProcessedOutputs,
  ]:
    """Performs a (possibly) adaptive simulation step."""
    evolving_names = runtime_params_t.numerics.evolving_names
    initial_dt = self.time_step_calculator.next_dt(
        input_state.t,
        runtime_params_t,
        geo_t,
        input_state.core_profiles,
        input_state.core_transport,
    )
    initial_dt = jnp.minimum(initial_dt, max_dt)
    initial_loop_stats = {
        'inner_solver_iterations': jnp.array(0, jax_utils.get_int_dtype()),
    }
    initial_state = adaptive_step.create_initial_state(
        input_state,
        evolving_names,
        initial_dt,
        runtime_params_t,
        geo_t,
    )

    result = whilei_loop.whilei_loop(
        adaptive_step.cond_fun,
        functools.partial(adaptive_step.compute_state, solver=self.solver),
        (
            initial_state,
            initial_loop_stats,
        ),
        initial_dt,
        runtime_params_t,
        geo_t,
        input_state,
        explicit_source_profiles,
        self.runtime_params_provider,
        self.geometry_provider,
    )
    assert isinstance(
        result.state, adaptive_step.AdaptiveStepState
    ), 'adaptive step state is not the expected type.'

    final_solver_numeric_outputs = state.SolverNumericOutputs(
        solver_error_state=jnp.array(
            result.state.solver_numeric_outputs.solver_error_state,
            jax_utils.get_int_dtype(),
        ),
        outer_solver_iterations=jnp.array(
            result.counter, jax_utils.get_int_dtype()
        ),
        inner_solver_iterations=jnp.array(
            result.loop_statistics['inner_solver_iterations'],
            jax_utils.get_int_dtype(),
        ),
        sawtooth_crash=False,
    )
    output_state, post_processed_outputs = (
        step_function_processing.finalize_outputs(
            t=input_state.t,
            dt=result.state.dt,
            x_new=result.state.x_new,
            solver_numeric_outputs=final_solver_numeric_outputs,
            runtime_params_t_plus_dt=result.state.runtime_params,
            geometry_t_plus_dt=result.state.geo,
            core_profiles_t=input_state.core_profiles,
            core_profiles_t_plus_dt=result.state.core_profiles,
            explicit_source_profiles=explicit_source_profiles,
            physics_models=self._solver.physics_models,
            evolving_names=evolving_names,
            input_post_processed_outputs=previous_post_processed_outputs,
        )
    )
    return output_state, post_processed_outputs

  def _fixed_step(
      self,
      max_dt: chex.Numeric,
      runtime_params_t: runtime_params_slice.RuntimeParams,
      geo_t: geometry.Geometry,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      input_state: sim_state.ToraxSimState,
      previous_post_processed_outputs: post_processing.PostProcessedOutputs,
  ) -> tuple[
      sim_state.ToraxSimState,
      post_processing.PostProcessedOutputs,
  ]:
    """Performs a single simulation step."""
    dt = self.time_step_calculator.next_dt(
        input_state.t,
        runtime_params_t,
        geo_t,
        input_state.core_profiles,
        input_state.core_transport,
    )
    dt = jnp.minimum(dt, max_dt)

    runtime_params_t_plus_dt, geo_t_plus_dt = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=input_state.t + dt,
            runtime_params_provider=self._runtime_params_provider,
            geometry_provider=self._geometry_provider,
        )
    )
    core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=input_state.core_profiles,
    )
    # The solver returned state is still "intermediate" since the CoreProfiles
    # need to be updated by the evolved CellVariables in x_new
    x_new, solver_numeric_outputs = self._solver(
        t=input_state.t,
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=input_state.core_profiles,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles,
    )
    output_state, post_processed_outputs = (
        step_function_processing.finalize_outputs(
            t=input_state.t,
            dt=dt,
            x_new=x_new,
            solver_numeric_outputs=solver_numeric_outputs,
            runtime_params_t_plus_dt=runtime_params_t_plus_dt,
            geometry_t_plus_dt=geo_t_plus_dt,
            core_profiles_t=input_state.core_profiles,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            explicit_source_profiles=explicit_source_profiles,
            physics_models=self._solver.physics_models,
            evolving_names=runtime_params_t.numerics.evolving_names,
            input_post_processed_outputs=previous_post_processed_outputs,
        )
    )
    return output_state, post_processed_outputs
