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
from torax._src import jax_utils
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src import xnp
from torax._src.config import build_runtime_params
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.core_profiles import getters
from torax._src.core_profiles import updaters
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.mhd.sawtooth import sawtooth_solver as sawtooth_solver_lib
from torax._src.orchestration import sim_state
from torax._src.output_tools import post_processing
from torax._src.physics import formulas
from torax._src.solver import solver as solver_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.time_step_calculator import time_step_calculator as ts
from torax._src.transport_model import transport_coefficients_builder

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
      dynamic_runtime_params_slice_provider: build_runtime_params.RuntimeParamsProvider,
      geometry_provider: geometry_provider_lib.GeometryProvider,
  ):
    """Initializes the SimulationStepFn.

    Args:
      solver: Evolves the core profiles.
      time_step_calculator: Calculates the dt for each time step.
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
    self._dynamic_runtime_params_slice_provider = (
        dynamic_runtime_params_slice_provider
    )

  @property
  def dynamic_runtime_params_slice_provider(
      self,
  ) -> build_runtime_params.RuntimeParamsProvider:
    return self._dynamic_runtime_params_slice_provider

  def tree_flatten(self):
    children = (
        self._dynamic_runtime_params_slice_provider,
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
        dynamic_runtime_params_slice_provider=children[0],
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

  @xnp.jit
  def __call__(
      self,
      input_state: sim_state.ToraxSimState,
      previous_post_processed_outputs: post_processing.PostProcessedOutputs,
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
    dynamic_runtime_params_slice_t, geo_t = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=input_state.t,
            runtime_params_provider=self._dynamic_runtime_params_slice_provider,
            geometry_provider=self._geometry_provider,
        )
    )

    # This only computes sources set to explicit in the
    # DynamicSourceConfigSlice. All implicit sources will have their profiles
    # set to 0.
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_t,
        geo=geo_t,
        core_profiles=input_state.core_profiles,
        source_models=self._solver.physics_models.source_models,
        neoclassical_models=self._solver.physics_models.neoclassical_models,
        explicit=True,
    )

    def _step():
      """Take either the adaptive or fixed step, depending on the config."""
      if dynamic_runtime_params_slice_t.numerics.adaptive_dt:
        return self._adaptive_step(
            dynamic_runtime_params_slice_t,
            geo_t,
            explicit_source_profiles,
            input_state,
            previous_post_processed_outputs,
        )
      else:
        return self._fixed_step(
            dynamic_runtime_params_slice_t,
            geo_t,
            explicit_source_profiles,
            input_state,
            previous_post_processed_outputs,
        )

    # If a sawtooth model is provided, and there was no previous
    # sawtooth crash, it will be checked to see if a sawtooth
    # should trigger. If it does, the sawtooth model will be applied and instead
    # of a full PDE solve, the step_fn will return early with a state following
    # sawtooth redistribution, at a t+dt set by the sawtooth model
    # configuration.
    if self._sawtooth_solver is not None:
      output_state, post_processed_outputs = xnp.cond(
          input_state.solver_numeric_outputs.sawtooth_crash,
          lambda *args: (input_state, previous_post_processed_outputs),
          self._sawtooth_step,
          dynamic_runtime_params_slice_t,
          geo_t,
          explicit_source_profiles,
          input_state,
          previous_post_processed_outputs,
      )

      output_state, post_processed_outputs = xnp.cond(
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

  def _sawtooth_step(
      self,
      dynamic_runtime_params_slice_t: runtime_params_slice.RuntimeParams,
      geo_t: geometry.Geometry,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      input_state: sim_state.ToraxSimState,
      previous_post_processed_outputs: post_processing.PostProcessedOutputs,
  ) -> tuple[
      sim_state.ToraxSimState,
      post_processing.PostProcessedOutputs,
  ]:
    """Performs a simulation step if a sawtooth crash is triggered."""
    assert dynamic_runtime_params_slice_t.mhd.sawtooth is not None
    dt_crash = dynamic_runtime_params_slice_t.mhd.sawtooth.crash_step_duration

    (
        dynamic_runtime_params_slice_t_plus_crash_dt,
        geo_t,
        geo_t_plus_crash_dt,
    ) = _get_geo_and_dynamic_runtime_params_at_t_plus_dt_and_phibdot(
        input_state.t,
        dt_crash,
        self._dynamic_runtime_params_slice_provider,
        geo_t,
        self._geometry_provider,
    )

    # If no sawtooth crash is triggered, output_state and
    # post_processed_outputs will be the same as the input state and
    # previous_post_processed_outputs.
    output_state, post_processed_outputs = _sawtooth_step(
        sawtooth_solver=self._sawtooth_solver,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
        dynamic_runtime_params_slice_t_plus_crash_dt=dynamic_runtime_params_slice_t_plus_crash_dt,
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
      dynamic_runtime_params_slice_t: runtime_params_slice.RuntimeParams,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.RuntimeParams,
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
        runtime_params_t=dynamic_runtime_params_slice_t,
        runtime_params_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
    )

    # Initial trial for solver. If did not converge (can happen for nonlinear
    # step with large dt) we apply the adaptive time step routine if requested.
    return self._solver(
        t=input_state.t,
        dt=dt,
        runtime_params_t=dynamic_runtime_params_slice_t,
        runtime_params_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles,
    )

  def _adaptive_step(
      self,
      dynamic_runtime_params_slice_t: runtime_params_slice.RuntimeParams,
      geo_t: geometry.Geometry,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      input_state: sim_state.ToraxSimState,
      previous_post_processed_outputs: post_processing.PostProcessedOutputs,
  ) -> tuple[
      sim_state.ToraxSimState,
      post_processing.PostProcessedOutputs,
  ]:
    """Performs a (possibly) adaptive simulation step."""
    evolving_names = dynamic_runtime_params_slice_t.numerics.evolving_names

    initial_dt = self.time_step_calculator.next_dt(
        input_state.t,
        dynamic_runtime_params_slice_t,
        geo_t,
        input_state.core_profiles,
        input_state.core_transport,
    )

    input_type = jax.Array
    output_type = tuple[
        tuple[cell_variable.CellVariable, ...],
        jax.Array,  # dt
        state.SolverNumericOutputs,
        runtime_params_slice.RuntimeParams,
        geometry.Geometry,
        state.CoreProfiles,
    ]

    def cond_fun(inputs: tuple[input_type, output_type]):
      next_dt, output = inputs
      solver_outputs = output[2]

      # Check for NaN in the next dt to avoid a recursive loop.
      is_nan_next_dt = xnp.isnan(next_dt)

      # If the solver did not converge we need to make a new step.
      solver_did_not_converge = solver_outputs.solver_error_state == 1

      # If t + dt  is exactly the final time we may need a smaller step than
      # min_dt to exactly reach the final time.
      if dynamic_runtime_params_slice_t.numerics.exact_t_final:
        at_exact_t_final = xnp.allclose(
            input_state.t + next_dt,
            dynamic_runtime_params_slice_t.numerics.t_final,
        )
      else:
        at_exact_t_final = xnp.array(False)

      next_dt_too_small = (
          next_dt < dynamic_runtime_params_slice_t.numerics.min_dt
      )

      take_another_step = xnp.cond(
          solver_did_not_converge,
          # If the solver did not converge then we check if we are at the exact
          # final time and should take a smaller step. If not we also check if
          # the next dt is too small, if so we should end the step.
          lambda: xnp.cond(
              at_exact_t_final, lambda: True, lambda: ~next_dt_too_small
          ),
          lambda: False,
      )

      return take_another_step & ~is_nan_next_dt

    def body_fun(inputs: tuple[input_type, output_type]):
      dt, output = inputs
      old_solver_outputs = output[2]

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
          self._dynamic_runtime_params_slice_provider,
          geo_t,
          self._geometry_provider,
      )

      core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
          dt=dt,
          runtime_params_t=dynamic_runtime_params_slice_t,
          runtime_params_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
          geo_t_plus_dt=geo_t_plus_dt,
          core_profiles_t=input_state.core_profiles,
      )
      # The solver returned state is still "intermediate" since the CoreProfiles
      # need to be updated by the evolved CellVariables in x_new
      x_new, solver_numeric_outputs = self._solver(
          t=input_state.t,
          dt=dt,
          runtime_params_t=dynamic_runtime_params_slice_t,
          runtime_params_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
          geo_t=geo_t_with_phibdot,
          geo_t_plus_dt=geo_t_plus_dt,
          core_profiles_t=input_state.core_profiles,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          explicit_source_profiles=explicit_source_profiles,
      )
      solver_numeric_outputs = state.SolverNumericOutputs(
          solver_error_state=solver_numeric_outputs.solver_error_state,
          outer_solver_iterations=old_solver_outputs.outer_solver_iterations
          + 1,
          inner_solver_iterations=old_solver_outputs.inner_solver_iterations
          + solver_numeric_outputs.inner_solver_iterations,
          sawtooth_crash=solver_numeric_outputs.sawtooth_crash,
      )
      next_dt = (
          dt
          / dynamic_runtime_params_slice_t_plus_dt.numerics.dt_reduction_factor
      )
      return next_dt, (
          x_new,
          dt,
          solver_numeric_outputs,
          dynamic_runtime_params_slice_t_plus_dt,
          geo_t_plus_dt,
          core_profiles_t_plus_dt,
      )

    _, result = xnp.while_loop(
        cond_fun,
        body_fun,
        (
            initial_dt,
            (
                convertors.core_profiles_to_solver_x_tuple(
                    input_state.core_profiles, evolving_names),
                initial_dt,
                state.SolverNumericOutputs(
                    # The solver has not converged yet as we have not performed
                    # any steps yet.
                    solver_error_state=1,
                    outer_solver_iterations=0,
                    inner_solver_iterations=0,
                    sawtooth_crash=False,
                ),
                dynamic_runtime_params_slice_t,
                geo_t,
                input_state.core_profiles,
            ),
        ),
    )
    output_state, post_processed_outputs = _finalize_outputs(
        t=input_state.t,
        dt=result[1],
        x_new=result[0],
        solver_numeric_outputs=result[2],
        dynamic_runtime_params_slice_t_plus_dt=result[3],
        geometry_t_plus_dt=result[4],
        core_profiles_t=input_state.core_profiles,
        core_profiles_t_plus_dt=result[5],
        explicit_source_profiles=explicit_source_profiles,
        physics_models=self._solver.physics_models,
        evolving_names=evolving_names,
        input_post_processed_outputs=previous_post_processed_outputs,
    )
    return output_state, post_processed_outputs

  def _fixed_step(
      self,
      dynamic_runtime_params_slice_t: runtime_params_slice.RuntimeParams,
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
        dynamic_runtime_params_slice_t,
        geo_t,
        input_state.core_profiles,
        input_state.core_transport,
    )
    # The solver needs the geo and dynamic_runtime_params_slice at time t + dt
    # for implicit computations in the solver. Once geo_t_plus_dt is calculated
    # we can use it to calculate Phibdot for both geo_t and geo_t_plus_dt, which
    # then update the initialized Phibdot=0 in the geo instances.
    dynamic_runtime_params_slice_t_plus_dt, geo_t, geo_t_plus_dt = (
        _get_geo_and_dynamic_runtime_params_at_t_plus_dt_and_phibdot(
            input_state.t,
            dt,
            self._dynamic_runtime_params_slice_provider,
            geo_t,
            self._geometry_provider,
        )
    )
    core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
        dt=dt,
        runtime_params_t=dynamic_runtime_params_slice_t,
        runtime_params_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=input_state.core_profiles,
    )
    # The solver returned state is still "intermediate" since the CoreProfiles
    # need to be updated by the evolved CellVariables in x_new
    x_new, solver_numeric_outputs = self._solver(
        t=input_state.t,
        dt=dt,
        runtime_params_t=dynamic_runtime_params_slice_t,
        runtime_params_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=input_state.core_profiles,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles,
    )
    output_state, post_processed_outputs = _finalize_outputs(
        t=input_state.t,
        dt=dt,
        x_new=x_new,
        solver_numeric_outputs=solver_numeric_outputs,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
        geometry_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=input_state.core_profiles,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles,
        physics_models=self._solver.physics_models,
        evolving_names=dynamic_runtime_params_slice_t.numerics.evolving_names,
        input_post_processed_outputs=previous_post_processed_outputs,
    )
    return output_state, post_processed_outputs


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def _finalize_outputs(
    t: jax.Array,
    dt: jax.Array,
    x_new: tuple[cell_variable.CellVariable, ...],
    solver_numeric_outputs: state.SolverNumericOutputs,
    geometry_t_plus_dt: geometry.Geometry,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.RuntimeParams,
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    evolving_names: tuple[str, ...],
    input_post_processed_outputs: post_processing.PostProcessedOutputs,
) -> tuple[sim_state.ToraxSimState, post_processing.PostProcessedOutputs]:
  """Returns the final state and post-processed outputs."""
  final_core_profiles, final_source_profiles = (
      updaters.update_core_and_source_profiles_after_step(
          dt=dt,
          x_new=x_new,
          runtime_params_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
          geo=geometry_t_plus_dt,
          core_profiles_t=core_profiles_t,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          explicit_source_profiles=explicit_source_profiles,
          source_models=physics_models.source_models,
          neoclassical_models=physics_models.neoclassical_models,
          evolving_names=evolving_names,
      )
  )
  final_total_transport = (
      transport_coefficients_builder.calculate_total_transport_coeffs(
          physics_models.pedestal_model,
          physics_models.transport_model,
          physics_models.neoclassical_models,
          dynamic_runtime_params_slice_t_plus_dt,
          geometry_t_plus_dt,
          final_core_profiles,
      )
  )

  output_state = sim_state.ToraxSimState(
      t=t + dt,
      dt=dt,
      core_profiles=final_core_profiles,
      core_sources=final_source_profiles,
      core_transport=final_total_transport,
      geometry=geometry_t_plus_dt,
      solver_numeric_outputs=solver_numeric_outputs,
  )
  post_processed_outputs = post_processing.make_post_processed_outputs(
      sim_state=output_state,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_dt,
      previous_post_processed_outputs=input_post_processed_outputs,
  )
  return output_state, post_processed_outputs


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'sawtooth_solver',
    ],
)
def _sawtooth_step(
    *,
    sawtooth_solver: sawtooth_solver_lib.SawtoothSolver | None,
    dynamic_runtime_params_slice_t: runtime_params_slice.RuntimeParams,
    dynamic_runtime_params_slice_t_plus_crash_dt: runtime_params_slice.RuntimeParams,
    geo_t: geometry.Geometry,
    geo_t_plus_crash_dt: geometry.Geometry,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    input_state: sim_state.ToraxSimState,
    input_post_processed_outputs: post_processing.PostProcessedOutputs,
) -> tuple[sim_state.ToraxSimState, post_processing.PostProcessedOutputs]:
  """Checks for and handles a sawtooth crash.

  If a sawtooth model is provided and a crash is triggered, this method
  computes the post-crash state and returns it. Otherwise, returns the input
  state and post-processed outputs unchanged.

  Consecutive sawtooth crashes are not allowed since standard PDE steps
  may then not take place. Therefore if the input state has sawtooth_crash set
  to True, then no crash is triggered.

  Args:
    sawtooth_solver: Sawtooth model which carries out sawtooth step..
    dynamic_runtime_params_slice_t: Dynamic slice at time t.
    dynamic_runtime_params_slice_t_plus_crash_dt: Dynamic slice at time t +
      crash_dt.
    geo_t: Geometry at time t.
    geo_t_plus_crash_dt: Geometry at time t + crash_dt.
    explicit_source_profiles: Explicit source profiles at time t.
    input_state: State at the start of the time step.
    input_post_processed_outputs: Post-processed outputs from the previous step.

  Returns:
    Returns a tuple (output_state, post_processed_outputs).
  """

  # Asserts needed for linter.
  assert dynamic_runtime_params_slice_t.mhd.sawtooth is not None
  assert sawtooth_solver is not None
  dt_crash = dynamic_runtime_params_slice_t.mhd.sawtooth.crash_step_duration

  # Prepare core_profiles_t_plus_crash_dt with new boundary conditions
  # and prescribed profiles if present.
  core_profiles_t_plus_crash_dt = updaters.provide_core_profiles_t_plus_dt(
      dt=dt_crash,
      runtime_params_t=dynamic_runtime_params_slice_t,
      runtime_params_t_plus_dt=dynamic_runtime_params_slice_t_plus_crash_dt,
      geo_t_plus_dt=geo_t_plus_crash_dt,
      core_profiles_t=input_state.core_profiles,
  )

  (
      x_candidate,
      solver_numeric_outputs,
  ) = sawtooth_solver(
      t=input_state.t,
      dt=dt_crash,
      runtime_params_t=dynamic_runtime_params_slice_t,
      runtime_params_t_plus_dt=dynamic_runtime_params_slice_t_plus_crash_dt,
      geo_t=geo_t,
      geo_t_plus_dt=geo_t_plus_crash_dt,
      core_profiles_t=input_state.core_profiles,
      core_profiles_t_plus_dt=core_profiles_t_plus_crash_dt,
      explicit_source_profiles=explicit_source_profiles,
  )

  def _make_post_crash_state_and_post_processed_outputs():
    """Returns the post-crash state and post-processed outputs."""

    # We also update the temperature profiles over the sawtooth time to
    # maintain constant dW/dt over the sawtooth period. While not strictly
    # realistic this avoids non-physical dW/dt=perturbations in
    # post-processing.
    # Following the sawtooth redistribution, the PDE will take over the
    # energy evolution and the physical dW/dt corresponding to the new profile
    # distribution will be calculated.
    # This must be done here and not in the sawtooth model since the Solver
    # API does not include the post-processed outputs.
    x_evolved = _evolve_x_after_sawtooth(
        x_redistributed=x_candidate,
        dynamic_runtime_params_slice_t_plus_crash_dt=dynamic_runtime_params_slice_t_plus_crash_dt,
        core_profiles_redistributed=core_profiles_t_plus_crash_dt,
        geo_t_plus_crash_dt=geo_t_plus_crash_dt,
        previous_post_processed_outputs=input_post_processed_outputs,
        evolving_names=dynamic_runtime_params_slice_t.numerics.evolving_names,
        dt_crash=dt_crash,
    )

    return _finalize_outputs(
        t=input_state.t,
        dt=dt_crash,
        x_new=x_evolved,
        solver_numeric_outputs=solver_numeric_outputs,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_crash_dt,
        geometry_t_plus_dt=geo_t_plus_crash_dt,
        core_profiles_t=input_state.core_profiles,
        core_profiles_t_plus_dt=core_profiles_t_plus_crash_dt,
        explicit_source_profiles=explicit_source_profiles,
        physics_models=sawtooth_solver.physics_models,
        evolving_names=dynamic_runtime_params_slice_t.numerics.evolving_names,
        input_post_processed_outputs=input_post_processed_outputs,
    )

  return jax.lax.cond(
      solver_numeric_outputs.sawtooth_crash,
      _make_post_crash_state_and_post_processed_outputs,
      lambda: (
          input_state,
          input_post_processed_outputs,
      ),
  )


def _get_geo_and_dynamic_runtime_params_at_t_plus_dt_and_phibdot(
    t: jax.Array,
    dt: jax.Array,
    dynamic_runtime_params_slice_provider: build_runtime_params.RuntimeParamsProvider,
    geo_t: geometry.Geometry,
    geometry_provider: geometry_provider_lib.GeometryProvider,
) -> tuple[
    runtime_params_slice.RuntimeParams,
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
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=t + dt,
          runtime_params_provider=dynamic_runtime_params_slice_provider,
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


def _evolve_x_after_sawtooth(
    x_redistributed: tuple[cell_variable.CellVariable, ...],
    dynamic_runtime_params_slice_t_plus_crash_dt: runtime_params_slice.RuntimeParams,
    core_profiles_redistributed: state.CoreProfiles,
    geo_t_plus_crash_dt: geometry.Geometry,
    previous_post_processed_outputs: post_processing.PostProcessedOutputs,
    evolving_names: tuple[str, ...],
    dt_crash: jax.Array,
) -> tuple[cell_variable.CellVariable, ...]:
  """Evolves the x_redistributed after the sawtooth redistribution."""

  updated_core_profiles = convertors.solver_x_tuple_to_core_profiles(
      x_new=x_redistributed,
      evolving_names=evolving_names,
      core_profiles=core_profiles_redistributed,
  )

  ions = getters.get_updated_ions(
      dynamic_runtime_params_slice_t_plus_crash_dt,
      geo_t_plus_crash_dt,
      updated_core_profiles.n_e,
      updated_core_profiles.T_e,
  )

  updated_core_profiles = dataclasses.replace(
      updated_core_profiles,
      n_i=ions.n_i,
      n_impurity=ions.n_impurity,
  )

  (
      pressure_thermal_el,
      pressure_thermal_ion,
      pressure_thermal_tot,
  ) = formulas.calculate_pressure(updated_core_profiles)

  _, _, W_thermal_tot = formulas.calculate_stored_thermal_energy(
      pressure_thermal_el,
      pressure_thermal_ion,
      pressure_thermal_tot,
      geo_t_plus_crash_dt,
  )

  # Update temperatures to maintain constant dW/dt over the sawtooth period.
  dW_target = previous_post_processed_outputs.dW_thermal_dt * dt_crash

  factor = 1 + dW_target / W_thermal_tot

  updated_core_profiles = dataclasses.replace(
      updated_core_profiles,
      T_e=dataclasses.replace(
          updated_core_profiles.T_e,
          value=updated_core_profiles.T_e.value * factor,
      ),
      T_i=dataclasses.replace(
          updated_core_profiles.T_i,
          value=updated_core_profiles.T_i.value * factor,
      ),
  )

  x_evolved = convertors.core_profiles_to_solver_x_tuple(
      updated_core_profiles,
      evolving_names,
  )

  return x_evolved
