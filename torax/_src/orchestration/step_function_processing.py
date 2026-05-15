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

"""Functions for pre and post processing used in the step function call."""

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import models as models_lib
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import updaters
from torax._src.edge import base as edge_base
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.orchestration import sim_state
from torax._src.output_tools import post_processing
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.formation import power_scaling_formation_model as power_scaling_formation_model_lib
from torax._src.physics import scaling_laws
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.time_step_calculator import time_step_calculator_state as time_step_calculator_state_lib
from torax._src.transport_model import transport_coefficients_builder

# pylint: disable=invalid-name

ConfinementMode = pedestal_transition_state_lib.ConfinementMode


def _update_pedestal_transition_state(
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState
    ),
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    core_sources: source_profiles_lib.SourceProfiles,
    models: models_lib.Models,
) -> pedestal_transition_state_lib.PedestalTransitionState:
  """Evaluates P_SOL vs P_LH and updates the pedestal transition state.

  Called once per timestep in pre_step. Computes P_SOL and P_LH, then
  delegates to the mode-specific update function.

  Args:
    pedestal_transition_state: Current transition state from previous timestep.
    runtime_params: Runtime parameters at time t.
    geo: Geometry at time t.
    core_profiles: Core plasma profiles at time t.
    core_sources: Source profiles at time t.
    models: Models for the simulation (used to access formation model config).

  Returns:
    Updated PedestalTransitionState.
  """
  formation_model = models.pedestal_model.formation_model
  # Explicitly check types (required for pytype).
  assert isinstance(
      formation_model,
      power_scaling_formation_model_lib.PowerScalingFormationModel,
  )
  assert isinstance(
      runtime_params.pedestal.formation,
      power_scaling_formation_model_lib.PowerScalingFormationRuntimeParams,
  )

  # Calculate P_SOL (total power crossing the separatrix).
  P_SOL = power_scaling_formation_model_lib.calculate_P_SOL_total(
      internal_plasma_energy=core_profiles.internal_plasma_energy,
      core_sources=core_sources,
      geo=geo,
  )

  # Calculate P_LH (L-H transition threshold power), with a configurable
  # prefactor to allow for tuning.
  P_LH, _ = scaling_laws.calculate_P_LH(
      geo=geo,
      core_profiles=core_profiles,
      scaling_law=formation_model.scaling_law,
      divertor_configuration=formation_model.divertor_configuration,
  )
  P_LH = P_LH * runtime_params.pedestal.formation.P_LH_prefactor

  if (
      runtime_params.pedestal.mode
      == pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT
  ):
    return _update_adaptive_transport(
        pedestal_transition_state, runtime_params, P_SOL, P_LH,
    )

  return _update_adaptive_source(
      pedestal_transition_state, runtime_params, geo, core_profiles,
      core_sources, models, P_SOL, P_LH,
  )


def _update_adaptive_transport(
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState
    ),
    runtime_params: runtime_params_lib.RuntimeParams,
    P_SOL: jax.Array,
    P_LH: jax.Array,
) -> pedestal_transition_state_lib.PedestalTransitionState:
  """Updates pedestal transition state for ADAPTIVE_TRANSPORT mode.

  Simple L-mode <-> H-mode transitions with hysteresis. No transitioning
  states, timers, or L-mode value capture. The sigmoid in the formation
  model handles the dynamics of the transition.

  Note that interpretation of ConfinementMode is on the expectation of the
  confinement regime at the end of the timestep interval based on the
  P_SOL/P_LH ratio at the beginning of the timestep interval. There is no
  information on the dynamics and where we are in the transition.

  Args:
    pedestal_transition_state: Current transition state from previous timestep.
    runtime_params: Runtime parameters at time t.
    P_SOL: Total power crossing the separatrix.
    P_LH: L-H transition threshold power (already rescaled by P_LH_prefactor).

  Returns:
    Updated PedestalTransitionState.
  """
  old_confinement_mode = pedestal_transition_state.confinement_mode
  new_confinement_mode = jnp.select(
      [
          # L-H transition.
          (old_confinement_mode == ConfinementMode.L_MODE) & (P_SOL > P_LH),
          # H-L back transition, with hysteresis.
          (old_confinement_mode == ConfinementMode.H_MODE)
          & (P_SOL < P_LH * runtime_params.pedestal.P_LH_hysteresis_factor),
      ],
      [ConfinementMode.H_MODE, ConfinementMode.L_MODE],
      default=old_confinement_mode,
  )

  return pedestal_transition_state_lib.PedestalTransitionState(
      confinement_mode=new_confinement_mode,
      transition_start_time=pedestal_transition_state.transition_start_time,
      T_i_ped_L_mode=pedestal_transition_state.T_i_ped_L_mode,
      T_e_ped_L_mode=pedestal_transition_state.T_e_ped_L_mode,
      n_e_ped_L_mode=pedestal_transition_state.n_e_ped_L_mode,
  )


def _update_adaptive_source(
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState
    ),
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    core_sources: source_profiles_lib.SourceProfiles,
    models: models_lib.Models,
    P_SOL: jax.Array,
    P_LH: jax.Array,
) -> pedestal_transition_state_lib.PedestalTransitionState:
  """Updates pedestal transition state for ADAPTIVE_SOURCE mode.

  Full 4-state machine with TRANSITIONING_TO_H/L states, transition timers,
  dithering support, and L-mode value capture for ramp interpolation.

  When transitioning from L-mode to H-mode (currently in L-mode and P_SOL >
  P_LH):
    - Records the current simulation time as transition_start_time
    - Saves the current kinetic profile values at the pedestal-top for the
      lower target values when setting up pedestal ramp up/down.

  When transitioning from H-mode to L-mode (currently in H-mode and P_SOL <
  P_LH):
    - Records the current simulation time as transition_start_time
    - Loads the saved L-mode pedestal-top values as a target for the end of
      the transition.

  If mid-transition and then starting to revert back to the original
  confinement mode:
    - Sets the transition_start_time such that the time spent in the
      reverse transition is the same as the time spent in the original
      transition.

  Args:
    pedestal_transition_state: Current transition state from previous timestep.
    runtime_params: Runtime parameters at time t.
    geo: Geometry at time t.
    core_profiles: Core plasma profiles at time t.
    core_sources: Source profiles at time t.
    models: Models for the simulation.
    P_SOL: Total power crossing the separatrix.
    P_LH: L-H transition threshold power (already rescaled by P_LH_prefactor).

  Returns:
    Updated PedestalTransitionState.
  """
  old_confinement_mode = pedestal_transition_state.confinement_mode

  # Has the transition time elapsed? Used for exiting transition states.
  elapsed_transition_time = (
      runtime_params.t - pedestal_transition_state.transition_start_time
  )
  transition_is_complete = (
      elapsed_transition_time >= runtime_params.pedestal.transition_time_width
  )

  # Update transition state based on P_SOL vs P_LH.
  # The transition has hysteresis, which boils down to the following:
  # - If P_SOL > P_LH, start transitioning to H-mode if not already in H-mode.
  # - If P_SOL < h*P_LH, where 0 < h < 1, start transitioning to L-mode if not
  #   already in L-mode.
  # - If in a transition and the transition time has elapsed, exit transition
  #   and enter the target mode.
  # - Otherwise, remain in the current state.
  conditions = [
      # Completed LH transition. Checked first so that completed transitions
      # take priority over starting new transitions in jnp.select.
      (old_confinement_mode == ConfinementMode.TRANSITIONING_TO_H_MODE)
      & transition_is_complete,
      # Completed HL transition.
      (old_confinement_mode == ConfinementMode.TRANSITIONING_TO_L_MODE)
      & transition_is_complete,
      # L-H transition.
      (old_confinement_mode != ConfinementMode.H_MODE) & (P_SOL > P_LH),
      # H-L back transition, with hysteresis.
      (old_confinement_mode != ConfinementMode.L_MODE)
      & (P_SOL < P_LH * runtime_params.pedestal.P_LH_hysteresis_factor),
  ]
  new_confinement_modes = [
      ConfinementMode.H_MODE,
      ConfinementMode.L_MODE,
      ConfinementMode.TRANSITIONING_TO_H_MODE,
      ConfinementMode.TRANSITIONING_TO_L_MODE,
  ]
  new_confinement_mode = jnp.select(
      conditions,
      new_confinement_modes,
      old_confinement_mode,
  )

  # Update the transition start time.
  # - If we've gone from L-mode to LH transition, or from H-mode to HL
  #   transition, set the start time to the current time.
  # - If we are dithering, set the start time so that we
  #   spend the same amount of time in the back-transition as we did in the
  #   forward transition.
  # - Otherwise, keep the current start time.
  standard_transition = (
      (old_confinement_mode == ConfinementMode.L_MODE)
      & (new_confinement_mode == ConfinementMode.TRANSITIONING_TO_H_MODE)
  ) | (
      (old_confinement_mode == ConfinementMode.H_MODE)
      & (new_confinement_mode == ConfinementMode.TRANSITIONING_TO_L_MODE)
  )
  dithering_transition = (
      (old_confinement_mode == ConfinementMode.TRANSITIONING_TO_H_MODE)
      & (new_confinement_mode == ConfinementMode.TRANSITIONING_TO_L_MODE)
  ) | (
      (old_confinement_mode == ConfinementMode.TRANSITIONING_TO_L_MODE)
      & (new_confinement_mode == ConfinementMode.TRANSITIONING_TO_H_MODE)
  )
  # At time t, suppose we have been in forward transition since t0, with desired
  # transition duration w. If we now begin a back-transition, we have been in
  # forward transition for t - t0, and so we want to end the back-transition at
  # time t + (t - t0) = 2t - t0. As we have specified the transition time width
  # as w, to achieve this we set the start time to (2t - t0) - w.
  new_transition_start_time = jnp.select(
      [standard_transition, dithering_transition],
      [
          runtime_params.t,
          2.0 * runtime_params.t
          - pedestal_transition_state.transition_start_time
          - runtime_params.pedestal.transition_time_width,
      ],
      # Otherwise, preserve the current transition start time. This covers
      # both ongoing transitions (where the mode hasn't changed) and
      # non-transition states (H_MODE/L_MODE where start_time is already inf).
      default=pedestal_transition_state.transition_start_time,
  )

  # Update the target values for transitions to L-mode.
  # Only needed for ADAPTIVE_SOURCE, which uses ramp interpolation.
  update_L_mode_values = (old_confinement_mode == ConfinementMode.L_MODE) & (
      new_confinement_mode == ConfinementMode.TRANSITIONING_TO_H_MODE
  )
  # TODO(b/500260959): Avoid calling the pedestal model again.
  pedestal_model_output = models.pedestal_model(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_profiles=core_sources,
      pedestal_transition_state=pedestal_transition_state,
  )
  ped_top_idx = jnp.argmin(
      jnp.abs(geo.rho_norm - pedestal_model_output.rho_norm_ped_top)
  )
  new_T_i_ped_L_mode = jnp.where(
      update_L_mode_values,
      core_profiles.T_i.value[ped_top_idx],
      pedestal_transition_state.T_i_ped_L_mode,
  )
  new_T_e_ped_L_mode = jnp.where(
      update_L_mode_values,
      core_profiles.T_e.value[ped_top_idx],
      pedestal_transition_state.T_e_ped_L_mode,
  )
  new_n_e_ped_L_mode = jnp.where(
      update_L_mode_values,
      core_profiles.n_e.value[ped_top_idx],
      pedestal_transition_state.n_e_ped_L_mode,
  )

  return pedestal_transition_state_lib.PedestalTransitionState(
      confinement_mode=new_confinement_mode,
      transition_start_time=new_transition_start_time,
      T_i_ped_L_mode=new_T_i_ped_L_mode,
      T_e_ped_L_mode=new_T_e_ped_L_mode,
      n_e_ped_L_mode=new_n_e_ped_L_mode,
  )


def pre_step(
    input_state: sim_state.SimState,
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    models: models_lib.Models,
) -> tuple[
    runtime_params_lib.RuntimeParams,
    geometry.Geometry,
    source_profiles_lib.SourceProfiles,
    edge_base.EdgeModelOutputs | None,
    pedestal_transition_state_lib.PedestalTransitionState | None,
]:
  """Performs the pre-step operations for the step function."""
  runtime_params_t, geo_t = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=input_state.t,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=geometry_provider,
          edge_outputs=input_state.edge_outputs,
          core_profiles=input_state.core_profiles,
      )
  )

  # This only computes sources set to explicit in the
  # SourceConfig.
  explicit_source_profiles = source_profile_builders.build_source_profiles(
      runtime_params=runtime_params_t,
      geo=geo_t,
      core_profiles=input_state.core_profiles,
      source_models=models.source_models,
      neoclassical_models=models.neoclassical_models,
      explicit=True,
  )

  # Execute the edge model if one is configured. The edge model uses the state
  # at time t to calculate new edge conditions for the next time step.
  edge_model = models.edge_model
  if edge_model is not None:

    # Update core sources with any newly calculated explicit sources.
    # This is because in input_state, the sources are those which were
    # used to compute the state. For explicit sources, these were computed with
    # core_profiles at time t_minus_dt, whereas the implicit sources are
    # consistent with time t. For the edge model, we want all sources consistent
    # with the state at time t, so we replace the explicit sources with the
    # newly calculated profiles.
    core_sources = dataclasses.replace(
        input_state.core_sources,
        T_e=input_state.core_sources.T_e | explicit_source_profiles.T_e,
        T_i=input_state.core_sources.T_i | explicit_source_profiles.T_i,
        n_e=input_state.core_sources.n_e | explicit_source_profiles.n_e,
        psi=input_state.core_sources.psi | explicit_source_profiles.psi,
    )
    edge_outputs = edge_model(
        runtime_params_t,
        geo_t,
        input_state.core_profiles,
        core_sources,
        previous_edge_outputs=input_state.edge_outputs,
    )
  else:
    edge_outputs = None

  # Update pedestal transition state for hysteresis tracking.
  # Called for both ADAPTIVE_SOURCE (with formation model) and
  # ADAPTIVE_TRANSPORT modes.
  pedestal_transition_state = input_state.pedestal_transition_state
  if (
      (
          runtime_params_t.pedestal.mode
          == pedestal_runtime_params_lib.Mode.ADAPTIVE_SOURCE
          and runtime_params_t.pedestal.use_formation_model_with_adaptive_source
      )
      or runtime_params_t.pedestal.mode
      == pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT
  ):
    # Merge explicit sources with previous implicit sources for accurate
    # P_SOL calculation (same pattern as the edge model above).
    merged_sources = dataclasses.replace(
        input_state.core_sources,
        T_e=input_state.core_sources.T_e | explicit_source_profiles.T_e,
        T_i=input_state.core_sources.T_i | explicit_source_profiles.T_i,
        n_e=input_state.core_sources.n_e | explicit_source_profiles.n_e,
        psi=input_state.core_sources.psi | explicit_source_profiles.psi,
    )
    pedestal_transition_state = _update_pedestal_transition_state(
        pedestal_transition_state=pedestal_transition_state,
        runtime_params=runtime_params_t,
        geo=geo_t,
        core_profiles=input_state.core_profiles,
        core_sources=merged_sources,
        models=models,
    )

  return (
      runtime_params_t,
      geo_t,
      explicit_source_profiles,
      edge_outputs,
      pedestal_transition_state,
  )


@jax.jit(
    static_argnames=[
        'models',
        'evolving_names',
    ],
)
def finalize_outputs(
    t: jax.Array,
    dt: jax.Array,
    x_new: tuple[cell_variable.CellVariable, ...],
    solver_numeric_outputs: state.SolverNumericOutputs,
    geometry_t_plus_dt: geometry.Geometry,
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    edge_outputs: edge_base.EdgeModelOutputs | None,
    models: models_lib.Models,
    evolving_names: tuple[str, ...],
    input_post_processed_outputs: post_processing.PostProcessedOutputs,
    time_step_calculator_state_t: time_step_calculator_state_lib.TimeStepCalculatorState,
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState
    ),
) -> tuple[sim_state.SimState, post_processing.PostProcessedOutputs]:
  """Returns the final state and post-processed outputs."""
  final_core_profiles, final_source_profiles = (
      updaters.update_core_and_source_profiles_after_step(
          dt=dt,
          x_new=x_new,
          runtime_params_t_plus_dt=runtime_params_t_plus_dt,
          geo=geometry_t_plus_dt,
          core_profiles_t=core_profiles_t,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          explicit_source_profiles=explicit_source_profiles,
          source_models=models.source_models,
          neoclassical_models=models.neoclassical_models,
          evolving_names=evolving_names,
      )
  )
  final_total_transport = (
      transport_coefficients_builder.calculate_all_transport_coeffs(
          models.pedestal_model,
          models.transport_model,
          models.neoclassical_models,
          runtime_params_t_plus_dt,
          geometry_t_plus_dt,
          final_core_profiles,
          final_source_profiles,
          pedestal_transition_state=pedestal_transition_state,
      )
  )
  output_state = sim_state.SimState(
      t=t + dt,
      dt=dt,
      core_profiles=final_core_profiles,
      core_sources=final_source_profiles,
      core_transport=final_total_transport,
      geometry=geometry_t_plus_dt,
      solver_numeric_outputs=solver_numeric_outputs,
      edge_outputs=edge_outputs,
      pedestal_transition_state=pedestal_transition_state,
      time_step_calculator_state=time_step_calculator_state_t,
  )

  # Update the time step calculator state.
  # time_step_calculator_state_t is the state before this time step, and
  # time_step_calculator_state_t_plus_dt is the state after this time step.
  time_step_calculator_state_t_plus_dt = (
      models.time_step_calculator.get_updated_state(
          sim_state=output_state,
      )
  )
  output_state = dataclasses.replace(
      output_state,
      time_step_calculator_state=time_step_calculator_state_t_plus_dt,
  )

  post_processed_outputs = post_processing.make_post_processed_outputs(
      sim_state=output_state,
      runtime_params=runtime_params_t_plus_dt,
      previous_post_processed_outputs=input_post_processed_outputs,
  )
  return output_state, post_processed_outputs
