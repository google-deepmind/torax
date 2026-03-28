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
import functools
import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import physics_models as physics_models_lib
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
from torax._src.pedestal_model.formation import power_scaling_formation_model as power_scaling_formation_model_lib
from torax._src.physics import scaling_laws
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.transport_model import transport_coefficients_builder

# pylint: disable=invalid-name


def _update_pedestal_transition_state(
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState
    ),
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    core_sources: source_profiles_lib.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
) -> pedestal_transition_state_lib.PedestalTransitionState:
  """Evaluates P_SOL vs P_LH and updates the pedestal transition state.

  Called once per timestep in pre_step. Determines whether the plasma should
  enter or exit H-mode based on the power crossing the separatrix (P_SOL)
  compared to the L-H transition threshold power (P_LH).

  When transitioning from L-mode to H-mode (P_SOL > P_LH):
    - Sets in_H_mode to True
    - Records the current simulation time as transition_start_time
    - Captures current pedestal-top profile values as L-mode baselines

  When transitioning from H-mode to L-mode (P_SOL < P_LH):
    - Sets in_H_mode to False
    - Records the current simulation time as transition_start_time
    - Preserves the existing L-mode baselines (from the most recent L→H start)

  Args:
    pedestal_transition_state: Current transition state from previous timestep.
    runtime_params: Runtime parameters at time t.
    geo: Geometry at time t.
    core_profiles: Core plasma profiles at time t.
    core_sources: Source profiles at time t.
    physics_models: Physics models (used to access formation model config).

  Returns:
    Updated PedestalTransitionState.
  """
  formation_model = physics_models.pedestal_model.formation_model
  assert isinstance(
      formation_model,
      power_scaling_formation_model_lib.PowerScalingFormationModel,
  ), (
      'use_formation_model_with_adaptive_source requires a'
      f' PowerScalingFormationModel, got {type(formation_model)}'
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

  # Calculate P_LH (L-H transition threshold power).
  P_LH, _ = scaling_laws.calculate_P_LH(
      geo=geo,
      core_profiles=core_profiles,
      scaling_law=formation_model.scaling_law,
      divertor_configuration=formation_model.divertor_configuration,
  )
  # Apply the user-specified prefactor.
  P_LH = P_LH * runtime_params.pedestal.formation.P_LH_prefactor

  # Determine transition direction.
  should_enter_h = ~pedestal_transition_state.in_H_mode & (P_SOL > P_LH)
  should_exit_h = pedestal_transition_state.in_H_mode & (P_SOL < P_LH)
  transitioning = should_enter_h | should_exit_h

  # Determine the updated H-mode state.
  in_H_mode = jnp.where(
      should_enter_h,
      True,
      jnp.where(should_exit_h, False, pedestal_transition_state.in_H_mode),
  )

  # Record transition start time when a mode change occurs.
  t = runtime_params.t
  new_transition_start_time = jnp.where(
      transitioning, t, pedestal_transition_state.transition_start_time
  )

  # Capture current pedestal-top values as L-mode baselines when entering
  # H-mode. The pedestal top location is stored in the transition state and
  # updated at the end of each timestep from the pedestal model output, so
  # models like EPEDNN that compute rho_norm_ped_top dynamically are supported.
  rho_norm_ped_top = pedestal_transition_state.rho_norm_ped_top
  ped_top_idx = jnp.argmin(jnp.abs(geo.rho_norm - rho_norm_ped_top))

  current_T_i_at_ped = core_profiles.T_i.value[ped_top_idx]
  current_T_e_at_ped = core_profiles.T_e.value[ped_top_idx]
  current_n_e_at_ped = core_profiles.n_e.value[ped_top_idx]

  # Only update L-mode baselines when entering H-mode.
  new_T_i_ped_L_mode = jnp.where(
      should_enter_h,
      current_T_i_at_ped,
      pedestal_transition_state.T_i_ped_L_mode,
  )
  new_T_e_ped_L_mode = jnp.where(
      should_enter_h,
      current_T_e_at_ped,
      pedestal_transition_state.T_e_ped_L_mode,
  )
  new_n_e_ped_L_mode = jnp.where(
      should_enter_h,
      current_n_e_at_ped,
      pedestal_transition_state.n_e_ped_L_mode,
  )

  return pedestal_transition_state_lib.PedestalTransitionState(
      in_H_mode=in_H_mode,
      transition_start_time=new_transition_start_time,
      T_i_ped_L_mode=new_T_i_ped_L_mode,
      T_e_ped_L_mode=new_T_e_ped_L_mode,
      n_e_ped_L_mode=new_n_e_ped_L_mode,
      rho_norm_ped_top=rho_norm_ped_top,
  )


def pre_step(
    input_state: sim_state.SimState,
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    physics_models: physics_models_lib.PhysicsModels,
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
      source_models=physics_models.source_models,
      neoclassical_models=physics_models.neoclassical_models,
      explicit=True,
  )

  # Execute the edge model if one is configured. The edge model uses the state
  # at time t to calculate new edge conditions for the next time step.
  edge_model = physics_models.edge_model
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

  # Update pedestal transition state if use_formation_model_with_adaptive_source
  # is enabled.
  pedestal_transition_state = input_state.pedestal_transition_state
  if runtime_params_t.pedestal.use_formation_model_with_adaptive_source:
    assert pedestal_transition_state is not None, (
        'pedestal_transition_state must not be None when'
        ' use_formation_model_with_adaptive_source is True.'
    )
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
        physics_models=physics_models,
    )

  return (
      runtime_params_t,
      geo_t,
      explicit_source_profiles,
      edge_outputs,
      pedestal_transition_state,
  )


@functools.partial(
    jax.jit,
    static_argnames=[
        'physics_models',
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
    physics_models: physics_models_lib.PhysicsModels,
    evolving_names: tuple[str, ...],
    input_post_processed_outputs: post_processing.PostProcessedOutputs,
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState | None
    ) = None,
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
          source_models=physics_models.source_models,
          neoclassical_models=physics_models.neoclassical_models,
          evolving_names=evolving_names,
      )
  )
  final_total_transport = (
      transport_coefficients_builder.calculate_all_transport_coeffs(
          physics_models.pedestal_model,
          physics_models.transport_model,
          physics_models.neoclassical_models,
          runtime_params_t_plus_dt,
          geometry_t_plus_dt,
          final_core_profiles,
          final_source_profiles,
          pedestal_transition_state=pedestal_transition_state,
      )
  )

  # Update rho_norm_ped_top in the pedestal transition state from the pedestal
  # model output at t+dt. This ensures that models which compute
  # rho_norm_ped_top dynamically (e.g. EPEDNN) propagate their value to the
  # next timestep's pre_step for accurate L-mode baseline extraction.
  if pedestal_transition_state is not None:
    pedestal_model_output = physics_models.pedestal_model(
        runtime_params_t_plus_dt,
        geometry_t_plus_dt,
        final_core_profiles,
        final_source_profiles,
    )
    pedestal_transition_state = dataclasses.replace(
        pedestal_transition_state,
        rho_norm_ped_top=jnp.array(
            pedestal_model_output.rho_norm_ped_top,
            dtype=jax_utils.get_dtype(),
        ),
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
  )
  post_processed_outputs = post_processing.make_post_processed_outputs(
      sim_state=output_state,
      runtime_params=runtime_params_t_plus_dt,
      previous_post_processed_outputs=input_post_processed_outputs,
  )
  return output_state, post_processed_outputs

