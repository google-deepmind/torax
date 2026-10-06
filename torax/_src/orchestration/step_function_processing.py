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
from torax._src import models as models_lib
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import updaters
from torax._src.edge import base as edge_base
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.internal_boundary_conditions import builder as internal_boundary_conditions_builder
from torax._src.orchestration import sim_state
from torax._src.output_tools import post_processing
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.time_step_calculator import time_step_calculator_state as time_step_calculator_state_lib
from torax._src.transport_model import transport_coefficients_builder

# pylint: disable=invalid-name


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
    pedestal_transition_state_lib.PedestalTransitionState,
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
      explicit=True,
  )

  # Update core sources with any newly calculated explicit sources.
  # This is because in input_state, the sources are those which were
  # used to compute the state. For explicit sources, these were computed with
  # core_profiles at time t_minus_dt, whereas the implicit sources are
  # consistent with time t. For the edge and pedestal models, we want all
  # sources consistent with the state at time t, so we replace the explicit
  # sources with the newly calculated profiles.
  merged_sources = dataclasses.replace(
      input_state.core_sources,
      T_e=input_state.core_sources.T_e | explicit_source_profiles.T_e,
      T_i=input_state.core_sources.T_i | explicit_source_profiles.T_i,
      n_e=input_state.core_sources.n_e | explicit_source_profiles.n_e,
      psi=input_state.core_sources.psi | explicit_source_profiles.psi,
  )

  # Execute the edge model if one is configured. The edge model uses the state
  # at time t to calculate new edge conditions for the next time step.
  edge_model = models.edge_model
  if edge_model is not None:
    edge_outputs = edge_model(
        runtime_params_t,
        geo_t,
        input_state.core_profiles,
        merged_sources,
        previous_edge_outputs=input_state.edge_outputs,
    )
  else:
    edge_outputs = None

  # Update the discrete pedestal transition state (confinement mode, transition
  # start time, and L-mode baselines) once per step at time t.
  pedestal_transition_state = models.pedestal_model.update_transition_state(
      runtime_params=runtime_params_t,
      geo=geo_t,
      core_profiles=input_state.core_profiles,
      source_profiles=merged_sources,
      pedestal_transition_state=input_state.pedestal_transition_state,
  )

  # When explicit_pedestal is True, evaluate the pedestal model once here
  # and freeze its output for the solver loop. calc_coeffs will skip
  # re-evaluation and use this stored output.
  if runtime_params_t.pedestal.explicit_pedestal:
    pedestal_model_output = models.pedestal_model(
        runtime_params_t,
        geo_t,
        input_state.core_profiles,
        merged_sources,
        pedestal_transition_state,
    )
    pedestal_transition_state = dataclasses.replace(
        pedestal_transition_state,
        pedestal_model_output=pedestal_model_output,
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
          neoclassical_model=models.neoclassical_model,
          evolving_names=evolving_names,
      )
  )
  # Compute pedestal model output and store on transition state.
  # Also update previous_pedestal_model_output if needed for EPED-NN
  # next-timestep width calculation.
  final_pedestal_model_output = models.pedestal_model(
      runtime_params_t_plus_dt,
      geometry_t_plus_dt,
      final_core_profiles,
      final_source_profiles,
      pedestal_transition_state,
  )
  pedestal_transition_state = dataclasses.replace(
      pedestal_transition_state,
      pedestal_model_output=final_pedestal_model_output,
      previous_pedestal_model_output=final_pedestal_model_output,
  )
  final_neoclassical_outputs = models.neoclassical_model(
      runtime_params_t_plus_dt, geometry_t_plus_dt, final_core_profiles
  )
  internal_boundary_conditions = (
      internal_boundary_conditions_builder.build_internal_boundary_conditions(
          runtime_params=runtime_params_t_plus_dt,
          geo=geometry_t_plus_dt,
          core_profiles=final_core_profiles,
          pedestal_transition_state=pedestal_transition_state,
          internal_boundary_condition_model=(
              models.internal_boundary_condition_model
          ),
          source_profiles=final_source_profiles,
      )
  )
  final_total_transport = (
      transport_coefficients_builder.calculate_all_transport_coeffs(
          transport_model=models.transport_model,
          runtime_params=runtime_params_t_plus_dt,
          geo=geometry_t_plus_dt,
          core_profiles=final_core_profiles,
          pedestal_transition_state=pedestal_transition_state,
          neoclassical_transport=final_neoclassical_outputs.transport,
          two_point_mask=internal_boundary_conditions.get_two_point_face_mask(
              geometry_t_plus_dt
          ),
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
