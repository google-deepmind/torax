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
from torax._src import model_state as model_state_lib
from torax._src import models as models_lib
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.edge import base as edge_base
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.orchestration import (
    explicit_model_outputs as explicit_model_outputs_lib,
)
from torax._src.orchestration import physics_pipeline
from torax._src.orchestration import sim_state
from torax._src.output_tools import post_processing
from torax._src.physics import psi_calculations
from torax._src.sources import source_profile_builders
from torax._src.time_step_calculator import time_step_calculator_state as time_step_calculator_state_lib

# pylint: disable=invalid-name


def pre_step(
    input_state: sim_state.SimState,
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    models: models_lib.Models,
) -> tuple[
    runtime_params_lib.RuntimeParams,
    geometry.Geometry,
    explicit_model_outputs_lib.ExplicitModelOutputs,
    edge_base.EdgeModelOutputs | None,
    model_state_lib.ModelState,
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
  explicit_source_outputs, source_states = (
      source_profile_builders.build_basic_sources(
          runtime_params=runtime_params_t,
          geo=geo_t,
          core_profiles=input_state.core_profiles,
          source_models=models.source_models,
          source_states=input_state.model_state.sources,
          explicit=True,
      )
  )

  # Update core sources with any newly calculated explicit sources.
  # This is because in input_state, the sources are those which were
  # used to compute the state. For explicit sources, these were computed with
  # core_profiles at time t_minus_dt, whereas the implicit sources are
  # consistent with time t. For the edge and pedestal models, we want all
  # sources consistent with the state at time t, so we replace the explicit
  # sources with the newly calculated profiles.
  merged_sources = input_state.core_sources.update_source_profiles(
      explicit_source_outputs,
      models.source_models,
  )

  # Execute the edge model if one is configured. The edge model uses the state
  # at time t to calculate new edge conditions for the next time step.
  edge_model = models.edge_model
  if edge_model is not None:
    assert runtime_params_t.edge is not None
    edge_outputs = edge_model(
        runtime_params_t.edge,
        runtime_params_t,
        geo_t,
        input_state.core_profiles,
        merged_sources,
        previous_edge_outputs=input_state.edge_outputs,
    )
  else:
    edge_outputs = None

  # Update the discrete pedestal state and compute explicit structure if needed.
  updated_pedestal_state, explicit_pedestal_structure = (
      models.pedestal_model.pre_step(
          runtime_params=runtime_params_t,
          geo=geo_t,
          core_profiles=input_state.core_profiles,
          source_profiles=merged_sources,
          pedestal_state=input_state.model_state.pedestal,
      )
  )

  model_state_t = dataclasses.replace(
      input_state.model_state,
      pedestal=updated_pedestal_state,
      sources=source_states,
  )
  explicit_model_outputs = explicit_model_outputs_lib.ExplicitModelOutputs(
      sources=explicit_source_outputs,
      pedestal=explicit_pedestal_structure,
  )
  return (
      runtime_params_t,
      geo_t,
      explicit_model_outputs,
      edge_outputs,
      model_state_t,
  )


@jax.jit(
    static_argnames=[
        'models',
        'evolving_names',
    ],
    inline=jax.Inline.XLA_LATE,
)
def finalize_outputs(
    t: jax.Array,
    dt: jax.Array,
    x_new: tuple[cell_variable.CellVariable, ...],
    solver_numeric_outputs: state.SolverNumericOutputs,
    geometry_t_plus_dt: geometry.Geometry,
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    core_profiles_t: state.CoreProfiles,
    solver_state_t_plus_dt: solver_state_lib.SolverState,
    explicit_model_outputs: explicit_model_outputs_lib.ExplicitModelOutputs,
    edge_outputs: edge_base.EdgeModelOutputs | None,
    models: models_lib.Models,
    evolving_names: tuple[str, ...],
    input_post_processed_outputs: post_processing.PostProcessedOutputs,
    time_step_calculator_state_t: time_step_calculator_state_lib.TimeStepCalculatorState,
    model_state_t: model_state_lib.ModelState,
) -> tuple[sim_state.SimState, post_processing.PostProcessedOutputs]:
  """Returns the final state and post-processed outputs."""
  updated_solver_state = convertors.solver_x_tuple_to_solver_state(
      x_new,
      evolving_names,
      solver_state_t_plus_dt,
  )
  time_reduction_factor = (
      runtime_params_t_plus_dt.numerics.current_diffusion_time_reduction_factor
  )
  profile_conditions = runtime_params_t_plus_dt.profile_conditions
  if profile_conditions.use_v_loop_lcfs_boundary_condition:
    v_loop_lcfs = profile_conditions.v_loop_lcfs
  else:
    v_loop_lcfs = psi_calculations.calculate_v_loop_lcfs_from_psi(
        core_profiles_t.flux.psi,
        updated_solver_state.psi,
        dt,
        current_diffusion_time_reduction_factor=time_reduction_factor,
    )
  physics_outputs, final_model_state = physics_pipeline.evaluate_step_physics(
      runtime_params=runtime_params_t_plus_dt,
      geo=geometry_t_plus_dt,
      solver_state=updated_solver_state,
      models=models,
      explicit_model_outputs=explicit_model_outputs,
      model_state=model_state_t,
      v_loop_lcfs=v_loop_lcfs,
      use_pereverzev=False,
  )
  output_state = sim_state.SimState(
      t=t + dt,
      dt=dt,
      core_profiles=physics_outputs.core_profiles,
      core_sources=physics_outputs.core_sources,
      core_transport=physics_outputs.core_transport,
      geometry=geometry_t_plus_dt,
      solver_numeric_outputs=solver_numeric_outputs,
      edge_outputs=edge_outputs,
      pedestal_output=physics_outputs.pedestal_output,
      model_state=final_model_state,
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
