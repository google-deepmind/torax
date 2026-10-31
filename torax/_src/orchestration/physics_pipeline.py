# Copyright 2026 DeepMind Technologies Limited
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

"""Unified physics pipeline for TORAX.

Evaluates base profiles, fast ions and psi sources, neoclassical current,
psidot, remaining sources, pedestal and internal boundary conditions,
and neoclassical and turbulent transport coefficients.
"""

import dataclasses

import jax
from torax._src import array_typing
from torax._src import model_state as model_state_lib
from torax._src import models as models_lib
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import builders
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import builder as internal_boundary_conditions_builder
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib
from torax._src.neoclassical import neoclassical_intermediates as neoclassical_intermediates_lib
from torax._src.orchestration import explicit_model_outputs as explicit_model_outputs_lib
from torax._src.pedestal_model import (
    pedestal_model_output as pedestal_model_output_lib,
)
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.transport_model import transport_coefficients_builder
from torax._src.transport_model import transport_state as transport_state_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PhysicsOutputs:
  """Outputs from evaluating the physics pipeline.

  Attributes:
    core_profiles: Fully evaluated core plasma profiles.
    core_sources: Fully evaluated core source profiles.
    core_transport: Fully evaluated transport coefficients.
    pedestal_output: Pedestal model output.
    internal_boundary_conditions: Active internal boundary conditions.
  """

  core_profiles: state.CoreProfiles
  core_sources: source_profiles_lib.SourceProfiles
  core_transport: state.CoreTransport
  pedestal_output: pedestal_model_output_lib.PedestalModelOutput
  internal_boundary_conditions: (
      internal_boundary_conditions_lib.InternalBoundaryConditions
  )


def evaluate_step_physics(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: state.SolverState,
    models: models_lib.Models,
    explicit_model_outputs: explicit_model_outputs_lib.ExplicitModelOutputs,
    model_state: model_state_lib.ModelState,
    *,
    v_loop_lcfs: array_typing.FloatScalar | None = None,
    use_pereverzev: bool = False,
) -> tuple[PhysicsOutputs, model_state_lib.ModelState]:
  """Evaluates the unified step physics pipeline.

  Args:
    runtime_params: Runtime configuration parameters.
    geo: Geometry of the tokamak.
    solver_state: Primary solvable and prescribed profiles.
    models: Physical models configured for the simulation.
    explicit_model_outputs: Pre-evaluated explicit outputs from time t.
    model_state: Model state carried across time steps.
    v_loop_lcfs: Optional loop voltage at the LCFS [V].
    use_pereverzev: Whether to calculate Pereverzev-Corrigan stabilization
      transport terms.

  Returns:
    A tuple of (PhysicsOutputs, next_model_state).
  """
  # Base core profiles, fast ions and psi sources, neoclassical current and
  # psidot.
  (
      core_profiles,
      basic_source_outputs,
      basic_source_states,
      bootstrap_current,
      neoclassical_intermediates,
  ) = builders.build_core_profiles(
      runtime_params=runtime_params,
      geo=geo,
      solver_state=solver_state,
      source_models=models.source_models,
      neoclassical_model=models.neoclassical_model,
      source_states=model_state.sources,
      explicit=False,
      explicit_source_outputs=explicit_model_outputs.sources,
      calculate_anyway=False,
      v_loop_lcfs=v_loop_lcfs,
  )

  # Remaining implicit source profiles.
  total_source_profiles, final_source_states = (
      source_profile_builders.build_remaining_sources(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          source_models=models.source_models,
          source_states=basic_source_states,
          basic_source_outputs=basic_source_outputs,
          bootstrap_current=bootstrap_current,
      )
  )

  # Pedestal model and internal boundary conditions.
  pedestal_output, candidate_pedestal_state = models.pedestal_model(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_profiles=total_source_profiles,
      pedestal_state=model_state.pedestal,
      explicit_pedestal_structure=explicit_model_outputs.pedestal,
  )
  (
      internal_boundary_conditions,
      core_transport,
      updated_transport_state,
  ) = _evaluate_ibc_and_transport(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      pedestal_output=pedestal_output,
      models=models,
      source_profiles=total_source_profiles,
      neoclassical_intermediates=neoclassical_intermediates,
      transport_state=model_state.transport,
      use_pereverzev=use_pereverzev,
  )

  next_model_state = model_state_lib.ModelState(
      pedestal=candidate_pedestal_state,
      transport=updated_transport_state,
      sources=final_source_states,
  )

  outputs = PhysicsOutputs(
      core_profiles=core_profiles,
      core_sources=total_source_profiles,
      core_transport=core_transport,
      pedestal_output=pedestal_output,
      internal_boundary_conditions=internal_boundary_conditions,
  )
  return outputs, next_model_state


def _evaluate_ibc_and_transport(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    pedestal_output: pedestal_model_output_lib.PedestalModelOutput,
    models: models_lib.Models,
    source_profiles: source_profiles_lib.SourceProfiles,
    neoclassical_intermediates: (
        neoclassical_intermediates_lib.NeoclassicalIntermediates
    ),
    transport_state: transport_state_lib.TransportState,
    use_pereverzev: bool,
) -> tuple[
    internal_boundary_conditions_lib.InternalBoundaryConditions,
    state.CoreTransport,
    transport_state_lib.TransportState,
]:
  """Evaluates internal boundary conditions and transport coefficients."""
  internal_boundary_conditions = (
      internal_boundary_conditions_builder.build_internal_boundary_conditions(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          pedestal_model_output=pedestal_output,
          internal_boundary_condition_model=(
              models.internal_boundary_condition_model
          ),
          source_profiles=source_profiles,
      )
  )

  # Neoclassical and turbulent transport coefficients.
  neoclassical_transport = models.neoclassical_model.calculate_transport(
      runtime_params,
      geo,
      core_profiles,
      neoclassical_intermediates,
  )
  core_transport, updated_transport_state = (
      transport_coefficients_builder.calculate_all_transport_coeffs(
          transport_model=models.transport_model,
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          pedestal_model_output=pedestal_output,
          transport_state=transport_state,
          neoclassical_transport=neoclassical_transport,
          two_point_mask=internal_boundary_conditions.get_two_point_face_mask(
              geo
          ),
          use_pereverzev=use_pereverzev,
      )
  )
  return internal_boundary_conditions, core_transport, updated_transport_state


def evaluate_initial_physics(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: state.SolverState,
    models: models_lib.Models,
) -> tuple[PhysicsOutputs, model_state_lib.ModelState]:
  """Evaluates the physics pipeline at t = 0 to establish the initial state.

  Args:
    runtime_params: Runtime configuration parameters at t = 0.
    geo: Magnetic geometry at t = 0.
    solver_state: Initial primary profiles.
    models: Physical models configured for the simulation.

  Returns:
    A tuple of (PhysicsOutputs, initial_model_state).
  """
  initial_source_states = models.source_models.initial_state(runtime_params)
  (
      core_profiles,
      basic_source_outputs,
      basic_source_states,
      bootstrap_current,
      neoclassical_intermediates,
  ) = builders.build_core_profiles(
      runtime_params=runtime_params,
      geo=geo,
      solver_state=solver_state,
      source_models=models.source_models,
      neoclassical_model=models.neoclassical_model,
      source_states=initial_source_states,
      explicit=True,
      calculate_anyway=True,
  )
  total_source_profiles, final_source_states = (
      source_profile_builders.build_remaining_sources(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          source_models=models.source_models,
          source_states=basic_source_states,
          basic_source_outputs=basic_source_outputs,
          bootstrap_current=bootstrap_current,
      )
  )
  initial_pedestal_output, initial_pedestal_state = (
      models.pedestal_model.initialize(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          source_profiles=total_source_profiles,
      )
  )
  (
      internal_boundary_conditions,
      core_transport,
      initial_transport_state,
  ) = _evaluate_ibc_and_transport(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      pedestal_output=initial_pedestal_output,
      models=models,
      source_profiles=total_source_profiles,
      neoclassical_intermediates=neoclassical_intermediates,
      transport_state=models.transport_model.initial_state(runtime_params),
      use_pereverzev=False,
  )
  outputs = PhysicsOutputs(
      core_profiles=core_profiles,
      core_sources=total_source_profiles,
      core_transport=core_transport,
      pedestal_output=initial_pedestal_output,
      internal_boundary_conditions=internal_boundary_conditions,
  )
  initial_model_state = model_state_lib.ModelState(
      pedestal=initial_pedestal_state,
      transport=initial_transport_state,
      sources=final_source_states,
  )
  return outputs, initial_model_state
