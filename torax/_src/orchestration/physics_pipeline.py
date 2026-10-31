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

from collections.abc import Mapping
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import model_state as model_state_lib
from torax._src import models as models_lib
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profile_conditions
from torax._src.core_profiles import profiles
from torax._src.core_profiles import (
    solver_profiles as solver_profiles_lib,
)
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import builder as internal_boundary_conditions_builder
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib
from torax._src.neoclassical import neoclassical_intermediates as neoclassical_intermediates_lib
from torax._src.neoclassical import neoclassical_model as neoclassical_model_lib
from torax._src.neoclassical.bootstrap_current import (
    base as bootstrap_current_base,
)
from torax._src.orchestration import explicit_model_outputs as explicit_model_outputs_lib
from torax._src.pedestal_model import (
    pedestal_model_output as pedestal_model_output_lib,
)
from torax._src.physics import psi_calculations
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.sources import source_state as source_state_lib
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

  core_profiles: profiles.CoreProfiles
  core_sources: source_profiles_lib.SourceProfiles
  core_transport: state.CoreTransport
  pedestal_output: pedestal_model_output_lib.PedestalModelOutput
  internal_boundary_conditions: (
      internal_boundary_conditions_lib.InternalBoundaryConditions
  )


def _build_core_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: solver_state_lib.SolverState,
    source_models: source_models_lib.SourceModels,
    neoclassical_model: neoclassical_model_lib.NeoclassicalModel,
    source_states: Mapping[str, source_state_lib.SourceState],
    *,
    explicit: bool,
    explicit_source_outputs: (
        Mapping[str, source_profiles_lib.SourceOutput] | None
    ) = None,
    calculate_anyway: bool = False,
    v_loop_lcfs: array_typing.FloatScalar | None = None,
) -> tuple[
    profiles.CoreProfiles,
    dict[str, source_profiles_lib.SourceOutput],
    dict[str, source_state_lib.SourceState],
    bootstrap_current_base.BootstrapCurrent,
    neoclassical_intermediates_lib.NeoclassicalIntermediates,
]:
  """Builds core profiles, basic source outputs, and neoclassical outputs.

  Args:
    runtime_params: Runtime configuration parameters.
    geo: Geometry of the tokamak.
    solver_state: The current set of primary profiles.
    source_models: Source models.
    neoclassical_model: Neoclassical model.
    source_states: Source states mapping.
    explicit: Whether to evaluate explicit sources.
    explicit_source_outputs: Pre-evaluated explicit source outputs, if any.
    calculate_anyway: Whether to calculate sources even if inactive.
    v_loop_lcfs: Optional loop voltage at the LCFS [V].

  Returns:
    A tuple containing:
      - core_profiles: Fully constructed CoreProfiles.
      - basic_source_outputs: Fast-ion and psi source outputs.
      - updated_source_states: Source states updated by basic sources.
      - bootstrap_current: Bootstrap current output.
      - neoclassical_intermediates: Neoclassical intermediates for transport.
  """
  if (
      v_loop_lcfs is None
      and runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
  ):
    v_loop_lcfs = jnp.asarray(
        runtime_params.profile_conditions.v_loop_lcfs,
        dtype=jax_utils.get_dtype(),
    )

  # Construct profiles that depend only on the solver variables.
  solver_profiles = solver_profiles_lib.SolverProfiles.from_solver_state(
      runtime_params=runtime_params,
      geo=geo,
      solver_state=solver_state,
      v_loop_lcfs=v_loop_lcfs,
  )

  # Evaluate sources that only depend on the solver profiles. These sources
  # may produce fast ions and current drive.
  source_outputs_pass, updated_source_states = (
      source_profile_builders.build_basic_sources(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=solver_profiles,
          source_models=source_models,
          source_states=source_states,
          explicit=explicit,
          calculate_anyway=calculate_anyway,
      )
  )
  if explicit_source_outputs is not None:
    basic_source_outputs = {
        **explicit_source_outputs,
        **source_outputs_pass,
    }
  else:
    basic_source_outputs = source_outputs_pass
  if runtime_params.numerics.enable_fast_ions:
    fast_ions = profile_conditions.apply_prescribed_fast_ions(
        source_profile_builders.extract_fast_ions(basic_source_outputs),
        runtime_params.profile_conditions.prescribed_fast_ions,
        geo.rho_face_norm,
    )
  else:
    fast_ions = ()

  # Evaluate neoclassical conductivity, bootstrap current, and poloidal
  # velocity. These depend on fast ions.
  (
      conductivity,
      bootstrap_current,
      poloidal_velocity,
      neoclassical_intermediates,
  ) = neoclassical_model.calculate_conductivity_current_and_rotation(
      runtime_params, geo, solver_profiles, fast_ions=fast_ions
  )

  total_psi_sources = source_profile_builders.sum_total_psi_sources(
      basic_source_outputs=basic_source_outputs,
      source_models=source_models,
      bootstrap_current=bootstrap_current,
      geo=geo,
  )
  # Combine psi sources with core profiles to compute psidot.
  psidot = psi_calculations.calculate_psidot(
      psi=solver_profiles.flux.psi,
      sigma=conductivity.sigma,
      psi_sources=total_psi_sources,
      geo=geo,
      evolve_current=runtime_params.numerics.evolve_current,
      prescribed_psidot=runtime_params.profile_conditions.psidot,
      v_loop_lcfs=v_loop_lcfs,
  )
  core_profiles = profiles.CoreProfiles(
      kinetic=solver_profiles.kinetic,
      ions=solver_profiles.ions,
      flux=solver_profiles.flux,
      toroidal_angular_velocity=solver_profiles.toroidal_angular_velocity,
      fast_ions=fast_ions,
      conductivity=conductivity,
      poloidal_velocity=poloidal_velocity.poloidal_velocity,
      psidot=psidot,
  )

  return (
      core_profiles,
      basic_source_outputs,
      updated_source_states,
      bootstrap_current,
      neoclassical_intermediates,
  )


def evaluate_step_physics(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: solver_state_lib.SolverState,
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
  ) = _build_core_profiles(
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
    core_profiles: profiles.CoreProfiles,
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
    solver_state: solver_state_lib.SolverState,
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
  ) = _build_core_profiles(
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
