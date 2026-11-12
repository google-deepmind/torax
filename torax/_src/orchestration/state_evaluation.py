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

"""Evaluates derived plasma state variables from SolverState and runtime inputs.

Computes all derived profiles, sources, neoclassical quantities, pedestal and
internal boundary conditions, and transport coefficients that depend on the
primary SolverState and runtime configuration.
"""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any

from absl import logging

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import model_state as model_state_lib
from torax._src import models as models_lib
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import core_profiles as core_profiles_lib
from torax._src.core_profiles import profile_conditions
from torax._src.core_profiles import (
    runtime_params as core_profile_runtime_params,
)
from torax._src.core_profiles import (
    solver_profiles as solver_profiles_lib,
)
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.geometry import geometry
from torax._src.geometry import geometry_state as geometry_state_lib
from torax._src.internal_boundary_conditions import (
    builder as internal_boundary_conditions_builder,
)
from torax._src.internal_boundary_conditions import (
    internal_boundary_conditions as internal_boundary_conditions_lib,
)
from torax._src.neoclassical import (
    neoclassical_intermediates as neoclassical_intermediates_lib,
)
from torax._src.neoclassical import neoclassical_model as neoclassical_model_lib
from torax._src.neoclassical.bootstrap_current import (
    base as bootstrap_current_base,
)
from torax._src.orchestration import (
    explicit_model_outputs as explicit_model_outputs_lib,
)
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.pedestal_model import (
    pedestal_model_output as pedestal_model_output_lib,
)
from torax._src.pedestal_model import pedestal_state as pedestal_state_lib
from torax._src.physics import psi_calculations
from torax._src.sources import core_sources as core_sources_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_state as source_state_lib
from torax._src.transport_model import transport_coefficients_builder
from torax._src.transport_model import transport_coeffs
from torax._src.transport_model import transport_state as transport_state_lib
import xarray as xr


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class EvaluatedState:
  """Derived plasma state variables evaluated from SolverState and inputs.

  Attributes:
    core_profiles: Fully evaluated core plasma profiles.
    core_sources: Fully evaluated core source profiles.
    core_transport: Fully evaluated transport coefficients.
    pedestal_output: Pedestal model output.
    internal_boundary_conditions: Active internal boundary conditions.
  """

  core_profiles: core_profiles_lib.CoreProfiles
  core_sources: core_sources_lib.CoreSources
  core_transport: transport_coeffs.CoreTransport
  pedestal_output: pedestal_model_output_lib.PedestalModelOutput
  internal_boundary_conditions: (
      internal_boundary_conditions_lib.InternalBoundaryConditions
  )


def evaluate_state(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: solver_state_lib.SolverState,
    models: models_lib.Models,
    explicit_model_outputs: (
        explicit_model_outputs_lib.ExplicitModelOutputs | None
    ),
    model_state: model_state_lib.ModelState,
    *,
    use_pereverzev: bool = False,
) -> tuple[EvaluatedState, model_state_lib.ModelState]:
  """Evaluates derived state variables and updated model state from SolverState.

  Args:
    runtime_params: Runtime configuration parameters.
    geo: Geometry of the tokamak.
    solver_state: Primary solvable and prescribed profiles.
    models: Physical models configured for the simulation.
    explicit_model_outputs: Pre-evaluated explicit outputs from time t, or None.
    model_state: Model state carried across time steps.
    use_pereverzev: Whether to calculate Pereverzev-Corrigan stabilization
      transport terms.

  Returns:
    A tuple of (EvaluatedState, updated_model_state).
  """
  explicit = explicit_model_outputs is None
  explicit_source_outputs = (
      explicit_model_outputs.sources
      if explicit_model_outputs is not None
      else None
  )
  explicit_pedestal_structure = (
      explicit_model_outputs.pedestal
      if explicit_model_outputs is not None
      else None
  )
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
      explicit=explicit,
      explicit_source_outputs=explicit_source_outputs,
      calculate_anyway=explicit,
      phi_b_dot=model_state.geometry.Phi_b_dot,
  )

  # Remaining implicit source profiles.
  total_source_profiles, updated_source_states = (
      source_profile_builders.evaluate_dependent_sources(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          source_models=models.source_models,
          source_states=basic_source_states,
          basic_source_outputs=basic_source_outputs,
          bootstrap_current=bootstrap_current,
          phi_b_dot=model_state.geometry.Phi_b_dot,
      )
  )

  # Pedestal model.
  pedestal_output, updated_pedestal_state = models.pedestal_model(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_profiles=total_source_profiles,
      pedestal_state=model_state.pedestal,
      explicit_pedestal_structure=explicit_pedestal_structure,
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

  updated_model_state = model_state_lib.ModelState(
      geometry=model_state.geometry,
      pedestal=updated_pedestal_state,
      transport=updated_transport_state,
      sources=updated_source_states,
      edge=model_state.edge,
  )

  evaluated_state = EvaluatedState(
      core_profiles=core_profiles,
      core_sources=total_source_profiles,
      core_transport=core_transport,
      pedestal_output=pedestal_output,
      internal_boundary_conditions=internal_boundary_conditions,
  )
  return evaluated_state, updated_model_state


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
        Mapping[str, core_sources_lib.SourceOutput] | None
    ) = None,
    calculate_anyway: bool = False,
    phi_b_dot: array_typing.FloatScalar,
) -> tuple[
    core_profiles_lib.CoreProfiles,
    dict[str, core_sources_lib.SourceOutput],
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
    phi_b_dot: Time derivative of the toroidal magnetic flux at the plasma
      boundary (LCFS) [Wb/s].

  Returns:
    A tuple containing:
      - core_profiles: Fully constructed CoreProfiles.
      - basic_source_outputs: Fast-ion and psi source outputs.
      - updated_source_states: Source states updated by basic sources.
      - bootstrap_current: Bootstrap current output.
      - neoclassical_intermediates: Neoclassical intermediates for transport.
  """
  # Construct profiles that depend only on the solver variables.
  solver_profiles = solver_profiles_lib.SolverProfiles.from_solver_state(
      runtime_params=runtime_params,
      geo=geo,
      solver_state=solver_state,
  )

  # Evaluate sources that only depend on the solver profiles. These sources
  # may produce fast ions and current drive.
  basic_source_outputs, updated_source_states = (
      source_profile_builders.evaluate_basic_sources(
          runtime_params=runtime_params,
          geo=geo,
          solver_profiles=solver_profiles,
          source_models=source_models,
          source_states=source_states,
          explicit=explicit,
          calculate_anyway=calculate_anyway,
      )
  )
  if explicit_source_outputs is not None:
    basic_source_outputs = {
        **explicit_source_outputs,
        **basic_source_outputs,
    }
  if runtime_params.numerics.enable_fast_ions:
    fast_ions = profile_conditions.apply_prescribed_fast_ions(
        core_sources_lib.extract_fast_ions(basic_source_outputs),
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

  total_psi_sources = core_sources_lib.sum_total_psi_sources(
      basic_source_outputs=basic_source_outputs,
      source_models=source_models,
      bootstrap_current=bootstrap_current,
      geo=geo,
  )
  match runtime_params.profile_conditions.psi_bc_source:
    case core_profile_runtime_params.PsiBoundaryConditionSource.I_P:
      right_face_constraint = None
      right_face_grad_constraint = (
          solver_state.psidot_right_face_grad_constraint
      )
    case core_profile_runtime_params.PsiBoundaryConditionSource.V_LOOP:
      right_face_constraint = jnp.asarray(
          runtime_params.profile_conditions.v_loop_lcfs,
          dtype=jax_utils.get_dtype(),
      )
      right_face_grad_constraint = None
    case _:
      raise ValueError(
          'Unknown psi_bc_source:'
          f' {runtime_params.profile_conditions.psi_bc_source}'
      )

  # Combine psi sources with core profiles to compute psidot.
  psidot = psi_calculations.calculate_psidot(
      psi=solver_profiles.poloidal_flux.psi,
      sigma=conductivity.sigma,
      psi_sources=total_psi_sources,
      geo=geo,
      evolve_current=runtime_params.numerics.evolve_current,
      prescribed_psidot=runtime_params.profile_conditions.psidot,
      right_face_constraint=right_face_constraint,
      right_face_grad_constraint=right_face_grad_constraint,
      phi_b_dot=phi_b_dot,
  )
  core_profiles = core_profiles_lib.CoreProfiles(
      kinetic=solver_profiles.kinetic,
      ions=solver_profiles.ions,
      poloidal_flux=solver_profiles.poloidal_flux,
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


def _evaluate_ibc_and_transport(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: core_profiles_lib.CoreProfiles,
    pedestal_output: pedestal_model_output_lib.PedestalModelOutput,
    models: models_lib.Models,
    source_profiles: core_sources_lib.CoreSources,
    neoclassical_intermediates: (
        neoclassical_intermediates_lib.NeoclassicalIntermediates
    ),
    transport_state: transport_state_lib.TransportState,
    *,
    use_pereverzev: bool,
) -> tuple[
    internal_boundary_conditions_lib.InternalBoundaryConditions,
    transport_coeffs.CoreTransport,
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


def get_child(datatree: xr.DataTree | None, key: str) -> xr.DataTree | None:
  """Returns child node `key` of `datatree`, or None if either is absent."""
  return datatree.children.get(key) if datatree is not None else None


def restore_or_cold_start(
    datatree: xr.DataTree | None,
    restore_fn: Callable[[xr.DataTree], Any],
    cold_start_fn: Callable[[], Any],
    *,
    label: str,
    t: object,
) -> tuple[Any, bool]:
  """Restores state from `datatree` if present and compatible, else cold-starts.

  Args:
    datatree: Time-sliced DataTree for the model, or None.
    restore_fn: Restores state from a non-empty `datatree`.
    cold_start_fn: Produces a fresh state when saved state is absent or invalid.
    label: Human-readable model/state label for warning logs.
    t: Restart time for warning logs.

  Returns:
    A tuple of (state, was_restored).
  """
  if output_grid_context.has_saved_state(datatree):
    try:
      return restore_fn(datatree), True
    except ValueError as err:
      logging.warning(
          'Incompatible saved %s at t=%s (%s); initializing fresh state.',
          label,
          t,
          err,
      )
      return cold_start_fn(), False
  state = cold_start_fn()
  if jax.tree.leaves(state):
    logging.warning(
        'No saved %s at t=%s; initializing fresh state.',
        label,
        t,
    )
  return state, False


def _restore_state(
    model: Any,
    datatree: xr.DataTree | None,
    runtime_params: runtime_params_lib.RuntimeParams,
    *,
    label: str,
) -> tuple[Any, bool]:
  """Restores one model's saved state, or cold-starts it."""
  return restore_or_cold_start(
      datatree,
      lambda dt: model.initial_state_from_datatree(dt, runtime_params),
      lambda: model.initial_state(runtime_params),
      label=label,
      t=runtime_params.t,
  )


def _restore_states(
    named_models: Mapping[str, Any],
    datatree: xr.DataTree | None,
    runtime_params: runtime_params_lib.RuntimeParams,
    *,
    kind: str,
) -> tuple[dict[str, Any], set[str]]:
  """Restores saved per-model states, or cold-starts models without one."""
  states: dict[str, Any] = {}
  restored: set[str] = set()
  for name, model in named_models.items():
    states[name], was_restored = _restore_state(
        model,
        get_child(datatree, name),
        runtime_params,
        label=f'{kind} {name!r} state',
    )
    if was_restored:
      restored.add(name)
  return states, restored


def _keep_restored(
    seed_states: Mapping[str, Any],
    restored: set[str],
    evaluated_states: Mapping[str, Any],
) -> dict[str, Any]:
  """Keeps restored states (for explicit models) and evaluated cold-starts."""
  return {
      name: seed_states[name] if name in restored else evaluated_states[name]
      for name in seed_states
  }


def _restore_geometry_state(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    models: models_lib.Models,
    restart_datatree: xr.DataTree,
) -> geometry_state_lib.GeometryState:
  """Restores GeometryState using a saved Phi_b_dot if available."""
  if runtime_params.geometry.calc_Phi_b_dot:
    scalars = restart_datatree.children[output_keys.SCALARS].dataset.squeeze()
    if output_keys.PHI_B_DOT in scalars.data_vars:
      return geometry_state_lib.GeometryState.from_geometry(
          geo,
          Phi_b_dot=jnp.asarray(
              scalars.data_vars[output_keys.PHI_B_DOT].to_numpy(),
              dtype=jax_utils.get_dtype(),
          ),
      )
  _, geometry_state = models.geometry_model.initialize(runtime_params)
  return geometry_state


def _restore_pedestal_state(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: core_profiles_lib.CoreProfiles,
    models: models_lib.Models,
    pedestal_tree: xr.DataTree | None,
) -> pedestal_state_lib.PedestalState | None:
  """Restores PedestalState from `pedestal_tree`, or returns None to init."""
  if not output_grid_context.has_saved_state(pedestal_tree):
    logging.warning(
        'No saved pedestal state at t=%s; initializing fresh state.',
        runtime_params.t,
    )
    return None
  try:
    top_level_state = models.pedestal_model.top_level_state_from_datatree(
        pedestal_tree
    )
  except ValueError as err:
    logging.warning(
        'Incompatible saved pedestal state at t=%s (%s); initializing fresh'
        ' state.',
        runtime_params.t,
        err,
    )
    return None

  def _restore_sub(sub_model: Any, key: str, label: str) -> Any:
    state, _ = restore_or_cold_start(
        get_child(pedestal_tree, key),
        lambda dt: sub_model.initial_state_from_datatree(dt, runtime_params),
        lambda: sub_model.initial_state(runtime_params, geo, core_profiles),
        label=label,
        t=runtime_params.t,
    )
    return state

  return dataclasses.replace(
      top_level_state,
      structure=_restore_sub(
          models.pedestal_model.structure_model,
          output_keys.PEDESTAL_STRUCTURE,
          'pedestal structure state',
      ),
      formation=_restore_sub(
          models.pedestal_model.formation_model,
          output_keys.PEDESTAL_FORMATION,
          'pedestal formation state',
      ),
  )


def evaluate_initial_state(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: solver_state_lib.SolverState,
    models: models_lib.Models,
    restart_datatree: xr.DataTree | None = None,
) -> tuple[EvaluatedState, model_state_lib.ModelState]:
  """Evaluates derived state variables and initial ModelState.

  On a fresh start, every stateful model is cold-started from the plasma state
  at `runtime_params.t`. On a file restart, each model's saved state in
  `restart_datatree` is restored when present and compatible; otherwise the
  model is cold-started and a warning is logged.

  Args:
    runtime_params: Runtime configuration parameters at the (re)start time.
    geo: Magnetic geometry at the (re)start time.
    solver_state: Initial primary profiles.
    models: Physical models configured for the simulation.
    restart_datatree: Time-sliced output DataTree to restart from, or None for a
      fresh start.

  Returns:
    A tuple of (EvaluatedState, initial_model_state).
  """
  model_state_tree = get_child(restart_datatree, output_keys.MODEL_STATE)

  # Geometry and sources.
  if restart_datatree is None:
    _, geometry_state = models.geometry_model.initialize(runtime_params)
    seed_source_states = models.source_models.initial_state(runtime_params)
    restored_sources: set[str] = set()
  else:
    geometry_state = _restore_geometry_state(
        runtime_params, geo, models, restart_datatree
    )
    seed_source_states, restored_sources = _restore_states(
        models.source_models.standard_sources,
        get_child(model_state_tree, output_keys.SOURCES),
        runtime_params,
        kind='source',
    )

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
      source_states=seed_source_states,
      explicit=True,
      calculate_anyway=True,
      phi_b_dot=geometry_state.Phi_b_dot,
  )
  total_source_profiles, evaluated_source_states = (
      source_profile_builders.evaluate_dependent_sources(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          source_models=models.source_models,
          source_states=basic_source_states,
          basic_source_outputs=basic_source_outputs,
          bootstrap_current=bootstrap_current,
          phi_b_dot=geometry_state.Phi_b_dot,
      )
  )

  # Pedestal.
  if restart_datatree is None:
    pedestal_state = None
  else:
    pedestal_state = _restore_pedestal_state(
        runtime_params,
        geo,
        core_profiles,
        models,
        get_child(model_state_tree, output_keys.PEDESTAL),
    )
  if pedestal_state is None:
    pedestal_output, pedestal_state = models.pedestal_model.initialize(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_profiles=total_source_profiles,
    )
  else:
    pedestal_output, _ = models.pedestal_model(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_profiles=total_source_profiles,
        pedestal_state=pedestal_state,
    )

  # Transport.
  if restart_datatree is None:
    seed_transport_state = models.transport_model.initial_state(runtime_params)
    restored_core_transport: set[str] = set()
    restored_ped_transport: set[str] = set()
  else:
    transport_tree = get_child(model_state_tree, output_keys.TRANSPORT)
    seed_core_transport, restored_core_transport = _restore_states(
        models.transport_model.core_transport_models,
        get_child(transport_tree, output_keys.CORE),
        runtime_params,
        kind='core transport model',
    )
    seed_ped_transport, restored_ped_transport = _restore_states(
        models.transport_model.pedestal_transport_models,
        get_child(transport_tree, output_keys.PEDESTAL),
        runtime_params,
        kind='pedestal transport model',
    )
    seed_transport_state = transport_state_lib.TransportState(
        core_transport_states=seed_core_transport,
        pedestal_transport_states=seed_ped_transport,
    )
  (
      internal_boundary_conditions,
      core_transport,
      evaluated_transport_state,
  ) = _evaluate_ibc_and_transport(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      pedestal_output=pedestal_output,
      models=models,
      source_profiles=total_source_profiles,
      neoclassical_intermediates=neoclassical_intermediates,
      transport_state=seed_transport_state,
      use_pereverzev=False,
  )

  evaluated_state = EvaluatedState(
      core_profiles=core_profiles,
      core_sources=total_source_profiles,
      core_transport=core_transport,
      pedestal_output=pedestal_output,
      internal_boundary_conditions=internal_boundary_conditions,
  )
  initial_model_state = model_state_lib.ModelState(
      geometry=geometry_state,
      pedestal=pedestal_state,
      transport=transport_state_lib.TransportState(
          core_transport_states=_keep_restored(
              seed_transport_state.core_transport_states,
              restored_core_transport,
              evaluated_transport_state.core_transport_states,
          ),
          pedestal_transport_states=_keep_restored(
              seed_transport_state.pedestal_transport_states,
              restored_ped_transport,
              evaluated_transport_state.pedestal_transport_states,
          ),
      ),
      sources=_keep_restored(
          seed_source_states, restored_sources, evaluated_source_states
      ),
  )
  return evaluated_state, initial_model_state
