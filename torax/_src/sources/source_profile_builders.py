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

"""Functions for building source profiles in TORAX."""

from collections.abc import Mapping
import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.bootstrap_current import (
    base as bootstrap_current_base,
)
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.physics import psi_calculations
from torax._src.sources import source as source_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profiles
from torax._src.sources import source_state as source_state_lib


def _update_standard_source_profiles(
    calculated_source_profiles: source_profiles.SourceProfiles,
    source_name: str,
    affected_core_profiles: tuple[source_lib.AffectedCoreProfile, ...],
    output: source_profiles.SourceOutput,
):
  """Updates the standard source profiles in calculated_source_profiles."""
  for profile, affected_core_profile in zip(
      output.profiles, affected_core_profiles, strict=True
  ):
    jax_profile = jnp.asarray(profile)
    match affected_core_profile:
      case source_lib.AffectedCoreProfile.PSI:
        calculated_source_profiles.psi[source_name] = jax_profile
      case source_lib.AffectedCoreProfile.NE:
        calculated_source_profiles.n_e[source_name] = jax_profile
      case source_lib.AffectedCoreProfile.TEMP_ION:
        calculated_source_profiles.T_i[source_name] = jax_profile
      case source_lib.AffectedCoreProfile.TEMP_EL:
        calculated_source_profiles.T_e[source_name] = jax_profile
  if output.fast_ions:
    calculated_source_profiles.fast_ions[source_name] = output.fast_ions


# TODO(b/572586768): Remove temporary compatibility helper once
# initialization / callers migrate.
def _evaluate_basic_sources(
    sources_to_evaluate: Mapping[str, source_lib.Source],
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.BaseCoreProfiles,
    calculated_source_profiles: source_profiles.SourceProfiles,
    source_states: Mapping[str, source_state_lib.SourceState],
    updated_source_states: dict[str, source_state_lib.SourceState],
    explicit: bool,
    calculate_anyway: bool = False,
) -> None:
  """Evaluates basic sources and updates profiles and source states."""
  for source_name, source in sources_to_evaluate.items():
    source_params = runtime_params.sources[source_name]
    if (explicit == source_params.is_explicit) | calculate_anyway:
      curr_state = source_states[source_name]
      output, new_state = source.get_value(
          runtime_params,
          geo,
          core_profiles,
          curr_state,
      )
      updated_source_states[source_name] = new_state
      _update_standard_source_profiles(
          calculated_source_profiles,
          source_name,
          source.AFFECTED_CORE_PROFILES,
          output,
      )


def _evaluate_core_profiles_sources(
    sources_to_evaluate: Mapping[str, source_lib.SourceUsingCoreProfiles],
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    calculated_source_profiles: source_profiles.SourceProfiles,
    source_states: Mapping[str, source_state_lib.SourceState],
    updated_source_states: dict[str, source_state_lib.SourceState],
) -> None:
  """Evaluates CoreProfiles-dependent sources and updates profiles and state."""
  for source_name, source in sources_to_evaluate.items():
    curr_state = source_states[source_name]
    output, new_state = source.get_value(
        runtime_params,
        geo,
        core_profiles,
        curr_state,
    )
    updated_source_states[source_name] = new_state
    _update_standard_source_profiles(
        calculated_source_profiles,
        source_name,
        source.AFFECTED_CORE_PROFILES,
        output,
    )


def _evaluate_final_sources(
    sources_to_evaluate: Mapping[str, source_lib.FinalSource],
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    calculated_source_profiles: source_profiles.SourceProfiles,
    source_states: Mapping[str, source_state_lib.SourceState],
    updated_source_states: dict[str, source_state_lib.SourceState],
) -> None:
  """Evaluates final aggregating sources and updates profiles and state."""
  for source_name, source in sources_to_evaluate.items():
    curr_state = source_states[source_name]
    output, new_state = source.get_value(
        runtime_params,
        geo,
        core_profiles,
        calculated_source_profiles,
        curr_state,
    )
    updated_source_states[source_name] = new_state
    _update_standard_source_profiles(
        calculated_source_profiles,
        source_name,
        source.AFFECTED_CORE_PROFILES,
        output,
    )


def build_basic_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.BaseCoreProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    explicit: bool,
    explicit_source_outputs: (
        Mapping[str, source_profiles.SourceOutput] | None
    ) = None,
    calculate_anyway: bool = False,
) -> tuple[
    dict[str, source_profiles.SourceOutput],
    dict[str, source_state_lib.SourceState],
]:
  """Evaluates basic standard sources depending only on BaseCoreProfiles.

  Basic sources depend only on base profiles and do not require psidot,
  conductivity, or outputs of other sources. This function is used to evaluate
  sources before neoclassical bootstrap current and psidot.

  Args:
    runtime_params: Runtime parameters for the time step.
    geo: Magnetic geometry.
    core_profiles: Base core plasma profiles.
    source_models: Source models container for TORAX sources.
    source_states: Cross-timestep source states mapping.
    explicit: Whether to evaluate explicit sources (True) or implicit sources
      (False).
    explicit_source_outputs: Explicit source outputs from the explicit pass,
      required when explicit is False.
    calculate_anyway: If True, evaluates sources regardless of the explicit
      matching condition.

  Returns:
    A tuple of (basic_source_outputs, updated_source_states).

  Raises:
    ValueError: If explicit is False and explicit_source_outputs is None.
  """
  if explicit:
    outputs: dict[str, source_profiles.SourceOutput] = {}
  else:
    if explicit_source_outputs is None:
      raise ValueError(
          '`explicit_source_outputs` must be provided if explicit is False.'
      )
    outputs = dict(explicit_source_outputs)
  updated_source_states = dict(source_states)
  for source_name, source in source_models.basic_sources.items():
    source_params = runtime_params.sources[source_name]
    if (explicit == source_params.is_explicit) | calculate_anyway:
      curr_state = source_states[source_name]
      output, new_state = source.get_value(
          runtime_params,
          geo,
          core_profiles,
          curr_state,
      )
      updated_source_states[source_name] = new_state
      outputs[source_name] = output
  return outputs, updated_source_states


def extract_fast_ions(
    source_outputs: Mapping[str, source_profiles.SourceOutput],
) -> tuple[fast_ion_lib.FastIon, ...]:
  """Returns all fast ions produced by source outputs."""
  fast_ions = []
  for out in source_outputs.values():
    fast_ions.extend(out.fast_ions)
  return tuple(fast_ions)


def extract_psi_sources(
    source_outputs: Mapping[str, source_profiles.SourceOutput],
    source_models: source_models_lib.SourceModels,
) -> dict[str, jax.Array]:
  """Extracts psi source profiles from SourceOutput mapping."""
  psi_sources = {}
  for name, out in source_outputs.items():
    if name in source_models.standard_sources:
      source = source_models.standard_sources[name]
      for profile, affected in zip(out.profiles, source.AFFECTED_CORE_PROFILES):
        if affected == source_lib.AffectedCoreProfile.PSI:
          psi_sources[name] = jnp.asarray(profile)
  return psi_sources


def sum_total_psi_sources(
    basic_source_outputs: Mapping[str, source_profiles.SourceOutput],
    source_models: source_models_lib.SourceModels,
    bootstrap_current: bootstrap_current_base.BootstrapCurrent,
    geo: geometry.Geometry,
) -> array_typing.Array:
  """Extracts and sums total psi PDE source term.

  Extracts and sums total psi PDE source term from basic sources and bootstrap.

  Args:
    basic_source_outputs: Output mapping from basic sources.
    source_models: Source models container.
    bootstrap_current: Bootstrap current model outputs.
    geo: Geometry context.

  Returns:
    Total psi source term on the grid.
  """
  psi_sources = extract_psi_sources(basic_source_outputs, source_models)
  total_j_parallel = bootstrap_current.j_parallel_bootstrap + sum(
      psi_sources.values(), start=jnp.zeros_like(geo.rho)
  )
  return source_profiles.j_parallel_to_psi_source(total_j_parallel, geo)


def update_source_profiles(
    core_sources: source_profiles.SourceProfiles,
    source_outputs: Mapping[str, source_profiles.SourceOutput],
    source_models: source_models_lib.SourceModels,
) -> source_profiles.SourceProfiles:
  """Updates core_sources with source_outputs, returning new SourceProfiles."""
  updated = source_profiles.SourceProfiles(
      bootstrap_current=core_sources.bootstrap_current,
      qei=core_sources.qei,
      T_e=dict(core_sources.T_e),
      T_i=dict(core_sources.T_i),
      n_e=dict(core_sources.n_e),
      psi=dict(core_sources.psi),
      fast_ions=dict(core_sources.fast_ions),
  )
  for name, output in source_outputs.items():
    if name in source_models.standard_sources:
      source = source_models.standard_sources[name]
      _update_standard_source_profiles(
          updated,
          name,
          source.AFFECTED_CORE_PROFILES,
          output,
      )
  return updated


def build_psi_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.BaseCoreProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    calculate_anyway: bool = False,
) -> tuple[
    dict[str, jax.Array],
    dict[str, source_state_lib.SourceState],
]:
  """Evaluates standard sources that affect psi."""
  psi_profiles: dict[str, jax.Array] = {}
  updated_source_states = dict(source_states)
  for source_name, source in source_models.psi_sources.items():
    source_params = runtime_params.sources[source_name]
    if source_params.is_explicit or calculate_anyway:
      curr_state = source_states[source_name]
      output, new_state = source.get_value(
          runtime_params,
          geo,
          core_profiles,
          curr_state,
      )
      updated_source_states[source_name] = new_state
      for profile, affected_core_profile in zip(
          output.profiles, source.AFFECTED_CORE_PROFILES, strict=True
      ):
        if affected_core_profile == source_lib.AffectedCoreProfile.PSI:
          psi_profiles[source_name] = jnp.asarray(profile)
  return psi_profiles, updated_source_states


def build_remaining_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    basic_source_outputs: Mapping[str, source_profiles.SourceOutput],
    bootstrap_current: bootstrap_current_base.BootstrapCurrent | None = None,
) -> tuple[
    source_profiles.SourceProfiles,
    dict[str, source_state_lib.SourceState],
]:
  """Evaluates remaining standard sources, qei, and final heating sinks.

  Evaluates qei, sources requiring full CoreProfiles (such as OhmicHeatSource),
  and final sources that aggregate total input heating power (such as
  ImpurityRadiationHeatSink).

  Args:
    runtime_params: Runtime parameters for the time step.
    geo: Magnetic geometry.
    core_profiles: Core plasma profiles.
    source_models: Source models container for TORAX sources.
    source_states: Cross-timestep source states mapping.
    basic_source_outputs: Outputs from basic sources evaluation.
    bootstrap_current: Bootstrap current calculated for this time step. If None,
      defaults to zeros.

  Returns:
    A tuple of (source_profiles, updated_source_states).
  """
  if bootstrap_current is None:
    bootstrap_current = bootstrap_current_base.BootstrapCurrent.zeros(geo)
  qei = source_models.qei_source.get_qei(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
  )
  profiles = source_profiles.SourceProfiles(
      bootstrap_current=bootstrap_current,
      qei=qei,
  )
  for name, output in basic_source_outputs.items():
    if name in source_models.standard_sources:
      source = source_models.standard_sources[name]
      _update_standard_source_profiles(
          profiles,
          name,
          source.AFFECTED_CORE_PROFILES,
          output,
      )
  updated_source_states = dict(source_states)
  _evaluate_core_profiles_sources(
      sources_to_evaluate=source_models.core_profiles_sources,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      calculated_source_profiles=profiles,
      source_states=source_states,
      updated_source_states=updated_source_states,
  )
  _evaluate_final_sources(
      sources_to_evaluate=source_models.final_sources,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      calculated_source_profiles=profiles,
      source_states=source_states,
      updated_source_states=updated_source_states,
  )
  return profiles, updated_source_states


# TODO(b/572586768): Remove temporary compatibility helper once
# initialization / callers migrate.
def build_source_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    explicit: bool,
    explicit_source_profiles: source_profiles.SourceProfiles | None = None,
    explicit_source_outputs: (
        Mapping[str, source_profiles.SourceOutput] | None
    ) = None,
    bootstrap_current: bootstrap_current_base.BootstrapCurrent | None = None,
) -> tuple[
    source_profiles.SourceProfiles,
    dict[str, source_state_lib.SourceState],
]:
  """Evaluates all sources and builds a complete SourceProfiles object."""
  del explicit_source_profiles
  if explicit:
    basic_outputs, updated_source_states = build_basic_sources(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=source_states,
        explicit=True,
    )
    dummy_bootstrap = bootstrap_current_base.BootstrapCurrent.zeros(geo)
    dummy_qei = source_profiles.QeiInfo.zeros(geo)
    profiles = source_profiles.SourceProfiles(
        bootstrap_current=dummy_bootstrap,
        qei=dummy_qei,
    )
    for name, output in basic_outputs.items():
      if name in source_models.standard_sources:
        _update_standard_source_profiles(
            profiles,
            name,
            source_models.standard_sources[name].AFFECTED_CORE_PROFILES,
            output,
        )
    return profiles, updated_source_states
  if explicit_source_outputs is None:
    explicit_source_outputs, _ = build_basic_sources(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=source_states,
        explicit=True,
    )
  basic_outputs, updated_source_states = build_basic_sources(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      source_states=source_states,
      explicit=False,
      explicit_source_outputs=explicit_source_outputs,
  )
  assert bootstrap_current is not None
  # TODO(b/572586768): Remove inline psidot recomputation once calc_coeffs
  # and initial_state migrate to physics_pipeline.
  v_loop_lcfs = core_profiles.psidot.right_face_constraint
  psidot = psi_calculations.calculate_psidot(
      psi=core_profiles.flux.psi,
      sigma=core_profiles.conductivity.sigma,
      psi_sources=sum_total_psi_sources(
          basic_outputs, source_models, bootstrap_current, geo
      ),
      geo=geo,
      evolve_current=runtime_params.numerics.evolve_current,
      prescribed_psidot=runtime_params.profile_conditions.psidot,
      v_loop_lcfs=v_loop_lcfs,
  )
  core_profiles = dataclasses.replace(core_profiles, psidot=psidot)
  return build_remaining_sources(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      source_states=updated_source_states,
      basic_source_outputs=basic_outputs,
      bootstrap_current=bootstrap_current,
  )


# TODO(b/572586768): Remove temporary compatibility helper once
# initialization / callers migrate.
def build_standard_source_profiles(
    *,
    calculated_source_profiles: source_profiles.SourceProfiles,
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    explicit: bool = True,
    calculate_anyway: bool = False,
    psi_only: bool = False,
) -> dict[str, source_state_lib.SourceState]:
  """Evaluates standard sources in place and returns updated source states."""
  updated_source_states = dict(source_states)
  if psi_only:
    _evaluate_basic_sources(
        sources_to_evaluate=source_models.psi_sources,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=calculated_source_profiles,
        source_states=source_states,
        updated_source_states=updated_source_states,
        explicit=explicit,
        calculate_anyway=calculate_anyway,
    )
    return updated_source_states

  standard_sources = getattr(source_models, 'standard_sources', {})
  if isinstance(standard_sources, Mapping):
    basic_sources = {
        name: src
        for name, src in standard_sources.items()
        if isinstance(src, source_lib.Source)
    }
    _evaluate_basic_sources(
        sources_to_evaluate=basic_sources,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=calculated_source_profiles,
        source_states=source_states,
        updated_source_states=updated_source_states,
        explicit=explicit,
        calculate_anyway=calculate_anyway,
    )
    core_sources = {
        name: src
        for name, src in standard_sources.items()
        if isinstance(src, source_lib.SourceUsingCoreProfiles)
    }
    _evaluate_core_profiles_sources(
        sources_to_evaluate=core_sources,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=calculated_source_profiles,
        source_states=source_states,
        updated_source_states=updated_source_states,
    )
    final_sources = {
        name: src
        for name, src in standard_sources.items()
        if isinstance(src, source_lib.FinalSource)
    }
    _evaluate_final_sources(
        sources_to_evaluate=final_sources,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=calculated_source_profiles,
        source_states=source_states,
        updated_source_states=updated_source_states,
    )
  return updated_source_states


# TODO(b/572586768): Remove temporary compatibility helper once
# initialization / callers migrate.
def get_all_source_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    bootstrap_current: bootstrap_current_base.BootstrapCurrent,
    source_states: Mapping[str, source_state_lib.SourceState],
) -> tuple[
    source_profiles.SourceProfiles, dict[str, source_state_lib.SourceState]
]:
  """Returns all source profiles for a given time."""
  basic_outputs, source_states = build_basic_sources(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      source_states=source_states,
      explicit=True,
      calculate_anyway=True,
  )
  return build_remaining_sources(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      source_states=source_states,
      basic_source_outputs=basic_outputs,
      bootstrap_current=bootstrap_current,
  )
