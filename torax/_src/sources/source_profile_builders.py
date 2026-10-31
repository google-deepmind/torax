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
from typing import Any
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profiles
from torax._src.core_profiles import solver_profiles
from torax._src.geometry import geometry
from torax._src.neoclassical.bootstrap_current import (
    base as bootstrap_current_base,
)
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profiles
from torax._src.sources import source_state as source_state_lib


def _evaluate_core_profiles_sources(
    sources_to_evaluate: Mapping[str, source_lib.SourceUsingCoreProfiles],
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: profiles.CoreProfiles,
    source_states: Mapping[str, source_state_lib.SourceState],
    updated_source_states: dict[str, source_state_lib.SourceState],
) -> dict[str, source_profiles.SourceOutput]:
  """Evaluates CoreProfiles-dependent sources and returns outputs and updated state."""
  outputs: dict[str, source_profiles.SourceOutput] = {}
  for source_name, source in sources_to_evaluate.items():
    curr_state = source_states[source_name]
    output, new_state = source.get_value(
        runtime_params,
        geo,
        core_profiles,
        curr_state,
    )
    updated_source_states[source_name] = new_state
    outputs[source_name] = output
  return outputs


def _assemble_source_profiles(
    source_outputs: Mapping[str, source_profiles.SourceOutput],
    standard_sources: Mapping[str, Any],
    bootstrap_current: bootstrap_current_base.BootstrapCurrent,
    qei: source_profiles.QeiInfo,
) -> source_profiles.SourceProfiles:
  """Assembles SourceProfiles from source outputs in a single pass."""
  T_e = {}
  T_i = {}
  n_e = {}
  psi = {}
  fast_ions = {}
  for name, output in source_outputs.items():
    if name in standard_sources:
      source = standard_sources[name]
      for profile, affected in zip(
          output.profiles, source.AFFECTED_CORE_PROFILES, strict=True
      ):
        jax_profile = jnp.asarray(profile)
        match affected:
          case source_profiles.AffectedCoreProfile.PSI:
            psi[name] = jax_profile
          case source_profiles.AffectedCoreProfile.NE:
            n_e[name] = jax_profile
          case source_profiles.AffectedCoreProfile.TEMP_ION:
            T_i[name] = jax_profile
          case source_profiles.AffectedCoreProfile.TEMP_EL:
            T_e[name] = jax_profile
      if output.fast_ions:
        fast_ions[name] = output.fast_ions
  return source_profiles.SourceProfiles(
      bootstrap_current=bootstrap_current,
      qei=qei,
      T_e=T_e,
      T_i=T_i,
      n_e=n_e,
      psi=psi,
      fast_ions=fast_ions,
  )


def build_basic_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: solver_profiles.SolverProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    explicit: bool,
    calculate_anyway: bool = False,
) -> tuple[
    dict[str, source_profiles.SourceOutput],
    dict[str, source_state_lib.SourceState],
]:
  """Evaluates basic standard sources depending only on SolverProfiles.

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
    calculate_anyway: If True, evaluates sources regardless of the explicit
      matching condition.

  Returns:
    A tuple of (basic_source_outputs, updated_source_states).
  """
  outputs: dict[str, source_profiles.SourceOutput] = {}
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


extract_fast_ions = source_profiles.extract_fast_ions
extract_psi_sources = source_profiles.extract_psi_sources
sum_total_psi_sources = source_profiles.sum_total_psi_sources


def build_psi_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: solver_profiles.SolverProfiles,
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
        if affected_core_profile == source_profiles.AffectedCoreProfile.PSI:
          psi_profiles[source_name] = jnp.asarray(profile)
  return psi_profiles, updated_source_states


def build_remaining_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: profiles.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    basic_source_outputs: Mapping[str, source_profiles.SourceOutput],
    bootstrap_current: bootstrap_current_base.BootstrapCurrent,
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
    bootstrap_current: Bootstrap current calculated for this time step.

  Returns:
    A tuple of (source_profiles, updated_source_states).
  """
  updated_source_states = dict(source_states)
  core_outputs = _evaluate_core_profiles_sources(
      sources_to_evaluate=source_models.core_profiles_sources,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_states=source_states,
      updated_source_states=updated_source_states,
  )
  all_standard_outputs = {**basic_source_outputs, **core_outputs}

  qei = source_models.qei_source.get_qei(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
  )
  profiles = _assemble_source_profiles(
      source_outputs=all_standard_outputs,
      standard_sources=source_models.standard_sources,
      bootstrap_current=bootstrap_current,
      qei=qei,
  )

  if source_models.final_sources:
    final_outputs: dict[str, source_profiles.SourceOutput] = {}
    for name, source in source_models.final_sources.items():
      curr_state = source_states[name]
      output, new_state = source.get_value(
          runtime_params,
          geo,
          core_profiles,
          profiles,
          curr_state,
      )
      updated_source_states[name] = new_state
      final_outputs[name] = output
    profiles = profiles.update_source_profiles(
        final_outputs,
        source_models,
    )

  return profiles, updated_source_states

