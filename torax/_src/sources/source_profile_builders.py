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
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import psi_calculations
from torax._src.sources import source as source_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profiles
from torax._src.sources import source_state as source_state_lib


def _evaluate_sources(
    sources_to_evaluate: Mapping[str, source_lib.Source],
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    calculated_source_profiles: source_profiles.SourceProfiles,
    source_states: Mapping[str, source_state_lib.SourceState],
    updated_source_states: dict[str, source_state_lib.SourceState],
    explicit: bool,
    conductivity: conductivity_base.Conductivity | None = None,
    psidot: array_typing.FloatVectorCell | None = None,
    calculate_anyway: bool = False,
) -> None:
  """Evaluates specified sources and updates profiles and source states."""
  for source_name, source in sources_to_evaluate.items():
    source_params = runtime_params.sources[source_name]
    if (explicit == source_params.is_explicit) | calculate_anyway:
      curr_state = source_states[source_name]
      value, new_state = source.get_value(
          runtime_params,
          geo,
          core_profiles,
          calculated_source_profiles,
          conductivity,
          curr_state,
          psidot=psidot,
      )
      updated_source_states[source_name] = new_state
      _update_standard_source_profiles(
          calculated_source_profiles,
          source_name,
          source.AFFECTED_CORE_PROFILES,
          value,
      )


def _copy_source_profiles(
    template: source_profiles.SourceProfiles,
    *,
    bootstrap_current: bootstrap_current_base.BootstrapCurrent,
    qei: source_profiles.QeiInfo,
) -> source_profiles.SourceProfiles:
  """Constructs a SourceProfiles shallow-copying channel dicts from template."""
  return source_profiles.SourceProfiles(
      bootstrap_current=bootstrap_current,
      qei=qei,
      T_e=dict(template.T_e),
      T_i=dict(template.T_i),
      n_e=dict(template.n_e),
      psi=dict(template.psi),
      fast_ions=dict(template.fast_ions),
  )


def build_all_zero_profiles(
    geo: geometry.Geometry,
) -> source_profiles.SourceProfiles:
  """Returns a SourceProfiles object with all zero profiles."""
  return source_profiles.SourceProfiles(
      bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
      qei=source_profiles.QeiInfo.zeros(geo),
  )


def build_basic_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    explicit: bool,
    explicit_source_profiles: source_profiles.SourceProfiles | None = None,
    calculate_anyway: bool = False,
) -> tuple[
    source_profiles.SourceProfiles,
    dict[str, source_state_lib.SourceState],
]:
  """Evaluates basic standard sources.

  Basic sources depend only on base profiles and do not require psidot,
  conductivity, or outputs of other sources. This function is used to evaluate
  sources before neoclassical bootstrap current and psidot.

  Args:
    runtime_params: Runtime parameters for the time step.
    geo: Magnetic geometry.
    core_profiles: Core plasma profiles.
    source_models: Source models container for TORAX sources.
    source_states: Cross-timestep source states mapping.
    explicit: Whether to evaluate explicit sources (True) or implicit sources
      (False).
    explicit_source_profiles: Explicit source profiles from the explicit pass,
      required when explicit is False.
    calculate_anyway: If True, evaluates sources regardless of the explicit
      matching condition.

  Returns:
    A tuple of (partial_source_profiles, updated_source_states).

  Raises:
    ValueError: If explicit is False and explicit_source_profiles is None.
  """
  if explicit:
    profiles = build_all_zero_profiles(geo)
  else:
    if explicit_source_profiles is None:
      raise ValueError(
          '`explicit_source_profiles` must be provided if explicit is False.'
      )
    profiles = _copy_source_profiles(
        explicit_source_profiles,
        bootstrap_current=explicit_source_profiles.bootstrap_current,
        qei=explicit_source_profiles.qei,
    )
  updated_source_states = dict(source_states)
  _evaluate_sources(
      sources_to_evaluate=source_models.basic_sources,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      calculated_source_profiles=profiles,
      source_states=source_states,
      updated_source_states=updated_source_states,
      explicit=explicit,
      conductivity=None,
      psidot=None,
      calculate_anyway=calculate_anyway,
  )
  return profiles, updated_source_states


def build_remaining_sources(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    partial_source_profiles: source_profiles.SourceProfiles,
    explicit: bool,
    conductivity: conductivity_base.Conductivity | None = None,
    bootstrap_current: bootstrap_current_base.BootstrapCurrent | None = None,
    calculate_anyway: bool = False,
) -> tuple[
    source_profiles.SourceProfiles,
    dict[str, source_state_lib.SourceState],
]:
  """Evaluates remaining standard sources, qei, and final heating sinks.

  Evaluates qei, sources requiring conductivity and psidot (such as
  OhmicHeatSource), and final sources that aggregate total input heating
  power (such as ImpurityRadiationHeatSink).

  When conductivity is provided, core_profiles.psidot must already be
  populated for the current state. build_remaining_sources is called downstream
  by fvm, core_profiles, and orchestration after psidot is computed.

  Args:
    runtime_params: Runtime parameters for the time step.
    geo: Magnetic geometry.
    core_profiles: Core plasma profiles.
    source_models: Source models container for TORAX sources.
    source_states: Cross-timestep source states mapping.
    partial_source_profiles: Source profiles containing evaluations from
      build_basic_sources.
    explicit: Whether to evaluate explicit sources (True) or implicit sources
      (False).
    conductivity: Neoclassical conductivity profile, required for implicit
      sources that compute psidot or ohmic heating.
    bootstrap_current: Bootstrap current calculated for this time step,
      required when explicit is False.
    calculate_anyway: If True, evaluates sources regardless of the explicit
      matching condition.

  Returns:
    A tuple of (profiles, updated_source_states) containing all evaluated
    source profiles and their updated states.

  Raises:
    ValueError: If explicit is False and bootstrap_current is None.
  """
  if explicit:
    qei = source_profiles.QeiInfo.zeros(geo)
    bootstrap_current = bootstrap_current_base.BootstrapCurrent.zeros(geo)
  else:
    if bootstrap_current is None:
      raise ValueError(
          '`bootstrap_current` must be provided if explicit is False.'
      )
    qei = source_models.qei_source.get_qei(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )
  profiles = _copy_source_profiles(
      template=partial_source_profiles,
      bootstrap_current=bootstrap_current,
      qei=qei,
  )
  psidot = (
      core_profiles.psidot.value
      if conductivity is not None
      else None
  )

  updated_source_states = dict(source_states)
  _evaluate_sources(
      sources_to_evaluate=source_models.remaining_and_final_sources,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      calculated_source_profiles=profiles,
      source_states=source_states,
      updated_source_states=updated_source_states,
      explicit=explicit,
      conductivity=conductivity,
      psidot=psidot,
      calculate_anyway=calculate_anyway,
  )
  return profiles, updated_source_states


# TODO(b/572586768): Remove once callers migrate to staged source builders.
@jax.jit(
    static_argnames=[
        'source_models',
        'explicit',
    ],
)
def build_source_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    explicit: bool,
    explicit_source_profiles: source_profiles.SourceProfiles | None = None,
    conductivity: conductivity_base.Conductivity | None = None,
    bootstrap_current: bootstrap_current_base.BootstrapCurrent | None = None,
) -> tuple[
    source_profiles.SourceProfiles, dict[str, source_state_lib.SourceState]
]:
  """Builds explicit profiles or the union of explicit and implicit profiles.

  Args:
    runtime_params: Input config for this time step. Can change from time step
      to time step.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    source_models: Functions computing profiles for all TORAX sources/sinks.
    source_states: Cross-timestep states for all sources.
    explicit: If True, this function will only return profiles for explicit
      sources. If False, then the explicit_source_profiles argument must be
      provided and the returned profiles will be the union of the explicit
      profiles in explicit_source_profiles and the implicit profiles computed
      here. Note that for the special-case sources (bootstrap and qei), the
      explicit argument will be used to determine whether to return a profile or
      all zeros.
    explicit_source_profiles: If explicit is False, this argument must be
      provided. It will be used to compute the union of the explicit profiles
      and the implicit profiles computed here.
    conductivity: Conductivity calculated for this time step. Not provided when
      calculating the explicit profiles.
    bootstrap_current: Bootstrap current calculated for this time step. Must be
      provided when explicit is False.

  Returns:
    A tuple (profiles, updated_source_states) calculated from the source
    models. If explicit is True, then only explicit profiles will be returned,
    and updated_source_states will contain updated states for the explicit
    sources with the remaining source states unchanged. If explicit is False,
    then the union of the explicit profiles in explicit_source_profiles and
    the implicit profiles computed here will be returned, and
    updated_source_states will have the implicit source states updated.
  """
  if not explicit and (
      explicit_source_profiles is None or bootstrap_current is None
  ):
    raise ValueError(
        '`explicit_source_profiles` and `bootstrap_current` must be provided'
        ' if explicit is False.'
    )
  partial_profiles, updated_source_states = build_basic_sources(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      source_states=source_states,
      explicit=explicit,
      explicit_source_profiles=explicit_source_profiles,
  )
  if not explicit and conductivity is not None:
    assert bootstrap_current is not None
    psidot = psi_calculations.calculate_psidot(
        psi=core_profiles.psi,
        sigma=conductivity.sigma,
        psi_sources=dataclasses.replace(
            partial_profiles, bootstrap_current=bootstrap_current
        ).total_psi_sources(geo),
        geo=geo,
        evolve_current=runtime_params.numerics.evolve_current,
        prescribed_psidot=runtime_params.profile_conditions.psidot,
    )
    core_profiles = dataclasses.replace(core_profiles, psidot=psidot)
  return build_remaining_sources(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      source_states=updated_source_states,
      partial_source_profiles=partial_profiles,
      explicit=explicit,
      conductivity=conductivity,
      bootstrap_current=bootstrap_current,
  )


# TODO(b/572586768): Remove once callers migrate to staged source builders.
def build_standard_source_profiles(
    *,
    calculated_source_profiles: source_profiles.SourceProfiles,
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState],
    explicit: bool = True,
    conductivity: conductivity_base.Conductivity | None = None,
    calculate_anyway: bool = False,
    psi_only: bool = False,
) -> dict[str, source_state_lib.SourceState]:
  """Updates calculated_source_profiles and returns updated source states."""
  updated_source_states = dict(source_states)

  _evaluate_sources(
      sources_to_evaluate=source_models.psi_sources,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      calculated_source_profiles=calculated_source_profiles,
      source_states=source_states,
      updated_source_states=updated_source_states,
      explicit=explicit,
      conductivity=conductivity,
      calculate_anyway=calculate_anyway,
  )
  if psi_only:
    return updated_source_states

  psidot = None
  if conductivity is not None:
    psidot = psi_calculations.calculate_psidot(
        psi=core_profiles.psi,
        sigma=conductivity.sigma,
        psi_sources=calculated_source_profiles.total_psi_sources(geo),
        geo=geo,
        evolve_current=runtime_params.numerics.evolve_current,
        prescribed_psidot=runtime_params.profile_conditions.psidot,
    ).value

  # pylint: disable-next=protected-access
  final_sources_set = source_models_lib._FINAL_SOURCES
  remaining_sources = {
      name: source
      for name, source in source_models.standard_sources.items()
      if name not in final_sources_set and name not in source_models.psi_sources
  }
  _evaluate_sources(
      sources_to_evaluate=remaining_sources,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      calculated_source_profiles=calculated_source_profiles,
      source_states=source_states,
      updated_source_states=updated_source_states,
      explicit=explicit,
      conductivity=conductivity,
      psidot=psidot,
      calculate_anyway=calculate_anyway,
  )
  final_sources = {
      name: source
      for name, source in source_models.standard_sources.items()
      if name in final_sources_set
  }
  _evaluate_sources(
      sources_to_evaluate=final_sources,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      calculated_source_profiles=calculated_source_profiles,
      source_states=source_states,
      updated_source_states=updated_source_states,
      explicit=explicit,
      conductivity=conductivity,
      psidot=psidot,
      calculate_anyway=calculate_anyway,
  )
  return updated_source_states


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


# TODO(b/572586768): Remove once callers migrate to staged source builders.
def get_all_source_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    conductivity: conductivity_base.Conductivity,
    bootstrap_current: bootstrap_current_base.BootstrapCurrent,
    source_states: Mapping[str, source_state_lib.SourceState],
) -> tuple[
    source_profiles.SourceProfiles, dict[str, source_state_lib.SourceState]
]:
  """Returns all source profiles for a given time.

  Used e.g. to initialize the source profiles at an initial time step.

  Args:
    runtime_params: Runtime parameters which may change from time step to time
      step without triggering recompilations.
    geo: The geometry of the torus during this time step of the simulation.
    core_profiles: Core profiles that may evolve throughout the course of a
      simulation. These values here are, of course, only the original states.
    source_models: Source models used to compute core source profiles.
    conductivity: Conductivity calculated for this time step.
    bootstrap_current: Bootstrap current calculated for this time step.
    source_states: Initial source states.

  Returns:
    (profiles, updated_source_states) from source models based on the core
    profiles from the starting state.
  """
  # Also add in the explicit sources to the initial sources.
  explicit_source_profiles, source_states = build_source_profiles(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      source_states=source_states,
      explicit=True,
  )
  return build_source_profiles(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      source_states=source_states,
      explicit=False,
      explicit_source_profiles=explicit_source_profiles,
      conductivity=conductivity,
      bootstrap_current=bootstrap_current,
  )
