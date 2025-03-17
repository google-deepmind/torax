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
import functools

import chex
from torax import jax_utils
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink

_FINAL_SOURCES = frozenset(
    [impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME]
)


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'source_models',
        'static_runtime_params_slice',
        'explicit',
    ],
)
def build_source_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    explicit: bool,
    explicit_source_profiles: source_profiles.SourceProfiles | None = None,
) -> source_profiles.SourceProfiles:
  """Builds explicit profiles or the union of explicit and implicit profiles.

  Args:
    static_runtime_params_slice: Input config. Cannot change from time step to
      time step.
    dynamic_runtime_params_slice: Input config for this time step. Can change
      from time step to time step.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    source_models: Functions computing profiles for all TORAX sources/sinks.
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

  Returns:
    SourceProfiles caclulated from the source models. If explicit is True, then
    only explicit profiles will be returned. If explicit is False, then the
    union of the explicit profiles in explicit_source_profiles and the implicit
    profiles computed here will be returned.
  """
  if not explicit and explicit_source_profiles is None:
    raise ValueError(
        '`explicit_source_profiles` must be provided if explicit is False.'
    )

  # Bootstrap current is a special-case source with multiple outputs, so handle
  # it here.
  static_bootstrap_runtime_params = static_runtime_params_slice.sources[
      source_models.j_bootstrap_name
  ]
  if explicit == static_bootstrap_runtime_params.is_explicit:
    bootstrap_profiles = source_models.j_bootstrap.get_bootstrap(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
  else:
    if explicit_source_profiles:
      # We have been passed the pre calculated explicit profiles, so use those.
      bootstrap_profiles = explicit_source_profiles.j_bootstrap
    else:
      # We have not been passed the pre calculated explicit profiles, so return
      # all zeros.
      bootstrap_profiles = source_profiles.BootstrapCurrentProfile.zero_profile(
          geo
      )
  if explicit:
    qei = source_profiles.QeiInfo.zeros(geo)
  else:
    qei = source_models.qei_source.get_qei(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
  profiles = source_profiles.SourceProfiles(
      j_bootstrap=bootstrap_profiles,
      qei=qei,
      temp_el=explicit_source_profiles.temp_el
      if explicit_source_profiles
      else {},
      temp_ion=explicit_source_profiles.temp_ion
      if explicit_source_profiles
      else {},
      ne=explicit_source_profiles.ne if explicit_source_profiles else {},
      psi=explicit_source_profiles.psi if explicit_source_profiles else {},
  )
  build_standard_source_profiles(
      calculated_source_profiles=profiles,
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      explicit=explicit,
  )
  return profiles


def build_standard_source_profiles(
    *,
    calculated_source_profiles: source_profiles.SourceProfiles,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    explicit: bool = True,
    calculate_anyway: bool = False,
    psi_only: bool = False,
):
  """Updates calculated_source_profiles with standard source profiles."""
  def calculate_source(source_name, source):
    static_source_runtime_params = static_runtime_params_slice.sources[
        source_name
    ]
    if (
        explicit == static_source_runtime_params.is_explicit
    ) | calculate_anyway:
      value = source.get_value(
          static_runtime_params_slice,
          dynamic_runtime_params_slice,
          geo,
          core_profiles,
          calculated_source_profiles,
      )
      _update_standard_source_profiles(
          calculated_source_profiles,
          source_name,
          source.affected_core_profiles,
          value,
      )

  # Calculate PSI sources first
  # These are used in the calculation of the ohmic_heat_source so we need to
  # calculate them first.
  for source_name, source in source_models.psi_sources.items():
    calculate_source(source_name, source)
  # The psi sources are used in the initialization of the core profiles
  # (specifically, the psidot, psi and currents), so we provide an option to
  # only calculate the psi sources to avoid extra work.
  if psi_only:
    return

  to_calculate = {}
  # Calculate the standard sources
  # This calculates all the remaining sources that have not been calculated
  # yet and are not final sources.
  # Final sources are sources that depend on the output of other sources, so
  # they are calculated after all other sources have been calculated.
  # The impurity_radiation_heat_sink is one such final source and needs to be
  # calculated after all the heat sources have been calculated.
  for source_name, source in source_models.standard_sources.items():
    if source_name in _FINAL_SOURCES:
      to_calculate[source_name] = source
      continue
    if source_name not in source_models.psi_sources:
      calculate_source(source_name, source)
  for source_name, source in to_calculate.items():
    calculate_source(source_name, source)


def _update_standard_source_profiles(
    calculated_source_profiles: source_profiles.SourceProfiles,
    source_name: str,
    affected_core_profiles: tuple[source_lib.AffectedCoreProfile, ...],
    profile: tuple[chex.Array, ...],
):
  """Updates the standard source profiles in calculated_source_profiles.

  Args:
    calculated_source_profiles: The source profiles to update with the newly
      calculated source profiles. Here we are adding to the standard source
      dictionaries so are modifying the calculated_source_profiles in place.
    source_name: The name of the source.
    affected_core_profiles: The core profiles affected by the source.
    profile: The profile of the source.
  """
  for profile, affected_core_profile in zip(
      profile, affected_core_profiles, strict=True):
    match affected_core_profile:
      case source_lib.AffectedCoreProfile.PSI:
        calculated_source_profiles.psi[source_name] = profile
      case source_lib.AffectedCoreProfile.NE:
        calculated_source_profiles.ne[source_name] = profile
      case source_lib.AffectedCoreProfile.TEMP_ION:
        calculated_source_profiles.temp_ion[source_name] = profile
      case source_lib.AffectedCoreProfile.TEMP_EL:
        calculated_source_profiles.temp_el[source_name] = profile


def build_all_zero_profiles(
    geo: geometry.Geometry,
) -> source_profiles.SourceProfiles:
  """Returns a SourceProfiles object with all zero profiles."""
  return source_profiles.SourceProfiles(
      j_bootstrap=source_profiles.BootstrapCurrentProfile.zero_profile(geo),
      qei=source_profiles.QeiInfo.zeros(geo),
  )


def get_initial_source_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> source_profiles.SourceProfiles:
  """Returns the source profiles for the initial state in run_simulation().

  Args:
    static_runtime_params_slice: Runtime parameters which, when they change,
      trigger recompilations. They should not change within a single run of the
      sim.
    dynamic_runtime_params_slice: Runtime parameters which may change from time
      step to time step without triggering recompilations.
    geo: The geometry of the torus during this time step of the simulation.
    core_profiles: Core profiles that may evolve throughout the course of a
      simulation. These values here are, of course, only the original states.
    source_models: Source models used to compute core source profiles.

  Returns:
    Implicit and explicit SourceProfiles from source models based on the core
    profiles from the starting state.
  """
  # Also add in the explicit sources to the initial sources.
  explicit_source_profiles = build_source_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      explicit=True,
  )
  return build_source_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      explicit=False,
      explicit_source_profiles=explicit_source_profiles,
  )
