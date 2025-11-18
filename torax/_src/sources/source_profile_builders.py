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

import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import source as source_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profiles
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink

_FINAL_SOURCES = frozenset(
    [impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME]
)


@functools.partial(
    jax.jit,
    static_argnames=[
        'source_models',
        'neoclassical_models',
        'explicit',
    ],
)
def build_source_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    explicit: bool,
    explicit_source_profiles: source_profiles.SourceProfiles | None = None,
    conductivity: conductivity_base.Conductivity | None = None,
) -> source_profiles.SourceProfiles:
  """Builds explicit profiles or the union of explicit and implicit profiles.

  Args:
    runtime_params: Input config for this time step. Can change from time step
      to time step.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    source_models: Functions computing profiles for all TORAX sources/sinks.
    neoclassical_models: Functions computing neoclassical physics.
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

  if explicit:
    qei = source_profiles.QeiInfo.zeros(geo)
    bootstrap_current = bootstrap_current_base.BootstrapCurrent.zeros(geo)
  else:
    qei = source_models.qei_source.get_qei(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )
    bootstrap_current = (
        neoclassical_models.bootstrap_current.calculate_bootstrap_current(
            runtime_params, geo, core_profiles
        )
    )
  profiles = source_profiles.SourceProfiles(
      bootstrap_current=bootstrap_current,
      qei=qei,
      T_e=explicit_source_profiles.T_e if explicit_source_profiles else {},
      T_i=explicit_source_profiles.T_i if explicit_source_profiles else {},
      n_e=explicit_source_profiles.n_e if explicit_source_profiles else {},
      psi=explicit_source_profiles.psi if explicit_source_profiles else {},
  )
  build_standard_source_profiles(
      calculated_source_profiles=profiles,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      explicit=explicit,
      conductivity=conductivity,
  )
  return profiles


def build_standard_source_profiles(
    *,
    calculated_source_profiles: source_profiles.SourceProfiles,
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    explicit: bool = True,
    conductivity: conductivity_base.Conductivity | None = None,
    calculate_anyway: bool = False,
    psi_only: bool = False,
):
  """Updates calculated_source_profiles with standard source profiles."""

  def calculate_source(source_name: str, source: source_lib.Source):
    source_params = runtime_params.sources[source_name]
    if (explicit == source_params.is_explicit) | calculate_anyway:
      value = source.get_value(
          runtime_params,
          geo,
          core_profiles,
          calculated_source_profiles,
          conductivity,
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
    profile: tuple[array_typing.FloatVectorCell, ...],
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
      profile, affected_core_profiles, strict=True
  ):
    match affected_core_profile:
      case source_lib.AffectedCoreProfile.PSI:
        calculated_source_profiles.psi[source_name] = profile
      case source_lib.AffectedCoreProfile.NE:
        calculated_source_profiles.n_e[source_name] = profile
      case source_lib.AffectedCoreProfile.TEMP_ION:
        calculated_source_profiles.T_i[source_name] = profile
      case source_lib.AffectedCoreProfile.TEMP_EL:
        calculated_source_profiles.T_e[source_name] = profile


def build_all_zero_profiles(
    geo: geometry.Geometry,
) -> source_profiles.SourceProfiles:
  """Returns a SourceProfiles object with all zero profiles."""
  return source_profiles.SourceProfiles(
      bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
      qei=source_profiles.QeiInfo.zeros(geo),
  )


def get_all_source_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    conductivity: conductivity_base.Conductivity,
) -> source_profiles.SourceProfiles:
  """Returns all source profiles for a given time.

  Used e.g. to initialize the source profiles at an initial time step.

  Args:
    runtime_params: Runtime parameters which may change from time step to time
      step without triggering recompilations.
    geo: The geometry of the torus during this time step of the simulation.
    core_profiles: Core profiles that may evolve throughout the course of a
      simulation. These values here are, of course, only the original states.
    source_models: Source models used to compute core source profiles.
    neoclassical_models: Neoclassical models.
    conductivity: Conductivity calculated for this time step.

  Returns:
    Implicit and explicit SourceProfiles from source models based on the core
    profiles from the starting state.
  """
  # Also add in the explicit sources to the initial sources.
  explicit_source_profiles = build_source_profiles(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
      explicit=True,
  )
  return build_source_profiles(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
      explicit=False,
      explicit_source_profiles=explicit_source_profiles,
      conductivity=conductivity,
  )
