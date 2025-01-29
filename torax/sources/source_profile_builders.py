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

from __future__ import annotations

import functools

import chex
import jax.numpy as jnp
from torax import jax_utils
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import bootstrap_current_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles


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
) -> source_profiles.SourceProfiles:
  """Builds explicit or implicit source profiles.

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
    explicit: If True, this function should return profiles for all explicit
      sources. All implicit sources should be set to 0. And same vice versa.

  Returns:
    SourceProfiles for either explicit or implicit sources (and all others set
    to zero).
  """
  # Bootstrap current is a special-case source with multiple outputs, so handle
  # it here.
  static_bootstrap_runtime_params = static_runtime_params_slice.sources[
      source_models.j_bootstrap_name
  ]
  bootstrap_profiles = build_bootstrap_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      static_source_runtime_params=static_bootstrap_runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      j_bootstrap_source=source_models.j_bootstrap,
      explicit=explicit,
  )
  other_profiles = build_standard_source_profiles(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
      explicit,
  )
  if not explicit:
    qei = source_models.qei_source.get_qei(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
  else:
    qei = source_profiles.QeiInfo.zeros(geo)
  return source_profiles.SourceProfiles(
      profiles=other_profiles,
      j_bootstrap=bootstrap_profiles,
      qei=qei,
  )


def build_bootstrap_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    static_source_runtime_params: runtime_params_lib.StaticRuntimeParams,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    j_bootstrap_source: bootstrap_current_source.BootstrapCurrentSource,
    explicit: bool = True,
    calculate_anyway: bool = False,
) -> source_profiles.BootstrapCurrentProfile:
  """Computes the bootstrap current profile.

  Args:
    static_runtime_params_slice: Input config. Cannot change from time step to
      time step.
    static_source_runtime_params: Input runtime parameters specific to the
      bootstrap current source that do not change from time step to time step.
    dynamic_runtime_params_slice: Input config for this time step. Can change
      from time step to time step.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    j_bootstrap_source: Bootstrap current source used to compute the profile.
    explicit: If True, this function should return the profile for an explicit
      source. If explicit is True and the bootstrap current source is not
      explicit, then this should return all zeros. And same with implicit (if
      explicit=False and the source is set to be explicit, then this will return
      all zeros).
    calculate_anyway: If True, returns values regardless of explicit

  Returns:
    Bootstrap current profile.
  """
  bootstrap_profile = j_bootstrap_source.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
  )
  if explicit == static_source_runtime_params.is_explicit | calculate_anyway:
    return bootstrap_profile
  else:
    return source_profiles.BootstrapCurrentProfile.zero_profile(geo)


def build_standard_source_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    explicit: bool = True,
    calculate_anyway: bool = False,
    affected_core_profiles: tuple[source_lib.AffectedCoreProfile, ...] = (
        source_lib.AffectedCoreProfile.PSI,
        source_lib.AffectedCoreProfile.NE,
        source_lib.AffectedCoreProfile.TEMP_ION,
        source_lib.AffectedCoreProfile.TEMP_EL,
    ),
) -> dict[str, chex.ArrayTree]:
  """Computes sources and builds a kwargs dict for SourceProfiles.

  Args:
    static_runtime_params_slice: Input config. Cannot change from time step to
      time step.
    dynamic_runtime_params_slice: Input config for this time step. Can change
      from time step to time step.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    source_models: Collection of all TORAX sources.
    explicit: If True, this function should return the profile for an explicit
      source. If explicit is True and a given source is not explicit, then this
      function will return zeros for that source. And same with implicit (if
      explicit=False and the source is set to be explicit, then this will return
      all zeros).
    calculate_anyway: If True, returns values regardless of explicit
    affected_core_profiles: Populate the output for sources that affect these
      core profiles.

  Returns:
    dict of source profiles excluding the two special-case sources (bootstrap
    and qei).
  """
  computed_source_profiles = {}
  affected_core_profiles_set = set(affected_core_profiles)
  for source_name, source in source_models.standard_sources.items():
    if affected_core_profiles_set.intersection(source.affected_core_profiles):
      static_source_runtime_params = static_runtime_params_slice.sources[
          source_name
      ]
      if (
          explicit
          == static_source_runtime_params.is_explicit | calculate_anyway
      ):
        computed_source_profiles[source_name] = source.get_value(
            static_runtime_params_slice,
            dynamic_runtime_params_slice,
            geo,
            core_profiles,
        )
      else:
        computed_source_profiles[source_name] = jnp.zeros(
            source.output_shape_getter(geo))
  return computed_source_profiles


def build_all_zero_profiles(
    geo: geometry.Geometry,
    source_models: source_models_lib.SourceModels,
) -> source_profiles.SourceProfiles:
  """Returns a SourceProfiles object with all zero profiles."""
  profiles = {
      source_name: jnp.zeros(source_model.output_shape_getter(geo))
      for source_name, source_model in source_models.standard_sources.items()
  }
  return source_profiles.SourceProfiles(
      profiles=profiles,
      j_bootstrap=source_profiles.BootstrapCurrentProfile.zero_profile(geo),
      qei=source_profiles.QeiInfo.zeros(geo),
  )
