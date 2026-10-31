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

"""Test utilities for constructing Torax sources and source profiles."""

from collections.abc import Mapping

from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles
from torax._src.sources import source_state as source_state_lib


def build_source_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    source_states: Mapping[str, source_state_lib.SourceState] | None = None,
    explicit: bool = True,
    bootstrap_current: bootstrap_current_base.BootstrapCurrent | None = None,
) -> tuple[
    source_profiles.SourceProfiles,
    dict[str, source_state_lib.SourceState],
]:
  """Helper to evaluate basic and remaining sources into complete SourceProfiles."""
  if source_states is None:
    source_states = source_models.initial_state(runtime_params)
  if bootstrap_current is None:
    bootstrap_current = bootstrap_current_base.BootstrapCurrent.zeros(geo)
  basic_source_outputs, source_states = (
      source_profile_builders.build_basic_sources(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          source_models=source_models,
          source_states=source_states,
          explicit=explicit,
      )
  )
  return source_profile_builders.build_remaining_sources(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      source_states=source_states,
      basic_source_outputs=basic_source_outputs,
      bootstrap_current=bootstrap_current,
  )
