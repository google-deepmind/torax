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

"""Module for a single source/sink term.

This module contains base classes for defining source terms. Other files in this
folder use these classes to define specific types of sources/sinks.
"""

import abc
from collections.abc import Callable
import dataclasses
from typing import ClassVar, TypeAlias

from jax import numpy as jnp
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source_profiles
from torax._src.sources import source_state as source_state_lib


AffectedCoreProfile = source_profiles.AffectedCoreProfile


_ModelFn: TypeAlias = Callable[
    [],
    tuple[source_profiles.SourceOutput, source_state_lib.SourceState],
]


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class SourceBase(static_dataclass.StaticDataclass, abc.ABC):
  """Shared base class for all TORAX source and sink terms.

  Attributes:
    SOURCE_ID: Identifier for the source type (e.g. `'ecrh'`, `'icrh'`,
      `'fusion'`). This corresponds to the field name on the `SourcesConfig`
      Pydantic config and the key in `runtime_params.sources` and
      `SourceModels.standard_sources`.
    AFFECTED_CORE_PROFILES: Core profiles affected by this source's profile(s).
      This attribute defines which equations the source profiles are terms for.
  """

  SOURCE_ID: ClassVar[str]
  AFFECTED_CORE_PROFILES: ClassVar[tuple[AffectedCoreProfile, ...]]

  def __post_init__(self):
    super().__post_init__()
    if not hasattr(self, 'SOURCE_ID'):
      raise ValueError('Source ID must be set.')
    if not getattr(self, 'AFFECTED_CORE_PROFILES', None):
      raise ValueError('Affected core profiles must be set.')

  def initial_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
  ) -> source_state_lib.SourceState:
    """Returns the initial state for this source."""
    del runtime_params
    return source_state_lib.SourceState()

  def _evaluate_with_mode(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      source_state: source_state_lib.SourceState,
      model_fn: _ModelFn,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Evaluates the source according to its configured runtime mode."""
    source_params = runtime_params.sources[self.SOURCE_ID]

    mode = source_params.mode
    match mode:
      case sources_runtime_params_lib.Mode.MODEL_BASED:
        output, source_state = model_fn()
      case sources_runtime_params_lib.Mode.PRESCRIBED:
        if len(source_params.prescribed_values) != len(
            self.AFFECTED_CORE_PROFILES
        ):
          raise ValueError(
              'When using PRESCRIBED mode, the number of prescribed values must'
              ' match the number of affected core profiles. Was: '
              f'{len(source_params.prescribed_values)} '
              f' Expected: {len(self.AFFECTED_CORE_PROFILES)}.'
          )
        output = source_profiles.SourceOutput(
            profiles=source_params.prescribed_values
        )
      case sources_runtime_params_lib.Mode.ZERO:
        zeros = jnp.zeros(geo.rho_norm.shape)
        output = source_profiles.SourceOutput(
            profiles=(zeros,) * len(self.AFFECTED_CORE_PROFILES)
        )
      case _:
        raise ValueError(f'Unknown mode: {mode}')

    return output, source_state


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class Source(SourceBase):
  """Standard source model depending on BaseCoreProfiles."""

  @abc.abstractmethod
  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.BaseCoreProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Returns the model-based value and updated state for this source."""

  def get_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.BaseCoreProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Returns the cell grid profile and updated state for this source."""
    return self._evaluate_with_mode(
        runtime_params,
        geo,
        source_state,
        lambda: self._get_model_value(
            runtime_params, geo, core_profiles, source_state
        ),
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class SourceUsingCoreProfiles(SourceBase):
  """Source requiring full CoreProfiles (such as conductivity and psidot)."""

  def __post_init__(self):
    super().__post_init__()
    if AffectedCoreProfile.PSI in self.AFFECTED_CORE_PROFILES:
      raise ValueError(
          f'{type(self).__name__} cannot affect PSI, but got'
          f' AFFECTED_CORE_PROFILES={self.AFFECTED_CORE_PROFILES}.'
      )

  @abc.abstractmethod
  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Returns the model-based value and updated state for this source."""

  def get_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Returns the cell grid profile and updated state for this source."""
    source_params = runtime_params.sources[self.SOURCE_ID]
    if source_params.is_explicit:
      raise ValueError(
          f'{type(self).__name__} requires implicit CoreProfiles (such as'
          ' conductivity and psidot) and cannot be configured with'
          ' is_explicit=True.'
      )
    output, source_state = self._evaluate_with_mode(
        runtime_params,
        geo,
        source_state,
        lambda: self._get_model_value(
            runtime_params, geo, core_profiles, source_state
        ),
    )
    if output.fast_ions:
      raise ValueError(
          f'{type(self).__name__} consumes CoreProfiles and cannot produce'
          ' fast_ions.'
      )
    return output, source_state


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class FinalSource(SourceBase):
  """Final source model evaluated after all other sources are computed.

  FinalSource aggregates already-computed source profiles. It cannot affect
  PSI, and at most one FinalSource may be registered in SourceModels so
  evaluation order is unambiguous.
  """

  def __post_init__(self):
    super().__post_init__()
    if AffectedCoreProfile.PSI in self.AFFECTED_CORE_PROFILES:
      raise ValueError(
          f'{type(self).__name__} cannot affect PSI, but got'
          f' AFFECTED_CORE_PROFILES={self.AFFECTED_CORE_PROFILES}.'
      )

  @abc.abstractmethod
  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Returns the model-based value and updated state for this source."""

  def get_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Returns the cell grid profile and updated state for this source."""
    source_params = runtime_params.sources[self.SOURCE_ID]
    if source_params.is_explicit:
      raise ValueError(
          f'{type(self).__name__} aggregates other source profiles and cannot'
          ' be configured with is_explicit=True.'
      )
    output, source_state = self._evaluate_with_mode(
        runtime_params,
        geo,
        source_state,
        lambda: self._get_model_value(
            runtime_params,
            geo,
            core_profiles,
            calculated_source_profiles,
            source_state,
        ),
    )
    if output.fast_ions:
      raise ValueError(
          f'{type(self).__name__} consumes CoreProfiles and cannot produce'
          ' fast_ions.'
      )
    return output, source_state
