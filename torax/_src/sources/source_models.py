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

"""Container for source models which build source profiles in TORAX."""
import dataclasses
import functools

import immutabledict
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_state as source_state_lib


@dataclasses.dataclass(frozen=True, eq=False)
class SourceModels(static_dataclass.StaticDataclass):
  """Source models for the different equations being evolved in Torax.

  This class is intended for use as a static argument to jitted Jax
  functions. It is thus immutable and supports comparison and hashing
  by value, and is not a pytree.
  """

  qei_source: qei_source_lib.QeiSource
  standard_sources: immutabledict.immutabledict[str, source_lib.SourceBase]

  def __post_init__(self):
    super().__post_init__()
    for name, src in self.standard_sources.items():
      if not isinstance(
          src,
          (
              source_lib.Source,
              source_lib.SourceUsingCoreProfiles,
              source_lib.FinalSource,
          ),
      ):
        raise TypeError(
            f'Unsupported source type {type(src).__name__} for {name!r}. Must'
            ' inherit from Source, SourceUsingCoreProfiles, or FinalSource.'
        )
    if len(self.final_sources) > 1:
      raise ValueError(
          'At most one FinalSource may be configured in SourceModels, got '
          f'{len(self.final_sources)}: {list(self.final_sources.keys())}.'
      )

  def initial_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
  ) -> dict[str, source_state_lib.SourceState]:
    """Returns the initial state mapping for all sources."""
    return {
        name: src.initial_state(runtime_params)
        for name, src in self.standard_sources.items()
    }

  @functools.cached_property
  def psi_sources(
      self,
  ) -> immutabledict.immutabledict[str, source_lib.Source]:
    """A derived dictionary of sources that affect the psi core_profile."""
    return immutabledict.immutabledict({
        name: source
        for name, source in self.standard_sources.items()
        if isinstance(source, source_lib.Source)
        and source_lib.AffectedCoreProfile.PSI in source.AFFECTED_CORE_PROFILES
    })

  @functools.cached_property
  def basic_sources(
      self,
  ) -> immutabledict.immutabledict[str, source_lib.Source]:
    """Derived mapping of standard sources depending only on SolverProfiles.

    These sources depend only on base profiles and do not require psidot,
    conductivity, or outputs of other sources.
    """
    return immutabledict.immutabledict({
        name: source
        for name, source in self.standard_sources.items()
        if isinstance(source, source_lib.Source)
    })

  @functools.cached_property
  def core_profiles_sources(
      self,
  ) -> immutabledict.immutabledict[str, source_lib.SourceUsingCoreProfiles]:
    """Derived mapping of standard sources requiring full CoreProfiles."""
    return immutabledict.immutabledict({
        name: source
        for name, source in self.standard_sources.items()
        if isinstance(source, source_lib.SourceUsingCoreProfiles)
    })

  @functools.cached_property
  def final_sources(
      self,
  ) -> immutabledict.immutabledict[str, source_lib.FinalSource]:
    """Derived mapping of final standard sources evaluated last."""
    return immutabledict.immutabledict({
        name: source
        for name, source in self.standard_sources.items()
        if isinstance(source, source_lib.FinalSource)
    })
