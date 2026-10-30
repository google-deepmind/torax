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
from torax._src.sources import ohmic_heat_source
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_state as source_state_lib
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink

# Sources that depend on live core_profiles (specifically psidot and
# conductivity) and therefore are evaluated with remaining sources after
# neoclassical conductivity is available.
_CORE_PROFILES_SOURCES = frozenset(
    [ohmic_heat_source.OhmicHeatSource.SOURCE_ID]
)

# Sources that depend on the output of other sources (e.g.
# ConstantFractionImpurityRadiationHeatSink, which scales with total input
# heating power) and therefore must be calculated last after all other heat
# sources are computed.
_FINAL_SOURCES = frozenset(
    [impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_ID]
)


@dataclasses.dataclass(frozen=True, eq=False)
class SourceModels(static_dataclass.StaticDataclass):
  """Source models for the different equations being evolved in Torax.

  This class is intended for use as a static argument to jitted Jax
  functions. It is thus immutable and supports comparison and hashing
  by value, and is not a pytree.
  """

  qei_source: qei_source_lib.QeiSource
  standard_sources: immutabledict.immutabledict[str, source_lib.Source]

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
  def psi_sources(self) -> immutabledict.immutabledict[str, source_lib.Source]:
    """A derived dictionary of sources that affect the psi core_profile."""
    return immutabledict.immutabledict({
        name: source
        for name, source in self.standard_sources.items()
        if source_lib.AffectedCoreProfile.PSI in source.AFFECTED_CORE_PROFILES
    })

  @functools.cached_property
  def basic_sources(
      self,
  ) -> immutabledict.immutabledict[str, source_lib.Source]:
    """Derived mapping of standard sources evaluated as basic sources.

    These sources depend only on base profiles and do not require psidot,
    conductivity, or outputs of other sources.
    """
    excluded = _CORE_PROFILES_SOURCES | _FINAL_SOURCES
    return immutabledict.immutabledict({
        name: source
        for name, source in self.standard_sources.items()
        if name not in excluded
    })

  @functools.cached_property
  def remaining_and_final_sources(
      self,
  ) -> immutabledict.immutabledict[str, source_lib.Source]:
    """Derived mapping of core_profiles sources followed by final sources."""
    sources = {
        name: source
        for name, source in self.standard_sources.items()
        if name in _CORE_PROFILES_SOURCES
    }
    sources.update({
        name: source
        for name, source in self.standard_sources.items()
        if name in _FINAL_SOURCES
    })
    return immutabledict.immutabledict(sources)
