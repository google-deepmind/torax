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

import functools

import chex
import immutabledict
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source as source_lib


@chex.dataclass(frozen=True)
class SourceModels:
  """Source models for the different equations being evolved in Torax."""

  bootstrap_current: bootstrap_current_base.BootstrapCurrentModel
  conductivity: conductivity_base.ConductivityModel
  qei_source: qei_source_lib.QeiSource
  standard_sources: immutabledict.immutabledict[str, source_lib.Source]

  @functools.cached_property
  def psi_sources(self) -> immutabledict.immutabledict[str, source_lib.Source]:
    """A derived dictionary of sources that affect the psi core_profile."""
    return immutabledict.immutabledict({
        name: source
        for name, source in self.standard_sources.items()
        if source_lib.AffectedCoreProfile.PSI in source.affected_core_profiles
    })

  def __hash__(self) -> int:
    hashes = [hash(self.standard_sources)]
    hashes.append(hash(self.bootstrap_current))
    hashes.append(hash(self.conductivity))
    hashes.append(hash(self.qei_source))
    return hash(tuple(hashes))

  def __eq__(self, other) -> bool:
    if set(self.standard_sources.keys()) == set(other.standard_sources.keys()):
      return (
          all(
              self.standard_sources[name] == other.standard_sources[name]
              for name in self.standard_sources.keys()
          )
          and self.bootstrap_current == other.bootstrap_current
          and self.conductivity == other.conductivity
          and self.qei_source == other.qei_source
      )
    return False
