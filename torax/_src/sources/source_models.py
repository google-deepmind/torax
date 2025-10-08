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
import jax
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source as source_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class SourceModels:
  """Source models for the different equations being evolved in Torax."""

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

  def _hash(self) -> int:
    """Custom hash implementation.

    Use this name to distinguish it from __hash__, which may be overwritten
    inadvertently by multiple inheritance.

    Returns:
      hash: the hash of this SourceModels
    """
    hashes = [hash(self.standard_sources)]
    hashes.append(hash(self.qei_source))
    return hash(tuple(hashes))

  def __hash__(self) -> int:
    return self._hash()

  def __eq__(self, other) -> bool:
    if set(self.standard_sources.keys()) == set(other.standard_sources.keys()):
      return (
          all(
              self.standard_sources[name] == other.standard_sources[name]
              for name in self.standard_sources.keys()
          )
          and self.qei_source == other.qei_source
      )
    return False

  def __post_init__(self):
    """Run post-init checks."""

    # Make sure our custom hash hasn't been overwritten by the dataclasses
    # decorator
    assert hash(self) == self._hash()
