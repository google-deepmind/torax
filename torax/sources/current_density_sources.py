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

"""Current density sources for the psi equation.

Simple current sources are in this library while more complex sources, like
bootstrap current, can be found in their own files.
"""

import dataclasses

from torax.sources import source


# The current sources below don't have any source-specific implementations, so
# their bodies are empty. You can refer to their base class to see the
# implementation. We define new classes here to:
#  a) support any future source-specific implementation.
#  b) better readability and human-friendly error messages when debugging.


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ECRHCurrentSource(source.SingleProfilePsiSource):
  """ECRH current density source for the psi equation."""


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ICRHCurrentSource(source.SingleProfilePsiSource):
  """ICRH current density source for the psi equation."""


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class LHCurrentSource(source.SingleProfilePsiSource):
  """LH current density source for the psi equation."""


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class NBICurrentSource(source.SingleProfilePsiSource):
  """NBI current density source for the psi equation."""

ECRHCurrentSourceBuilder = source.make_source_builder(ECRHCurrentSource)
ICRHCurrentSourceBuilder = source.make_source_builder(ICRHCurrentSource)
LHCurrentSourceBuilder = source.make_source_builder(LHCurrentSource)
NBICurrentSourceBuilder = source.make_source_builder(NBICurrentSource)
