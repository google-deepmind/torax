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

"""Sources that influence the ion and/or electron temperature.

The sources in this file are simple. More complex sources can usually be found
in their own files.
"""

import dataclasses

from torax.sources import source


# The heat sources/sinks below don't have any source-specific implementations,
# so their bodies are empty. You can refer to their base class to see the
# implementation. We define new classes here to:
#  a) support any future source-specific implementation.
#  b) better readability and human-friendly error messages when debugging.
@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ChargeExchangeHeatSink(source.Source):
  """Charge exchange loss term for the ion temp equation."""

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_ION,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class CyclotronRadiationHeatSink(source.Source):
  """Cyclotron radiation loss term for the electron temp equation."""

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_EL,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ECRHHeatSource(source.Source):
  """ECRH heat source for the electron temp equation."""

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_EL,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ICRHHeatSource(source.Source):
  """ICRH heat source for the ion temp equation."""

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_ION,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class LHHeatSource(source.Source):
  """LH heat source for the electron temp equation."""

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_EL,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class LineRadiationHeatSink(source.Source):
  """Line radiation loss sink for the electron temp equation."""

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_EL,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class NBIElectronHeatSource(source.Source):
  """NBI heat source for the electron temp equation."""

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_EL,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class NBIIonHeatSource(source.Source):
  """NBI heat source for the ion temp equation."""

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_ION,)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RecombinationHeatSink(source.Source):
  """Recombination loss sink for the electron temp equation."""

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_EL,)
