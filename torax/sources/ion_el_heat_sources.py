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


@dataclasses.dataclass(frozen=True, kw_only=True)
class BremsstrahlungHeatSink(source.SingleProfileTempElSource):
  """Bremsstrahlung loss sink for the electron temp equation."""

  name: str = 'bremsstrahlung_heat_sink'


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChargeExchangeHeatSink(source.SingleProfileTempIonSource):
  """Charge exchange loss term for the ion temp equation."""

  name: str = 'charge_exchange_heat_sink'


@dataclasses.dataclass(frozen=True, kw_only=True)
class CyclotronRadiationHeatSink(source.SingleProfileTempElSource):
  """Cyclotron radiation loss term for the electron temp equation."""

  name: str = 'cyclotron_radiation_heat_sink'


@dataclasses.dataclass(frozen=True, kw_only=True)
class ECRHHeatSource(source.SingleProfileTempElSource):
  """ECRH heat source for the electron temp equation."""

  name: str = 'ecrh_heat_source'


@dataclasses.dataclass(frozen=True, kw_only=True)
class ICRHHeatSource(source.SingleProfileTempIonSource):
  """ICRH heat source for the ion temp equation."""

  name: str = 'icrh_heat_source'


@dataclasses.dataclass(frozen=True, kw_only=True)
class LHHeatSource(source.SingleProfileTempElSource):
  """LH heat source for the electron temp equation."""

  name: str = 'lh_heat_source'


@dataclasses.dataclass(frozen=True, kw_only=True)
class LineRadiationHeatSink(source.SingleProfileTempElSource):
  """Line radiation loss sink for the electron temp equation."""

  name: str = 'line_radiation_heat_sink'


@dataclasses.dataclass(frozen=True, kw_only=True)
class NBIElectronHeatSource(source.SingleProfileTempElSource):
  """NBI heat source for the electron temp equation."""

  name: str = 'nbi_electron_heat_source'


@dataclasses.dataclass(frozen=True, kw_only=True)
class NBIIonHeatSource(source.SingleProfileTempIonSource):
  """NBI heat source for the ion temp equation."""

  name: str = 'nbi_ion_heat_source'


@dataclasses.dataclass(frozen=True, kw_only=True)
class RecombinationHeatSink(source.SingleProfileTempElSource):
  """Recombination loss sink for the electron temp equation."""

  name: str = 'recombination_heat_sink'
