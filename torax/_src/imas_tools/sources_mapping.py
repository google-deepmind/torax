# Copyright 2026 DeepMind Technologies Limited
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

"""Maps TORAX source names to IMAS source IDS and vice-versa."""

from typing import Final, NamedTuple

from torax._src.sources import bremsstrahlung_heat_sink
from torax._src.sources import cyclotron_radiation_heat_sink
from torax._src.sources import electron_cyclotron_source
from torax._src.sources import fusion_heat_source
from torax._src.sources import gas_puff_source
from torax._src.sources import ohmic_heat_source
from torax._src.sources import pellet_source
from torax._src.sources import qei_source
from torax._src.sources import source as source_module
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink
from torax._src.sources.ion_cyclotron_source import base as ion_cyclotron_source


class _SourceMappingEntry(NamedTuple):
  affected_core_profiles: tuple[source_module.AffectedCoreProfile, ...]
  torax_source_name: str
  is_external: bool


IMAS_SOURCE_ID_TO_TORAX_SOURCE_MAPPING: Final[
    dict[str, _SourceMappingEntry]
] = {
    # External fuelling and HCD sources
    "pellet": _SourceMappingEntry(
        pellet_source.PelletSource.AFFECTED_CORE_PROFILES,
        pellet_source.PelletSource.SOURCE_NAME,
        True,
    ),
    "gas_puff": _SourceMappingEntry(
        gas_puff_source.GasPuffSource.AFFECTED_CORE_PROFILES,
        gas_puff_source.GasPuffSource.SOURCE_NAME,
        True,
    ),
    "ec": _SourceMappingEntry(
        electron_cyclotron_source.ElectronCyclotronSource.AFFECTED_CORE_PROFILES,
        electron_cyclotron_source.ElectronCyclotronSource.SOURCE_NAME,
        True,
    ),
    "ic": _SourceMappingEntry(
        ion_cyclotron_source.IonCyclotronSource.AFFECTED_CORE_PROFILES,
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME,
        True,
    ),
    "ohmic": _SourceMappingEntry(
        ohmic_heat_source.OhmicHeatSource.AFFECTED_CORE_PROFILES,
        ohmic_heat_source.OhmicHeatSource.SOURCE_NAME,
        False,
    ),
    "fusion": _SourceMappingEntry(
        fusion_heat_source.FusionHeatSource.AFFECTED_CORE_PROFILES,
        fusion_heat_source.FusionHeatSource.SOURCE_NAME,
        False,
    ),
    "collisional_equipartition": _SourceMappingEntry(
        qei_source.QeiSource.AFFECTED_CORE_PROFILES,
        qei_source.QeiSource.SOURCE_NAME,
        False,
    ),
    "cyclotron_radiation": _SourceMappingEntry(
        cyclotron_radiation_heat_sink.CyclotronRadiationHeatSink.AFFECTED_CORE_PROFILES,
        cyclotron_radiation_heat_sink.CyclotronRadiationHeatSink.SOURCE_NAME,
        False,
    ),
    "bremsstrahlung": _SourceMappingEntry(
        bremsstrahlung_heat_sink.BremsstrahlungHeatSink.AFFECTED_CORE_PROFILES,
        bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_NAME,
        False,
    ),
    "impurity_radiation": _SourceMappingEntry(
        impurity_radiation_heat_sink.ImpurityRadiationHeatSink.AFFECTED_CORE_PROFILES,
        impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME,
        False,
    ),
}

TORAX_SOURCE_NAME_TO_IMAS_SOURCE_ID: Final[dict[str, str]] = {
    entry.torax_source_name: imas_id
    for imas_id, entry in IMAS_SOURCE_ID_TO_TORAX_SOURCE_MAPPING.items()
}
