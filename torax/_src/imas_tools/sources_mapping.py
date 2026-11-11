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
from torax._src.sources import core_sources
from torax._src.sources import cyclotron_radiation_heat_sink
from torax._src.sources import electron_cyclotron_source
from torax._src.sources import fusion_heat_source
from torax._src.sources import gas_puff_source
from torax._src.sources import neutral_beam_injection_source
from torax._src.sources import ohmic_heat_source
from torax._src.sources import pellet_source
from torax._src.sources import qei_source
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink
from torax._src.sources.ion_cyclotron_source import base as ion_cyclotron_source


class _SourceMappingEntry(NamedTuple):
  populated_source_outputs: tuple[core_sources.PopulatedSourceOutput, ...]
  torax_source_name: str
  is_external: bool


IMAS_SOURCE_ID_TO_TORAX_SOURCE_MAPPING: Final[
    dict[str, _SourceMappingEntry]
] = {
    # External fuelling and HCD sources
    "pellet": _SourceMappingEntry(
        pellet_source.PelletSource.POPULATED_SOURCE_OUTPUTS,
        pellet_source.PelletSource.SOURCE_ID,
        True,
    ),
    "gas_puff": _SourceMappingEntry(
        gas_puff_source.GasPuffSource.POPULATED_SOURCE_OUTPUTS,
        gas_puff_source.GasPuffSource.SOURCE_ID,
        True,
    ),
    "ec": _SourceMappingEntry(
        electron_cyclotron_source.ElectronCyclotronSource.POPULATED_SOURCE_OUTPUTS,
        electron_cyclotron_source.ElectronCyclotronSource.SOURCE_ID,
        True,
    ),
    "ic": _SourceMappingEntry(
        ion_cyclotron_source.IonCyclotronSource.POPULATED_SOURCE_OUTPUTS,
        ion_cyclotron_source.IonCyclotronSource.SOURCE_ID,
        True,
    ),
    "nbi": _SourceMappingEntry(
        neutral_beam_injection_source.NeutralBeamInjectionSource.POPULATED_SOURCE_OUTPUTS,
        neutral_beam_injection_source.NeutralBeamInjectionSource.SOURCE_ID,
        True,
    ),
    "ohmic": _SourceMappingEntry(
        ohmic_heat_source.OhmicHeatSource.POPULATED_SOURCE_OUTPUTS,
        ohmic_heat_source.OhmicHeatSource.SOURCE_ID,
        False,
    ),
    "fusion": _SourceMappingEntry(
        fusion_heat_source.FusionHeatSource.POPULATED_SOURCE_OUTPUTS,
        fusion_heat_source.FusionHeatSource.SOURCE_ID,
        False,
    ),
    "collisional_equipartition": _SourceMappingEntry(
        (),
        qei_source.QeiSource.SOURCE_ID,
        False,
    ),
    "cyclotron_radiation": _SourceMappingEntry(
        cyclotron_radiation_heat_sink.CyclotronRadiationHeatSink.POPULATED_SOURCE_OUTPUTS,
        cyclotron_radiation_heat_sink.CyclotronRadiationHeatSink.SOURCE_ID,
        False,
    ),
    "bremsstrahlung": _SourceMappingEntry(
        bremsstrahlung_heat_sink.BremsstrahlungHeatSink.POPULATED_SOURCE_OUTPUTS,
        bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_ID,
        False,
    ),
    "impurity_radiation": _SourceMappingEntry(
        impurity_radiation_heat_sink.ImpurityRadiationHeatSink.POPULATED_SOURCE_OUTPUTS,
        impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_ID,
        False,
    ),
}

TORAX_SOURCE_NAME_TO_IMAS_SOURCE_ID: Final[dict[str, str]] = {
    entry.torax_source_name: imas_id
    for imas_id, entry in IMAS_SOURCE_ID_TO_TORAX_SOURCE_MAPPING.items()
}
