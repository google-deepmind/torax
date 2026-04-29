# Copyright 2025 DeepMind Technologies Limited
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
"""Useful functions to load IMAS core_sources IDSs."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, NamedTuple, Self

from absl import logging
from imas import ids_structure
from imas import ids_toplevel
import numpy as np
from torax._src.imas_tools.input import loader
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


_IMAS_SOURCE_ID_TO_TORAX_SOURCE_MAPPING = {
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
    # Physics-based and radiation sources
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


@dataclasses.dataclass
class _SourceProfiles:
  """Dataclass holding extracted profiles from an IMAS source."""

  time: Sequence[float]
  rho_norm: Sequence[Sequence[float]]
  affected_profiles: tuple[source_module.AffectedCoreProfile, ...]
  # Dict containing the affected profiles.
  profiles: dict[source_module.AffectedCoreProfile, Sequence[float]]

  def combine_sources(self, new_source: Self) -> None:
    """Combines 2 sources of the same type together (adds the profiles).

    Needed if the input IDS has several inputs for the same source (e.g. 1
    entry for each ec launcher).

    Args:
      new_source: The source to combine with this one.
    """
    # Checks that time and grid are the same in each source.
    if not np.allclose(self.time, new_source.time) or not np.allclose(
        self.rho_norm, new_source.rho_norm
    ):
      raise ValueError(
          "Can't combine sources with different time or radial coordinates."
      )
    if self.affected_profiles != new_source.affected_profiles:
      raise ValueError(
          "combines_sources must be used on sources of the same type."
      )

    for affected_profile in self.affected_profiles:
      initial_profile = self.profiles[affected_profile]
      new_profile = new_source.profiles[affected_profile]
      self.profiles[affected_profile] = np.add(initial_profile, new_profile)


@dataclasses.dataclass
class _SourceCollection:
  """Accumulator object to store sources and convert them to a TORAX dict.

  Attributes:
    _data: Dictionary of source name to _SourceProfiles, holding the sources
      read from the IDS. Note that source_name is the TORAX name, which can
      differ from the IMAS name.
  """

  _data: dict[str, _SourceProfiles] = dataclasses.field(default_factory=dict)

  def add(self, source_name: str, source_data: _SourceProfiles) -> None:
    """Add source to accumulator. If the source already exists, combine them."""
    if source_name not in self._data:
      self._data[source_name] = source_data
    else:
      self._data[source_name].combine_sources(source_data)

  def to_dict(self) -> dict[str, Any]:
    """Finalize output to expected TORAX sources dict."""
    output = {}
    for name, source_data in self._data.items():
      values_list = []
      for affected_profile in source_data.affected_profiles:
        # TODO(b/502494342): Handle fast ion sources from IMAS. For now we skip
        # them here as IMAS fast ion data seems to be in core_profiles.
        if affected_profile == source_module.AffectedCoreProfile.FAST_IONS:
          continue
        profile = source_data.profiles[affected_profile]
        values_list.append((source_data.time, source_data.rho_norm, profile))

      if values_list:
        output[name] = {
            "mode": "PRESCRIBED",
            "prescribed_values": tuple(values_list),
        }
    return output


# pylint: disable=invalid-name
def sources_from_IMAS(
    ids: ids_toplevel.IDSToplevel,
    t_initial: float | None = None,
    load_only_external_sources: bool = False,
) -> Mapping[str, Any]:
  """Converts core_sources IDS to a sources dict for TORAX config.

  Args:
    ids: A core_sources IDS object. The IDS can contain multiple time slices.
    t_initial: Initial time used to map the profiles in the dicts. If None the
      initial time will be the time of the first time slice of the ids. Else all
      time slices will be shifted such that the first time slice has time =
      t_initial.
    load_only_external_sources: If True, only loads sources from auxiliary
      systems for which models exist in TORAX (pellet, gas puff, ecrh, icrh). If
      False, also loads physics-based sources such as radiation sources.

  Returns:
    The fields read from the IDS that can be used to completely or partially
    fill the `sources` section of a TORAX `CONFIG`.
  """
  # Checks that the IDS is of the correct type.
  if ids.metadata.name != "core_sources":
    raise ValueError(f"Expected core_sources IDS, got {ids.metadata.name} IDS.")

  accumulator = _SourceCollection()
  for source in ids.source:
    imas_source_name = str(source.identifier.name)
    if imas_source_name in _IMAS_SOURCE_ID_TO_TORAX_SOURCE_MAPPING:
      source_mapping = _IMAS_SOURCE_ID_TO_TORAX_SOURCE_MAPPING[imas_source_name]
      if load_only_external_sources and not source_mapping.is_external:
        continue
      source_data = _extract_source_profiles(
          source, source_mapping.affected_core_profiles, t_initial
      )
      accumulator.add(source_mapping.torax_source_name, source_data)
    else:
      logging.info(
          "IMAS Source %s does not have a corresponding TORAX source model."
          " Skipping this source.",
          imas_source_name,
      )

  return accumulator.to_dict()


def _extract_source_profiles(
    source: ids_structure.IDSStructure,
    affected_profiles: tuple[source_module.AffectedCoreProfile, ...],
    t_initial: float | None = None,
) -> _SourceProfiles:
  """Extracts profiles for a given source from a core_sources IDS.

  Args:
      source: Individual source from the core_sources IDS.
      affected_profiles: Tuple of profiles to extract. See
        AFFECTED_CORE_PROFILES class for the possible values.
      t_initial: Initial time used to map the profiles in the dicts.

  Returns:
      _SourceProfiles object containing the extracted profiles.
  """
  profiles_1d, rhon_array, time_array = loader.get_time_and_radial_arrays(
      source, t_initial
  )

  profiles = {}
  for affected_profile in affected_profiles:
    if affected_profile == source_module.AffectedCoreProfile.PSI:
      # Switch sign due to the difference between input COCOS conventions and
      # TORAX ones. IMAS Data Dictionary version >4.0.0 (TORAX compatible
      # versions) uses COCOS=17. Previous DD versions used COCOS=11, which had
      # the opposite sign convention for psi.
      profiles[affected_profile] = [
          -1.0 * profile.j_parallel for profile in profiles_1d
      ]
    elif affected_profile == source_module.AffectedCoreProfile.TEMP_ION:
      profiles[affected_profile] = [
          profile.total_ion_energy for profile in profiles_1d
      ]
    elif affected_profile == source_module.AffectedCoreProfile.TEMP_EL:
      profiles[affected_profile] = [
          profile.electrons.energy for profile in profiles_1d
      ]
    elif affected_profile == source_module.AffectedCoreProfile.NE:
      profiles[affected_profile] = [
          profile.electrons.particles for profile in profiles_1d
      ]
    # Handling of fast ions from ICRH not taken into account yet.
    elif affected_profile == source_module.AffectedCoreProfile.FAST_IONS:
      profiles[affected_profile] = None

  return _SourceProfiles(
      time=time_array,
      rho_norm=rhon_array,
      affected_profiles=affected_profiles,
      profiles=profiles,
  )
