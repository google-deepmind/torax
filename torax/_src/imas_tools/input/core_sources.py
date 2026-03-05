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
from dataclasses import dataclass
from typing import Any, Self

from imas import ids_structure
from imas import ids_toplevel
import numpy as np
from torax._src.imas_tools.input import loader
from torax._src.sources import bremsstrahlung_heat_sink as brehmsstrahlung
from torax._src.sources import cyclotron_radiation_heat_sink as cyclotron_radiation
from torax._src.sources import electron_cyclotron_source as ecrh
from torax._src.sources import fusion_heat_source as fusion
from torax._src.sources import gas_puff_source as gas_puff
from torax._src.sources import ion_cyclotron_source as icrh
from torax._src.sources import ohmic_heat_source as ohmic
from torax._src.sources import pellet_source as pellet
from torax._src.sources import qei_source as qei
from torax._src.sources import source as source_module
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink as impurity_radiation


@dataclass
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
    """
    # Checks that time and grid are the same in each source.
    if not np.allclose(self.time, new_source.time) or not np.allclose(
        self.rho_norm, new_source.rho_norm
    ):
      raise ValueError(
          "Can't combine sources with different time or radial coordinates."
      )
    if not self.affected_profiles == new_source.affected_profiles:
      raise ValueError(
          "combines_sources must be used on sources of the same type."
      )

    for affected_profile in self.affected_profiles:
      initial_profile = self.profiles[affected_profile]
      new_profile = new_source.profiles[affected_profile]
      self.profiles[affected_profile] = np.add(initial_profile, new_profile)


class _SourceCollection:
  """Accumulator object to store sources and convert them to a TORAX dict."""

  def __init__(self):
    # Mapping of source name to _SourceProfiles. Note that source_name is the
    # TORAX name, which can differ from the IMAS name.
    self._data: dict[str, _SourceProfiles] = {}

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
      # Automatically ordered thanks to the affected_profiles class atribute.
      for affected_profile in source_data.affected_profiles:
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
      systems for which models exist in TORAX (pellet, gas puff, ecrh, icrh).
      If False, also loads physics-based sources such as radiation sources.

  Returns:
    The fields read from the IDS that can be used to completely or partially
    fill the `sources` section of a TORAX `CONFIG`.
  """
  # Checks that the IDS is of the correct type.
  if ids.metadata.name != "core_sources":
    raise ValueError(f"Expected core_sources IDS, got {ids.metadata.name} IDS.")

  accumulator = _SourceCollection()
  for source in ids.source:
    source_name = str(source.identifier.name)

    # External fuelling sources.
    if source_name == "pellet":
      # Pellet: particle source.
      source_data = _extract_source_profiles(
          source, pellet.PelletSource.AFFECTED_CORE_PROFILES, t_initial
      )
      accumulator.add("pellet", source_data)

    elif source_name == "gas_puff":
      # Gas puff: particle source.
      source_data = _extract_source_profiles(
          source, gas_puff.GasPuffSource.AFFECTED_CORE_PROFILES, t_initial
      )
      accumulator.add("gas_puff", source_data)

    # External HCD sources.
    elif source_name == "ec":
      # ECRH: electron heating and current drive.
      source_data = _extract_source_profiles(
          source, ecrh.ElectronCyclotronSource.AFFECTED_CORE_PROFILES, t_initial
      )
      accumulator.add("ecrh", source_data)

    elif source_name == "ic":
      # ICRH: ion heating and current drive.
      source_data = _extract_source_profiles(
          source, icrh.IonCyclotronSource.AFFECTED_CORE_PROFILES, t_initial
      )
      accumulator.add("icrh", source_data)

    if not load_only_external_sources:
      # Physics-based sources
      if source_name == "ohmic":
        source_data = _extract_source_profiles(
            source, ohmic.OhmicHeatSource.AFFECTED_CORE_PROFILES, t_initial
        )
        accumulator.add("ohmic", source_data)

      elif source_name == "fusion":
        source_data = _extract_source_profiles(
            source, fusion.FusionHeatSource.AFFECTED_CORE_PROFILES, t_initial
        )
        accumulator.add("fusion", source_data)

      elif source_name == "collisional_equipartition":
        source_data = _extract_source_profiles(
            source, qei.QeiSource.AFFECTED_CORE_PROFILES, t_initial
        )
        accumulator.add("ei_exchange", source_data)

      # Radiation sources
      elif source_name == "cyclotron_radiation":
        source_data = _extract_source_profiles(
            source,
            cyclotron_radiation.CyclotronRadiationHeatSink.AFFECTED_CORE_PROFILES,
            t_initial,
        )
        accumulator.add("cyclotron_radiation", source_data)

      elif source_name == "bremsstrahlung":
        source_data = _extract_source_profiles(
            source,
            brehmsstrahlung.BremsstrahlungHeatSink.AFFECTED_CORE_PROFILES,
            t_initial,
        )
        accumulator.add("bremsstrahlung", source_data)

      elif source_name == "impurity_radiation":
        source_data = _extract_source_profiles(
            source,
            impurity_radiation.ImpurityRadiationHeatSink.AFFECTED_CORE_PROFILES,
            t_initial,
        )
        accumulator.add("impurity_radiation", source_data)

  return accumulator.to_dict()


def _extract_source_profiles(
    source: ids_structure.IDSStructure,
    affected_profiles: tuple[source_module.AffectedCoreProfile, ...],
    t_initial: float | None = None,
) -> _SourceProfiles:
  """
  Extracts profiles for a given source from a core_sources IDS.

  Args:
      source: Individual source from the core_sources IDS.
      affected_profiles: Tuple of profiles to extract. See
        AFFECTED_CORE_PROFILES class for the possible values.
      t_initial: Initial time used to map the profiles in the dicts.

  """
  profiles_1d, rhon_array, time_array = loader.get_time_and_radial_arrays(
      source, t_initial
  )

  profiles = {}
  for affected_profile in affected_profiles:
    if affected_profile == source_module.AffectedCoreProfile.PSI:
      # Switch sign due to the difference between input COCOS conventions and
      # TORAX ones
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

  return _SourceProfiles(
      time=time_array,
      rho_norm=rhon_array,
      affected_profiles=affected_profiles,
      profiles=profiles,
  )
