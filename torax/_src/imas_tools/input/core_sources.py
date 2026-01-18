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
from typing import Any, Final, Self

from imas import ids_structure
from imas import ids_toplevel
import numpy as np
from torax._src.imas_tools.input import loader

_ALL_AFFECTED_PROFILES: Final[Sequence[str]] = (
    "psi",
    "n_e",
    "T_i",
    "T_e",
)


@dataclass
class SourceProfiles:
  """Dataclass holding extracted profiles from an IMAS source."""

  time: Sequence[float]
  rhon: Sequence[Sequence[float]]
  # Optional profiles, depending on the source type.
  current: Sequence[float] | None = None
  ion_heat: Sequence[float] | None = None
  elec_heat: Sequence[float] | None = None
  particle: Sequence[float] | None = None

  def combine_sources(self, new_source: Self) -> None:
    """Combines 2 sources of the same type together (adds the profiles).

    Needed if the input IDS has several inputs for the same source (e.g. 1
    entry for each ec launcher).
    """
    # Checks that time and grid are the same in each source.
    if not np.allclose(self.time, new_source.time) or not np.allclose(
        self.rhon, new_source.rhon
    ):
      raise ValueError(
          "Can't combine sources with different time or radial coordinates."
      )

    def sum_profiles(profile1, profile2):
      # Profile not affected for this type of source: keep None.
      if profile1 is None and profile2 is None:
        return None
      # Unexpected case: Called outside of sources_from_IMAS with different
      # sources not affecting the same profiles.
      if profile1 is None or profile2 is None:
        raise ValueError(
            "combine_sources must be used on sources of the same type."
        )
      return np.add(profile1, profile2)

    self.current = sum_profiles(self.current, new_source.current)
    self.ion_heat = sum_profiles(self.ion_heat, new_source.ion_heat)
    self.elec_heat = sum_profiles(self.elec_heat, new_source.elec_heat)
    self.particle = sum_profiles(self.particle, new_source.particle)


class SourceCollection:
  """Accumulator object to store sources and convert them to a TORAX dict."""

  def __init__(self):
    """Mapping of source name to SourceProfiles. Note that source_name is
    the TORAX name, which can differ from the IMAS name.
    """
    self._data: dict[str, SourceProfiles] = {}

  def add(self, source_name: str, profiles: SourceProfiles) -> None:
    """Add source to accumulator. If the source already exists, combine them."""
    if source_name not in self._data:
      self._data[source_name] = profiles
    else:
      self._data[source_name].combine_sources(profiles)

  def to_dict(self) -> dict[str, Any]:
    """Finalize output to expected TORAX sources dict."""
    output = {}
    for name, profiles in self._data.items():
      values_list = []

      # Ordered to respect the expected order for each source.
      # Ion heat
      if profiles.ion_heat is not None:
        values_list.append((profiles.time, profiles.rhon, profiles.ion_heat))
      # Elec heat
      if profiles.elec_heat is not None:
        values_list.append((profiles.time, profiles.rhon, profiles.elec_heat))
      # Current
      if profiles.current is not None:
        values_list.append((profiles.time, profiles.rhon, profiles.current))
      # Particles
      if profiles.particle is not None:
        values_list.append((profiles.time, profiles.rhon, profiles.particle))

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

  Returns:
    The fields read from the IDS that can be used to completely or partially
    fill the `sources` section of a TORAX `CONFIG`.
  """
  # Checks that the IDS is of the correct type.
  if ids.metadata.name != "core_sources":
    raise ValueError(f"Expected core_sources IDS, got {ids.metadata.name} IDS.")

  accumulator = SourceCollection()
  for source in ids.source:
    source_name = str(source.identifier.name)

    # External fuelling sources.
    if "pellet" in source_name:
      # Pellet: particle source.
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["n_e"]
      )
      accumulator.add("pellet", profiles)

    elif "gas_puff" in source_name:
      # Gas puff: particle source.
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["n_e"]
      )
      accumulator.add("gas_puff", profiles)

    # External HCD sources.
    elif "ec" in source_name:
      # ECRH: electron heating and current drive.
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["T_e", "psi"]
      )
      accumulator.add("ecrh", profiles)

    elif "ic" in source_name:
      # ICRH: ion heating and current drive.
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["T_i", "T_e"]
      )
      accumulator.add("icrh", profiles)

    if not load_only_external_sources:
      # Physics-based sources
      if "ohmic" in source_name:
        profiles = _extract_source_profiles(
            source, t_initial, affected_profiles=["T_e"]
        )
        accumulator.add("ohmic", profiles)

      elif "fusion" in source_name:
        profiles = _extract_source_profiles(
            source, t_initial, affected_profiles=["T_i", "T_e"]
        )
        accumulator.add("fusion", profiles)

      elif "collisional_equipartition" in source_name:
        profiles = _extract_source_profiles(
            source, t_initial, affected_profiles=["T_i", "T_e"]
        )
        accumulator.add("ei_exchange", profiles)

      # Radiation sources
      elif "cyclotron_radiation" in source_name:
        profiles = _extract_source_profiles(
            source, t_initial, affected_profiles=["T_e"]
        )
        accumulator.add("cyclotron_radiation", profiles)

      elif "bremsstrahlung" in source_name:
        profiles = _extract_source_profiles(
            source, t_initial, affected_profiles=["T_e"]
        )
        accumulator.add("bremsstrahlung", profiles)

      elif "impurity_radiation" in source_name:
        profiles = _extract_source_profiles(
            source, t_initial, affected_profiles=["T_e"]
        )
        accumulator.add("impurity_radiation", profiles)

  return accumulator.to_dict()


def _extract_source_profiles(
    source: ids_structure.IDSStructure,
    t_initial: float | None = None,
    affected_profiles: Sequence[str] = _ALL_AFFECTED_PROFILES,
) -> SourceProfiles:
  """
  Extracts profiles for a given source from a core_sources IDS.

  Args:
      source: individual source from the core_sources IDS.
      t_initial: Initial time used to map the profiles in the dicts
      affected_profiles: List of profiles to extract. Possible values: 
        ['psi', 'n_e', 'T_e', 'T_i']. If None, extracts all profiles.
  """
  profiles_1d, rhon_array, time_array = loader.get_time_and_radial_arrays(
      source, t_initial
  )
  profiles = SourceProfiles(time=time_array, rhon=rhon_array)

  # Extract current profile
  if "psi" in affected_profiles:
    # Switch sign due to the difference between input COCOS conventions
    # and TORAX ones
    profiles.current = [-1.0 * profile.j_parallel for profile in profiles_1d]

  # Extract heating profiles
  if "T_i" in affected_profiles:
    profiles.ion_heat = [profile.total_ion_energy for profile in profiles_1d]
  if "T_e" in affected_profiles:
    profiles.elec_heat = [profile.electrons.energy for profile in profiles_1d]
  # Extract fuelling profile
  if "n_e" in affected_profiles:
    profiles.particle = [profile.electrons.particles for profile in profiles_1d]

  return profiles
