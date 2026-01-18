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


@dataclass
class _SourceProfiles:
  """Dataclass holding extracted profiles from an IMAS source."""

  time: Sequence[float]
  rho_norm: Sequence[Sequence[float]]
  # Optional profiles, depending on the source type.
  psi: Sequence[float] | None = None
  T_i: Sequence[float] | None = None
  T_e: Sequence[float] | None = None
  n_e: Sequence[float] | None = None

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

    self.psi = sum_profiles(self.psi, new_source.psi)
    self.T_i = sum_profiles(self.T_i, new_source.T_i)
    self.T_e = sum_profiles(self.T_e, new_source.T_e)
    self.n_e = sum_profiles(self.n_e, new_source.n_e)


class _SourceCollection:
  """Accumulator object to store sources and convert them to a TORAX dict."""

  def __init__(self):
    # Mapping of source name to _SourceProfiles. Note that source_name is the
    # TORAX name, which can differ from the IMAS name.
    self._data: dict[str, _SourceProfiles] = {}

  def add(self, source_name: str, profiles: _SourceProfiles) -> None:
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
      if profiles.T_i is not None:
        values_list.append((profiles.time, profiles.rho_norm, profiles.T_i))
      # Elec heat
      if profiles.T_e is not None:
        values_list.append((profiles.time, profiles.rho_norm, profiles.T_e))
      # Current
      if profiles.psi is not None:
        values_list.append((profiles.time, profiles.rho_norm, profiles.psi))
      # Particles
      if profiles.n_e is not None:
        values_list.append((profiles.time, profiles.rho_norm, profiles.n_e))

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
      profiles = _extract_source_profiles(source, ["n_e"], t_initial)
      accumulator.add("pellet", profiles)

    elif source_name == "gas_puff":
      # Gas puff: particle source.
      profiles = _extract_source_profiles(source, ["n_e"], t_initial)
      accumulator.add("gas_puff", profiles)

    # External HCD sources.
    elif source_name == "ec":
      # ECRH: electron heating and current drive.
      profiles = _extract_source_profiles(source, ["T_e", "psi"], t_initial)
      accumulator.add("ecrh", profiles)

    elif source_name == "ic":
      # ICRH: ion heating and current drive.
      profiles = _extract_source_profiles(source, ["T_i", "T_e"], t_initial)
      accumulator.add("icrh", profiles)

    if not load_only_external_sources:
      # Physics-based sources
      if source_name == "ohmic":
        profiles = _extract_source_profiles(source, ["T_e"], t_initial)
        accumulator.add("ohmic", profiles)

      elif source_name == "fusion":
        profiles = _extract_source_profiles(source, ["T_i", "T_e"], t_initial)
        accumulator.add("fusion", profiles)

      elif source_name == "collisional_equipartition":
        profiles = _extract_source_profiles(source, ["T_i", "T_e"], t_initial)
        accumulator.add("ei_exchange", profiles)

      # Radiation sources
      elif source_name == "cyclotron_radiation":
        profiles = _extract_source_profiles(source, ["T_e"], t_initial)
        accumulator.add("cyclotron_radiation", profiles)

      elif source_name == "bremsstrahlung":
        profiles = _extract_source_profiles(source, ["T_e"], t_initial)
        accumulator.add("bremsstrahlung", profiles)

      elif source_name == "impurity_radiation":
        profiles = _extract_source_profiles(source, ["T_e"], t_initial)
        accumulator.add("impurity_radiation", profiles)

  return accumulator.to_dict()


def _extract_source_profiles(
    source: ids_structure.IDSStructure,
    affected_profiles: Sequence[str],
    t_initial: float | None = None,
) -> _SourceProfiles:
  """
  Extracts profiles for a given source from a core_sources IDS.

  Args:
      source: Individual source from the core_sources IDS.
      affected_profiles: List of profiles to extract. Possible values:
        ['psi', 'n_e', 'T_e', 'T_i'].
      t_initial: Initial time used to map the profiles in the dicts.

  """
  profiles_1d, rhon_array, time_array = loader.get_time_and_radial_arrays(
      source, t_initial
  )
  profiles = _SourceProfiles(time=time_array, rho_norm=rhon_array)

  # Extract current profile
  if "psi" in affected_profiles:
    # Switch sign due to the difference between input COCOS conventions
    # and TORAX ones
    profiles.psi = [-1.0 * profile.j_parallel for profile in profiles_1d]

  # Extract heating profiles
  if "T_i" in affected_profiles:
    profiles.T_i = [profile.total_ion_energy for profile in profiles_1d]
  if "T_e" in affected_profiles:
    profiles.T_e = [profile.electrons.energy for profile in profiles_1d]
  # Extract fuelling profile
  if "n_e" in affected_profiles:
    profiles.n_e = [profile.electrons.particles for profile in profiles_1d]

  return profiles
