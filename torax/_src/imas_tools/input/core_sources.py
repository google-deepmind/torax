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
from typing import Any, Final

from imas import ids_structure
from imas import ids_toplevel
from torax._src.imas_tools.input import loader

_ALL_AFFECTED_PROFILES: Final[Sequence[str]] = (
    "psi",
    "n_e",
    "T_i",
    "T_e",
)


# pylint: disable=invalid-name
def sources_from_IMAS(
    ids: ids_toplevel.IDSToplevel,
    t_initial: float | None = None,
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
  sources_output = {}
  for source in ids.source:
    source_name = str(source.identifier.name)

    # External fuelling sources.
    if "pellet" in source_name:
      # Pellet: particle source.
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["ne"]
      )

      sources_output["pellet"] = {
          "mode": "PRESCRIBED",
          "prescribed_values": (
              (profiles["time"], profiles["rhon"], profiles["particle"]),
          ),
      }

    elif "gas_puff" in source_name:
      # Gas puff: particle source.
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["ne"]
      )

      sources_output["gas_puff"] = {
          "mode": "PRESCRIBED",
          "prescribed_values": (
              (profiles["time"], profiles["rhon"], profiles["particle"]),
          ),
      }
    # External HCD sources.
    elif "ec" in source_name:
      # ECRH: electron heating and current drive.
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["temp_el", "current"]
      )

      sources_output["ecrh"] = {
          "mode": "PRESCRIBED",
          "prescribed_values": (
              (profiles["time"], profiles["rhon"], profiles["elec_heat"]),
              (profiles["time"], profiles["rhon"], profiles["current"]),
          ),
      }

    elif "ic" in source_name:
      # ICRH: ion heating and current drive.
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["temp_ion", "temp_el"]
      )

      sources_output["icrh"] = {
          "mode": "PRESCRIBED",
          "prescribed_values": (
              (profiles["time"], profiles["rhon"], profiles["ion_heat"]),
              (profiles["time"], profiles["rhon"], profiles["elec_heat"]),
          ),
      }
    # Physics-based sources
    elif "ohmic" in source_name:
      # Ohmic: electron heating
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["temp_el"]
      )

      sources_output["ohmic"] = {
          "mode": "PRESCRIBED",
          "prescribed_values": (
              (profiles["time"], profiles["rhon"], profiles["elec_heat"]),
          ),
      }

    elif "fusion" in source_name:
      # Fusion: ion and electron heating
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["temp_ion", "temp_el"]
      )

      sources_output["fusion"] = {
          "mode": "PRESCRIBED",
          "prescribed_values": (
              (profiles["time"], profiles["rhon"], profiles["ion_heat"]),
              (profiles["time"], profiles["rhon"], profiles["elec_heat"]),
          ),
      }

    elif "collisional_equipartition" in source_name:
      # Collisional equipartition: ion and electron heating
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["temp_ion", "temp_el"]
      )
      sources_output["ei_exchange"] = {
          "mode": "PRESCRIBED",
          "prescribed_values": (
              (profiles["time"], profiles["rhon"], profiles["ion_heat"]),
              (profiles["time"], profiles["rhon"], profiles["elec_heat"]),
          ),
      }
    # Radiation sources
    elif "cyclotron_radiation" in source_name:
      # cyclotron radiation sink.
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["temp_el"]
      )

      sources_output["cyclotron_radiation"] = {
          "mode": "PRESCRIBED",
          "prescribed_values": (
              (profiles["time"], profiles["rhon"], profiles["elec_heat"]),
          ),
      }
    elif "bremsstrahlung" in source_name:
      # bremsstrahlung radiation sink.
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["temp_el"]
      )

      sources_output["bremsstrahlung"] = {
          "mode": "PRESCRIBED",
          "prescribed_values": (
              (profiles["time"], profiles["rhon"], profiles["elec_heat"]),
          ),
      }
    elif "impurity_radiation" in source_name:
      # impurity radiation sink.
      profiles = _extract_source_profiles(
          source, t_initial, affected_profiles=["temp_el"]
      )

      sources_output["impurity_radiation"] = {
          "mode": "PRESCRIBED",
          "prescribed_values": (
              (profiles["time"], profiles["rhon"], profiles["elec_heat"]),
          ),
      }
  return sources_output


def _extract_source_profiles(
    source: ids_structure.IDSStructure,
    t_initial: float | None = None,
    affected_profiles: Sequence[str] = _ALL_AFFECTED_PROFILES,
) -> Mapping[str, Any]:
  """
  Extract profiles for a given source from a core_sources IDS.

  Args:
      source: individual source from the core_sources IDS.
      t_initial: Initial time used to map the profiles in the dicts
      affected_profiles: List of profiles to extract.
        Possible values: ['psi', 'ne', 'temp_ion', 'temp_el']. If None, extract
        all profiles.
  """
  profiles_1d, rhon_array, time_array = loader.get_time_and_radial_arrays(
      source, t_initial
  )
  profiles = {}
  profiles["time"] = time_array
  profiles["rhon"] = rhon_array

  # Extract current profile
  if "psi" in affected_profiles:
    # Switch sign due to the difference between input COCOS conventions
    # and TORAX ones
    profiles["current"] = [-1.0 * profile.j_parallel for profile in profiles_1d]

  # Extract heating profiles
  if "T_i" in affected_profiles:
    profiles["ion_heat"] = [profile.total_ion_energy for profile in profiles_1d]
  if "T_e" in affected_profiles:
    profiles["elec_heat"] = [
        profile.electrons.energy for profile in profiles_1d
    ]
  # Extract fuelling profile
  if "n_e" in affected_profiles:
    profiles["particle"] = [
        profile.electrons.particles for profile in profiles_1d
    ]

  return profiles
