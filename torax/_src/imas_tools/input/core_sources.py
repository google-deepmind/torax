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

from imas import ids_struct_array
from imas import ids_toplevel
import numpy as np
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
    The updated fields read from the IDS that can be used to completely or
    partially fill the `sources` section of a TORAX `CONFIG`.
  """
  # Checks that the IDS is of the correct type.
  if not ids.metadata.name == "core_sources":
    raise TypeError(
        f"Expected core_sources IDS, got {ids.metadata.name} IDS."
    )
  sources_output = {}
  for source in ids.source:
    source_name = source.identifier.name
    # TODO: Basic output structure to be replaced by parsing of the different
    # sources building the expected structure for TORAX sources runtime_params.
    sources_output[source_name] = _extract_source_profiles(
        source,
        t_initial=t_initial,
        affected_profiles=_ALL_AFFECTED_PROFILES,
    )
  return sources_output


def _extract_source_profiles(
    source: ids_struct_array.IDSStructArray,
    t_initial: float | None = None,
    affected_profiles: Sequence[str] = _ALL_AFFECTED_PROFILES,
) -> Mapping[str, np.ndarray]:
  """
  Extract profiles for a given source from a core_sources IDS.

  Args:
      source: individual source from the core_sources IDS.
      time_array: Time array of the source.
      affected_profiles: List of profiles to extract.
        Possible values: ['psi', 'ne', 'temp_ion', 'temp_el']. If None, extract
        all profiles(default).

  Returns:
      A dictionary containing the extracted profiles.
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
