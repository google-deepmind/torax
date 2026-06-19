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

"""Functions to save IMAS core_sources IDSs from TORAX."""

from collections.abc import Sequence
import datetime

import imas
from imas import ids_toplevel
from torax._src import array_typing
from torax._src.geometry import geometry as geometry_lib
from torax._src.imas_tools import sources_mapping
from torax._src.sources import qei_source
from torax._src.sources import source_profiles


# pylint: disable=invalid-name
def core_sources_to_IMAS(
    core_sources: Sequence[source_profiles.SourceProfiles],
    geometry: Sequence[geometry_lib.Geometry],
    times: array_typing.FloatVector,
) -> ids_toplevel.IDSToplevel:
  """Save TORAX outputs into an IMAS core_sources IDS."""
  ids = imas.IDSFactory().core_sources()
  _fill_metadata(ids)
  ids.time = times
  num_times = len(times)
  # Collect active TORAX source names.
  active_sources = set()
  for source in core_sources:
    active_sources.update(source.T_e.keys())
    active_sources.update(source.T_i.keys())
    active_sources.update(source.n_e.keys())
    active_sources.update(source.psi.keys())
    active_sources.update(source.fast_ions.keys())
    # qei source treated differently so added separately.
    active_sources.add(qei_source.QeiSource.SOURCE_NAME)

  ids.source.resize(len(active_sources))

  for source_name, source_node in zip(active_sources, ids.source, strict=True):
    imas_name = sources_mapping.TORAX_SOURCE_NAME_TO_IMAS_SOURCE_ID[source_name]
    source_node.identifier.name = imas_name
    # TODO(b/323504363): b/459479939 - i/2233: Add identifier in once a mapping is available
    # in IMAS-python https://github.com/iterorganization/IMAS-Python/issues/134
    source_node.profiles_1d.resize(num_times)
    source_node.global_quantities.resize(num_times)

    for i in range(num_times):
      t = times[i]
      geo = geometry[i]

      source_node.profiles_1d[i].time = t
      source_node.global_quantities[i].time = t

      _fill_grid_coordinates(source_node.profiles_1d[i], geo)

  return ids


def _fill_metadata(ids: ids_toplevel.IDSToplevel):
  """Fills metadata in-place for the core_sources IDS."""
  ids.ids_properties.comment = (
      "IDS built from TORAX simulation source profiles. Grid based on TORAX "
      "cell grid."
  )
  # Homogeneous_time = 1 means time values are stored in the time node just
  # below the root of the IDS.
  ids.ids_properties.homogeneous_time = 1
  ids.ids_properties.creation_date = datetime.date.today().isoformat()
  ids.code.name = "TORAX"
  ids.code.description = (
      "TORAX is a differentiable tokamak core transport simulator."
  )
  ids.code.repository = "https://github.com/google-deepmind/torax"


def _fill_grid_coordinates(
    profiles_1d_slice: imas.ids_structure.IDSStructure,
    geo: geometry_lib.Geometry,
) -> None:
  """Fills 1D grid coordinates for a given time slice."""
  grid = profiles_1d_slice.grid
  grid.rho_tor_norm = geo.rho_norm
  grid.rho_tor = geo.rho
  grid.volume = geo.volume
  grid.area = geo.area
