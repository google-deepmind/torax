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
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src import state as state_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.imas_tools.input.core_sources import _IMAS_SOURCE_ID_TO_TORAX_SOURCE_MAPPING
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.physics import psi_calculations
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import model_config

_TORAX_SOURCE_NAME_TO_IMAS_SOURCE_ID = {
    entry.torax_source_name: imas_id
    for imas_id, entry in _IMAS_SOURCE_ID_TO_TORAX_SOURCE_MAPPING.items()
}

_IMAS_SOURCE_ID_TO_IDENTIFIER_INDEX = {
    "total": 1,
    "ec": 3,
    "ic": 5,
    "fusion": 6,
    "ohmic": 7,
    "bremsstrahlung": 8,
    "collisional_equipartition": 11,
    "bootstrap_current": 13,
    "pellet": 14,
    "gas_puff": 108,
    "cyclotron_synchrotron_radiation": 202,
    "impurity_radiation": 203,
}


def core_sources_to_IMAS(
    torax_config: model_config.ToraxConfig,
    post_processed_outputs: Sequence[post_processing.PostProcessedOutputs],
    core_profiles: Sequence[state_lib.CoreProfiles],
    core_sources: Sequence[source_profiles.SourceProfiles],
    geometry: Sequence[geometry_lib.Geometry],
    times: array_typing.FloatVector,
    ids: ids_toplevel.IDSToplevel | None = None,
) -> ids_toplevel.IDSToplevel:
  """Save TORAX source profiles into an IMAS core_sources IDS."""
  if ids is None:
    ids = imas.IDSFactory().core_sources()
  elif ids.metadata.name != "core_sources":
    raise TypeError(f"Expected core_sources IDS, got {ids.metadata.name} IDS.")

  _fill_metadata(ids)
  ids.time = times

  # Identify all active sources across dictionaries
  active_sources = set()
  for cs in core_sources:
    active_sources.update(cs.T_e.keys())
    active_sources.update(cs.T_i.keys())
    active_sources.update(cs.n_e.keys())
    active_sources.update(cs.psi.keys())
    active_sources.update(cs.fast_ions.keys())
    # Add qei if coefficients are computed.
    if hasattr(cs, "qei") and cs.qei is not None:
      if np.any(np.abs(cs.qei.qei_coef) > 0.0):
        active_sources.add("qei")

  sorted_sources = sorted(list(active_sources))
  ids.source.resize(len(sorted_sources))

  for idx, src_name in enumerate(sorted_sources):
    imas_name = _TORAX_SOURCE_NAME_TO_IMAS_SOURCE_ID.get(src_name, src_name)
    source_node = ids.source[idx]
    source_node.identifier.name = imas_name
    source_node.identifier.index = _IMAS_SOURCE_ID_TO_IDENTIFIER_INDEX.get(
        imas_name, 0
    )

    source_node.profiles_1d.resize(len(times))
    source_node.global_quantities.resize(len(times))

    for i in range(len(times)):
      t = times[i]
      geo = geometry[i]
      cs_state = core_sources[i]
      core_profile_state = core_profiles[i]
      ppo = post_processed_outputs[i]

      source_node.profiles_1d[i].time = t
      source_node.global_quantities[i].time = t

      _fill_grid_coordinates(source_node.profiles_1d[i], geo)
          source_node.profiles_1d[i],
          source_node.global_quantities[i],

  return ids


def _fill_metadata(ids: ids_toplevel.IDSToplevel):
  """Fills metadata in-place for the core_sources IDS."""
  ids.ids_properties.comment = (
      "IDS built from TORAX simulation source profiles. Grid based on TORAX "
      "cell grid + boundaries."
  )
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
  grid.rho_tor_norm = np.concatenate([[0.0], geo.rho_norm, [1.0]])
  grid.rho_tor = np.concatenate([[0.0], geo.rho, [geo.rho_b]])
  grid.volume = _extend_cell_profile_to_boundaries(geo.volume, geo)
  grid.area = _extend_cell_profile_to_boundaries(geo.area, geo)


def _extend_cell_profile_to_boundaries(
    cell_val: np.ndarray,
    geo: geometry_lib.Geometry,
) -> np.ndarray:
  """Extends cell-centered profile to boundaries."""
  face_val = math_utils.cell_to_face(cell_val, geo)
  return output.extend_cell_grid_to_boundaries(
      [cell_val], np.array([face_val])
  )[0]

