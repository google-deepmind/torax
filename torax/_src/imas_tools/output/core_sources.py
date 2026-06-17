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
      _fill_profiles_1d(
          source_node.profiles_1d[i],
          cs_state,
          core_profile_state,
          ppo,
          geo,
          torax_config,
          src_name,
      )
      _fill_global_quantities(
          source_node.global_quantities[i],
          cs_state,
          core_profile_state,
          ppo,
          geo,
          torax_config,
          src_name,
      )

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


def _fill_profiles_1d(
    profiles_1d_slice: imas.ids_structure.IDSStructure,
    cs_state: source_profiles.SourceProfiles,
    core_profile_state: state_lib.CoreProfiles,
    ppo: post_processing.PostProcessedOutputs,
    geo: geometry_lib.Geometry,
    torax_config: model_config.ToraxConfig,
    src_name: str,
) -> None:
  """Fills 1D profiles for a source at a single time slice."""
  energy_el_cell = np.zeros_like(geo.rho)
  energy_ion_cell = np.zeros_like(geo.rho)
  particles_el_cell = np.zeros_like(geo.rho)
  j_par_cell = np.zeros_like(geo.rho)
  # qei source stored differently so needs to be handled separately
  if src_name == "qei":
    # qei represents power to ions
    qei_val = cs_state.qei.qei_coef * (
        core_profile_state.T_e.value - core_profile_state.T_i.value
    )
    energy_ion_cell = qei_val
    energy_el_cell = -qei_val
  else:
    energy_el_cell = cs_state.T_e.get(src_name, energy_el_cell)
    energy_ion_cell = cs_state.T_i.get(src_name, energy_ion_cell)
    particles_el_cell = cs_state.n_e.get(src_name, particles_el_cell)
    j_par_cell = cs_state.psi.get(src_name, j_par_cell)

  profiles_1d_slice.electrons.energy = _extend_cell_profile_to_boundaries(
      energy_el_cell, geo
  )
  profiles_1d_slice.total_ion_energy = _extend_cell_profile_to_boundaries(
      energy_ion_cell, geo
  )
  profiles_1d_slice.electrons.particles = _extend_cell_profile_to_boundaries(
      particles_el_cell, geo
  )
  # TODO(b/335204606): Clean this up once we finalize our COCOS convention.
  # Currents sign flipped due to the difference between TORAX COCOS convention
  # and IMAS COCOS.
  profiles_1d_slice.j_parallel = -1.0 * _extend_cell_profile_to_boundaries(
      j_par_cell, geo
  )

  # TODO: Map Ion Species-Specific Particle Sources. For this we
  # will need to know which ion(s) each source is affecting. For now we just
  # create the ion substructure for all ions (including impurities) without
  # filling the profiles.
  main_ions = list(core_profile_state.main_ion_fractions.keys())
  impurities = list(core_profile_state.impurity_fractions.keys())
  all_ions = main_ions + impurities
  profiles_1d_slice.ion.resize(len(all_ions))
  for k, symbol in enumerate(all_ions):
    ion_properties = constants.ION_PROPERTIES_DICT[symbol]
    profiles_1d_slice.ion[k].name = symbol
    profiles_1d_slice.ion[k].element.resize(1)
    profiles_1d_slice.ion[k].element[0].a = ion_properties.A
    profiles_1d_slice.ion[k].element[0].z_n = ion_properties.Z


def _fill_global_quantities(
    global_quantities_slice: imas.ids_structure.IDSStructure,
    cs_state: source_profiles.SourceProfiles,
    core_profile_state: state_lib.CoreProfiles,
    ppo: post_processing.PostProcessedOutputs,
    geo: geometry_lib.Geometry,
    torax_config: model_config.ToraxConfig,
    src_name: str,
) -> None:
  """Fills integrated global quantities for a source at a single time slice."""
  energy_el_cell = np.zeros_like(geo.rho)
  energy_ion_cell = np.zeros_like(geo.rho)
  particles_el_cell = np.zeros_like(geo.rho)
  j_par_cell = np.zeros_like(geo.rho)

  if src_name == "qei":
    qei_val = cs_state.qei.qei_coef * (
        core_profile_state.T_e.value - core_profile_state.T_i.value
    )
    energy_ion_cell = qei_val
    energy_el_cell = -qei_val
  else:
    energy_el_cell = cs_state.T_e.get(src_name, energy_el_cell)
    energy_ion_cell = cs_state.T_i.get(src_name, energy_ion_cell)
    particles_el_cell = cs_state.n_e.get(src_name, particles_el_cell)
    j_par_cell = cs_state.psi.get(src_name, j_par_cell)

  power_el = math_utils.volume_integration(energy_el_cell, geo)
  power_ion = math_utils.volume_integration(energy_ion_cell, geo)
  particles_el_int = math_utils.volume_integration(particles_el_cell, geo)
  # TODO(b/335204606): Clean this up once we finalize our COCOS convention.
  # Currents sign flipped due to the difference between TORAX COCOS convention
  # and IMAS COCOS.
  j_curr_int = -1.0 * math_utils.area_integration(j_par_cell, geo)

  global_quantities_slice.electrons.power = power_el
  global_quantities_slice.total_ion_power = power_ion
  global_quantities_slice.power = power_el + power_ion
  global_quantities_slice.electrons.particles = particles_el_int
  global_quantities_slice.current_parallel = j_curr_int
