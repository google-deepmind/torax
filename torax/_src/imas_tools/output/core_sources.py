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
from torax._src import state
from torax._src.geometry import geometry as geometry_lib
from torax._src.imas_tools import sources_mapping
from torax._src.sources import qei_source
from torax._src.sources import source_profiles
from torax._src import math_utils


# pylint: disable=invalid-name
def core_sources_to_IMAS(
    core_profiles: Sequence[state.CoreProfiles],
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
    if not "generic" in source_name:
      imas_name = sources_mapping.TORAX_SOURCE_NAME_TO_IMAS_SOURCE_ID[source_name]
      source_node.identifier.name = imas_name
      # TODO(b/323504363): b/459479939 - i/2233: Add identifier in once a mapping is available
      # in IMAS-python https://github.com/iterorganization/IMAS-Python/issues/134
      if imas_name == "custom_1":
        source_node.identifier.description = "TORAX generic current source"
      if imas_name == "custom_2":
        source_node.identifier.description = "TORAX generic heating source"
      if imas_name == "custom_3":
        source_node.identifier.description = "TORAX generic particle source"

      source_node.profiles_1d.resize(num_times)
      source_node.global_quantities.resize(num_times)

      for i in range(num_times):
        t = times[i]
        geo = geometry[i]
        core_source_state = core_sources[i]
        core_profile_state = core_profiles[i]

        source_node.profiles_1d[i].time = t
        source_node.global_quantities[i].time = t

        _fill_grid_coordinates(source_node.profiles_1d[i], geo)
        _fill_profiles_1d(
            source_node.profiles_1d[i],
            core_source_state,
            core_profile_state,
            geo,
            source_name,
        )
        _fill_global_quantities(
            source_node.global_quantities[i],
            core_source_state,
            core_profile_state,
            geo,
            source_name,
        )

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


def _fill_profiles_1d(
    profiles_1d_slice: imas.ids_structure.IDSStructure,
    core_source_state: source_profiles.SourceProfiles,
    core_profile_state: state.CoreProfiles,
    geo: geometry_lib.Geometry,
    source_name: str,
) -> None:
  """Fills 1D profiles for a source at a single time slice."""
  energy_el = np.zeros_like(geo.rho)
  energy_ion = np.zeros_like(geo.rho)
  particles_el = np.zeros_like(geo.rho)
  j_par = np.zeros_like(geo.rho)
  # qei source stored differently so needs to be handled separately
  if source_name == qei_source.QeiSource.SOURCE_NAME:
    # qei represents power to ions
    qei_val = core_source_state.qei.qei_coef * (
        core_profile_state.T_e.value - core_profile_state.T_i.value
    )
    energy_ion = qei_val
    energy_el = -qei_val
  else:
    energy_el = core_source_state.T_e.get(source_name, energy_el)
    energy_ion = core_source_state.T_i.get(source_name, energy_ion)
    particles_el = core_source_state.n_e.get(source_name, particles_el)
    j_par = core_source_state.psi.get(source_name, j_par)

  profiles_1d_slice.electrons.energy = energy_el
  profiles_1d_slice.total_ion_energy = energy_ion
  profiles_1d_slice.electrons.particles = particles_el
  # TODO(b/335204606): Clean this up once we finalize our COCOS convention.
  # Currents sign flipped due to the difference between TORAX COCOS convention
  # and IMAS COCOS.
  profiles_1d_slice.j_parallel = -1.0 * j_par

  # TODO: Map Ion Species-Specific Particle Sources. For this we
  # will need to know which ion(s) each source is affecting. For now we just
  # create the ion substructure for all ions (including impurities) without
  # filling the profiles.
  main_ions = list(core_profile_state.main_ion_fractions.keys())
  impurities = list(core_profile_state.impurity_fractions.keys())
  all_ions = main_ions + impurities
  profiles_1d_slice.ion.resize(len(all_ions))
  for ion_node, ion_symbol in zip(profiles_1d_slice.ion, all_ions):
    ion_properties = constants.ION_PROPERTIES_DICT[ion_symbol]
    ion_node.name = ion_symbol
    ion_node.element.resize(1)
    ion_node.element[0].a = ion_properties.A
    ion_node.element[0].z_n = ion_properties.Z


def _fill_global_quantities(
    global_quantities_slice: imas.ids_structure.IDSStructure,
    core_source_state: source_profiles.SourceProfiles,
    core_profile_state: state.CoreProfiles,
    geo: geometry_lib.Geometry,
    source_name: str,
) -> None:
  """Fills global quantities for a source at a single time slice."""
  energy_el = np.zeros_like(geo.rho)
  energy_ion = np.zeros_like(geo.rho)
  particles_el = np.zeros_like(geo.rho)
  j_par = np.zeros_like(geo.rho)

  if source_name == qei_source.QeiSource.SOURCE_NAME:
    qei_val = core_source_state.qei.qei_coef * (
        core_profile_state.T_e.value - core_profile_state.T_i.value
    )
    energy_ion = qei_val
    energy_el = -qei_val
  else:
    energy_el = core_source_state.T_e.get(source_name, energy_el)
    energy_ion = core_source_state.T_i.get(source_name, energy_ion)
    particles_el = core_source_state.n_e.get(source_name, particles_el)
    j_par = core_source_state.psi.get(source_name, j_par)

  power_el = math_utils.volume_integration(energy_el, geo)
  power_ion = math_utils.volume_integration(energy_ion, geo)
  particles_el_int = math_utils.volume_integration(particles_el, geo)
  # TODO(b/335204606): Clean this up once we finalize our COCOS convention.
  # Currents sign flipped due to the difference between TORAX COCOS convention
  # and IMAS COCOS.
  j_curr_int = -1.0 * math_utils.area_integration(j_par, geo)

  global_quantities_slice.electrons.power = power_el
  global_quantities_slice.total_ion_power = power_ion
  global_quantities_slice.power = power_el + power_ion
  global_quantities_slice.electrons.particles = particles_el_int
  global_quantities_slice.current_parallel = j_curr_int
