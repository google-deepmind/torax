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

"""Functions to save IMAS core_transport IDSs from TORAX."""

from collections.abc import Sequence
import datetime

import imas
from imas import ids_toplevel
import numpy as np
from torax._src import array_typing
from torax._src import state
from torax._src.geometry import geometry as geometry_lib
from torax._src.output_tools import post_processing
from torax._src.torax_pydantic import model_config

_IMAS_MODEL_ID_TO_IDENTIFIER_INDEX = {
    "combined": 1,
    "transport_solver": 2,
    "neoclassical": 5,
    "anomalous": 6,
}


# pylint: disable=invalid-name,unused-argument,too-many-positional-arguments
def core_transport_to_IMAS(
    torax_config: model_config.ToraxConfig,
    post_processed_outputs: Sequence[post_processing.PostProcessedOutputs],
    core_profiles: Sequence[state.CoreProfiles],
    core_transport: Sequence[state.CoreTransport],
    geometry: Sequence[geometry_lib.Geometry],
    times: array_typing.FloatVector,
    ids: ids_toplevel.IDSToplevel | None = None,
) -> ids_toplevel.IDSToplevel:
  """Save TORAX transport coefficients into an IMAS core_transport IDS.

  The output grid for all 1D quantities is the "cell_plus_boundaries".
  The function can be used to save an entire trajectory or a single time slice.
  If you want to use this function programatically and save a single time
  slice, please make sure the inputs are `Sequence`s of length 1.

  Args:
    torax_config: ToraxConfig object to get number of main ions.
    post_processed_outputs: Sequence of TORAX PostProcessedOutputs objects.
    core_profiles: Sequence of TORAX CoreProfiles objects.
    core_transport: Sequence of TORAX CoreTransport objects.
    geometry: Sequence of TORAX Geometry objects.
    times: Time array of the slices to save.
    ids: Optional IDS object to be filled. If not provided a core_transport IDS
      will be created and output.

  Returns:
    Filled core_transport IDS object.
  """
  if ids is None:
    ids = imas.IDSFactory().core_transport()
  elif ids.metadata.name != "core_transport":
    raise TypeError(
        f"Expected core_transport IDS, got {ids.metadata.name} IDS."
    )

  _fill_metadata(ids)
  ids.time = times
  model_names = list(_IMAS_MODEL_ID_TO_IDENTIFIER_INDEX.keys())
  ids.model.resize(len(model_names))

  for idx, model_name in enumerate(model_names):
    model_node = ids.model[idx]
    model_node.identifier = imas.identifiers.core_transport_identifier[
        model_name
    ]

    if model_name == "anomalous":
      transport_cfg = torax_config.transport
      if transport_cfg.model_name == "combined":
        non_constant_names = [
            m.model_name
            for m in transport_cfg.transport_models
            if m.model_name != "constant"
        ]
        if non_constant_names:
          model_node.code.name = "+".join(non_constant_names)
      elif not model_name == "constant":
        model_node.code.name = model_name
    elif model_name == "neoclassical":
      model_node.code.name = "TORAX angioni-sauter"

    model_node.profiles_1d.resize(len(times))

    for i in range(len(times)):
      t = times[i]
      geo = geometry[i]
      core_transport_state = core_transport[i]
      core_profiles_state = core_profiles[i]

      model_node.profiles_1d[i].time = t
      _fill_grid_coordinates(
          model_node.profiles_1d[i], geo, core_profiles_state
      )
      _fill_profiles_1d(
          model_node.profiles_1d[i],
          core_transport_state,
          core_profiles_state,
          model_name,
      )

  return ids


def _fill_metadata(ids: ids_toplevel.IDSToplevel):
  """Fills metadata in-place for the core_transport IDS."""
  ids.ids_properties.comment = (
      "IDS built from TORAX simulation transport coefficients. Grid based on "
      "TORAX face grid."
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
    core_profiles_state: state.CoreProfiles,
) -> None:
  """Fills 1D grid coordinates for a given time slice."""
  grid = profiles_1d_slice.grid_d
  grid.rho_tor_norm = geo.rho_face_norm
  grid.rho_tor = geo.rho_face
  grid.psi = core_profiles_state.psi.face_value()
  grid.psi_magnetic_axis = core_profiles_state.psi.left_face_value[0]
  grid.psi_boundary = core_profiles_state.psi.right_face_value[0]
  grid.rho_pol_norm = np.sqrt(
      (grid.psi - grid.psi_magnetic_axis)
      / (grid.psi_boundary - grid.psi_magnetic_axis)
  )
  grid.volume = geo.volume_face
  grid.area = geo.area_face

  # Assign the same grid to grid_v
  grid_v = profiles_1d_slice.grid_v
  grid_v.rho_tor_norm = grid.rho_tor_norm
  grid_v.rho_tor = grid.rho_tor
  grid_v.psi = grid.psi
  grid_v.psi_magnetic_axis = grid.psi_magnetic_axis
  grid_v.psi_boundary = grid.psi_boundary
  grid_v.rho_pol_norm = grid.rho_pol_norm
  grid_v.volume = grid.volume
  grid_v.area = grid.area


def _fill_profiles_1d(
    profiles_1d_slice: imas.ids_structure.IDSStructure,
    core_transport_state: state.CoreTransport,
    core_profiles_state: state.CoreProfiles,
    model_name: str,
) -> None:
  """Fills 1D profiles for a transport model at a single time slice."""

  if model_name in ("combined", "transport_solver"):
    chi_e = core_transport_state.chi_face_el_total
    chi_i = core_transport_state.chi_face_ion_total
    d_e = core_transport_state.d_face_el_total
    v_e = core_transport_state.v_face_el_total
  elif model_name == "neoclassical":
    chi_e = core_transport_state.chi_neo_e
    chi_i = core_transport_state.chi_neo_i
    d_e = core_transport_state.D_neo_e
    v_e = core_transport_state.V_neo_e + core_transport_state.V_neo_ware_e
    profiles_1d_slice.conductivity_parallel = core_profiles_state.sigma_face
  elif model_name == "anomalous":
    chi_e = (
        core_transport_state.chi_face_el
        + core_transport_state.chi_face_el_pereverzev
    )
    chi_i = (
        core_transport_state.chi_face_ion
        + core_transport_state.chi_face_ion_pereverzev
    )
    d_e = (
        core_transport_state.d_face_el
        + core_transport_state.d_face_el_pereverzev
    )
    v_e = (
        core_transport_state.v_face_el
        + core_transport_state.v_face_el_pereverzev
    )
  else:
    raise ValueError(f"Unknown model_name: {model_name}")

  profiles_1d_slice.electrons.energy.d = chi_e
  profiles_1d_slice.total_ion_energy.d = chi_i
  profiles_1d_slice.electrons.particles.d = d_e
  profiles_1d_slice.electrons.particles.v = v_e
