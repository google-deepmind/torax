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
from torax._src import state as state_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.output_tools import output
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
    core_profiles: Sequence[state_lib.CoreProfiles],
    core_transport: Sequence[state_lib.CoreTransport],
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

  model_names = ["combined", "transport_solver", "neoclassical", "anomalous"]
  ids.model.resize(len(model_names))

  for idx, model_name in enumerate(model_names):
    model_node = ids.model[idx]
    model_node.identifier.name = model_name
    model_node.identifier.index = _IMAS_MODEL_ID_TO_IDENTIFIER_INDEX.get(
        model_name, 0
    )

    if model_name == "anomalous":
      transport_cfg = torax_config.transport
      if transport_cfg.model_name == "combined":
        non_constant_names = [
            m.model_name
            for m in transport_cfg.transport_models
            if m.model_name != "constant"
        ]
        if not non_constant_names:
          model_node.code.name = "TORAX"
        else:
          model_node.code.name = "+".join(non_constant_names)
      elif transport_cfg.model_name == "constant":
        model_node.code.name = "TORAX"
      else:
        model_node.code.name = transport_cfg.model_name
    elif model_name == "neoclassical":
      model_node.code.name = "TORAX angioni-sauter"
    else:
      model_node.code.name = "TORAX"

    model_node.profiles_1d.resize(len(times))

    for i in range(len(times)):
      t = times[i]
      geo = geometry[i]
      ct_state = core_transport[i]
      cp_state = core_profiles[i]

      model_node.profiles_1d[i].time = t
      _fill_grid_coordinates(model_node.profiles_1d[i], geo, cp_state)
      _fill_profiles_1d(
          model_node.profiles_1d[i],
          ct_state,
          cp_state,
          model_name,
      )

  return ids


def _fill_metadata(ids: ids_toplevel.IDSToplevel):
  """Fills metadata in-place for the core_transport IDS."""
  ids.ids_properties.comment = (
      "IDS built from TORAX simulation transport coefficients. Grid based on "
      "TORAX cell grid + boundaries."
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
    cp_state: state_lib.CoreProfiles,
) -> None:
  """Fills 1D grid coordinates for a given time slice."""
  grid = profiles_1d_slice.grid_d
  grid.rho_tor_norm = np.concatenate([[0.0], geo.rho_norm, [1.0]])
  grid.rho_tor = np.concatenate([[0.0], geo.rho, [geo.rho_b]])
  grid.psi = cp_state.psi.cell_plus_boundaries()
  grid.psi_magnetic_axis = cp_state.psi.left_face_value[0]
  grid.psi_boundary = cp_state.psi.right_face_value[0]
  grid.rho_pol_norm = np.sqrt(
      (grid.psi - grid.psi_magnetic_axis)
      / (grid.psi_boundary - grid.psi_magnetic_axis)
  )
  grid.volume = output.extend_cell_grid_to_boundaries(
      [geo.volume], np.array([geo.volume_face])
  )[0]
  grid.area = output.extend_cell_grid_to_boundaries(
      [geo.area], np.array([geo.area_face])
  )[0]

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


def _extend_face_profile_to_boundaries(
    face_val: np.ndarray,
) -> np.ndarray:
  """Extends face-centered profile to cell_plus_boundaries grid."""
  cell_val = geometry_lib.face_to_cell(face_val)
  return output.extend_cell_grid_to_boundaries(
      [cell_val], np.array([face_val])
  )[0]


def _fill_profiles_1d(
    profiles_1d_slice: imas.ids_structure.IDSStructure,
    ct_state: state_lib.CoreTransport,
    cp_state: state_lib.CoreProfiles,
    model_name: str,
) -> None:
  """Fills 1D profiles for a transport model at a single time slice."""

  if model_name in ("combined", "transport_solver"):
    chi_e = ct_state.chi_face_el_total
    chi_i = ct_state.chi_face_ion_total
    d_e = ct_state.d_face_el_total
    v_e = ct_state.v_face_el_total
  elif model_name == "neoclassical":
    chi_e = ct_state.chi_neo_e
    chi_i = ct_state.chi_neo_i
    d_e = ct_state.D_neo_e
    v_e = ct_state.V_neo_e + ct_state.V_neo_ware_e
    profiles_1d_slice.conductivity_parallel = (
        output.extend_cell_grid_to_boundaries(
            [cp_state.sigma], np.array([cp_state.sigma_face])
        )[0]
    )
  elif model_name == "anomalous":
    chi_e = ct_state.chi_face_el + ct_state.chi_face_el_pereverzev
    chi_i = ct_state.chi_face_ion + ct_state.chi_face_ion_pereverzev
    d_e = ct_state.d_face_el + ct_state.d_face_el_pereverzev
    v_e = ct_state.v_face_el + ct_state.v_face_el_pereverzev
  else:
    raise ValueError(f"Unknown model_name: {model_name}")

  profiles_1d_slice.electrons.energy.d = _extend_face_profile_to_boundaries(
      chi_e
  )
  profiles_1d_slice.total_ion_energy.d = _extend_face_profile_to_boundaries(
      chi_i
  )
  profiles_1d_slice.electrons.particles.d = _extend_face_profile_to_boundaries(
      d_e
  )
  profiles_1d_slice.electrons.particles.v = _extend_face_profile_to_boundaries(
      v_e
  )
