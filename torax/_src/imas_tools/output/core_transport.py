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

"""Functions to save IMAS core_transport or plasma_transport IDSs from TORAX."""

from collections.abc import Sequence
import datetime

import imas
from imas import ids_toplevel
import numpy as np
from torax._src import array_typing
from torax._src.core_profiles import core_profiles as core_profiles_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import config as neoclassical_config
from torax._src.torax_pydantic import config
from torax._src.transport_model import transport_coeffs


# pylint: disable=invalid-name
def core_transport_to_IMAS(
    torax_config: config.ToraxConfig,
    core_profiles: Sequence[core_profiles_lib.CoreProfiles],
    core_transport: Sequence[transport_coeffs.CoreTransport],
    geometry: Sequence[geometry_lib.Geometry],
    times: array_typing.FloatVector,
    ids: ids_toplevel.IDSToplevel | None = None,
) -> ids_toplevel.IDSToplevel:
  """Saves TORAX transport coefficients into an IMAS transport IDS.

  The output grid for all 1D quantities is the TORAX face grid.
  The function can be used to save an entire trajectory or a single time slice.
  To save a single time slice, pass `Sequence`s of length 1.

  Args:
    torax_config: ToraxConfig describing the simulation transport and
      neoclassical models.
    core_profiles: Sequence of TORAX CoreProfiles objects.
    core_transport: Sequence of TORAX CoreTransport objects.
    geometry: Sequence of TORAX Geometry objects.
    times: Time array of the slices to save.
    ids: Optional IDS object to be filled. Can be either `core_transport` or
      `plasma_transport`. If not provided, a `core_transport` IDS is created.

  Returns:
    Filled `core_transport` or `plasma_transport` IDS object.
  """
  if ids is None:
    ids = imas.IDSFactory().core_transport()
  elif ids.metadata.name not in ('core_transport', 'plasma_transport'):
    raise TypeError(
        'Expected core_transport or plasma_transport IDS, got'
        f' {ids.metadata.name} IDS.'
    )

  identifier_enum = (
      imas.identifiers.plasma_transport_identifier
      if ids.metadata.name == 'plasma_transport'
      else imas.identifiers.core_transport_identifier
  )

  _fill_metadata(ids)
  ids.time = times

  active_models = _get_active_transport_models(torax_config)
  ids.model.resize(len(active_models))

  for idx, (model_name, code_name) in enumerate(active_models):
    model_node = ids.model[idx]
    model_node.identifier = identifier_enum[model_name]
    model_node.code.name = code_name

    model_node.profiles_1d.resize(len(times))

    for i, t in enumerate(times):
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


def _fill_metadata(ids: ids_toplevel.IDSToplevel) -> None:
  """Fills metadata in-place for the transport IDS."""
  ids.ids_properties.comment = (
      'IDS built from TORAX simulation transport coefficients. Grid based on'
      ' TORAX face grid.'
  )
  ids.ids_properties.homogeneous_time = 1
  ids.ids_properties.creation_date = datetime.date.today().isoformat()
  ids.code.name = 'TORAX'
  ids.code.description = (
      'TORAX is a differentiable tokamak core transport simulator.'
  )
  ids.code.repository = 'https://github.com/google-deepmind/torax'


def _fill_grid_coordinates(
    profiles_1d_slice: imas.ids_structure.IDSStructure,
    geo: geometry_lib.Geometry,
    core_profiles_state: core_profiles_lib.CoreProfiles,
) -> None:
  """Fills 1D grid coordinates for a given time slice."""
  psi = core_profiles_state.poloidal_flux.psi
  grid = profiles_1d_slice.grid_d
  grid.rho_tor_norm = geo.rho_face_norm
  grid.rho_tor = geo.rho_face
  grid.psi = psi.face_value()
  grid.psi_magnetic_axis = psi.left_face_value[0]
  grid.psi_boundary = psi.right_face_value[0]
  grid.rho_pol_norm = np.sqrt(
      (grid.psi - grid.psi_magnetic_axis)
      / (grid.psi_boundary - grid.psi_magnetic_axis)
  )
  grid.volume = geo.volume_face
  grid.area = geo.area_face

  # Assign the same grid to grid_v.
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
    core_transport_state: transport_coeffs.CoreTransport,
    core_profiles_state: core_profiles_lib.CoreProfiles,
    model_name: str,
) -> None:
  """Fills 1D profiles for a transport model at a single time slice."""
  if model_name == 'combined':
    # Physical total transport excludes Pereverzev-Corrigan numerical
    # stabilization terms (unlike CoreTransport.total).
    coeffs = transport_coeffs.sum_transport_coeffs(
        core_transport_state.turbulent.total,
        core_transport_state.neoclassical,
    )
  elif model_name == 'transport_solver':
    coeffs = core_transport_state.total
  elif model_name == 'anomalous':
    coeffs = core_transport_state.turbulent.core
  elif model_name == 'pedestal':
    coeffs = core_transport_state.turbulent.pedestal
  elif model_name == 'neoclassical':
    coeffs = core_transport_state.neoclassical
    profiles_1d_slice.conductivity_parallel = (
        core_profiles_state.conductivity.sigma_face
    )
  else:
    raise ValueError(f'Unknown model_name: {model_name}')

  profiles_1d_slice.electrons.energy.d = coeffs.chi_face_el
  profiles_1d_slice.total_ion_energy.d = coeffs.chi_face_ion
  profiles_1d_slice.electrons.particles.d = coeffs.d_face_el
  profiles_1d_slice.electrons.particles.v = coeffs.v_face_el


def _get_active_transport_models(
    torax_config: config.ToraxConfig,
) -> list[tuple[str, str]]:
  """Returns active `(model_name, code_name)` pairs for the given config."""
  models: list[tuple[str, str]] = [
      ('combined', 'TORAX'),
      ('transport_solver', 'TORAX'),
  ]

  core_models = torax_config.transport.core_transport_models
  if core_models:
    non_prescribed_names = [
        model.model_name
        for model in core_models.values()
        if model.model_name != 'prescribed'
    ]
    if non_prescribed_names:
      models.append(('anomalous', '+'.join(non_prescribed_names)))
    else:
      models.append(('anomalous', 'prescribed'))

  pedestal_models = torax_config.transport.pedestal_transport_models
  if pedestal_models:
    pedestal_code_name = '+'.join(
        model.model_name for model in pedestal_models.values()
    )
    models.append(('pedestal', pedestal_code_name))

  # Neoclassical conductivity is always computed, so always save it.
  neo_cfg = torax_config.neoclassical
  if isinstance(neo_cfg, neoclassical_config.AnalyticalNeoclassicalConfig):
    if neo_cfg.transport.model_name != 'zeros':
      neo_code_name = (
          f'TORAX {neo_cfg.transport.model_name}'
          f'+{neo_cfg.conductivity.model_name}'
      )
    else:
      neo_code_name = f'TORAX {neo_cfg.conductivity.model_name}'
  else:
    neo_code_name = getattr(neo_cfg, 'model_name', 'TORAX neoclassical')
  models.append(('neoclassical', neo_code_name))

  return models
