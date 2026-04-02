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
"""Generic loader that can load any IDS from a netCDF file or from an IMASdb."""

import os
import pathlib
from typing import Literal

from absl import logging
import imas
from imas import ids_struct_array
from imas import ids_structure
from imas import ids_toplevel
import numpy as np
from torax._src import path_utils
import xarray as xr

# pylint: disable=invalid-name

# Names of the IDSs that can be loaded and used in TORAX.
IDS = Literal['core_profiles', 'plasma_profiles', 'equilibrium']
_TORAX_IMAS_DD_VERSION = '4.0.0'


def load_imas_data(
    uri: str,
    ids_name: str,
    directory: pathlib.Path | None = None,
    explicit_convert: bool = False,
) -> ids_toplevel.IDSToplevel:
  """Loads a full IDS for a given uri or path_name and a given ids_name.

  It can load either an IMAS netCDF file with filename as uri and given
  directory or from an IMASdb by giving the full uri of the IDS and the
  directory arg will be ignored. Note that loading from an IMASdb requires
  IMAS-core. The loaded IDS can then be used as input to
  core_profiles_from_IMAS().

  Args:
    uri: Path to netCDF file or full uri of the IDS (this requires IMAS-core).
    ids_name: The name of the IDS to load.
    directory: The directory of the IDS to load.
    explicit_convert: Whether to explicitly convert the IDS to the current DD
      version. If True, an explicit conversion will be attempted. Explicit
      conversion is recommended when converting between major DD versions.
      https://imas-python.readthedocs.io/en/latest/multi-dd.html#conversion-of-idss-between-dd-versions

  Returns:
    An IDS object.
  Raises:
    AttributeError: If the IDS cannot be autoconverted.
  """
  # Differentiate between netCDF and IMASdb uris. For IMASdb files the full
  # filepath is already provided in the uri.
  if uri.endswith('.nc'):
    if directory is None:
      directory = path_utils.torax_path().joinpath("data/third_party/imas")
    uri = os.path.join(directory, uri)
  with imas.DBEntry(uri=uri, mode='r', dd_version=_TORAX_IMAS_DD_VERSION) as db:
    if not explicit_convert:
      ids = db.get(ids_name=ids_name, autoconvert=True)
    else:
      ids = db.get(ids_name=ids_name, autoconvert=False)
      ids = imas.convert_ids(ids, _TORAX_IMAS_DD_VERSION)
  return ids


def get_time_and_radial_arrays(
    ids_node: ids_structure.IDSStructure,
    t_initial: float | None = None,
) -> tuple[ids_struct_array.IDSStructArray, list[list[float]], list[float]]:
  """Extracts time and radial arrays from a given IDS node.

  Can be used with core_profiles IDSTopLevel or with core_sources.source[i]
  nodes.

  Args:
      ids_node: The IDS node from which to extract the arrays.
      t_initial: Initial time used to map the profiles in the dicts. If None the
        initial time will be the time of the first time slice of the ids. Else
        all time slices will be shifted such that the first time slice has time
        = t_initial.

  Returns:
      A tuple containing:
      - profiles_1d: IDS's profiles_1d array of structures.
      - rhon_array: A list of lists containing the radial grid points for each
        time slice.
      - time_array: A list containing the time points.
  """
  profiles_1d = ids_node.profiles_1d
  time_array = [profile.time for profile in profiles_1d]
  if t_initial:
    time_array = [t - time_array[0] + t_initial for t in time_array]
  rhon_array = [profile.grid.rho_tor_norm for profile in profiles_1d]
  return profiles_1d, rhon_array, time_array


def imas_to_torax_xr(
    uri: str,
    directory: str | None = None,
    explicit_convert: bool = False,
) -> xr.DataTree:
  """Loads IMAS data and returns a TORAX-structured xr.DataTree.

  Args:
    uri: Path to netCDF file or full uri of the IDS.
    directory: The directory of the IDS to load.
    explicit_convert: Whether to explicitly convert the IDS to current DD.

  Returns:
    An xr.DataTree containing the IMAS data in TORAX format.
  """
  try:
    core_profiles_ids = load_imas_data(
        uri, 'core_profiles', directory, explicit_convert
    )
  except Exception as e:
    logging.warning('Failed to load core_profiles IDS: %s', e)
    raise

  try:
    equilibrium_ids = load_imas_data(
        uri, 'equilibrium', directory, explicit_convert
    )
  except Exception as e:
    logging.warning('Failed to load equilibrium IDS: %s', e)
    raise

  try:
    core_sources_ids = load_imas_data(
        uri, 'core_sources', directory, explicit_convert
    )
  except Exception as e:
    logging.warning('Failed to load core_sources IDS: %s', e)
    raise

  try:
    core_transport_ids = load_imas_data(
        uri, 'core_transport', directory, explicit_convert
    )
  except Exception as e:
    logging.warning('Failed to load core_transport IDS: %s', e)
    raise

  try:
    summary_ids = load_imas_data(uri, 'summary', directory, explicit_convert)
  except Exception as e:
    logging.warning('Failed to load summary IDS: %s', e)
    raise

  # Load coords
  # Assume that all IDSs have the same time and radial arrays.
  rho_norm = np.asarray(core_profiles_ids.profiles_1d[0].grid.rho_tor_norm)
  time = np.asarray(summary_ids.time)

  rho_cell_norm = rho_norm[1:-1]
  midpoints = (rho_cell_norm[1:] + rho_cell_norm[:-1]) / 2.0
  rho_face_norm = np.concatenate([[0.0], midpoints, [1.0]])

  coords = {
      'time': xr.DataArray(time, dims=['time'], name='time'),
      'rho_norm': xr.DataArray(rho_norm, dims=['rho_norm'], name='rho_norm'),
      'rho_cell_norm': xr.DataArray(
          rho_cell_norm, dims=['rho_cell_norm'], name='rho_cell_norm'
      ),
      'rho_face_norm': xr.DataArray(
          rho_face_norm, dims=['rho_face_norm'], name='rho_face_norm'
      ),
  }

  profiles_dict = {}
  scalars_dict = {}

  def _get_from_path(x, path):
    if hasattr(x, path):
      return getattr(x, path)
    else:
      next_path_element = path.split('.')[0]
      next_path = '.'.join(path.split('.')[1:])
      next_x = getattr(x, next_path_element)
      return _get_from_path(next_x, next_path)

  def _get_core_profile(path):
    return np.asarray(
        [_get_from_path(p, path) for p in core_profiles_ids.profiles_1d]
    )

  transport_model_names = [
      t.identifier.name.value for t in core_transport_ids.model
  ]
  transport_model_name_to_idx_mapping = {
      name: idx for idx, name in enumerate(transport_model_names)
  }

  def _get_transport(model_name, path):
    model_idx = transport_model_name_to_idx_mapping[model_name]
    return np.asarray([
        _get_from_path(p, path)
        for p in core_transport_ids.model[model_idx].profiles_1d
    ])

  source_names = [s.identifier.name.value for s in core_sources_ids.source]
  source_name_to_idx_mapping = {
      name: idx for idx, name in enumerate(source_names)
  }

  def _get_source_profile(source_name, path):
    source_idx = source_name_to_idx_mapping[source_name]
    return np.asarray([
        _get_from_path(p, path)
        for p in core_sources_ids.source[source_idx].profiles_1d
    ])

  def _get_source_global_quantity(source_name, path):
    source_idx = source_name_to_idx_mapping[source_name]
    return np.asarray([
        _get_from_path(g, path)
        for g in core_sources_ids.source[source_idx].global_quantities
    ])

  def _get_equilibrium(path):
    return np.asarray(
        [_get_from_path(t, path) for t in equilibrium_ids.time_slice]
    )

  def _get_summary(path):
    return np.asarray(_get_from_path(summary_ids, path).value)

  # COCOS - we want current to be positive
  current_sign = np.sign(_get_summary('global_quantities.ip'))

  # Load
  profiles_dict['T_e'] = xr.DataArray(
      _get_core_profile('electrons.temperature') / 1e3,
      dims=['time', 'rho_norm'],
      name='T_e',
  )
  profiles_dict['T_i'] = xr.DataArray(
      _get_core_profile('t_i_average') / 1e3,
      dims=['time', 'rho_norm'],
      name='T_i',
  )
  profiles_dict['n_e'] = xr.DataArray(
      _get_core_profile('electrons.density'),
      dims=['time', 'rho_norm'],
      name='n_e',
  )
  ion_names = [str(ion.name) for ion in core_profiles_ids.profiles_1d[0].ion]
  n_ions = len(ion_names)
  allowed_main_ion_names = ['D', 'T']
  impurity_ion_idxs = [
      i for i in range(n_ions) if ion_names[i] not in allowed_main_ion_names
  ]
  all_ion_densities = np.asarray([
      [p.ion[i].density.value for p in core_profiles_ids.profiles_1d]
      for i in range(n_ions)
  ])
  n_i_total = np.sum(all_ion_densities, axis=0)
  impurity_ions_Z = np.asarray([
      [p.ion[i].z_ion_1d for p in core_profiles_ids.profiles_1d]
      for i in impurity_ion_idxs
  ])
  impurity_ion_Z_average = np.sum(impurity_ions_Z, axis=0)
  profiles_dict['n_i'] = xr.DataArray(
      n_i_total,
      dims=['time', 'rho_norm'],
      name='n_i',
  )
  profiles_dict['Z_eff'] = xr.DataArray(
      _get_core_profile('zeff'),
      dims=['time', 'rho_norm'],
      name='Z_eff',
  )
  profiles_dict['Z_impurity'] = xr.DataArray(
      impurity_ion_Z_average,
      dims=['time', 'rho_norm'],
      name='Z_impurity',
  )
  profiles_dict['q'] = xr.DataArray(
      _get_core_profile('q'),
      dims=['time', 'rho_norm'],
      name='q',
  )
  profiles_dict['magnetic_shear'] = xr.DataArray(
      _get_core_profile('magnetic_shear'),
      dims=['time', 'rho_norm'],
      name='magnetic_shear',
  )
  # Convert currents
  j_total_toroidal = _get_core_profile('j_phi')
  j_total_parallel = _get_core_profile('j_total')
  j_toroidal_over_j_parallel = j_total_toroidal / j_total_parallel
  profiles_dict['j_total'] = xr.DataArray(
      current_sign * j_total_toroidal,
      dims=['time', 'rho_norm'],
      name='j_total',
  )
  profiles_dict['j_bootstrap'] = xr.DataArray(
      current_sign
      * j_toroidal_over_j_parallel
      * _get_core_profile('j_bootstrap'),
      dims=['time', 'rho_norm'],
      name='j_bootstrap',
  )
  profiles_dict['j_ohmic'] = xr.DataArray(
      current_sign * j_toroidal_over_j_parallel * _get_core_profile('j_ohmic'),
      dims=['time', 'rho_norm'],
      name='j_ohmic',
  )
  profiles_dict['j_non_inductive'] = xr.DataArray(
      current_sign
      * j_toroidal_over_j_parallel
      * _get_core_profile('j_non_inductive'),
      dims=['time', 'rho_norm'],
      name='j_non_inductive',
  )

  profiles_dict['psi'] = xr.DataArray(
      _get_equilibrium('profiles_1d.psi'),
      dims=['time', 'rho_norm'],
      name='psi',
  )

  profiles_dict['p_ohmic_e'] = xr.DataArray(
      _get_source_profile('ohmic', 'electrons.energy'),
      dims=['time', 'rho_norm'],
      name='p_ohmic_e',
  )
  profiles_dict['ei_exchange'] = xr.DataArray(
      _get_source_profile('collisional_equipartition', 'electrons.energy'),
      dims=['time', 'rho'],
      name='ei_exchange',
  )
  profiles_dict['p_alpha_e'] = xr.DataArray(
      _get_source_profile('fusion', 'electrons.energy'),
      dims=['time', 'rho'],
      name='p_alpha_e',
  )
  profiles_dict['p_alpha_i'] = xr.DataArray(
      _get_source_profile('fusion', 'total_ion_energy'),
      dims=['time', 'rho'],
      name='p_alpha_i',
  )
  profiles_dict['p_impurity_radiation_e'] = xr.DataArray(
      _get_source_profile('radiation', 'electrons.energy'),
      dims=['time', 'rho'],
      name='p_impurity_radiation_e',
  )
  profiles_dict['p_ecrh_e'] = xr.DataArray(
      _get_source_profile('ec', 'electrons.energy'),
      dims=['time', 'rho'],
      name='p_ecrh_e',
  )
  profiles_dict['p_generic_heat_e'] = xr.DataArray(
      _get_source_profile('nbi', 'electrons.energy')
      + _get_source_profile('lh', 'electrons.energy')
      + _get_source_profile('ic', 'electrons.energy'),
      dims=['time', 'rho'],
      name='p_generic_heat_e',
  )
  profiles_dict['p_generic_heat_i'] = xr.DataArray(
      _get_source_profile('nbi', 'total_ion_energy')
      + _get_source_profile('lh', 'total_ion_energy')
      + _get_source_profile('ic', 'total_ion_energy'),
      dims=['time', 'rho'],
      name='p_generic_heat_i',
  )
  scalars_dict['P_aux_total'] = xr.DataArray(
      _get_source_global_quantity('nbi', 'electrons.power')
      + _get_source_global_quantity('nbi', 'total_ion_power')
      + _get_source_global_quantity('lh', 'electrons.power')
      + _get_source_global_quantity('lh', 'total_ion_power')
      + _get_source_global_quantity('ic', 'electrons.power')
      + _get_source_global_quantity('ic', 'total_ion_power')
      + _get_source_global_quantity('ec', 'electrons.power'),
      dims=['time'],
      name='P_aux_total',
  )
  scalars_dict['P_ohmic_e'] = xr.DataArray(
      _get_source_global_quantity('ohmic', 'electrons.power'),
      dims=['time'],
      name='P_ohmic_e',
  )
  scalars_dict['P_alpha_total'] = xr.DataArray(
      _get_source_global_quantity('fusion', 'electrons.power')
      + _get_source_global_quantity('fusion', 'total_ion_power'),
      dims=['time'],
      name='P_alpha_total',
  )
  scalars_dict['P_radiation_e'] = xr.DataArray(
      _get_source_global_quantity('radiation', 'electrons.power'),
      dims=['time'],
      name='P_radiation_e',
  )

  # Derived values
  scalars_dict['P_fusion'] = 5 * scalars_dict['P_alpha_total']
  scalars_dict['P_external'] = (
      scalars_dict['P_aux_total'] + scalars_dict['P_ohmic_e']
  )
  scalars_dict['Q_fusion'] = (
      scalars_dict['P_fusion'] / scalars_dict['P_external']
  )

  profiles_dict['chi_total_e'] = xr.DataArray(
      _get_transport('combined', 'electrons.energy.d'),
      dims=['time', 'rho_norm'],
      name='chi_total_e',
  )
  profiles_dict['chi_total_i'] = xr.DataArray(
      _get_transport('combined', 'total_ion_energy.d'),
      dims=['time', 'rho_norm'],
      name='chi_total_i',
  )

  scalars_dict['Ip'] = xr.DataArray(
      current_sign * _get_summary('global_quantities.ip'),
      dims=['time'],
      name='Ip',
  )
  scalars_dict['I_bootstrap'] = xr.DataArray(
      current_sign * _get_summary('global_quantities.current_bootstrap'),
      dims=['time'],
      name='I_bootstrap',
  )
  scalars_dict['I_aux_generic'] = xr.DataArray(
      current_sign
      * (
          summary_ids.heating_current_drive.lh[0].current.value
          + summary_ids.heating_current_drive.ic[0].current.value
          + summary_ids.heating_current_drive.nbi[0].current.value
      ),
      dims=['time'],
      name='I_aux_generic',
  )
  scalars_dict['I_ecrh'] = xr.DataArray(
      current_sign * summary_ids.heating_current_drive.ec[0].current.value,
      dims=['time'],
      name='I_ecrh',
  )

  # Not supported values
  unsupported_profiles = [
      'Ip_profile',
      'v_loop',
      'D_total_e',
      'V_total_e',
  ]
  for key in unsupported_profiles:
    profiles_dict[key] = xr.DataArray(
        np.full((len(time), len(rho_norm)), np.nan),
        dims=['time', 'rho_norm'],
        name=key,
    )
  unsupported_scalars = [
      'P_bremsstrahlung_e',
      'P_cyclotron_e',
  ]
  for key in unsupported_scalars:
    scalars_dict[key] = xr.DataArray(
        np.full(len(time), np.nan),
        dims=['time'],
        name=key,
    )

  profiles = xr.Dataset(profiles_dict)
  scalars = xr.Dataset(scalars_dict)
  numerics = xr.Dataset()

  children = {
      'profiles': xr.DataTree(dataset=profiles),
      'scalars': xr.DataTree(dataset=scalars),
      'numerics': xr.DataTree(dataset=numerics),
  }

  data_tree = xr.DataTree(
      children=children,
      dataset=xr.Dataset(coords=coords),
  )

  return data_tree
