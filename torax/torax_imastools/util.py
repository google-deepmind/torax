# Copyright 2024 DeepMind Technologies Limited
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

"""Useful functions for handling of IMAS IDSs and converts them into TORAX objects"""
from typing import Dict, Any
import os
import datetime
import importlib

import numpy as np
import yaml
import scipy
try:
    import imaspy
    from imaspy.ids_toplevel import IDSToplevel
except:
    IDSToplevel = Any
    pass

from torax.geometry import geometry_loader


def requires_module(module_name):
    """
    Decorator that checks if a module can be imported.
    Returns the function if the module is available,
    otherwise raises an ImportError.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                importlib.import_module(module_name)
            except ImportError:
                raise ImportError(
                    f"Required module '{module_name}' is not installed. "
                    "Make sure you install the needed optional dependencies."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


@requires_module('imaspy')
def save_netCDF(directory_path: str | None,
                file_name: str | None,
                IDS):
    """Generates a netcdf file from an IDS.

    Args:
      directory_path: Directory where to save the netCDF output file. If None, it will be saved in data/third_party/geo.
      file_name: Desired output file name. If None will default to IDS_file_ + Time.
      IDS: IMAS Interface Data Structure object that will be written in the IMAS netCDF file.

    Returns:
      None
    """
    if directory_path is None:
      directory_path = 'torax/data/third_party/geo'
    if file_name is None:
       file_name = "IDS_file_" + datetime.datetime.now().strftime(
      '%Y%m%d_%H%M%S')
    filepath = os.path.join(directory_path, file_name) + '.nc'
    with imaspy.DBEntry(filepath, "w") as netcdf_entry:
       netcdf_entry.put(IDS)


@requires_module('imaspy')
def load_IMAS_equilibrium_from_Data_entry(file_path: str) -> dict[str, np.ndarray]:
  """Loads the equilibrium IDS for a single time slice from a specific data entry / scenario using the IMAS Access Layer

  Args:
    file_path: Absolute path to the yaml file containing the DB specifications of the desired IDS : input_database, shot, run, input_user_or_path and time_begin for the desired time_slice

  Returns:
    Equilibrium IDS"""
  file = open(file_path, 'r')
  scenario = yaml.load(file,Loader=yaml.CLoader)
  file.close()
  if scenario['scenario_backend'] == 'hdf5':
        sc_backend = imaspy.ids_defs.HDF5_BACKEND
  else:
      sc_backend = imaspy.ids_defs.MDSPLUS_BACKEND
  input = imaspy.DBEntry(sc_backend,scenario['input_database'],scenario['shot'],scenario['run_in'],scenario['input_user_or_path'])
  input.open()
  timenow = scenario['time_begin']
  IMAS_data = input.get_slice('equilibrium',timenow,1)

  return IMAS_data


@requires_module('imaspy')
def load_IMAS_data_from_netCDF(file_path: str) -> dict[str, np.ndarray]:
  """Loads the equilibrium IDS for a single time slice from an IMAS netCDF file path"""
  input = imaspy.DBEntry(file_path, "r")
  equilibrium_ids = input.get('equilibrium')
  return equilibrium_ids

@requires_module('imaspy')
def load_IMAS_data_from_hdf5(directory_path: str) -> dict[str, np.ndarray]:
  """Loads the equilibrium IDS for a single time slice from the path of a local directory containing it stored with hdf5 backend. The repository must contain the master.h5 file."""
  imasuri = "imas:hdf5?path=" + directory_path
  input = imaspy.DBEntry(imasuri, "r")
  equilibrium_ids = input.get('equilibrium')
  return equilibrium_ids

@requires_module('imaspy')
def load_IMAS_data(path: str) -> dict[str, np.ndarray]:
  """Loads the equilibrium IDS for a single time slice either from a netCDF file path or from an hdf5 file in the given directory path."""
  if path[-3:] == ".nc":
    return load_IMAS_data_from_netCDF(file_path=path)
  else:
    return load_IMAS_data_from_hdf5(directory_path=path)



@requires_module('imaspy')
def write_ids_equilibrium_into_config(config: dict, equilibrium)->dict[str,np.ndarray]:
  """Loads the equilibrium into the geometry config.
  Args:
  config: TORAX config object.
  equilibrium: equilibrium IDS to put into the config as the equilibrium_object.

  Returns:
  Full config object will the IDS inside."""
  config["geometry"]["geometry_type"] = 'imas'
  config["geometry"]["equilibrium_object"] = equilibrium
  return config


@requires_module('imaspy')
def core_profiles_to_IMAS(ids, state, geometry):
  """Converts torax core_profiles to IMAS IDS.
  Takes the cell grid as a basis and converts values on face grid to cell.
  Args:
  ids: IDS object
  state: torax state object

  Returns:
  filled IDS object"""
  t = state.t
  cp_state = state.core_profiles
  ids.ids_properties.comment = "Grid based on torax cell grid, used cell grid values and interpolated face grid values"
  ids.ids_properties.homogeneous_time = 1
  ids.time = [t]
  ids.vacuum_toroidal_field.b0.resize(1)
  ids.global_quantities.current_non_inductive.resize(1)
  ids.profiles_1d.resize(1)
  ids.profiles_1d[0].ion.resize(2)
  ids.profiles_1d[0].ion[0].element.resize(1)
  ids.profiles_1d[0].ion[1].element.resize(1)
  ids.vacuum_toroidal_field.r0 = geometry.Rmaj
  ids.vacuum_toroidal_field.b0[0] = geometry.B0
  ids.global_quantities.current_non_inductive[0] = cp_state.currents.I_bootstrap
  ids.profiles_1d[0].grid.rho_tor_norm = geometry.rho_norm
  ids.profiles_1d[0].grid.rho_tor = geometry.rho
  ids.profiles_1d[0].grid.psi = cp_state.psi.value
  ids.profiles_1d[0].grid.volume = geometry.volume
  ids.profiles_1d[0].grid.area = geometry.area
  ids.profiles_1d[0].electrons.temperature = cp_state.temp_el.value
  ids.profiles_1d[0].electrons.density = cp_state.ne.value
  ids.profiles_1d[0].ion[0].z_ion = cp_state.Zi
  ids.profiles_1d[0].ion[0].temperature = cp_state.temp_ion.value
  ids.profiles_1d[0].ion[0].density = cp_state.ni.value
  # assume no molecules, revisit later
  ids.profiles_1d[0].ion[0].element[0].a = cp_state.Zi
  ids.profiles_1d[0].ion[0].element[0].z_n = cp_state.Ai
  ids.profiles_1d[0].ion[1].z_ion = cp_state.Zimp
  ids.profiles_1d[0].ion[1].temperature = cp_state.temp_ion.value
  ids.profiles_1d[0].ion[1].density = cp_state.nimp.value
  # assume no molecules, revisit later
  ids.profiles_1d[0].ion[1].element[0].a = cp_state.Zimp
  ids.profiles_1d[0].ion[1].element[0].z_n = cp_state.Aimp
  ids.profiles_1d[0].q = face_to_cell(cp_state.q_face)
  ids.profiles_1d[0].magnetic_shear = face_to_cell(cp_state.s_face)
  ids.profiles_1d[0].j_total = cp_state.currents.jtot
  ids.profiles_1d[0].j_ohmic = cp_state.currents.johm
  ids.profiles_1d[0].j_non_inductive = cp_state.currents.external_current_source
  ids.profiles_1d[0].j_bootstrap = cp_state.currents.j_bootstrap
  ids.profiles_1d[0].conductivity_parallel = cp_state.currents.sigma
  # ids. = cp_state.psidot.value
  # ids. = face_to_cell(cp_state.currents.jtot_face)
  # ids. = face_to_cell(cp_state.currents.Ip_profile_face)
  # ids. = face_to_cell(cp_state.currents.j_bootstrap_face)
  return ids


@requires_module('imaspy')
def geometry_from_IMAS(
  equilibrium_object: str | IDSToplevel,
  geometry_dir: str | None = None,
  Ip_from_parameters: bool = False,
  n_rho: int = 25,
  hires_fac: int = 4,
) -> Dict:
  """Constructs a StandardGeometryIntermediates from a IMAS equilibrium IDS.
  Args:
    equilibrium_object: Either directly the equilbrium IDS containing the relevant data, or the name of the IMAS netCDF file containing the equilibrium.
    geometry_dir: Directory where to find the scenario file ontaining the parameters of the Data entry to read.
      If None, uses the environment variable TORAX_GEOMETRY_DIR if
      available. If that variable is not set and geometry_dir is not provided,
      then it defaults to another dir. See `load_geo_data` implementation.
    Ip_from_parameters: If True, the Ip is taken from the parameters and the
      values in the Geometry are resacled to match the new Ip.
    n_rho: Radial grid points (num cells)
    hires_fac: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
  Returns:
    A StandardGeometry instance based on the input file. This can then be
    used to build a StandardGeometry by passing to `build_standard_geometry`.
  """
  #If the equilibrium_object is the file name, loads the ids from the netCDF.
  if isinstance(equilibrium_object, str):
    equilibrium = geometry_loader.load_geo_data(geometry_dir, equilibrium_object, geometry_loader.GeometrySource.IMAS)
  elif isinstance(equilibrium_object, IDSToplevel):
    equilibrium = equilibrium_object
  else:
    raise ValueError('equilibrium_object must be a string (file path) or an IDS')
  IMAS_data = equilibrium.time_slice[0]
  # b_field_phi has to be used for version >3.42.0, in previous versions it was b_field_tor.
  B0 = np.abs(IMAS_data.global_quantities.magnetic_axis.b_field_phi)
  Rmaj = np.asarray(IMAS_data.boundary.geometric_axis.r)

  # Poloidal flux (switch sign between ddv3 and ddv4)
  #psi = -1 * IMAS_data.profiles_1d.psi #ddv3
  psi = 1 * IMAS_data.profiles_1d.psi #ddv4

  # toroidal flux
  Phi = np.abs(IMAS_data.profiles_1d.phi)

  # midplane radii
  Rin = IMAS_data.profiles_1d.r_inboard
  Rout = IMAS_data.profiles_1d.r_outboard
  # toroidal field flux function
  F = np.abs(IMAS_data.profiles_1d.f)

  #Flux surface integrals of various geometry quantities
  #IDS Contour integrals
  if len(IMAS_data.profiles_1d.dvolume_dpsi) > 0:
      dvoldpsi = -1 * IMAS_data.profiles_1d.dvolume_dpsi
  else:
      dvoldpsi = -1 * np.gradient(IMAS_data.profiles_1d.volume) \
          / np.gradient(IMAS_data.profiles_1d.psi)
  #dpsi_drho_tor (switch sign between ddv3 and ddv4)
  if len(IMAS_data.profiles_1d.dpsi_drho_tor) > 0:
      #dpsidrhotor = -1 * IMAS_data.profiles_1d.dpsi_drho_tor #ddv3
      dpsidrhotor = 1 * IMAS_data.profiles_1d.dpsi_drho_tor #ddv4
  else:
      #dpsidrhotor = -1 * np.gradient(IMAS_data.profiles_1d.psi) \
      #    / np.gradient(IMAS_data.profiles_1d.rho_tor)           #ddv3
      dpsidrhotor = 1 * np.gradient(IMAS_data.profiles_1d.psi) \
          / np.gradient(IMAS_data.profiles_1d.rho_tor) #ddv4
  flux_surf_avg_RBp = IMAS_data.profiles_1d.gm7 * dpsidrhotor / (2 * np.pi) # dpsi, C0/C1
  flux_surf_avg_R2Bp2 = IMAS_data.profiles_1d.gm3 * (dpsidrhotor **2) / (4 * np.pi**2) # C4/C1
  flux_surf_avg_Bp2 = IMAS_data.profiles_1d.gm2 * (dpsidrhotor **2) / (4 * np.pi**2) # C3/C1
  int_dl_over_Bp = dvoldpsi #C1
  flux_surf_avg_1_over_R2 = IMAS_data.profiles_1d.gm1 # C2/C1

  #jtor = dI/drhon / (drho/dS) = dI/drhon / spr
  # spr = vpr / ( 2 * np.pi * Rmaj)
  # -> Ip_profile = integrate(y = spr * jtor, x= rhon, initial = 0.0)
  jtor= -1 * IMAS_data.profiles_1d.j_phi
  rhon = np.sqrt(Phi / Phi[-1])
  vpr = 4 * np.pi * Phi[-1] * rhon / (F * flux_surf_avg_1_over_R2)
  spr = vpr / (2*np.pi * Rmaj)
  Ip_profile = scipy.integrate.cumulative_trapezoid(y=spr * jtor, x=rhon, initial=0.0)

  # To check
  z_magnetic_axis = np.asarray(IMAS_data.global_quantities.magnetic_axis.z)

  return {
    'Ip_from_parameters': Ip_from_parameters,
    'Rmaj': Rmaj,
    'Rmin': np.asarray(IMAS_data.boundary.minor_radius),
    'B': B0,
    'psi': psi,
    'Ip_profile': Ip_profile,
    'Phi': Phi,
    'Rin': Rin,
    'Rout': Rout,
    'F': F,
    'int_dl_over_Bp': int_dl_over_Bp,
    'flux_surf_avg_1_over_R2': flux_surf_avg_1_over_R2,
    'flux_surf_avg_Bp2': flux_surf_avg_Bp2,
    'flux_surf_avg_RBp': flux_surf_avg_RBp,
    'flux_surf_avg_R2Bp2': flux_surf_avg_R2Bp2,
    'delta_upper_face': IMAS_data.profiles_1d.triangularity_upper,
    'delta_lower_face': IMAS_data.profiles_1d.triangularity_lower,
    'elongation': IMAS_data.profiles_1d.elongation,
    'vpr': vpr,
    'n_rho': n_rho,
    'hires_fac': hires_fac,
    'z_magnetic_axis': z_magnetic_axis,
  }

# todo check if we can copy form geometry without weird dependency loops
def face_to_cell(face):
  """Infers cell values corresponding to a vector of face values.
  Args:
    face: An array containing face values.

  Returns:
    cell: An array containing cell values.
  """

  return 0.5 * (face[:-1] + face[1:])
