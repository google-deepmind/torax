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
import imaspy
import yaml
import datetime
import os
import numpy as np


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
