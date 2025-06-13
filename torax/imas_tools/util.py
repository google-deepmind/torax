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

"""Useful functions for handling of IMAS IDSs and converts them into TORAX
objects"""
import datetime
import importlib
import os
from typing import Any

import yaml

try:
    import imas
    from imas.ids_toplevel import IDSToplevel
except ImportError:
    IDSToplevel = Any


def requires_module(module_name: str):
    """
    Decorator that checks if a module can be imported.
    Returns the function if the module is available,
    otherwise raises an ImportError.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if importlib.util.find_spec(module_name) is None:
                raise ImportError(
                    f"Required module '{module_name}' is not installed. "
                    "Make sure you install the needed optional dependencies."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


@requires_module("imas_core")
def save_netCDF(
    directory_path: str | None,
    file_name: str | None,
    IDS: IDSToplevel,
):
    """Generates a netcdf file from an IDS.

    Args:
      directory_path: Directory where to save the netCDF output file. If None,
        it will be saved in data/third_party/geo.
      file_name: Desired output file name.
        If None will default to IDS_file_ + Time.
      IDS: IMAS Interface Data Structure object that will be written in the
        IMAS netCDF file.

    Returns:
      None
    """
    if directory_path is None:
        directory_path = "torax/data/third_party/geo"
    if file_name is None:
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = "IDS_file_" + date_str
    filepath = os.path.join(directory_path, file_name) + ".nc"
    with imas.DBEntry(filepath, "w") as netcdf_entry:
        netcdf_entry.put(IDS)


@requires_module("imas_core")
def load_ids_from_Data_entry(file_path: str, ids_name: str) -> IDSToplevel:
    """Loads the IDS for a single time slice from a specific data
    entry / scenario using the IMAS Access Layer

    Args:
      file_path: Absolute path to the yaml file containing the DB
      specifications of the desired IDS : input_database, shot, run,
        input_user_or_path and time_begin for the desired time_slice
      ids_name: name of IDS to load

    Returns:
      IDS"""
    file = open(file_path, "r")
    scenario = yaml.load(file, Loader=yaml.CLoader)
    file.close()
    if scenario["scenario_backend"] == "hdf5":
        sc_backend = imas.ids_defs.HDF5_BACKEND
    else:
        sc_backend = imas.ids_defs.MDSPLUS_BACKEND
    input = imas.DBEntry(
        sc_backend,
        scenario["input_database"],
        scenario["shot"],
        scenario["run_in"],
        scenario["input_user_or_path"],
    )
    input.open()
    timenow = scenario["time_begin"]
    IMAS_data = input.get_slice(ids_name, timenow, 1)

    return IMAS_data


def load_IDS_from_netCDF(file_path: str, ids_name: str) -> IDSToplevel:
    """Loads an IDS for a single time slice from an IMAS netCDF
    file path"""
    input = imas.DBEntry(file_path, "r")
    ids = input.get(ids_name)
    return ids


@requires_module("imas_core")
def load_IDS_from_hdf5(directory_path: str, ids_name: str) -> IDSToplevel:
    """Loads an IDS for a single time slice from the path of a
    local directory containing it stored with hdf5 backend. The repository must
    contain the master.h5 file."""
    imasuri = "imas:hdf5?path=" + directory_path
    input = imas.DBEntry(imasuri, "r")
    ids = input.get(ids_name)
    return ids


def load_IMAS_data(path: str, ids_name: str) -> IDSToplevel:
    """Loads an IDS for a single time slice either from a netCDF
    file path or from an hdf5 file in the given directory path."""
    if path[-3:] == ".nc":
        return load_IDS_from_netCDF(file_path=path, ids_name=ids_name)
    else:
        return load_IDS_from_hdf5(directory_path=path, ids_name=ids_name)


# todo check if we can copy form geometry without weird dependency loops
def face_to_cell(face):
    """Infers cell values corresponding to a vector of face values.
    Args:
      face: An array containing face values.

    Returns:
      cell: An array containing cell values.
    """

    return 0.5 * (face[:-1] + face[1:])
