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
import pathlib
import os
from typing import Any

import yaml
import imas
from imas.ids_toplevel import IDSToplevel


def save_netCDF(
    directory_path: str | pathlib.Path | None,
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
    from torax._src.config.config_loader import torax_path
    directory_path =  pathlib.Path(directory_path) if isinstance(directory_path, str) else directory_path
    if not directory_path.is_dir():
        #Checks that the directory can be found.
        new_path = torax_path().joinpath(directory_path)
        if not new_path.is_dir():
          raise ValueError(f'Directory {directory_path} could not be found. If it is a relative path, it'
            ' could not be resolved relative to the working directory'
            f' {os.getcwd()} or the Torax directory {torax_path()}.'
          )
        directory_path = new_path

    if directory_path is None:
        directory_path = torax_path().joinpath('torax/data/third_party/geo')
    if file_name is None:
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = "IDS_file_" + date_str
    filepath = os.path.join(directory_path, file_name) + ".nc"
    with imas.DBEntry(uri=filepath, mode="w") as netcdf_entry:
        netcdf_entry.put(ids=IDS)
        print(f'Successfully saved file {filepath}')


def load_IMAS_data(uri: str, ids_name: str) -> IDSToplevel:
    """
    Loads a full IDS for a given uri or path_name and a given ids_name.
    """
    db = imas.DBEntry(uri=uri, mode="r")
    ids = db.get(ids_name=ids_name)
    return ids
