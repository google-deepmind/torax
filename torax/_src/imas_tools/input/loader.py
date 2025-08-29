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

"""Generic loader function that can load any IDS from a netCDF file or from an
IMASdb.
"""
import os
from typing import Final

import imas
from imas import ids_toplevel
from torax._src.config import config_loader

# Names of the IDSs that can be loaded and used in TORAX.
CORE_PROFILES: Final[str] = "core_profiles"
PLASMA_PROFILES: Final[str] = "plasma_profiles"
EQUILIBRIUM: Final[str] = "equilibrium"


def load_imas_data(
    uri: str,
    ids_name: str,
    directory: str | None = None,
) -> ids_toplevel.IDSToplevel:
  """Loads a full IDS for a given uri or path_name and a given ids_name.

  It can load either an IMAS netCDF file with filename as uri and given
  directory or from an IMASdb by giving the full uri of the IDS and the
  directory arg will be ignored. Note that loading from an IMASdb requires
  IMAS-core. The loaded IDS can then be used as input to
  core_profiles_from_IMAS().
  """
  # Differentiate between netCDF and IMASdb uris. For IMASdb files the full
  # filepath is already provided in the uri.
  if uri.endswith(".nc"):
    if directory is None:
      directory = os.path.join(
          config_loader.torax_path(), "data/third_party/imas_data"
      )
    uri = os.path.join(directory, uri)
  with imas.DBEntry(uri=uri, mode="r") as db:
    ids = db.get(ids_name=ids_name)
  return ids
