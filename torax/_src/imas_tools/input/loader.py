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

import imas
from imas import ids_toplevel
from torax._src import path_utils

# Names of the IDSs that can be loaded and used in TORAX.
IDS = Literal["core_profiles", "plasma_profiles", "equilibrium"]
_TORAX_IMAS_DD_VERSION = "4.0.0"


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
  if uri.endswith(".nc"):
    if directory is None:
      directory = path_utils.torax_path().joinpath("data/third_party/imas")
    uri = os.path.join(directory, uri)
  with imas.DBEntry(uri=uri, mode="r", dd_version=_TORAX_IMAS_DD_VERSION) as db:
    if not explicit_convert:
      ids = db.get(ids_name=ids_name, autoconvert=True)
    else:
      ids = db.get(ids_name=ids_name, autoconvert=False)
      ids = imas.convert_ids(ids, _TORAX_IMAS_DD_VERSION)
  return ids
