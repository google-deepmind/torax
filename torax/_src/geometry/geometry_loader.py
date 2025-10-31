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

"""File I/O for loading geometry files."""

import enum
import os
from typing import IO
from typing import Literal

import eqdsk
import numpy as np
import scipy
import torax

COCOSInt = Literal[1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]


@enum.unique
class GeometrySource(enum.Enum):
  """Integer enum for geometry source."""

  CHEASE = 0
  FBT = 1
  EQDSK = 2


def _load_CHEASE_data(  # pylint: disable=invalid-name
    file_path: str,
) -> dict[str, np.ndarray]:
  """Loads the data from a CHEASE file into a dictionary."""

  with open(file_path, 'r') as file:
    chease_data = {}
    var_labels = file.readline().strip().split()[1:]  # ignore % comment column

    for var_label in var_labels:
      chease_data[var_label] = []

    # store data in respective keys
    for line in file:
      values = line.strip().split()
      for var_label, value in zip(var_labels, values):
        chease_data[var_label].append(float(value))

  # Convert lists to jax arrays.
  return {
      var_label: np.asarray(chease_data[var_label]) for var_label in chease_data
  }


def _load_fbt_data(file_path: str | IO[bytes]) -> dict[str, np.ndarray]:
  """Loads data into a dictionary from an FBT-LY file or file path."""

  meq_data = scipy.io.loadmat(file_path, squeeze_me=True)

  if 'LY' not in meq_data:
    # L file or LY file not the output of MEQ meqlpack. Return as is.
    return meq_data
  else:
    # LY bundle file. numpy structured array likely resulting from
    # scipy.io.loadmat returning the output of MEQ meqlpack.m.
    # Reformat to return a dict. Shapes of 2D arrays are (radius, time)
    LY_bundle_dict = {}  # pylint: disable=invalid-name
    LY_bundle_array = meq_data['LY']  # pylint: disable=invalid-name
    field_names = LY_bundle_array.dtype.names
    for name in field_names:
      LY_bundle_dict[name] = LY_bundle_array[name].item()

    return LY_bundle_dict


def _load_eqdsk_data(file_path: str, cocos: COCOSInt) -> dict[str, np.ndarray]:
  eqdsk_data = eqdsk.EQDSKInterface.from_file(
      file_path, from_cocos=cocos, to_cocos=11
  )
  eqdsk_dict = eqdsk_data.__dict__  # dict(eqdsk_data) is broken

  # Set Ip and B0 positive
  # This is a choice not necessarily governed by COCOS (see Table Ib in COCOS
  # paper). TODO: The logic here may need to change when rotation is added.
  # https://doi.org/10.1016/j.cpc.2012.09.010)
  eqdsk_dict["cplasma"] = np.abs(eqdsk_dict["cplasma"])
  eqdsk_dict["bcentre"] = np.abs(eqdsk_dict["bcentre"])
  if eqdsk_dict["psimag"] > eqdsk_dict["psibdry"]:
    eqdsk_dict["psi"] = -eqdsk_dict["psi"]
    eqdsk_dict["psimag"] = -eqdsk_dict["psimag"]
    eqdsk_dict["psibdry"] = -eqdsk_dict["psibdry"]
  return eqdsk_dict


def get_geometry_dir(geometry_dir: str | None = None) -> str:
  """Gets the default geometry directory if no geometry_dir is provided."""
  if geometry_dir is None:
    geometry_dir = os.path.join(torax.__path__[0], 'data/third_party/geo')
  return geometry_dir


def load_geo_data(
    geometry_dir: str | None,
    geometry_file: str,
    geometry_source: GeometrySource,
    cocos: COCOSInt | None = None,
) -> dict[str, np.ndarray]:
  """Loads the data from a geometry file into a dictionary.

  If geometry_source=EQDSK, cocos must be provided."""
  geometry_dir = get_geometry_dir(geometry_dir)
  filepath = os.path.join(geometry_dir, geometry_file)

  # initialize geometry from file
  match geometry_source:
    case GeometrySource.CHEASE:
      return _load_CHEASE_data(file_path=filepath)
    case GeometrySource.FBT:
      return _load_fbt_data(file_path=filepath)
    case GeometrySource.EQDSK:
      return _load_eqdsk_data(file_path=filepath, cocos=cocos)
    case _:
      raise ValueError(f'Unknown geometry source: {geometry_source}')
