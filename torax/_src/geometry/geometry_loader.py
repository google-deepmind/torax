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
import logging
import os
from typing import IO

import eqdsk
import numpy as np
import scipy
import torax
from torax.imas_tools import util as imas_util


@enum.unique
class GeometrySource(enum.Enum):
  """Integer enum for geometry source."""

  CHEASE = 0
  FBT = 1
  EQDSK = 2
  IMAS = 3


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


def _load_eqdsk_data(file_path: str) -> dict[str, np.ndarray]:
  eqdsk_data = eqdsk.EQDSKInterface.from_file(file_path, no_cocos=True)
  # TODO(b/335204606): handle COCOS shenanigans
  logging.warning(
      """
               WARNING: Using EQDSK geometry.
               The TORAX EQDSK converter has only been tested against CHEASE-generated EQDSK which is COCOS=2.
               The converter is not guaranteed to work as expected with arbitrary EQDSK input. Please verify carefully.
               Future work will be done to correctly handle EQDSK inputs provided with a specific COCOS value."""
  )
  return eqdsk_data.__dict__  # dict(eqdsk_data)


def load_geo_data(
    geometry_dir: str | None,
    geometry_file: str,
    geometry_source: GeometrySource,
) -> dict[str, np.ndarray]:
  """Loads the data from a CHEASE file into a dictionary."""
  if geometry_dir is None:
    geometry_dir = os.path.join(torax.__path__[0], 'data/third_party/geo')
  filepath = os.path.join(geometry_dir, geometry_file)

  # initialize geometry from file
  match geometry_source:
    case GeometrySource.CHEASE:
      return _load_CHEASE_data(file_path=filepath)
    case GeometrySource.FBT:
      return _load_fbt_data(file_path=filepath)
    case GeometrySource.EQDSK:
      return _load_eqdsk_data(file_path=filepath)
    case GeometrySource.IMAS:
      return imas_util.load_IMAS_data(path=filepath, ids_name='equilibrium')
    case _:
      raise ValueError(f'Unknown geometry source: {geometry_source}')
