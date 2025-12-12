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

import eqdsk
import numpy as np
import scipy

from torax._src import path_utils


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

  with open(file_path, "r") as file:
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
  """Loads data from an FBT-LY file or file path.

  Handles MEQ nesting patterns such as L['L'] and LY['LY'].
  """
  meq_data = scipy.io.loadmat(file_path)

  if "LY" not in meq_data:
    # L file or LY file not the output of MEQ meqlpack.
    # Generalize MEQ loader to handle L nesting: some .mat files have nested
    # structures where the value is a dict or object array containing the key
    # itself (e.g., meq_data["foo"]["foo"]). We unwrap these cases here.
    cleaned = {}
    for key, value in meq_data.items():
      # Skip scipy.io.loadmat metadata.
      if key.startswith("__"):
        continue
      # Handle dict nesting: value is a dict containing the same key.
      if isinstance(value, dict) and key in value:
        cleaned[key] = value[key]
        continue
      # Handle object-array nesting produced by some MEQ workflows.
      if isinstance(value, np.ndarray) and value.dtype == object:
        try:
          obj = value.item()
          if isinstance(obj, dict) and key in obj:
            cleaned[key] = obj[key]
          else:
            cleaned[key] = obj
          continue
        except Exception:  # pylint: disable=broad-exception-caught
          pass
      # Handle structured array with nested dict (from scipy.io.savemat).
      if isinstance(value, np.ndarray) and value.dtype.names:
        try:
          inner = value[key].item()
          if isinstance(inner, dict) and key in inner:
            cleaned[key] = inner[key]
          else:
            cleaned[key] = inner
          continue
        except Exception:  # pylint: disable=broad-exception-caught
          pass
      # Plain ndarray: keep as-is.
      if isinstance(value, np.ndarray):
        cleaned[key] = value
        continue
      # Scalar fallback.
      cleaned[key] = np.array(value)
    return cleaned
  else:
    # LY bundle file. numpy structured array likely resulting from
    # scipy.io.loadmat returning the output of MEQ meqlpack.m.
    # Reformat to return a dict. Shapes of 2D arrays are (radius, time)
    LY_bundle_dict = {}  # pylint: disable=invalid-name
    LY_bundle_array = meq_data["LY"]  # pylint: disable=invalid-name
    field_names = LY_bundle_array.dtype.names
    for name in field_names:
      LY_bundle_dict[name] = LY_bundle_array[name].item()

    return LY_bundle_dict


def _load_eqdsk_data(file_path: str, cocos: int) -> dict[str, np.ndarray]:
  """Loads data into a dictionary from an EQDSK file."""
  eqdsk_data = eqdsk.EQDSKInterface.from_file(
      file_path, from_cocos=cocos, to_cocos=11
  )
  eqdsk_dict = eqdsk_data.__dict__  # dict(eqdsk_data) is broken

  # Set Ip and B0 positive
  # This is a choice not necessarily governed by COCOS (see Table Ib in COCOS
  # paper). The logic here may need to change when rotation is added,
  # particularly combined vExB and parallel rotation, and momentum transport.
  # https://doi.org/10.1016/j.cpc.2012.09.010)
  # TODO(b/461727869): consolidate logic when full rotation features are added.
  eqdsk_dict["cplasma"] = np.abs(eqdsk_dict["cplasma"])
  eqdsk_dict["bcentre"] = np.abs(eqdsk_dict["bcentre"])
  # Ensure that psi grows with increasing radius, consistent with positive Ip
  if eqdsk_dict["psimag"] > eqdsk_dict["psibdry"]:
    eqdsk_dict["psi"] = -eqdsk_dict["psi"]
    eqdsk_dict["psimag"] = -eqdsk_dict["psimag"]
    eqdsk_dict["psibdry"] = -eqdsk_dict["psibdry"]
  # pprime and ffprime are gradients with respect to the poloidal flux (psi),
  # and its sign depends on the sign of the poloidal flux. Since pressure on
  # average should decrease as the radius increases, positive mean pprime means
  # decreasing psi which is not consistent with positive Ip, therefore we need
  # to flip the values.
  if np.mean(eqdsk_dict["pprime"]) > 0:
    eqdsk_dict["pprime"] = -eqdsk_dict["pprime"]
    eqdsk_dict["ffprime"] = -eqdsk_dict["ffprime"]
  return eqdsk_dict


def get_geometry_dir(geometry_dir: str | None = None) -> str:
  """Gets the default geometry directory if no geometry_dir is provided."""
  if geometry_dir is None:
    relative_dir = "data/third_party/geo"
    geometry_dir = os.path.join(path_utils.torax_path(), relative_dir)
  return geometry_dir


def load_geo_data(
    geometry_dir: str | None,
    geometry_file: str,
    geometry_source: GeometrySource,
    cocos: int | None = None,
) -> dict[str, np.ndarray]:
  """Loads the data from a geometry file into a dictionary.

  Args:
    geometry_dir: Directory containing the geometry file. If None, a default
      directory is used.
    geometry_file: Name of the geometry file.
    geometry_source: The type of geometry file, from the GeometrySource enum.
    cocos: The COCOS convention of the EQDSK file. If geometry_source=EQDSK,
      input cocos must be provided, and the output dict will be in cocos=11.

  Returns:
    A dictionary containing the geometry data.

  Raises:
    ValueError: If geometry_source is not a valid GeometrySource.
  """
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
      raise ValueError(f"Unknown geometry source: {geometry_source}")
