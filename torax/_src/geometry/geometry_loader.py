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

import difflib
import enum
import logging
import os
from typing import IO

import eqdsk
import numpy as np
import scipy

from torax._src import path_utils
from torax._src.geometry import geometry_errors


@enum.unique
class GeometrySource(enum.Enum):
  """Integer enum for geometry source."""

  CHEASE = 0
  FBT = 1
  EQDSK = 2


def _find_similar_files(
    target_file: str,
    directory: str,
    max_suggestions: int = 5,
) -> list[str]:
  """Find files in directory similar to target_file.

  Args:
    target_file: The target filename to match against.
    directory: Directory to search for similar files.
    max_suggestions: Maximum number of suggestions to return.

  Returns:
    List of similar filenames, sorted by similarity score.
  """
  if not os.path.exists(directory) or not os.path.isdir(directory):
    return []

  try:
    all_files = os.listdir(directory)
  except PermissionError:
    logging.warning("Permission denied when listing directory: %s", directory)
    return []

  # Use difflib to find similar filenames
  similar_files = difflib.get_close_matches(
      target_file,
      all_files,
      n=max_suggestions,
      cutoff=0.6,  # Similarity threshold
  )

  return similar_files


def _validate_file_access(filepath: str, geometry_dir: str) -> None:
  """Validate that a file exists and is accessible.

  Args:
    filepath: Path to the file to validate.
    geometry_dir: Directory containing the geometry file.

  Raises:
    GeometryFileNotFoundError: If file does not exist.
    GeometryFilePermissionError: If file cannot be accessed.
    GeometryFileEmptyError: If file is empty.
  """
  # Check file existence
  if not os.path.exists(filepath):
    logging.error("Geometry file not found: %s", filepath)
    # Find similar files for suggestions
    filename = os.path.basename(filepath)
    similar_files = _find_similar_files(filename, geometry_dir)
    raise geometry_errors.GeometryFileNotFoundError(
        filepath=filepath,
        similar_files=similar_files,
        geometry_dir=geometry_dir,
    )

  # Check file permissions
  if not os.access(filepath, os.R_OK):
    logging.error("Permission denied for geometry file: %s", filepath)
    raise geometry_errors.GeometryFilePermissionError(filepath=filepath)

  # Check if file is empty
  if os.path.getsize(filepath) == 0:
    logging.error("Geometry file is empty: %s", filepath)
    raise geometry_errors.GeometryFileEmptyError(filepath=filepath)

  logging.info("File validation passed for: %s", filepath)


def _load_CHEASE_data(  # pylint: disable=invalid-name
    file_path: str,
) -> dict[str, np.ndarray]:
  """Loads the data from a CHEASE file into a dictionary.

  Args:
    file_path: Path to the CHEASE file.

  Returns:
    Dictionary containing the CHEASE data.

  Raises:
    GeometryFileFormatError: If the file format is invalid.
  """
  logging.info("Loading CHEASE data from: %s", file_path)

  try:
    with open(file_path, "r") as file:
      chease_data = {}

      # Read header line
      header_line = file.readline().strip()
      if not header_line:
        raise geometry_errors.GeometryFileFormatError(
            filepath=file_path,
            expected_format="CHEASE",
            details="File header is missing or empty",
        )

      var_labels = header_line.split()[1:]  # ignore % comment column

      if not var_labels:
        raise geometry_errors.GeometryFileFormatError(
            filepath=file_path,
            expected_format="CHEASE",
            details="No variable labels found in header",
        )

      for var_label in var_labels:
        chease_data[var_label] = []

      # store data in respective keys
      for line_num, line in enumerate(file, start=2):
        values = line.strip().split()
        if not values:  # Skip empty lines
          continue

        if len(values) != len(var_labels):
          raise geometry_errors.GeometryFileFormatError(
              filepath=file_path,
              expected_format="CHEASE",
              details=(
                  f"Line {line_num}: Expected {len(var_labels)} values, "
                  f"got {len(values)}"
              ),
          )

        try:
          for var_label, value in zip(var_labels, values):
            chease_data[var_label].append(float(value))
        except ValueError as e:
          raise geometry_errors.GeometryFileFormatError(
              filepath=file_path,
              expected_format="CHEASE",
              details=f"Line {line_num}: Cannot convert '{value}' to float",
          ) from e

    # Convert lists to numpy arrays
    result = {
        var_label: np.asarray(chease_data[var_label])
        for var_label in chease_data
    }

    logging.info(
        "Successfully loaded CHEASE data with %d variables", len(result)
    )
    return result

  except geometry_errors.GeometryFileError:
    # Re-raise our custom exceptions
    raise
  except Exception as e:
    # Catch any other unexpected errors
    logging.exception("Unexpected error loading CHEASE file: %s", file_path)
    raise geometry_errors.GeometryFileFormatError(
        filepath=file_path,
        expected_format="CHEASE",
        details=f"Unexpected error: {type(e).__name__}: {str(e)}",
    ) from e


def _load_fbt_data(file_path: str | IO[bytes]) -> dict[str, np.ndarray]:
  """Loads data into a dictionary from an FBT-LY file or file path.

  Args:
    file_path: Path to the FBT file or file-like object.

  Returns:
    Dictionary containing the FBT data.

  Raises:
    GeometryFileFormatError: If the file format is invalid.
  """
  filepath_str = file_path if isinstance(file_path, str) else "<file object>"
  logging.info("Loading FBT data from: %s", filepath_str)

  try:
    meq_data = scipy.io.loadmat(file_path, squeeze_me=True)

    if not meq_data:
      raise geometry_errors.GeometryFileFormatError(
          filepath=filepath_str,
          expected_format="FBT (MATLAB .mat file)",
          details="File loaded but contains no data",
      )

    if "LY" not in meq_data:
      # L file or LY file not the output of MEQ meqlpack. Return as is.
      logging.info("Loaded FBT data (L format) with %d keys", len(meq_data))
      return meq_data
    else:
      # LY bundle file. numpy structured array likely resulting from
      # scipy.io.loadmat returning the output of MEQ meqlpack.m.
      # Reformat to return a dict. Shapes of 2D arrays are (radius, time)
      LY_bundle_dict = {}  # pylint: disable=invalid-name
      LY_bundle_array = meq_data["LY"]  # pylint: disable=invalid-name

      if not hasattr(LY_bundle_array, "dtype") or not hasattr(
          LY_bundle_array.dtype, "names"
      ):
        raise geometry_errors.GeometryFileFormatError(
            filepath=filepath_str,
            expected_format="FBT (MATLAB .mat file with LY bundle)",
            details="LY field does not have expected structured array format",
        )

      field_names = LY_bundle_array.dtype.names
      if not field_names:
        raise geometry_errors.GeometryFileFormatError(
            filepath=filepath_str,
            expected_format="FBT (MATLAB .mat file with LY bundle)",
            details="LY bundle has no field names",
        )

      for name in field_names:
        LY_bundle_dict[name] = LY_bundle_array[name].item()

      logging.info(
          "Loaded FBT data (LY format) with %d fields", len(LY_bundle_dict)
      )
      return LY_bundle_dict

  except geometry_errors.GeometryFileError:
    # Re-raise our custom exceptions
    raise
  except (OSError, IOError) as e:
    # File I/O errors
    logging.exception("I/O error loading FBT file: %s", filepath_str)
    raise geometry_errors.GeometryFileFormatError(
        filepath=filepath_str,
        expected_format="FBT (MATLAB .mat file)",
        details=f"I/O error: {str(e)}",
    ) from e
  except Exception as e:
    # Catch scipy.io.loadmat errors and other unexpected errors
    logging.exception("Error loading FBT file: %s", filepath_str)
    raise geometry_errors.GeometryFileFormatError(
        filepath=filepath_str,
        expected_format="FBT (MATLAB .mat file)",
        details=f"Error: {type(e).__name__}: {str(e)}",
    ) from e


def _load_eqdsk_data(file_path: str, cocos: int) -> dict[str, np.ndarray]:
  """Loads data into a dictionary from an EQDSK file.

  Args:
    file_path: Path to the EQDSK file.
    cocos: The COCOS convention of the input file.

  Returns:
    Dictionary containing the EQDSK data (converted to COCOS=11).

  Raises:
    GeometryFileFormatError: If the file format is invalid or COCOS is invalid.
  """
  logging.info("Loading EQDSK data from: %s (COCOS=%d)", file_path, cocos)

  # Validate COCOS parameter
  if cocos not in range(1, 19):  # Valid COCOS values are 1-18
    raise geometry_errors.GeometryFileFormatError(
        filepath=file_path,
        expected_format="EQDSK",
        details=f"Invalid COCOS value: {cocos}. Must be between 1 and 18.",
    )

  try:
    eqdsk_data = eqdsk.EQDSKInterface.from_file(
        file_path, from_cocos=cocos, to_cocos=11
    )
  except FileNotFoundError as e:
    # This shouldn't happen as we validate first, but handle it anyway
    logging.error("EQDSK file not found during eqdsk library load: %s", file_path)
    raise geometry_errors.GeometryFileNotFoundError(
        filepath=file_path,
        similar_files=None,
        geometry_dir=os.path.dirname(file_path),
    ) from e
  except Exception as e:
    # Catch eqdsk library errors
    logging.exception("Error parsing EQDSK file: %s", file_path)
    raise geometry_errors.GeometryFileFormatError(
        filepath=file_path,
        expected_format="EQDSK",
        details=f"EQDSK parsing error: {type(e).__name__}: {str(e)}",
    ) from e

  try:
    eqdsk_dict = eqdsk_data.__dict__  # dict(eqdsk_data) is broken

    # Validate required fields
    required_fields = ["cplasma", "bcentre", "psimag", "psibdry", "psi", "pprime", "ffprime"]
    missing_fields = [f for f in required_fields if f not in eqdsk_dict]
    if missing_fields:
      raise geometry_errors.GeometryFileFormatError(
          filepath=file_path,
          expected_format="EQDSK",
          details=f"Missing required fields: {', '.join(missing_fields)}",
      )

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

    logging.info("Successfully loaded EQDSK data with %d fields", len(eqdsk_dict))
    return eqdsk_dict

  except geometry_errors.GeometryFileError:
    # Re-raise our custom exceptions
    raise
  except Exception as e:
    # Catch any unexpected errors during post-processing
    logging.exception("Error processing EQDSK data: %s", file_path)
    raise geometry_errors.GeometryFileFormatError(
        filepath=file_path,
        expected_format="EQDSK",
        details=f"Post-processing error: {type(e).__name__}: {str(e)}",
    ) from e


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
    GeometryFileNotFoundError: If the file does not exist.
    GeometryFilePermissionError: If the file cannot be accessed.
    GeometryFileEmptyError: If the file is empty.
    GeometryFileFormatError: If the file format is invalid.
  """
  logging.info(
      "Loading geometry data: file=%s, source=%s, dir=%s",
      geometry_file,
      geometry_source.name,
      geometry_dir,
  )

  geometry_dir = get_geometry_dir(geometry_dir)
  filepath = os.path.join(geometry_dir, geometry_file)

  # Validate file exists and is accessible
  _validate_file_access(filepath, geometry_dir)

  # Validate EQDSK-specific requirements
  if geometry_source == GeometrySource.EQDSK and cocos is None:
    raise ValueError(
        "COCOS parameter must be provided when loading EQDSK files. "
        "Please specify the cocos parameter."
    )

  # Load geometry from file based on source type
  try:
    match geometry_source:
      case GeometrySource.CHEASE:
        result = _load_CHEASE_data(file_path=filepath)
      case GeometrySource.FBT:
        result = _load_fbt_data(file_path=filepath)
      case GeometrySource.EQDSK:
        result = _load_eqdsk_data(file_path=filepath, cocos=cocos)
      case _:
        raise ValueError(f"Unknown geometry source: {geometry_source}")

    logging.info(
        "Successfully loaded geometry data from %s (%s format)",
        geometry_file,
        geometry_source.name,
    )
    return result

  except geometry_errors.GeometryFileError:
    # Re-raise our custom exceptions
    raise
  except Exception as e:
    # Catch any other unexpected errors
    logging.exception(
        "Unexpected error loading geometry file: %s", filepath
    )
    raise geometry_errors.GeometryFileError(
        f"Unexpected error loading geometry file: {type(e).__name__}: {str(e)}",
        filepath=filepath,
    ) from e
