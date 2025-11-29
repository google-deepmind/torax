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

"""Helper utilities for geometry file loading error handling."""

import difflib
import logging
import os

from torax._src.geometry import geometry_errors


def find_similar_files(
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


def validate_file_access(filepath: str, geometry_dir: str) -> None:
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
    similar_files = find_similar_files(filename, geometry_dir)
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
