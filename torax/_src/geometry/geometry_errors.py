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

"""Custom exceptions for geometry file loading."""


class GeometryFileError(Exception):
  """Base exception for geometry file loading errors."""

  def __init__(self, message: str, filepath: str | None = None):
    self.filepath = filepath
    super().__init__(message)


class GeometryFileNotFoundError(GeometryFileError):
  """Exception raised when a geometry file cannot be found."""

  def __init__(
      self,
      filepath: str,
      similar_files: list[str] | None = None,
      geometry_dir: str | None = None,
  ):
    self.similar_files = similar_files
    self.geometry_dir = geometry_dir

    message = f"Geometry file not found: '{filepath}'"

    if geometry_dir:
      message += f"\n  Searched in directory: {geometry_dir}"

    if similar_files:
      message += "\n\n  Did you mean one of these?"
      for sim_file in similar_files[:5]:  # Limit to top 5 suggestions
        message += f"\n    - {sim_file}"

    message += "\n\n  Action: Check the file path and ensure the file exists."

    super().__init__(message, filepath)


class GeometryFileFormatError(GeometryFileError):
  """Exception raised when a geometry file has an invalid format."""

  def __init__(
      self,
      filepath: str,
      expected_format: str,
      details: str | None = None,
  ):
    self.expected_format = expected_format
    self.details = details

    message = (
        f"Invalid geometry file format: '{filepath}'\n"
        f"  Expected format: {expected_format}"
    )

    if details:
      message += f"\n  Details: {details}"

    message += (
        "\n\n  Action: Verify the file format matches the expected type "
        f"({expected_format})."
    )

    super().__init__(message, filepath)


class GeometryFilePermissionError(GeometryFileError):
  """Exception raised when a geometry file cannot be accessed due to permissions."""

  def __init__(self, filepath: str, original_error: Exception | None = None):
    self.original_error = original_error

    message = f"Permission denied when accessing geometry file: '{filepath}'"

    if original_error:
      message += f"\n  System error: {str(original_error)}"

    message += (
        "\n\n  Action: Check file permissions and ensure you have read access."
    )

    super().__init__(message, filepath)


class GeometryFileEmptyError(GeometryFileError):
  """Exception raised when a geometry file is empty."""

  def __init__(self, filepath: str):
    message = (
        f"Geometry file is empty: '{filepath}'\n"
        "  Action: Ensure the file contains valid geometry data."
    )
    super().__init__(message, filepath)


class GeometryDataValidationError(GeometryFileError):
  """Exception raised when geometry data fails validation."""

  def __init__(
      self,
      filepath: str,
      field_name: str,
      issue: str,
  ):
    self.field_name = field_name
    self.issue = issue

    message = (
        f"Geometry data validation failed for '{filepath}'\n"
        f"  Field: {field_name}\n"
        f"  Issue: {issue}\n"
        "  Action: Check the geometry file data for physical constraints."
    )

    super().__init__(message, filepath)
