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

"""Utility functions for geometry file loading with enhanced error handling."""

import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
from torax._src.geometry import geometry_loader
from torax._src.geometry import geometry_errors
from torax._src.geometry import file_validation


logger = logging.getLogger(__name__)


def load_geometry_file(
    file_path: Union[str, Path], file_type: str, validate_format: bool = True
) -> dict[str, np.ndarray]:
    """Load geometry file with comprehensive error handling.

    Args:
      file_path: Path to geometry file (CHEASE, FBT, or EQDSK)
      file_type: Type of geometry file ('chease', 'fbt', 'eqdsk')
      validate_format: Whether to validate file format before parsing

    Returns:
      Dictionary containing parsed geometry data

    Raises:
      GeometryFileNotFoundError: If file doesn't exist
      GeometryFilePermissionError: If file can't be read
      GeometryFileFormatError: If file format is invalid
      GeometryDataValidationError: If data fails validation
    """
    type_mapping = {
        "chease": geometry_loader.GeometrySource.CHEASE,
        "fbt": geometry_loader.GeometrySource.FBT,
        "eqdsk": geometry_loader.GeometrySource.EQDSK,
    }

    file_type_lower = file_type.lower()
    if file_type_lower not in type_mapping:
        supported_types = list(type_mapping.keys())
        raise ValueError(
            f"Unsupported file type: {
                file_type}. Supported types: {supported_types}"
        )

    path = Path(file_path)
    geometry_dir = str(path.parent)
    geometry_file = path.name
    geometry_source = type_mapping[file_type_lower]

    return geometry_loader.load_geo_data(geometry_dir, geometry_file, geometry_source)


def load_chease_file(file_path: Union[str, Path]) -> dict[str, np.ndarray]:
    """Load CHEASE geometry file with error handling."""
    return load_geometry_file(file_path, "chease")


def load_fbt_file(file_path: Union[str, Path]) -> dict[str, np.ndarray]:
    """Load FBT geometry file with error handling."""
    return load_geometry_file(file_path, "fbt")


def load_eqdsk_file(file_path: Union[str, Path]) -> dict[str, np.ndarray]:
    """Load EQDSK geometry file with error handling."""
    return load_geometry_file(file_path, "eqdsk")


def validate_geometry_data(data: dict[str, np.ndarray], file_type: str) -> None:
    """Validate loaded geometry data meets physical constraints."""
    file_validation.validate_geometry_data(data, file_type, "geometry_data")
