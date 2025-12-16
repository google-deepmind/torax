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

"""File validation utilities for geometry file loading."""

import logging
import os
from pathlib import Path
from typing import Union, List

import numpy as np
from torax._src.geometry import geometry_errors

logger = logging.getLogger(__name__)


def validate_file_access(
    file_path: Union[str, Path], file_type: str, validate_format: bool = True
) -> Path:
    """Validate file exists and is accessible.

    Args:
      file_path: Path to geometry file
      file_type: Type of geometry file ('chease', 'fbt', 'eqdsk')
      validate_format: Whether to validate file extension

    Returns:
      Path object for the validated file

    Raises:
      GeometryFileNotFoundError: If file doesn't exist
      GeometryFilePermissionError: If file can't be read
      GeometryFileFormatError: If file is empty or has wrong format
    """
    path = Path(file_path)

    # Check file exists
    if not path.exists():
        error_msg = (
            f"Geometry file not found: {file_path}\n"
            "Please check:\n"
            "  1. File path is correct\n"
            "  2. File exists at the specified location\n"
            "  3. Filename spelling is correct"
        )

        logger.error(error_msg)
        raise geometry_errors.GeometryFileNotFoundError(error_msg)

    # Check file permissions
    if not os.access(path, os.R_OK):
        error_msg = (
            f"No read permission for geometry file: {file_path}\n"
            f"Please check file permissions and run:\n"
            f"  chmod +r {file_path}"
        )
        logger.error(error_msg)
        raise geometry_errors.GeometryFilePermissionError(error_msg)

    # Check file is not empty
    if path.stat().st_size == 0:
        error_msg = f"Geometry file is empty: {file_path}"
        logger.error(error_msg)
        raise geometry_errors.GeometryFileFormatError(error_msg)

    # Validate file type
    if validate_format:
        expected_extensions = {
            "chease": [".chease", ".txt", ".mat2cols"],
            "fbt": [".fbt", ".dat", ".mat"],
            "eqdsk": [".eqdsk", ".geqdsk"],
        }

        if file_type.lower() in expected_extensions:
            valid_ext = path.suffix.lower(
            ) in expected_extensions[file_type.lower()]
            if not valid_ext:
                logger.warning(
                    "File extension '%s' unusual for %s files. Expected: %s",
                    path.suffix,
                    file_type.upper(),
                    expected_extensions[file_type.lower()],
                )

    return path


def validate_geometry_data(data: dict, file_type: str, file_path: str) -> None:
    """Validate loaded geometry data structure.

    This performs basic structural validation to catch corrupt files,
    but does not enforce strict physical constraints that might vary
    by use case.

    Args:
      data: Loaded geometry data dictionary
      file_type: Type of geometry file
      file_path: Path to file for error messages

    Raises:
      GeometryDataValidationError: If data fails validation
    """
    issues = []

    # Format-specific validation
    if file_type.lower() == "chease":
        _validate_chease_data(data, issues)
    elif file_type.lower() == "fbt":
        _validate_fbt_data(data, issues)
    elif file_type.lower() == "eqdsk":
        _validate_eqdsk_data(data, issues)

    if issues:
        error_msg = (
            f"Geometry data validation failed for {file_type.upper()} file: {
                file_path}\n"
            + "\n".join(f"  - {issue}" for issue in issues)
        )
        logger.error(error_msg)
        raise geometry_errors.GeometryDataValidationError(error_msg)


def _validate_chease_data(data: dict, issues: List[str]) -> None:
    """Validate CHEASE-specific data structure.

    Checks for required fields and data consistency, but does not
    enforce strict physical constraints to allow for edge cases.
    """
    required_fields = [
        "PSIchease=psi/2pi",
        "Ipprofile",
        "RHO_TOR=sqrt(Phi/pi/B0)",
        "R_INBOARD",
        "R_OUTBOARD",
        "T=RBphi",
    ]

    for field in required_fields:
        if field not in data:
            issues.append(f"Missing required CHEASE field: {field}")

    # Check data structure consistency
    if data:
        array_fields = {k: v for k,
                        v in data.items() if isinstance(v, np.ndarray)}
        if array_fields:
            lengths = [len(v) for v in array_fields.values()]
            if len(set(lengths)) > 1:
                issues.append(
                    f"CHEASE arrays have inconsistent lengths: "
                    f"{dict(zip(array_fields.keys(), lengths))}"
                )

    # Basic sanity checks (only flag clearly invalid data)
    if "R_INBOARD" in data and "R_OUTBOARD" in data:
        r_in = np.asarray(data["R_INBOARD"])
        r_out = np.asarray(data["R_OUTBOARD"])

        # Only check for clearly invalid values (negative radii)
        if np.any(r_in < 0):
            issues.append(
                "Inboard radius (R_INBOARD) contains negative values")
        if np.any(r_out < 0):
            issues.append(
                "Outboard radius (R_OUTBOARD) contains negative values")

        # Only flag if data is consistently invalid (not just edge cases)
        # Use mean values to avoid issues with individual grid points
        mean_r_in = np.mean(r_in[r_in > 0]) if np.any(r_in > 0) else 0
        mean_r_out = np.mean(r_out[r_out > 0]) if np.any(r_out > 0) else 0

        if mean_r_out > 0 and mean_r_in > 0 and mean_r_out < mean_r_in:
            issues.append(
                "Average outboard radius less than inboard radius "
                "(possible column swap in file)"
            )


def _validate_fbt_data(data: dict, issues: List[str]) -> None:
    """Validate FBT-specific data structure."""
    if "LY" in data:
        ly_data = data["LY"]
        # Check if it's a structured array
        if not hasattr(ly_data, "dtype"):
            issues.append("FBT LY bundle has unexpected structure")
            return

        # Check for some expected fields (not all required for compatibility)
        common_fields = ["rBt", "aminor", "rgeom"]
        if hasattr(ly_data.dtype, "names") and ly_data.dtype.names:
            available_fields = ly_data.dtype.names
            missing_common = [
                f for f in common_fields if f not in available_fields]
            if len(missing_common) == len(common_fields):
                issues.append(
                    f"FBT LY bundle missing common fields. "
                    f"Expected at least one of: {common_fields}"
                )
    else:
        # Non-LY format - just check for basic fields
        common_fields = ["rBt", "aminor", "rgeom"]
        missing = [f for f in common_fields if f not in data]
        if len(missing) == len(common_fields):
            issues.append(
                f"FBT file missing common fields. "
                f"Expected at least one of: {common_fields}"
            )


def _validate_eqdsk_data(data: dict, issues: List[str]) -> None:
    """Validate EQDSK-specific data structure."""
    required_fields = [
        "bcentre",
        "xmag",
        "zmag",
        "psimag",
        "psibdry",
        "xbdry",
        "zbdry",
        "fpol",
        "qpsi",
        "psi",
    ]

    for field in required_fields:
        if field not in data:
            issues.append(f"Missing required EQDSK field: {field}")

    # Basic structural checks
    if "xmag" in data:
        try:
            xmag_val = float(data["xmag"])
            if xmag_val <= 0:
                issues.append(
                    "Magnetic axis major radius (xmag) must be positive")
        except (ValueError, TypeError):
            issues.append(
                "Magnetic axis major radius (xmag) has invalid format")

    if "xbdry" in data and "zbdry" in data:
        try:
            x_bdy = np.asarray(data["xbdry"])
            z_bdy = np.asarray(data["zbdry"])

            if len(x_bdy) != len(z_bdy):
                issues.append(
                    "Boundary x and z coordinates must have same length")
            elif len(x_bdy) < 3:
                issues.append("Boundary must have at least 3 points")
        except (ValueError, TypeError):
            issues.append("Boundary coordinates have invalid format")
