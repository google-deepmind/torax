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
from pathlib import Path
from typing import IO, Union

import eqdsk
import numpy as np
import scipy
import torax
from torax._src.geometry import geometry_errors
from torax._src.geometry import file_validation

logger = logging.getLogger(__name__)


@enum.unique
class GeometrySource(enum.Enum):
    """Integer enum for geometry source."""

    CHEASE = 0
    FBT = 1
    EQDSK = 2


def _load_CHEASE_data(file_path: Union[str, Path]) -> dict[str, np.ndarray]:
    """Load data from a CHEASE file into a dictionary."""
    try:
        validated_path = file_validation.validate_file_access(
            file_path, "chease", validate_format=True
        )
        logger.info("Loading CHEASE geometry file: %s", file_path)

        with open(validated_path, "r", encoding="utf-8") as file:
            chease_data = {}

            header_line = file.readline().strip()
            if not header_line:
                raise geometry_errors.GeometryFileFormatError(
                    "CHEASE file empty or missing header: %s" % file_path
                )

            var_labels = header_line.split()[1:]
            if not var_labels:
                raise geometry_errors.GeometryFileFormatError(
                    "No variable labels in CHEASE header: %s" % file_path
                )

            for label in var_labels:
                chease_data[label] = []

            line_num = 1
            for line in file:
                line_num += 1
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    values = line.split()
                    if len(values) != len(var_labels):
                        raise geometry_errors.GeometryFileFormatError(
                            "Line %d in CHEASE file has %d values, expected %d: %s" %
                            (line_num, len(values), len(var_labels), file_path)
                        )
                    for label, value in zip(var_labels, values):
                        chease_data[label].append(float(value))
                except ValueError as e:
                    raise geometry_errors.GeometryFileFormatError(
                        "Failed parsing line %d in CHEASE file: %s\n%s" %
                        (line_num, file_path, str(e))
                    ) from e

        result = {k: np.asarray(v) for k, v in chease_data.items()}
        file_validation.validate_geometry_data(
            result, "chease", str(file_path))

        logger.info(
            "Successfully loaded CHEASE file with %d variables and %d grid points",
            len(result), len(next(iter(result.values())))
        )
        return result

    except UnicodeDecodeError as e:
        msg = ("Failed to decode CHEASE file: %s. File may be binary or wrong encoding."
               " Expected UTF-8/ASCII") % file_path
        logger.error(msg)
        raise geometry_errors.GeometryFileFormatError(msg) from e

    except (OSError, IOError) as e:
        msg = "Failed to read CHEASE file: %s. Error: %s" % (file_path, str(e))
        logger.error(msg)
        raise geometry_errors.GeometryFileError(msg) from e


def _load_fbt_data(file_path: Union[str, Path, IO[bytes]]) -> dict[str, np.ndarray]:
    """Load data from an FBT-LY file or file handle."""
    try:
        if isinstance(file_path, (str, Path)):
            validated_path = file_validation.validate_file_access(
                file_path, "fbt", validate_format=True
            )
            logger.info("Loading FBT geometry file: %s", file_path)
            load_path = validated_path
        else:
            load_path = file_path
            logger.info("Loading FBT geometry data from file handle")

        meq_data = scipy.io.loadmat(load_path, squeeze_me=True)

        if "LY" not in meq_data:
            result = meq_data
        else:
            LY_bundle_dict = {}
            LY_array = meq_data["LY"]
            field_names = LY_array.dtype.names
            if not field_names:
                raise geometry_errors.GeometryFileFormatError(
                    "FBT LY bundle has no field names: %s" % file_path
                )
            for name in field_names:
                LY_bundle_dict[name] = LY_array[name].item()
            result = LY_bundle_dict

        if isinstance(file_path, (str, Path)):
            file_validation.validate_geometry_data(
                result, "fbt", str(file_path))
            logger.info(
                "Successfully loaded FBT file with %d fields", len(result))

        return result

    except (scipy.io.matlab.miobase.MatReadError, ValueError) as e:
        msg = ("Failed to parse FBT file: %s. Error: %s\n"
               "Verify file format, MATLAB .mat file, MEQ compatibility") % (
            file_path, str(e))
        logger.error(msg)
        raise geometry_errors.GeometryFileFormatError(msg) from e

    except (OSError, IOError) as e:
        if isinstance(file_path, (str, Path)):
            msg = "Failed to read FBT file: %s. Error: %s" % (
                file_path, str(e))
        else:
            msg = "Failed to read FBT file handle: %s" % str(e)
        logger.error(msg)
        raise geometry_errors.GeometryFileError(msg) from e


def _load_eqdsk_data(file_path: Union[str, Path]) -> dict[str, np.ndarray]:
    """Load data from an EQDSK file."""
    try:
        validated_path = file_validation.validate_file_access(
            file_path, "eqdsk", validate_format=True
        )
        logger.info("Loading EQDSK geometry file: %s", file_path)

        eqdsk_data = eqdsk.EQDSKInterface.from_file(
            str(validated_path), no_cocos=True)

        logger.warning(
            "WARNING: Using EQDSK geometry. Only tested against CHEASE-generated EQDSK."
        )

        result = eqdsk_data.__dict__
        file_validation.validate_geometry_data(result, "eqdsk", str(file_path))
        logger.info(
            "Successfully loaded EQDSK file with %d fields", len(result))
        return result

    except Exception as e:
        if isinstance(e, geometry_errors.GeometryFileError):
            raise
        msg = ("Failed to parse EQDSK file: %s. Error: %s" %
               (file_path, str(e)))
        logger.error(msg)
        raise geometry_errors.GeometryFileFormatError(msg) from e


def get_geometry_dir(geometry_dir: str | None = None) -> str:
    """Return default geometry directory if none provided."""
    if geometry_dir is None:
        geometry_dir = os.path.join(torax.__path__[0], "data/third_party/geo")
    return geometry_dir


def load_geo_data(
    geometry_dir: str | None,
    geometry_file: str,
    geometry_source: GeometrySource
) -> dict[str, np.ndarray]:
    """Load data from a geometry file into a dictionary."""
    try:
        geometry_dir = get_geometry_dir(geometry_dir)
        filepath = os.path.join(geometry_dir, geometry_file)

        match geometry_source:
            case GeometrySource.CHEASE:
                return _load_CHEASE_data(filepath)
            case GeometrySource.FBT:
                return _load_fbt_data(filepath)
            case GeometrySource.EQDSK:
                return _load_eqdsk_data(filepath)
            case _:
                msg = "Unknown geometry source: %s" % geometry_source
                logger.error(msg)
                raise ValueError(msg)

    except geometry_errors.GeometryFileError:
        raise
    except Exception as e:
        msg = ("Unexpected error loading geometry file: %s. Type: %s. Msg: %s" %
               (filepath, type(e).__name__, str(e)))
        logger.error(msg)
        raise geometry_errors.GeometryFileError(msg) from e
