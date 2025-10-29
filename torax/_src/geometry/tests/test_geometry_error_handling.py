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

"""Tests for geometry file error handling."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import numpy as np

from torax._src.geometry import geometry_errors
from torax._src.geometry import geometry_loader
from torax._src.geometry import file_validation


class TestFileValidation:
    """Test file validation functionality."""

    def test_file_not_found(self, tmp_path):
        """Test clear error when file doesn't exist."""
        nonexistent_file = tmp_path / "nonexistent.chease"

        with pytest.raises(geometry_errors.GeometryFileNotFoundError) as exc_info:
            file_validation.validate_file_access(nonexistent_file, 'chease')

        assert "not found" in str(exc_info.value).lower()
        assert "Please check" in str(exc_info.value)

    def test_file_not_found_with_similar_files(self, tmp_path):
        """Test that similar files are suggested when file not found."""
        # Create some similar files
        (tmp_path / "file1.chease").touch()
        (tmp_path / "file2.chease").touch()
        (tmp_path / "other.txt").touch()

        nonexistent_file = tmp_path / "file3.chease"

        with pytest.raises(geometry_errors.GeometryFileNotFoundError) as exc_info:
            file_validation.validate_file_access(nonexistent_file, 'chease')

        assert "Similar files found" in str(exc_info.value)
        assert "file1.chease" in str(exc_info.value)
        assert "file2.chease" in str(exc_info.value)
        # Should not suggest files with different extensions
        assert "other.txt" not in str(exc_info.value)

    def test_permission_error(self, tmp_path):
        """Test clear error for permission issues."""
        file_path = tmp_path / "readonly.chease"
        file_path.write_text("data")
        file_path.chmod(0o000)  # Remove all permissions

        try:
            with pytest.raises(geometry_errors.GeometryFilePermissionError) as exc_info:
                file_validation.validate_file_access(file_path, 'chease')

            assert "permission" in str(exc_info.value).lower()
            assert "chmod" in str(exc_info.value)
        finally:
            # Restore permissions for cleanup
            file_path.chmod(0o644)

    def test_empty_file(self, tmp_path):
        """Test clear error for empty files."""
        file_path = tmp_path / "empty.chease"
        file_path.touch()

        with pytest.raises(geometry_errors.GeometryFileFormatError) as exc_info:
            file_validation.validate_file_access(file_path, 'chease')

        assert "empty" in str(exc_info.value).lower()

    def test_file_extension_warning(self, tmp_path, caplog):
        """Test warning for unusual file extension."""
        file_path = tmp_path / "geometry.unusual"
        file_path.write_text("some data")

        # Should not raise exception but should log warning
        result = file_validation.validate_file_access(
            file_path, 'chease', validate_format=True
        )

        assert result == file_path
        assert "unusual" in caplog.text.lower()

    def test_valid_file(self, tmp_path):
        """Test successful validation of valid file."""
        file_path = tmp_path / "valid.chease"
        file_path.write_text("some data")

        result = file_validation.validate_file_access(file_path, 'chease')
        assert result == file_path


class TestCHEASEErrorHandling:
    """Test CHEASE-specific error handling."""

    def test_chease_missing_header(self, tmp_path):
        """Test error for CHEASE file with missing header."""
        file_path = tmp_path / "no_header.chease"
        file_path.write_text("")  # Empty file

        with pytest.raises(geometry_errors.GeometryFileFormatError) as exc_info:
            geometry_loader._load_CHEASE_data(file_path)

        assert "empty" in str(exc_info.value).lower()

    def test_chease_invalid_header(self, tmp_path):
        """Test error for CHEASE file with invalid header."""
        file_path = tmp_path / "invalid_header.chease"
        file_path.write_text("%\n1.0 2.0\n")  # Header with no variable names

        with pytest.raises(geometry_errors.GeometryFileFormatError) as exc_info:
            geometry_loader._load_CHEASE_data(file_path)

        assert "variable labels" in str(exc_info.value).lower()

    def test_chease_inconsistent_columns(self, tmp_path):
        """Test error for CHEASE file with inconsistent column counts."""
        file_path = tmp_path / "inconsistent.chease"
        file_path.write_text(
            "% var1 var2 var3\n"
            "1.0 2.0 3.0\n"
            "4.0 5.0\n"  # Missing third column
            "7.0 8.0 9.0\n"
        )

        with pytest.raises(geometry_errors.GeometryFileFormatError) as exc_info:
            geometry_loader._load_CHEASE_data(file_path)

        assert "line 3" in str(exc_info.value).lower()
        assert "2 values" in str(exc_info.value)
        assert "expected 3" in str(exc_info.value)

    def test_chease_invalid_float(self, tmp_path):
        """Test error for CHEASE file with non-numeric data."""
        file_path = tmp_path / "invalid_float.chease"
        file_path.write_text(
            "% var1 var2\n"
            "1.0 2.0\n"
            "invalid 5.0\n"
        )

        with pytest.raises(geometry_errors.GeometryFileFormatError) as exc_info:
            geometry_loader._load_CHEASE_data(file_path)

        assert "parse line 3" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_chease_binary_file(self, tmp_path):
        """Test error for binary file passed as CHEASE."""
        file_path = tmp_path / "binary.chease"
        file_path.write_bytes(b'\x00\x01\x02\x03' * 100)

        with pytest.raises(geometry_errors.GeometryFileFormatError) as exc_info:
            geometry_loader._load_CHEASE_data(file_path)

        # Binary files may be read but fail at header parsing, or fail with decode error
        error_msg = str(exc_info.value).lower()
        assert "variable labels" in error_msg or "decode" in error_msg or "binary" in error_msg

    def test_chease_valid_file(self, tmp_path):
        """Test successful loading of valid CHEASE file."""
        file_path = tmp_path / "valid.chease"
        content = (
            "% PSIchease=psi/2pi Ipprofile RHO_TOR=sqrt(Phi/pi/B0) R_INBOARD R_OUTBOARD T=RBphi\n"
            "0.0 0.0 0.0 3.0 3.5 5.2\n"
            "0.1 100.0 0.316 2.9 3.6 5.3\n"
            "0.2 200.0 0.447 2.8 3.7 5.4\n"
        )
        file_path.write_text(content)

        result = geometry_loader._load_CHEASE_data(file_path)

        assert "PSIchease=psi/2pi" in result
        assert "Ipprofile" in result
        assert len(result["PSIchease=psi/2pi"]) == 3
        assert np.allclose(result["R_INBOARD"], [3.0, 2.9, 2.8])


class TestFBTErrorHandling:
    """Test FBT-specific error handling."""

    @patch('scipy.io.loadmat')
    def test_fbt_mat_read_error(self, mock_loadmat, tmp_path):
        """Test error for corrupted MAT file."""
        file_path = tmp_path / "corrupted.fbt"
        # Write enough data to avoid empty file error
        file_path.write_bytes(b"not a mat file" * 100)

        # Use MatReadError which is caught by the implementation
        import scipy.io.matlab.miobase
        mock_loadmat.side_effect = scipy.io.matlab.miobase.MatReadError(
            "Failed to read MAT file")

        with pytest.raises(geometry_errors.GeometryFileFormatError) as exc_info:
            geometry_loader._load_fbt_data(file_path)

        assert "Failed to parse FBT" in str(exc_info.value)
        assert "MATLAB .mat file" in str(exc_info.value)

    @patch('scipy.io.loadmat')
    def test_fbt_missing_ly_fields(self, mock_loadmat, tmp_path):
        """Test error for FBT file missing required LY fields."""
        file_path = tmp_path / "incomplete.fbt"
        # Write enough data to avoid empty file error
        file_path.write_bytes(b"dummy mat file content" * 100)

        # Mock LY bundle with missing fields - needs to be subscriptable
        class MockArray:
            def __init__(self):
                self.dtype = type('MockDtype', (), {
                                  'names': ('incomplete_field',)})()

            def __getitem__(self, key):
                # Return a mock object that has an item() method
                class MockField:
                    def item(self):
                        return np.array([1.0])
                return MockField()

        mock_loadmat.return_value = {'LY': MockArray()}

        with pytest.raises(geometry_errors.GeometryDataValidationError) as exc_info:
            geometry_loader._load_fbt_data(file_path)

        assert "validation failed" in str(exc_info.value).lower()


class TestEQDSKErrorHandling:
    """Test EQDSK-specific error handling."""

    @patch('eqdsk.EQDSKInterface.from_file')
    def test_eqdsk_parse_error(self, mock_from_file, tmp_path):
        """Test error for invalid EQDSK format."""
        file_path = tmp_path / "invalid.eqdsk"
        file_path.write_text("not an eqdsk file")

        mock_from_file.side_effect = Exception("Invalid EQDSK format")

        with pytest.raises(geometry_errors.GeometryFileFormatError) as exc_info:
            geometry_loader._load_eqdsk_data(file_path)

        assert "Failed to parse EQDSK" in str(exc_info.value)
        assert "EQDSK specification" in str(exc_info.value)


class TestLoadGeoData:
    """Test the main load_geo_data function."""

    def test_unknown_geometry_source(self):
        """Test error for unknown geometry source."""
        # The load_geo_data function wraps ValueError in GeometryFileError
        with pytest.raises(geometry_errors.GeometryFileError) as exc_info:
            geometry_loader.load_geo_data(
                None, "test.dat", "UNKNOWN_SOURCE"
            )

        assert "Unknown geometry source" in str(exc_info.value)
        assert "Supported sources" in str(exc_info.value)

    @patch('torax._src.geometry.geometry_loader._load_CHEASE_data')
    def test_unexpected_error_handling(self, mock_load_chease, tmp_path):
        """Test handling of unexpected errors."""
        mock_load_chease.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(geometry_errors.GeometryFileError) as exc_info:
            geometry_loader.load_geo_data(
                str(tmp_path), "test.chease", geometry_loader.GeometrySource.CHEASE
            )

        assert "Unexpected error loading geometry file" in str(exc_info.value)
        assert "RuntimeError" in str(exc_info.value)
        assert "github.com/google-deepmind/torax/issues" in str(exc_info.value)


class TestGeometryDataValidation:
    """Test geometry data validation."""

    def test_chease_missing_required_fields(self):
        """Test validation failure for missing CHEASE fields."""
        incomplete_data = {
            'PSIchease=psi/2pi': np.array([0.0, 0.1]),
            # Missing other required fields
        }

        with pytest.raises(geometry_errors.GeometryDataValidationError) as exc_info:
            file_validation.validate_geometry_data(
                incomplete_data, 'chease', 'test.chease'
            )

        assert "validation failed" in str(exc_info.value).lower()
        assert "Missing required CHEASE field" in str(exc_info.value)

    def test_chease_invalid_radii(self):
        """Test validation failure for invalid radius data."""
        invalid_data = {
            'PSIchease=psi/2pi': np.array([0.0, 0.1]),
            'Ipprofile': np.array([0.0, 100.0]),
            'RHO_TOR=sqrt(Phi/pi/B0)': np.array([0.0, 0.1]),
            'R_INBOARD': np.array([3.0, 2.9]),
            'R_OUTBOARD': np.array([2.5, 2.8]),  # Outboard < Inboard (invalid)
            'T=RBphi': np.array([5.2, 5.3]),
        }

        with pytest.raises(geometry_errors.GeometryDataValidationError) as exc_info:
            file_validation.validate_geometry_data(
                invalid_data, 'chease', 'test.chease'
            )

        assert "Outboard radius must be greater than inboard radius" in str(
            exc_info.value)

    def test_eqdsk_missing_boundary_data(self):
        """Test validation failure for incomplete EQDSK boundary data."""
        incomplete_data = {
            'bcentre': 5.3,
            'xmag': 6.2,
            'zmag': 0.0,
            'psimag': 0.0,
            'psibdry': 1.0,
            'xbdry': np.array([6.0, 7.0]),
            'zbdry': np.array([1.0]),  # Mismatched length
            'fpol': np.array([31.0, 32.0]),
            'qpsi': np.array([1.0, 2.0]),
            'psi': np.array([[0.0, 0.1], [0.2, 0.3]]),
        }

        with pytest.raises(geometry_errors.GeometryDataValidationError) as exc_info:
            file_validation.validate_geometry_data(
                incomplete_data, 'eqdsk', 'test.eqdsk'
            )

        assert "same length" in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__])
