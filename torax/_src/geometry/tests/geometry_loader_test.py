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

"""Tests for geometry file loader error handling."""

import os
import tempfile
import unittest.mock as mock

from absl.testing import absltest
from absl.testing import parameterized
from torax._src.geometry import geometry_errors
from torax._src.geometry import geometry_loader


class GeometryLoaderErrorTest(parameterized.TestCase):
  """Tests for geometry loader error handling."""

  def setUp(self):
    super().setUp()
    # Create a temporary directory for test files
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    # Clean up temporary directory
    import shutil
    shutil.rmtree(self.test_dir, ignore_errors=True)

  def test_file_not_found_error(self):
    """Test that a file not found error is raised with helpful message."""
    with self.assertRaises(geometry_errors.GeometryFileNotFoundError) as cm:
      geometry_loader.load_geo_data(
          geometry_dir=self.test_dir,
          geometry_file="nonexistent_file.txt",
          geometry_source=geometry_loader.GeometrySource.CHEASE,
      )

    self.assertIn("Geometry file not found", str(cm.exception))
    self.assertIn("nonexistent_file.txt", str(cm.exception))
    self.assertIn("Action", str(cm.exception))

  def test_file_not_found_with_similar_files(self):
    """Test that similar files are suggested when file is not found."""
    # Create some similar files
    similar_files = [
        "geometry_file.txt",
        "geometry_data.txt",
        "geom_file.txt",
    ]
    for filename in similar_files:
      filepath = os.path.join(self.test_dir, filename)
      with open(filepath, "w") as f:
        f.write("test data")

    with self.assertRaises(geometry_errors.GeometryFileNotFoundError) as cm:
      geometry_loader.load_geo_data(
          geometry_dir=self.test_dir,
          geometry_file="geometry_fle.txt",  # Typo
          geometry_source=geometry_loader.GeometrySource.CHEASE,
      )

    exception_message = str(cm.exception)
    self.assertIn("Did you mean one of these?", exception_message)
    # At least one similar file should be suggested
    self.assertTrue(
        any(f in exception_message for f in similar_files),
        f"None of {similar_files} found in: {exception_message}",
    )

  def test_empty_file_error(self):
    """Test that an empty file error is raised."""
    # Create an empty file
    empty_file = os.path.join(self.test_dir, "empty.txt")
    with open(empty_file, "w") as f:
      pass  # Create empty file

    with self.assertRaises(geometry_errors.GeometryFileEmptyError) as cm:
      geometry_loader.load_geo_data(
          geometry_dir=self.test_dir,
          geometry_file="empty.txt",
          geometry_source=geometry_loader.GeometrySource.CHEASE,
      )

    self.assertIn("Geometry file is empty", str(cm.exception))
    self.assertIn("empty.txt", str(cm.exception))

  @mock.patch("os.access")
  def test_permission_error(self, mock_access):
    """Test that a permission error is raised."""
    # Create a file
    test_file = os.path.join(self.test_dir, "test.txt")
    with open(test_file, "w") as f:
      f.write("test data")

    # Mock os.access to return False (no read permission)
    mock_access.return_value = False

    with self.assertRaises(geometry_errors.GeometryFilePermissionError) as cm:
      geometry_loader.load_geo_data(
          geometry_dir=self.test_dir,
          geometry_file="test.txt",
          geometry_source=geometry_loader.GeometrySource.CHEASE,
      )

    self.assertIn("Permission denied", str(cm.exception))
    self.assertIn("test.txt", str(cm.exception))

  def test_chease_format_error_no_header(self):
    """Test CHEASE format error when header is missing."""
    # Create a file with no header
    test_file = os.path.join(self.test_dir, "bad_chease.txt")
    with open(test_file, "w") as f:
      f.write("\n")  # Empty header

    with self.assertRaises(geometry_errors.GeometryFileFormatError) as cm:
      geometry_loader.load_geo_data(
          geometry_dir=self.test_dir,
          geometry_file="bad_chease.txt",
          geometry_source=geometry_loader.GeometrySource.CHEASE,
      )

    self.assertIn("Invalid geometry file format", str(cm.exception))
    self.assertIn("CHEASE", str(cm.exception))
    self.assertIn("header", str(cm.exception).lower())

  def test_chease_format_error_invalid_data(self):
    """Test CHEASE format error when data cannot be converted to float."""
    # Create a file with invalid data
    test_file = os.path.join(self.test_dir, "bad_chease_data.txt")
    with open(test_file, "w") as f:
      f.write("% var1 var2\n")
      f.write("1.0 not_a_number\n")

    with self.assertRaises(geometry_errors.GeometryFileFormatError) as cm:
      geometry_loader.load_geo_data(
          geometry_dir=self.test_dir,
          geometry_file="bad_chease_data.txt",
          geometry_source=geometry_loader.GeometrySource.CHEASE,
      )

    self.assertIn("Invalid geometry file format", str(cm.exception))
    self.assertIn("CHEASE", str(cm.exception))
    self.assertIn("not_a_number", str(cm.exception))

  def test_chease_format_error_mismatched_columns(self):
    """Test CHEASE format error when data has wrong number of columns."""
    # Create a file with mismatched columns
    test_file = os.path.join(self.test_dir, "bad_chease_columns.txt")
    with open(test_file, "w") as f:
      f.write("% var1 var2 var3\n")
      f.write("1.0 2.0\n")  # Only 2 values, expected 3

    with self.assertRaises(geometry_errors.GeometryFileFormatError) as cm:
      geometry_loader.load_geo_data(
          geometry_dir=self.test_dir,
          geometry_file="bad_chease_columns.txt",
          geometry_source=geometry_loader.GeometrySource.CHEASE,
      )

    self.assertIn("Invalid geometry file format", str(cm.exception))
    self.assertIn("Expected 3 values", str(cm.exception))

  def test_eqdsk_missing_cocos_error(self):
    """Test that EQDSK without COCOS raises ValueError."""
    # Create a dummy file
    test_file = os.path.join(self.test_dir, "test.eqdsk")
    with open(test_file, "w") as f:
      f.write("dummy data")

    with self.assertRaises(ValueError) as cm:
      geometry_loader.load_geo_data(
          geometry_dir=self.test_dir,
          geometry_file="test.eqdsk",
          geometry_source=geometry_loader.GeometrySource.EQDSK,
          cocos=None,  # Missing COCOS
      )

    self.assertIn("COCOS", str(cm.exception))
    self.assertIn("must be provided", str(cm.exception))

  def test_eqdsk_invalid_cocos_error(self):
    """Test that EQDSK with invalid COCOS raises error."""
    # Create a dummy file
    test_file = os.path.join(self.test_dir, "test.eqdsk")
    with open(test_file, "w") as f:
      f.write("dummy data")

    with self.assertRaises(geometry_errors.GeometryFileFormatError) as cm:
      geometry_loader.load_geo_data(
          geometry_dir=self.test_dir,
          geometry_file="test.eqdsk",
          geometry_source=geometry_loader.GeometrySource.EQDSK,
          cocos=99,  # Invalid COCOS value
      )

    self.assertIn("Invalid COCOS value", str(cm.exception))
    self.assertIn("99", str(cm.exception))

  def test_find_similar_files(self):
    """Test the similar files finder function."""
    # Create some test files
    test_files = [
        "geometry_001.txt",
        "geometry_002.txt",
        "geometry_test.txt",
        "other_file.txt",
    ]
    for filename in test_files:
      filepath = os.path.join(self.test_dir, filename)
      with open(filepath, "w") as f:
        f.write("test")

    # Find similar files to a typo
    similar = geometry_loader._find_similar_files(
        "geometry_003.txt", self.test_dir
    )

    # Should find geometry files
    self.assertTrue(len(similar) > 0)
    self.assertTrue(any("geometry" in f for f in similar))

  def test_validate_file_access_success(self):
    """Test that file validation succeeds for valid file."""
    # Create a valid file
    test_file = os.path.join(self.test_dir, "valid.txt")
    with open(test_file, "w") as f:
      f.write("valid data")

    # Should not raise any exception
    try:
      geometry_loader._validate_file_access(test_file, self.test_dir)
    except Exception as e:
      self.fail(f"Validation raised unexpected exception: {e}")

  def test_exception_hierarchy(self):
    """Test that custom exceptions have correct hierarchy."""
    # All custom exceptions should inherit from GeometryFileError
    self.assertTrue(
        issubclass(
            geometry_errors.GeometryFileNotFoundError,
            geometry_errors.GeometryFileError,
        )
    )
    self.assertTrue(
        issubclass(
            geometry_errors.GeometryFileFormatError,
            geometry_errors.GeometryFileError,
        )
    )
    self.assertTrue(
        issubclass(
            geometry_errors.GeometryFilePermissionError,
            geometry_errors.GeometryFileError,
        )
    )
    self.assertTrue(
        issubclass(
            geometry_errors.GeometryFileEmptyError,
            geometry_errors.GeometryFileError,
        )
    )

  def test_exception_attributes(self):
    """Test that exceptions store correct attributes."""
    # Test GeometryFileNotFoundError attributes
    exc = geometry_errors.GeometryFileNotFoundError(
        filepath="/path/to/file.txt",
        similar_files=["file1.txt", "file2.txt"],
        geometry_dir="/path/to",
    )
    self.assertEqual(exc.filepath, "/path/to/file.txt")
    self.assertEqual(exc.similar_files, ["file1.txt", "file2.txt"])
    self.assertEqual(exc.geometry_dir, "/path/to")

    # Test GeometryFileFormatError attributes
    exc = geometry_errors.GeometryFileFormatError(
        filepath="/path/to/file.txt",
        expected_format="CHEASE",
        details="Invalid header",
    )
    self.assertEqual(exc.filepath, "/path/to/file.txt")
    self.assertEqual(exc.expected_format, "CHEASE")
    self.assertEqual(exc.details, "Invalid header")


class CHEASEDataLoadingTest(absltest.TestCase):
  """Tests for CHEASE data loading with valid files."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    import shutil
    shutil.rmtree(self.test_dir, ignore_errors=True)

  def test_load_valid_chease_file(self):
    """Test loading a valid CHEASE file."""
    # Create a valid CHEASE file
    test_file = os.path.join(self.test_dir, "valid_chease.txt")
    with open(test_file, "w") as f:
      f.write("% rho psi q\n")
      f.write("0.0 0.0 1.0\n")
      f.write("0.5 0.25 1.5\n")
      f.write("1.0 1.0 2.0\n")

    data = geometry_loader.load_geo_data(
        geometry_dir=self.test_dir,
        geometry_file="valid_chease.txt",
        geometry_source=geometry_loader.GeometrySource.CHEASE,
    )

    # Check that data is loaded correctly
    self.assertIn("rho", data)
    self.assertIn("psi", data)
    self.assertIn("q", data)
    self.assertEqual(len(data["rho"]), 3)
    self.assertAlmostEqual(data["rho"][1], 0.5)

  def test_load_chease_with_empty_lines(self):
    """Test loading CHEASE file with empty lines (should be skipped)."""
    # Create a CHEASE file with empty lines
    test_file = os.path.join(self.test_dir, "chease_empty_lines.txt")
    with open(test_file, "w") as f:
      f.write("% rho psi\n")
      f.write("0.0 0.0\n")
      f.write("\n")  # Empty line
      f.write("0.5 0.25\n")
      f.write("   \n")  # Whitespace line
      f.write("1.0 1.0\n")

    data = geometry_loader.load_geo_data(
        geometry_dir=self.test_dir,
        geometry_file="chease_empty_lines.txt",
        geometry_source=geometry_loader.GeometrySource.CHEASE,
    )

    # Should have 3 data points (empty lines skipped)
    self.assertEqual(len(data["rho"]), 3)


if __name__ == "__main__":
  absltest.main()
