# Copyright 2025 DeepMind Technologies Limited
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

"""Tests for the geometry_loader module."""

import pathlib
from unittest import mock
from absl.testing import absltest
import numpy as np
from torax._src.geometry import geometry_loader

# pylint: disable=invalid-name


class GeometryLoaderTest(absltest.TestCase):

  @mock.patch("scipy.io.loadmat")
  def test_load_data_flat(self, mock_loadmat):
    """Tests loading MEQ data that is already flat in the mat file."""
    mock_data = {
        "rBt": np.array([5.0]),
        "aminor": np.linspace(0.1, 1.0, 10),
    }
    mock_loadmat.return_value = mock_data

    result = geometry_loader._load_meq_file("dummy_path.mat")
    self.assertEqual(result, mock_data)

  @mock.patch("scipy.io.loadmat")
  def test_load_meq_file_nested_LY(self, mock_loadmat):
    """Tests loading MEQ data nested inside an 'LY' structured array."""
    # Build a structured array to mimic scipy.io.loadmat's behavior for
    # meqlpack outputs.
    aminor_data = np.linspace(0.1, 1.0, 10)
    dtype = [("rBt", "O"), ("aminor", "O")]
    # Structured array with 1 element, fields contain arrays.
    LY_array = np.array([(np.array([5.0]), aminor_data)], dtype=dtype)

    mock_loadmat.return_value = {"LY": LY_array}

    result = geometry_loader._load_meq_file("dummy_path.mat")

    self.assertIn("rBt", result)
    self.assertIn("aminor", result)
    np.testing.assert_array_equal(result["rBt"], np.array([5.0]))
    np.testing.assert_array_equal(result["aminor"], aminor_data)

  @mock.patch("scipy.io.loadmat")
  def test_load_meq_file_nested_L(self, mock_loadmat):
    """Tests loading MEQ data nested inside an 'L' structured array."""
    pQ_data = np.linspace(0.0, 1.0, 20)
    dtype = [("pQ", "O")]
    L_array = np.array([(pQ_data,)], dtype=dtype)

    mock_loadmat.return_value = {"L": L_array}

    result = geometry_loader._load_meq_file("dummy_path.mat")

    self.assertIn("pQ", result)
    np.testing.assert_array_equal(result["pQ"], pQ_data)

  @mock.patch("scipy.io.loadmat")
  def test_load_meq_file_invalid_item_type(self, mock_loadmat):
    """Tests that a ValueError is raised if a nested item is not an array."""
    dtype = [("bad_field", "O")]
    # Create a structured array where the item is NOT a numpy array (e.g., int)
    LY_array = np.array([(123,)], dtype=dtype)

    mock_loadmat.return_value = {"LY": LY_array}

    with self.assertRaisesRegex(
        ValueError, "MEQ data field 'bad_field' is not a numpy array"
    ):
      geometry_loader._load_meq_file("dummy_path.mat")

  def test_get_geometry_dir_with_pathlib_path(self):
    """Tests get_geometry_dir when given a pathlib.Path."""
    path = pathlib.Path("/tmp/test_geo_dir")
    self.assertEqual(geometry_loader.get_geometry_dir(path), path)

  @mock.patch.object(geometry_loader, "_load_CHEASE_file")
  def test_load_geo_data_with_pathlib_path(self, mock_load_chease):
    """Tests load_geo_data accepts pathlib.Path for dir and file."""
    mock_load_chease.return_value = {"dummy": np.array([1.0])}
    geo_dir = pathlib.Path("/tmp/test_dir")
    geo_file = pathlib.Path("test_chease.mat")
    data = geometry_loader.load_geo_data(
        geometry_dir=geo_dir,
        geometry_file=geo_file,
        geometry_source=geometry_loader.GeometrySource.CHEASE,
    )
    self.assertEqual(data, {"dummy": np.array([1.0])})
    mock_load_chease.assert_called_once_with(geo_dir / geo_file)


if __name__ == "__main__":
  absltest.main()
