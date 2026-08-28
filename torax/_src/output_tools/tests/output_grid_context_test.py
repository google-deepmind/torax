# Copyright 2026 DeepMind Technologies Limited
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

"""Unit tests for OutputGridContext and OutputKey grid metadata."""

import enum
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys


class OutputGridContextTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.n_times = 5
    self.n_face = 10
    self.n_cell = 9
    self.n_cell_plus_boundaries = 11

    self.times = np.linspace(0.0, 1.0, self.n_times)
    self.rho_face_norm = np.linspace(0.0, 1.0, self.n_face)
    self.rho_cell_norm = np.linspace(0.05, 0.95, self.n_cell)
    self.rho_cell_plus_boundaries_norm = np.linspace(
        0.0, 1.0, self.n_cell_plus_boundaries
    )

    self.context = output_grid_context.OutputGridContext(
        times=self.times,
        rho_face_norm=self.rho_face_norm,
        rho_cell_norm=self.rho_cell_norm,
        rho_cell_plus_boundaries_norm=self.rho_cell_plus_boundaries_norm,
    )

  def test_pack_scalar(self):
    key = output_keys.OutputKey(
        "test_scalar",
        units=output_keys.Units.KEV,
        grid_type=output_keys.GridType.SCALAR,
    )
    data = np.ones((self.n_times,))
    packed = self.context.pack(key, data)
    dims, val, attrs = packed
    self.assertEqual(dims, (output_keys.TIME,))
    self.assertEqual(attrs, {"units": output_keys.Units.KEV})
    np.testing.assert_array_equal(val, data)

  def test_pack_enum_scalar(self):
    class Status(enum.Enum):
      OK = 0
      ERROR = 1

    key = output_keys.OutputKey(
        "sim_status",
        units=output_keys.Units.NOT_APPLICABLE,
        grid_type=output_keys.GridType.SCALAR,
    )
    # Pack array of enums or single enum broadcasted
    data = np.array([Status.OK.value] * self.n_times)
    packed = self.context.pack(key, data)
    dims, val, attrs = packed
    self.assertEqual(dims, (output_keys.TIME,))
    self.assertEqual(attrs, {})
    np.testing.assert_array_equal(val, np.zeros(self.n_times))

  def test_pack_face(self):
    key = output_keys.OutputKey(
        "test_face",
        units=output_keys.Units.SQUARE_METER_PER_SECOND,
        grid_type=output_keys.GridType.FACE,
    )
    data = np.ones((self.n_times, self.n_face))
    packed = self.context.pack(key, data)
    dims, val, attrs = packed
    self.assertEqual(dims, (output_keys.TIME, output_keys.RHO_FACE_NORM))
    self.assertEqual(
        attrs, {"units": output_keys.Units.SQUARE_METER_PER_SECOND}
    )
    np.testing.assert_array_equal(val, data)

  def test_pack_cell(self):
    key = output_keys.OutputKey(
        "test_cell",
        units=output_keys.Units.KEV,
        grid_type=output_keys.GridType.CELL,
    )
    data = np.ones((self.n_times, self.n_cell))
    packed = self.context.pack(key, data)
    dims, val, attrs = packed
    self.assertEqual(dims, (output_keys.TIME, output_keys.RHO_CELL_NORM))
    self.assertEqual(attrs, {"units": output_keys.Units.KEV})
    np.testing.assert_array_equal(val, data)

  def test_pack_cell_plus_boundaries(self):
    key = output_keys.OutputKey(
        "test_cpb",
        units=output_keys.Units.KEV,
        grid_type=output_keys.GridType.CELL_PLUS_BOUNDARIES,
    )
    data = np.ones((self.n_times, self.n_cell_plus_boundaries))
    packed = self.context.pack(key, data)
    dims, val, attrs = packed
    self.assertEqual(dims, (output_keys.TIME, output_keys.RHO_NORM))
    self.assertEqual(attrs, {"units": output_keys.Units.KEV})
    np.testing.assert_array_equal(val, data)

  def test_pack_shape_mismatch_raises(self):
    key = output_keys.OutputKey(
        "test_mismatch",
        units=output_keys.Units.KEV,
        grid_type=output_keys.GridType.SCALAR,
    )
    data = np.ones((self.n_times, 5))
    with self.assertRaises(ValueError):
      self.context.pack(key, data)

  def test_pack_missing_grid_type_raises(self):
    key = output_keys.OutputKey(
        "test_no_grid",
        units=output_keys.Units.KEV,
        grid_type=output_keys.GridType.NOT_APPLICABLE,
    )
    data = np.ones((self.n_times,))
    with self.assertRaises(ValueError):
      self.context.pack(key, data)

  def test_build_dataset(self):
    key_scalar = output_keys.OutputKey(
        "test_scalar",
        units=output_keys.Units.KEV,
        grid_type=output_keys.GridType.SCALAR,
    )
    key_face = output_keys.OutputKey(
        "test_face",
        units=output_keys.Units.SQUARE_METER_PER_SECOND,
        grid_type=output_keys.GridType.FACE,
    )
    var_dict = {
        str(key_scalar): self.context.pack(
            key_scalar, np.ones((self.n_times,))
        ),
        str(key_face): self.context.pack(
            key_face, np.ones((self.n_times, self.n_face))
        ),
    }
    ds = self.context.build_dataset(var_dict)
    self.assertIn("test_scalar", ds)
    self.assertIn("test_face", ds)
    self.assertEqual(ds["test_scalar"].dims, (output_keys.TIME,))
    self.assertEqual(
        ds["test_face"].dims, (output_keys.TIME, output_keys.RHO_FACE_NORM)
    )
    np.testing.assert_array_equal(
        ds.coords[output_keys.TIME].values, self.times
    )
    np.testing.assert_array_equal(
        ds.coords[output_keys.RHO_FACE_NORM].values, self.rho_face_norm
    )

  def test_build_dataset_with_custom_coords(self):
    key_scalar = output_keys.OutputKey(
        "test_scalar",
        units=output_keys.Units.KEV,
        grid_type=output_keys.GridType.SCALAR,
    )
    var_dict = {
        str(key_scalar): self.context.pack(
            key_scalar, np.ones((self.n_times,))
        ),
    }
    ds = self.context.build_dataset(
        var_dict, coords={output_keys.TIME: self.times}
    )
    self.assertIn("test_scalar", ds)
    self.assertEqual(list(ds.coords.keys()), [output_keys.TIME])
    np.testing.assert_array_equal(
        ds.coords[output_keys.TIME].values, self.times
    )

  def test_build_dataset_with_extra_coords(self):
    key = output_keys.OutputKey(
        "test_impurity",
        units=output_keys.Units.INVERSE_CUBIC_METER,
        grid_type=output_keys.GridType.SCALAR,
    )
    var_dict = {
        str(key): (
            ("impurity", output_keys.TIME),
            np.ones((2, self.n_times)),
            {"units": "m^-3"},
        ),
    }
    coords = {output_keys.TIME: self.times, "impurity": ["Ar", "Ne"]}
    ds = self.context.build_dataset(var_dict, coords=coords)
    self.assertIn("test_impurity", ds)
    self.assertEqual(ds["test_impurity"].dims, ("impurity", output_keys.TIME))
    self.assertEqual(list(ds.coords["impurity"].values), ["Ar", "Ne"])

  def test_extend_cell_grid_to_boundaries(self):
    cell_var = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
    face_var = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    extended = output_grid_context.extend_cell_grid_to_boundaries(
        cell_var, face_var
    )
    expected = np.array(
        [[1.0, 10.0, 20.0, 30.0, 4.0], [1.0, 10.0, 20.0, 30.0, 4.0]]
    )
    np.testing.assert_array_equal(extended, expected)


if __name__ == "__main__":
  absltest.main()
