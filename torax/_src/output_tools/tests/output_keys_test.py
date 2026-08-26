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

"""Unit tests for OutputKey and GridType."""

import copy
import os
import pickle
from absl.testing import absltest
from absl.testing import parameterized
from torax._src.output_tools import output
from torax._src.output_tools import output_keys
from torax._src.test_utils import paths


class OutputKeysTest(parameterized.TestCase):

  def test_output_key_attributes(self):
    key = output_keys.OutputKey(
        "test_var",
        units=output_keys.Units.KEV,
        grid_type=output_keys.GridType.CELL_PLUS_BOUNDARIES,
    )
    self.assertEqual(key, "test_var")
    self.assertEqual(key.units, output_keys.Units.KEV)
    self.assertEqual(key.grid_type, output_keys.GridType.CELL_PLUS_BOUNDARIES)

  def test_output_key_not_applicable_grid_type(self):
    key = output_keys.OutputKey(
        "dataset_name",
        units=output_keys.Units.NOT_APPLICABLE,
        grid_type=output_keys.GridType.NOT_APPLICABLE,
    )
    self.assertEqual(key.grid_type, output_keys.GridType.NOT_APPLICABLE)

  def test_output_key_equality_and_hashing(self):
    key = output_keys.OutputKey(
        "test_var",
        units=output_keys.Units.KEV,
        grid_type=output_keys.GridType.CELL_PLUS_BOUNDARIES,
    )
    self.assertEqual(key, "test_var")
    self.assertEqual(hash(key), hash("test_var"))

  @parameterized.named_parameters(
      ("copy", copy.copy),
      ("deepcopy", copy.deepcopy),
      ("pickle", lambda obj: pickle.loads(pickle.dumps(obj))),  # pylint: disable=g-unsafe-pickle-load
  )
  def test_output_key_copy_and_serialization(self, transform_fn):
    key = output_keys.OutputKey(
        "test_var",
        units=output_keys.Units.KEV,
        grid_type=output_keys.GridType.CELL_PLUS_BOUNDARIES,
    )
    reconstructed_key = transform_fn(key)
    self.assertEqual(reconstructed_key, "test_var")
    self.assertIsInstance(reconstructed_key, output_keys.OutputKey)
    assert isinstance(reconstructed_key, output_keys.OutputKey)
    self.assertEqual(reconstructed_key.units, output_keys.Units.KEV)
    self.assertEqual(
        reconstructed_key.grid_type,
        output_keys.GridType.CELL_PLUS_BOUNDARIES,
    )
    self.assertEqual(hash(reconstructed_key), hash(key))

  def test_predefined_keys_have_grid_types(self):
    self.assertEqual(
        output_keys.PROFILES.grid_type,
        output_keys.GridType.NOT_APPLICABLE,
    )
    self.assertEqual(
        output_keys.T_E.grid_type,
        output_keys.GridType.CELL_PLUS_BOUNDARIES,
    )
    self.assertEqual(
        output_keys.PSI.grid_type, output_keys.GridType.CELL_PLUS_BOUNDARIES
    )
    self.assertEqual(output_keys.IP.grid_type, output_keys.GridType.SCALAR)
    self.assertEqual(
        output_keys.TOROIDAL_ANGULAR_VELOCITY.grid_type,
        output_keys.GridType.CELL_PLUS_BOUNDARIES,
    )
    self.assertEqual(
        output_keys.J_PARALLEL_OHMIC.grid_type, output_keys.GridType.CELL
    )
    self.assertEqual(output_keys.DELTA.grid_type, output_keys.GridType.FACE)
    self.assertEqual(
        output_keys.Q_PARALLEL.grid_type, output_keys.GridType.SCALAR
    )

  def test_validate_grid_types_against_nc_benchmark(self):
    """Validates grid_type annotations in output_keys.py against NetCDF benchmark file."""
    data_dir = paths.test_data_dir()
    nc_file = os.path.join(data_dir, "test_iterhybrid_predictor_corrector.nc")
    self.assertTrue(
        os.path.exists(nc_file),
        msg=f"NetCDF benchmark file not found: {nc_file}",
    )

    all_keys = {
        str(getattr(output_keys, attr)): getattr(output_keys, attr)
        for attr in dir(output_keys)
        if isinstance(getattr(output_keys, attr), output_keys.OutputKey)
    }

    mismatches = []
    dt = output.load_state_file(nc_file)
    for group_node in dt.subtree:
      for var_name, data_array in group_node.data_vars.items():
        last_dim = data_array.dims[-1] if data_array.dims else None
        match last_dim:
          case output_keys.RHO_CELL_NORM:
            actual_grid = output_keys.GridType.CELL
          case output_keys.RHO_FACE_NORM:
            actual_grid = output_keys.GridType.FACE
          case output_keys.RHO_NORM:
            actual_grid = output_keys.GridType.CELL_PLUS_BOUNDARIES
          case _:
            if group_node.path.startswith("/numerics"):
              actual_grid = output_keys.GridType.NOT_APPLICABLE
            else:
              actual_grid = output_keys.GridType.SCALAR

        if var_name in all_keys:
          expected_grid = all_keys[var_name].grid_type
          if expected_grid != actual_grid:
            mismatches.append(
                f"Var '{var_name}' (path '{group_node.path}'): annotated as"
                f" '{expected_grid}', but in NetCDF has dims"
                f" {data_array.dims} -> '{actual_grid}'"
            )

    self.assertEmpty(
        mismatches,
        msg=(
            f"Found {len(mismatches)} grid_type mismatches against NetCDF"
            " benchmark:\n"
            + "\n".join(mismatches)
        ),
    )


if __name__ == "__main__":
  absltest.main()
