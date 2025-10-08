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

import json
import os
import pathlib
import shutil
from unittest import mock
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
from torax._src import path_utils
from torax._src.test_utils import torax_refs
from torax.tests.scripts import regenerate_torax_refs

_LEN = 25  # Number of elements in the reference data arrays.


class RegenerateRefsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = pathlib.Path(self.create_tempdir())

    # Get the original torax_path before mocking
    original_torax_path = path_utils.torax_path()
    original_refs_path = (
        original_torax_path / '_src' / 'test_utils' / torax_refs.JSON_FILENAME
    )

    self.mock_torax_path_patch = mock.patch.object(
        path_utils, 'torax_path', return_value=self.temp_dir
    )
    self.mock_torax_path = self.mock_torax_path_patch.start()
    self.addCleanup(self.mock_torax_path_patch.stop)

    self.output_path = (
        self.temp_dir / '_src' / 'test_utils' / torax_refs.JSON_FILENAME
    )

    # Copy the original references.json to the temp directory. This is needed
    # to ensure that the file exists for all test cases.
    self.output_path.parent.mkdir(parents=True, exist_ok=True)
    if original_refs_path.exists():
      shutil.copyfile(original_refs_path, self.output_path)

  @parameterized.named_parameters(
      (case_name, case_name) for case_name in torax_refs.REFERENCES_REGISTRY
  )
  def test_smoke_test_all_cases(self, case_name):
    """Smoke test to ensure the script runs for all registered cases."""
    with flagsaver.flagsaver(case=[case_name]):
      # Should run without raising any exceptions.
      regenerate_torax_refs.main([])

  @flagsaver.flagsaver(write_to_file=True, case=['circular_references'])
  def test_write_to_file_creates_json(self):
    """Tests that the --write_to_file flag creates a valid JSON file."""
    # Ensure the parent directory exists
    self.output_path.parent.mkdir(parents=True, exist_ok=True)

    regenerate_torax_refs.main([])

    self.assertTrue(os.path.exists(self.output_path))
    with open(self.output_path, 'r') as f:
      data = json.load(f)

    self.assertIn('circular_references', data)
    case_data = data['circular_references']
    self.assertIsInstance(case_data, dict)
    expected_keys = {'psi', 'psi_face_grad', 'psidot', 'j_total', 'q', 's'}
    self.assertContainsSubset(expected_keys, case_data.keys())
    self.assertIsInstance(case_data['psi'], list)  # Check for NumPy conversion

  @flagsaver.flagsaver(case=['invalid_case_name'])
  def test_invalid_case_raises_error(self):
    """Tests that providing an unknown case name raises a ValueError."""
    with self.assertRaisesRegex(
        ValueError, "Case 'invalid_case_name' not found"
    ):
      regenerate_torax_refs.main([])

  @flagsaver.flagsaver(case=['circular_references'], print_summary=True)
  def test_summary_flag_prints_output(self):
    """Tests that the --print_summary flag prints detailed output."""
    with self.assertLogs(level='INFO') as cm:
      regenerate_torax_refs.main([])
      log_output = '\n'.join(cm.output)
      self.assertIn(
          'Regenerating references for: circular_references...', log_output
      )
      self.assertIn('Full data for circular_references', log_output)
      self.assertIn('circular_references.psi', log_output)
      self.assertIn('circular_references.psi_face_grad', log_output)
      self.assertIn('circular_references.psidot', log_output)
      self.assertIn('circular_references.j_total', log_output)
      self.assertIn('circular_references.q', log_output)
      self.assertIn('circular_references.s', log_output)

  @flagsaver.flagsaver(case=['circular_references'])
  def test_summary_is_not_printed_by_default(self):
    """Tests that detailed output is not printed by default."""
    with self.assertLogs(level='INFO') as cm:
      regenerate_torax_refs.main([])
      log_output = '\n'.join(cm.output)
      self.assertIn(
          'Regenerating references for: circular_references...', log_output
      )
      self.assertNotIn('Full data for circular_references', log_output)

  @flagsaver.flagsaver(case=['circular_references'], write_to_file=True)
  def test_write_to_file_preserves_other_cases(self):
    """Tests that regenerating one case does not delete others in the file."""
    # Create a dummy existing references file
    self.output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_circular_data = {
        'psi': np.ones(_LEN).tolist(),
        'psi_face_grad': np.ones(_LEN + 1).tolist(),
        'psidot': np.ones(_LEN).tolist(),
        'j_total': np.ones(_LEN).tolist(),
        'q': np.ones(_LEN + 1).tolist(),
        's': np.ones(_LEN + 1).tolist(),
    }

    existing_data = {
        'dummy_case': {'psi': [1, 2, 3]},
        'circular_references': (
            dummy_circular_data
        ),  # Old data to be overwritten
    }
    with open(self.output_path, 'w') as f:
      json.dump(existing_data, f)

    regenerate_torax_refs.main([])

    with open(self.output_path, 'r') as f:
      final_data = json.load(f)

    # Check that the dummy case is still there
    self.assertIn('dummy_case', final_data)
    self.assertEqual(final_data['dummy_case']['psi'], [1, 2, 3])

    # Check that the circular_references case has been updated
    self.assertIn('circular_references', final_data)
    for key, old_value in existing_data['circular_references'].items():
      self.assertNotEqual(final_data['circular_references'][key], old_value)


if __name__ == '__main__':
  absltest.main()
