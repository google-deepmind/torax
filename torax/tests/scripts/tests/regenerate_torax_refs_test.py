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
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from torax._src.config import config_loader
from torax._src.test_utils import torax_refs
from torax.tests.scripts import regenerate_torax_refs


class RegenerateRefsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = pathlib.Path(self.create_tempdir())
    self.mock_torax_path_patch = mock.patch.object(
        config_loader, 'torax_path', return_value=self.temp_dir
    )
    self.mock_torax_path = self.mock_torax_path_patch.start()
    self.addCleanup(self.mock_torax_path_patch.stop)
    self.output_path = (
        self.temp_dir
        / '_src'
        / 'test_utils'
        / regenerate_torax_refs._JSON_FILENAME
    )

  @parameterized.named_parameters(
      (case_name, case_name) for case_name in torax_refs.REFERENCES_REGISTRY
  )
  @flagsaver.flagsaver(no_summary=True)
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

  @flagsaver.flagsaver(case=['circular_references'], no_summary=True)
  def test_no_summary_flag_suppresses_output(self):
    """Tests that the --no_summary flag prevents detailed output."""
    with self.assertLogs(level='INFO') as cm:
      regenerate_torax_refs.main([])
      log_output = '\n'.join(cm.output)
      self.assertIn(
          'Regenerating references for: circular_references...', log_output
      )
      self.assertNotIn('Full data for circular_references', log_output)

  @flagsaver.flagsaver(case=['circular_references'])
  def test_summary_is_printed_by_default(self):
    """Tests that detailed output is printed by default."""
    with self.assertLogs(level='INFO') as cm:
      regenerate_torax_refs.main([])
      log_output = '\n'.join(cm.output)
      self.assertIn(
          'Regenerating references for: circular_references...', log_output
      )
      self.assertIn('Full data for circular_references', log_output)
      self.assertIn('circular_references.psi', log_output)

  @flagsaver.flagsaver(write_to_file=True, case=['circular_references'])
  def test_write_to_file_preserves_other_cases(self):
    """Tests that regenerating one case does not delete others in the file."""
    # Create a dummy existing references file
    self.output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_data = {
        'dummy_case': {'psi': [1, 2, 3]},
        'circular_references': {'psi': [0, 0, 0]},  # Old data to be overwritten
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
    self.assertNotEqual(final_data['circular_references']['psi'], [0, 0, 0])


if __name__ == '__main__':
  absltest.main()
