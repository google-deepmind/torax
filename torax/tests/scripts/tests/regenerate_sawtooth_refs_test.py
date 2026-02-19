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

import json
import os
import pathlib
import shutil
from unittest import mock
from absl.testing import absltest
from absl.testing import flagsaver
import numpy as np
from torax._src import path_utils
from torax.tests.scripts import regenerate_sawtooth_refs

# Internal import.

_NRHO = regenerate_sawtooth_refs.NRHO


class RegenerateSawtoothRefsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = pathlib.Path(self.create_tempdir())

    original_torax_path = path_utils.torax_path()
    original_refs_path = (
        original_torax_path
        / '_src'
        / 'mhd'
        / 'sawtooth'
        / 'tests'
        / regenerate_sawtooth_refs.REFERENCES_FILE
    )

    self.mock_torax_path_patch = mock.patch.object(
        path_utils, 'torax_path', return_value=self.temp_dir
    )
    self.mock_torax_path = self.mock_torax_path_patch.start()
    self.addCleanup(self.mock_torax_path_patch.stop)

    self.output_path = (
        self.temp_dir
        / '_src'
        / 'mhd'
        / 'sawtooth'
        / 'tests'
        / regenerate_sawtooth_refs.REFERENCES_FILE
    )

    self.output_path.parent.mkdir(parents=True, exist_ok=True)
    if original_refs_path.exists():
      shutil.copyfile(original_refs_path, self.output_path)

  def test_smoke_test(self):
    regenerate_sawtooth_refs.main([])

  @flagsaver.flagsaver(write_to_file=True)
  def test_write_to_file_creates_json(self):
    self.output_path.parent.mkdir(parents=True, exist_ok=True)

    regenerate_sawtooth_refs.main([])

    self.assertTrue(self.output_path.exists())
    with open(str(self.output_path), 'r') as f:
      data = json.load(f)

    self.assertIsInstance(data, dict)
    expected_keys = {
        'post_crash_temperature',
        'post_crash_n',
        'post_crash_psi',
    }
    self.assertContainsSubset(expected_keys, data.keys())
    for key in expected_keys:
      self.assertIsInstance(data[key], list)
      self.assertLen(data[key], _NRHO)

  @flagsaver.flagsaver(print_summary=True)
  def test_summary_flag_prints_output(self):
    with self.assertLogs(level='INFO') as cm:
      regenerate_sawtooth_refs.main([])
      log_output = '\n'.join(cm.output)
      self.assertIn('Regenerating sawtooth crash references', log_output)
      self.assertIn('Sawtooth crash reference values', log_output)
      self.assertIn('post_crash_temperature', log_output)
      self.assertIn('post_crash_n', log_output)
      self.assertIn('post_crash_psi', log_output)

  def test_summary_is_not_printed_by_default(self):
    with self.assertLogs(level='INFO') as cm:
      regenerate_sawtooth_refs.main([])
      log_output = '\n'.join(cm.output)
      self.assertIn('Regenerating sawtooth crash references', log_output)
      self.assertNotIn('Sawtooth crash reference values', log_output)

  @flagsaver.flagsaver(write_to_file=True)
  def test_write_to_file_overwrites_existing(self):
    self.output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_data = {
        'post_crash_temperature': np.ones(_NRHO).tolist(),
        'post_crash_n': np.ones(_NRHO).tolist(),
        'post_crash_psi': np.ones(_NRHO).tolist(),
    }
    with open(str(self.output_path), 'w') as f:
      json.dump(dummy_data, f)

    regenerate_sawtooth_refs.main([])

    with open(str(self.output_path), 'r') as f:
      final_data = json.load(f)

    for key, old_value in dummy_data.items():
      self.assertNotEqual(final_data[key], old_value)

  def test_calculate_sawtooth_crash_references_returns_expected_keys(self):
    result = regenerate_sawtooth_refs.calculate_sawtooth_crash_references()
    expected_keys = {
        'post_crash_temperature',
        'post_crash_n',
        'post_crash_psi',
    }
    self.assertEqual(set(result.keys()), expected_keys)
    for key in expected_keys:
      self.assertLen(np.asarray(result[key]), _NRHO)

  @flagsaver.flagsaver(write_to_file=True)
  def test_output_dir_flag(self):
    custom_dir = pathlib.Path(self.create_tempdir('custom_output'))
    with flagsaver.flagsaver(output_dir=str(custom_dir)):
      regenerate_sawtooth_refs.main([])

    custom_output_path = custom_dir / regenerate_sawtooth_refs.REFERENCES_FILE
    self.assertTrue(custom_output_path.exists())
    with open(str(custom_output_path), 'r') as f:
      data = json.load(f)
    expected_keys = {
        'post_crash_temperature',
        'post_crash_n',
        'post_crash_psi',
    }
    self.assertContainsSubset(expected_keys, data.keys())

  def test_works_without_jax_enable_x64_env_var(self):
    original_value = os.environ.pop('JAX_ENABLE_X64', None)
    try:
      regenerate_sawtooth_refs.main([])
    finally:
      if original_value is not None:
        os.environ['JAX_ENABLE_X64'] = original_value


if __name__ == '__main__':
  absltest.main()
