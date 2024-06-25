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

"""Script to copy newly generated sim tests as reference sim tests."""

import os
import shutil
from typing import Sequence

from absl import app
from absl import flags

from torax.tests import test_lib

_FAILED_TEST_OUTPUT_DIR = flags.DEFINE_string(
    'failed_test_output_dir',
    '/tmp/torax_failed_sim_test_outputs/',
    'File path to the directory containing failed sim test output'
    ' subdirectories.',
)

_REFERENCE_TEST_DATA_DIR = flags.DEFINE_string(
    'reference_dir',
    'torax/tests/test_data',
    'File path to the directory containing reference data files',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  _copy_all_failed_sim_test_outputs()


def _copy_all_failed_sim_test_outputs() -> None:
  """Copies newly generated sim tests to the tests/test_data directory.

  Raises:
    FileNotFoundError: if _FAILED_TEST_OUTPUT_DIR does not exist.
  """
  failed_test_output_dir = _FAILED_TEST_OUTPUT_DIR.value

  if not os.path.exists(failed_test_output_dir):
    raise FileNotFoundError(
        f'Directory not found: {failed_test_output_dir}. '
        'Make sure failed tests exist in output directory.'
    )

  for failed_test_dir in os.listdir(failed_test_output_dir):
    _copy_sim_test_outputs(failed_test_dir)


def _copy_sim_test_outputs(failed_test_dir: str) -> None:
  """Compares xarray outputs of failed sim tests to their references.

  Args:
    failed_test_dir: Name of the failed test directory.
  """
  failed_test_output_dir = _FAILED_TEST_OUTPUT_DIR.value
  reference_test_data_dir = _REFERENCE_TEST_DATA_DIR.value
  old_file = os.path.join(reference_test_data_dir, failed_test_dir + '.nc')
  new_file = os.path.join(
      failed_test_output_dir, failed_test_dir, 'state_history.nc'
  )

  # Copy old_file to new_file, overwriting new_file if it exists.
  # Only copy references where the test name matches the reference name.
  if failed_test_dir == test_lib.get_data_file(failed_test_dir)[:-3]:
    shutil.copy(new_file, old_file)
    print(f'\nCopied {new_file} to {old_file}\n')


if __name__ == '__main__':
  app.run(main)
