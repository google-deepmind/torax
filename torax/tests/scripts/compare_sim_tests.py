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

"""Script for comparing failed sim tests to reference sim tests."""

import os
from typing import Sequence

from absl import app
from absl import flags
import numpy as np
from torax._src.output_tools import output
from torax.tests import scripts
import xarray as xr


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
  _compare_all_failed_sim_test_outputs()


def _compare_all_failed_sim_test_outputs() -> None:
  """Compares xarray outputs of all failed sim tests to their references.

  Raises:
    FileNotFoundError: if _FAILED_TEST_OUTPUT_DIR does not exist.
  """
  failed_test_output_dir = _FAILED_TEST_OUTPUT_DIR.value

  if not os.path.exists(failed_test_output_dir):
    raise FileNotFoundError(
        f'Directory not found: {failed_test_output_dir}. '
        'Make sure failed tests exist in output directory.'
    )

  for failed_test_file in os.listdir(failed_test_output_dir):
    _compare_sim_test_outputs(failed_test_file)


def _compare_sim_test_outputs(failed_test_file: str) -> None:
  """Compares xarray outputs of failed sim tests to their references.

  Args:
    failed_test_file: Name of the failed test file, this is corresponds to the
      test which failed.
  """
  test_name = failed_test_file.split('.')[0]
  failed_test_output_dir = _FAILED_TEST_OUTPUT_DIR.value
  reference_test_data_dir = _REFERENCE_TEST_DATA_DIR.value

  old_file = os.path.join(
      reference_test_data_dir, scripts.get_data_file(test_name)
  )
  new_file = os.path.join(
      failed_test_output_dir, failed_test_file
  )

  # The variables in the nc files which to compare and print out the diffs
  profile_names = [
      'T_i',
      'T_e',
      'n_e',
      'psi',
      'q',
  ]
  # Load the Datasets
  ds_old = output.safe_load_dataset(old_file).children['profiles']
  ds_new = output.safe_load_dataset(new_file).children['profiles']
  print(f'Comparing {old_file} and {new_file}:')
  for profile_name in profile_names:
    _print_diff(profile_name, ds_old, ds_new)
  print('\n')


def _print_diff(profile_name: str, ds_old: xr.Dataset, ds_new: xr.Dataset):
  """Prints the difference between the last time steps of two profiles.

  Args:
    profile_name: Name of the profile to compare.
    ds_old: Dataset containing the old simulation output.
    ds_new: Dataset containing the new simulation output.
  """

  if (profile_name == 'psi'):
    # Avoid potential 0.0 on-axis
    old_value = ds_old[profile_name].isel(time=-1).to_numpy()[1:]
    new_value = ds_new[profile_name].isel(time=-1).to_numpy()[1:]
  else:
    old_value = ds_old[profile_name].isel(time=-1).to_numpy()
    new_value = ds_new[profile_name].isel(time=-1).to_numpy()

  abs_diff = np.mean(np.abs(old_value - new_value))
  rel_diff = np.mean(np.abs((old_value - new_value) / old_value))
  print(
      f'\t{profile_name}: Mean abs diff: {abs_diff:.2e}, '
      f'mean relative diff: {rel_diff:.2e}'
  )


if __name__ == '__main__':
  app.run(main)
