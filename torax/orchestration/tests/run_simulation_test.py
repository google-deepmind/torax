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
import logging
import os

from absl.testing import absltest
import numpy as np
from torax import output
from torax.orchestration import run_simulation
from torax.tests.test_lib import sim_test_case
import xarray as xr


_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


class RunSimulationTest(sim_test_case.SimTestCase):

  def test_change_config(self):
    torax_config = self._get_torax_config('test_iterhybrid_mockup.py')
    history = run_simulation.run_simulation(torax_config)

    original_value = torax_config.runtime_params.profile_conditions.nbar
    new_value = original_value.value * 1.1

    torax_config.update_fields(
        {'runtime_params.profile_conditions.nbar': new_value}
    )
    new_history = run_simulation.run_simulation(torax_config)

    self.assertFalse(
        np.array_equal(
            history.core_profiles.ne.value[-1],
            new_history.core_profiles.ne.value[-1],
        )
    )

  def test_restart(self):
    test_config_state_file = 'test_iterhybrid_rampup.nc'
    restart_config = 'test_iterhybrid_rampup_restart.py'

    torax_config = self._get_torax_config(restart_config)
    history = run_simulation.run_simulation(torax_config)

    data_tree_restart = history.simulation_output_to_xr()

    # Load the reference dataset.
    datatree_ref = output.load_state_file(
        os.path.join(self.test_data_dir, test_config_state_file)
    )

    # Stitch the restart state file to the beginning of the reference dataset.
    datatree_new = output.stitch_state_files(
        torax_config.restart, data_tree_restart
    )

    # Check equality for all time-dependent variables.
    def check_equality(ds1: xr.Dataset, ds2: xr.Dataset):
      for var_name in ds1.data_vars:
        if 'time' in ds1[var_name].dims:
          with self.subTest(var_name=var_name):
            np.testing.assert_allclose(
                ds1[var_name].values,
                ds2[var_name].values,
                err_msg=f'Mismatch for {var_name} in restart test',
                rtol=1e-6,
            )

    xr.map_over_datasets(check_equality, datatree_ref, datatree_new)

  def test_no_compile_for_second_run(self):
    # Access the jax logger and set its level to DEBUG.
    jax_logger = logging.getLogger('jax')
    jax_logger.setLevel(logging.DEBUG)
    with self.assertLogs(logger=jax_logger, level=logging.DEBUG) as l:
      torax_config = self._get_torax_config('test_iterhybrid_rampup.py')
      run_simulation.run_simulation(torax_config)
      # Check that the messages we expect to see for tracing and compilation
      # are present in the first run.
      self.assertTrue(any('Finished tracing' in line for line in l.output))
      self.assertTrue(any('Compiling' in line for line in l.output))
      self.assertTrue(
          any('Finished XLA compilation' in line for line in l.output)
      )
    with self.assertLogs(jax_logger, level=logging.DEBUG) as l:
      jax_logger.debug('Second run')
      torax_config = self._get_torax_config('test_iterhybrid_rampup.py')
      run_simulation.run_simulation(torax_config)
      # Check that the same messages are not present in the second run.
      self.assertFalse(any('Finished tracing' in line for line in l.output))
      self.assertFalse(any('Compiling f' in line for line in l.output))
      self.assertFalse(
          any('Finished XLA compilation' in line for line in l.output)
      )


if __name__ == '__main__':
  absltest.main()
