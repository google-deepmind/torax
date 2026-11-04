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
from absl.testing import parameterized
import numpy as np
import pydantic
from torax._src import state
from torax._src.orchestration import jit_run_loop
from torax._src.orchestration import run_simulation
from torax._src.output_tools import output
from torax._src.test_utils import default_configs
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import config as torax_config_lib
import xarray as xr

_ALL_PROFILES = ('T_i', 'T_e', 'psi', 'q_face', 's_face', 'n_e')


class RunSimulationTest(sim_test_case.SimTestCase):

  def test_change_config(self):
    torax_config = self._get_torax_config('test_iterhybrid_mockup.py')
    simulation_xr = run_simulation.run_simulation(
        torax_config, max_steps=100
    ).simulation_output_to_xr()

    original_value = torax_config.profile_conditions.n_e_bar
    new_value = original_value.value * 1.1

    torax_config.update_fields({'profile_conditions.n_e_bar': new_value})
    new_simulation_xr = run_simulation.run_simulation(
        torax_config, max_steps=100
    ).simulation_output_to_xr()

    self.assertFalse(
        np.array_equal(
            simulation_xr.profiles.n_e.values[-1],
            new_simulation_xr.profiles.n_e.values[-1],
        )
    )

  def test_restart(self):
    test_config_state_file = 'test_iterhybrid_rampup.nc'
    restart_config = 'test_iterhybrid_rampup_restart.py'

    torax_config = self._get_torax_config(restart_config)
    data_tree_restart = run_simulation.run_simulation(
        torax_config
    ).simulation_output_to_xr()

    # Load the reference dataset.
    datatree_ref = output.load_state_file(
        os.path.join(self.test_data_dir, test_config_state_file)
    )

    # Stitch the restart state file to the beginning of the reference dataset.
    datatree_new = output.stitch_state_files(
        torax_config.restart, data_tree_restart  # pyrefly: ignore[bad-argument-type]
    )

    # Check that time coordinates agree.
    with self.subTest(var_name='time'):
      np.testing.assert_allclose(
          datatree_new.time.values,
          datatree_ref.time.values,
          err_msg='Time coordinates do not agree',
          rtol=1e-6,
      )

    # Check equality for all time-dependent variables.
    def check_equality(actual: xr.Dataset, desired: xr.Dataset):
      for var_name in actual.data_vars:
        if 'time' in actual[var_name].dims:
          with self.subTest(var_name=var_name):
            np.testing.assert_allclose(
                actual[var_name].values,
                desired[var_name].values,
                err_msg=f'Mismatch for {var_name} in restart test',
                rtol=1e-5,
                atol=1e-5,
            )

    xr.map_over_datasets(check_equality, datatree_new, datatree_ref)

  @parameterized.named_parameters(
      ('static_geometry_QLKNN', 'test_iterhybrid_rampup.py'),
  )
  def test_no_compile_for_second_run(self, config_name: str):
    # Access the jax logger and set its level to DEBUG.
    jax_logger = logging.getLogger('jax')
    jax_logger.setLevel(logging.DEBUG)
    with self.assertLogs(logger=jax_logger, level=logging.DEBUG) as l:
      torax_config = self._get_torax_config(config_name)
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
      torax_config = self._get_torax_config(config_name)
      run_simulation.run_simulation(torax_config)
      # Check that the same messages are not present in the second run.
      self.assertFalse(
          any('Finished tracing' in line for line in l.output), msg=l
      )
      self.assertFalse(any('Compiling f' in line for line in l.output), msg=l)
      self.assertFalse(
          any('Finished XLA compilation' in line for line in l.output),
          msg=l,
      )

  def test_did_not_reach_t_final_error_when_max_steps_too_low(self):
    torax_config = self._get_torax_config('test_implicit.py')

    # Use max_steps=1 which is far too few to reach t_final=1.
    state_history = run_simulation.run_simulation(torax_config, max_steps=1)

    self.assertEqual(
        state_history.sim_error, state.SimError.DID_NOT_REACH_T_FINAL
    )

  @parameterized.named_parameters(
      ('with_progress_bar', True),
      ('without_progress_bar', False),
  )
  def test_jitted_run_loop_with_logging(self, progress_bar: bool):
    torax_config = self._get_torax_config('test_iterhybrid_mockup.py')
    state_history = run_simulation.run_simulation(
        torax_config,
        log_timestep_info=True,
        progress_bar=progress_bar,
        log_n_steps=5,
        max_steps=100,
    )
    self.assertIsInstance(state_history.simulation_output_to_xr(), xr.DataTree)
    self.assertIsInstance(state_history, output.StateHistory)
    self.assertEqual(state_history.sim_error, state.SimError.NO_ERROR)
    with jit_run_loop._LOCK:
      self.assertEmpty(jit_run_loop._TQDM_REF)

  def test_max_steps_override_does_not_mutate_input_config(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        't_initial': 0.0,
        't_final': 1.0,
        'dt': 0.1,
        'allow_backtracking': False,
        'max_steps': 50,
    }
    torax_config = torax_config_lib.ToraxConfig.from_dict(config_dict)

    state_history = run_simulation.run_simulation(
        torax_config, max_steps=2, progress_bar=False
    )

    self.assertEqual(torax_config.time_step_calculator.max_steps, 50)
    self.assertEqual(
        state_history.torax_config.time_step_calculator.max_steps, 2
    )
    self.assertEqual(
        state_history.sim_error, state.SimError.DID_NOT_REACH_T_FINAL
    )
    self.assertLen(state_history.times, 3)

    for invalid_max_steps in (0, -1):
      with self.assertRaises(pydantic.ValidationError):
        run_simulation.run_simulation(
            torax_config, max_steps=invalid_max_steps, progress_bar=False
        )


if __name__ == '__main__':
  absltest.main()
