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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.orchestration import run_simulation
from torax._src.test_utils import default_configs
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.torax_pydantic import model_config


class TimeStepCalculatorTest(parameterized.TestCase):

  @parameterized.parameters(
      (0.0, True, 2.0), (4.0, True, 1.0), (4.0, False, 2.0)
  )
  def test_next_dt_constant(self, t, exact_t_final, expected_dt):
    time_step_calculator_instance = (
        fixed_time_step_calculator.FixedTimeStepCalculator()
    )
    config_dict = default_configs.get_default_config_dict()
    config_dict['numerics'] = {
        'fixed_dt': 2.0,
        't_initial': t,
        't_final': 5.0,
        'exact_t_final': exact_t_final,
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    sim_state, _, step_fn = run_simulation.prepare_simulation(torax_config)

    runtime_params_t = step_fn.runtime_params_provider(t=t)
    dt = time_step_calculator_instance.next_dt(
        runtime_params=runtime_params_t,
        sim_state=sim_state,
    )
    self.assertEqual(dt, expected_dt)

  @parameterized.parameters(
      (0.0, True, 2.0),
      (2.0, True, 1.0),
      (3.0, True, 1.0),
      (4.0, True, 1.0),
      (4.0, False, 2.0),
  )
  def test_next_dt_time_dependent(self, t, exact_t_final, expected_dt):
    time_step_calculator_instance = (
        fixed_time_step_calculator.FixedTimeStepCalculator()
    )
    config_dict = default_configs.get_default_config_dict()
    # TODO(b/454891040): Change the value at the boundary for the STEP
    # interpolation mode.
    epsilon = 1e-5
    config_dict['numerics'] = {
        'fixed_dt': {
            0.0: 2.0,
            2.0 - epsilon: 1.0,
            4.0 - epsilon: 2.0,
        },
        't_initial': t,
        't_final': 5.0,
        'exact_t_final': exact_t_final,
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    sim_state, _, step_fn = run_simulation.prepare_simulation(torax_config)

    runtime_params_t = step_fn.runtime_params_provider(t=t)
    dt = time_step_calculator_instance.next_dt(
        runtime_params=runtime_params_t,
        sim_state=sim_state,
    )
    self.assertEqual(dt, expected_dt)

  @parameterized.named_parameters(
      ('exact_t_final', True, [0.0, 2.0, 4.0, 5.0]),
      ('overshoot_t_final', False, [0.0, 2.0, 4.0, 6.0]),
  )
  def test_get_time_grid(self, exact_t_final, expected_times):
    times = fixed_time_step_calculator.get_time_grid(
        t_initial=0.0,
        t_final=5.0,
        fixed_dt=2.0,
        exact_t_final=exact_t_final,
        tolerance=1e-7,
    )
    np.testing.assert_allclose(times, expected_times)

  def test_get_time_grid_matches_simulation_times(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['numerics'] = {
        'fixed_dt': 0.3,
        't_initial': 0.0,
        't_final': 1.0,
        'adaptive_dt': False,
    }
    config_dict['time_step_calculator'] = {'calculator_type': 'fixed'}
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    _, state_history = run_simulation.run_simulation(
        torax_config, progress_bar=False
    )
    times = fixed_time_step_calculator.get_time_grid(
        t_initial=torax_config.numerics.t_initial,
        t_final=torax_config.numerics.t_final,
        fixed_dt=0.3,
        exact_t_final=torax_config.numerics.exact_t_final,
        tolerance=torax_config.time_step_calculator.tolerance,
    )
    np.testing.assert_array_equal(times, state_history.times)

  def test_get_time_grid_stops_at_max_num_times(self):
    times = fixed_time_step_calculator.get_time_grid(
        t_initial=0.0,
        t_final=1_000_000.0,
        fixed_dt=0.1,
        exact_t_final=True,
        tolerance=1e-7,
        max_num_times=3,
    )
    self.assertIsNone(times)

  def test_get_time_grid_rejects_non_positive_dt(self):
    with self.assertRaisesRegex(ValueError, 'must be positive'):
      fixed_time_step_calculator.get_time_grid(
          t_initial=0.0,
          t_final=1.0,
          fixed_dt=0.0,
          exact_t_final=True,
          tolerance=1e-7,
      )


if __name__ == '__main__':
  absltest.main()
