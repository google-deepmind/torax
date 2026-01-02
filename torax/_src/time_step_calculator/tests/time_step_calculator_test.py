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
        't_initial': 0.0,
        't_final': 5.0,
        'exact_t_final': exact_t_final,
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    sim_state, _, step_fn = run_simulation.prepare_simulation(torax_config)

    runtime_params_t = step_fn.runtime_params_provider(t=t)
    dt = time_step_calculator_instance.next_dt(
        t=t,
        runtime_params=runtime_params_t,
        geo=sim_state.geometry,
        core_profiles=sim_state.core_profiles,
        core_transport=sim_state.core_transport,
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
        't_initial': 0.0,
        't_final': 5.0,
        'exact_t_final': exact_t_final,
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    sim_state, _, step_fn = run_simulation.prepare_simulation(torax_config)

    runtime_params_t = step_fn.runtime_params_provider(t=t)
    dt = time_step_calculator_instance.next_dt(
        t=t,
        runtime_params=runtime_params_t,
        geo=sim_state.geometry,
        core_profiles=sim_state.core_profiles,
        core_transport=sim_state.core_transport,
    )
    self.assertEqual(dt, expected_dt)


if __name__ == '__main__':
  absltest.main()
