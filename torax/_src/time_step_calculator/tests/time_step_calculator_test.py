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
import dataclasses

from absl.testing import absltest
from torax._src.orchestration import run_simulation
from torax._src.test_utils import default_configs
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.torax_pydantic import model_config


class TimeStepCalculatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    config_dict = default_configs.get_default_config_dict()
    config_dict['numerics'] = {
        'fixed_dt': 2.0,
        't_initial': 0.0,
        't_final': 5.0,
        'exact_t_final': True,
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    print(torax_config.numerics.fixed_dt)
    self.static_params, dynamic_provider, self.sim_state, _, _, _ = (
        run_simulation.prepare_simulation(torax_config)
    )
    self.dynamic_slice = dynamic_provider(self.sim_state.t)

  def test_next_dt_basic_case(self):
    time_step_calculator_instance = (
        fixed_time_step_calculator.FixedTimeStepCalculator()
    )
    dt = time_step_calculator_instance.next_dt(
        t=0.0,
        static_runtime_params_slice=self.static_params,
        dynamic_runtime_params_slice=self.dynamic_slice,
        geo=self.sim_state.geometry,
        core_profiles=self.sim_state.core_profiles,
        core_transport=self.sim_state.core_transport,
    )
    self.assertEqual(dt, 2.0)

  def test_next_dt_when_t_final_is_reached(self):
    time_step_calculator_instance = (
        fixed_time_step_calculator.FixedTimeStepCalculator()
    )
    dt = time_step_calculator_instance.next_dt(
        t=4.0,
        static_runtime_params_slice=self.static_params,
        dynamic_runtime_params_slice=self.dynamic_slice,
        geo=self.sim_state.geometry,
        core_profiles=self.sim_state.core_profiles,
        core_transport=self.sim_state.core_transport,
    )
    self.assertEqual(dt, 1.0)

  def test_next_dt_when_t_final_is_reached_not_exact(self):
    static_slice = dataclasses.replace(
        self.static_params,
        numerics=dataclasses.replace(
            self.static_params.numerics, exact_t_final=False
        ),
    )
    time_step_calculator_instance = (
        fixed_time_step_calculator.FixedTimeStepCalculator()
    )
    dt = time_step_calculator_instance.next_dt(
        t=4.0,
        static_runtime_params_slice=static_slice,
        dynamic_runtime_params_slice=self.dynamic_slice,
        geo=self.sim_state.geometry,
        core_profiles=self.sim_state.core_profiles,
        core_transport=self.sim_state.core_transport,
    )
    self.assertEqual(dt, 2.0)


if __name__ == '__main__':
  absltest.main()
