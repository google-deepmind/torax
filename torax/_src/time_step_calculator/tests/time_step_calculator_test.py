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
from absl.testing import parameterized
import jax.numpy as jnp
from torax._src.orchestration import run_simulation
from torax._src.test_utils import default_configs
from torax._src.time_step_calculator import chi_time_step_calculator
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.time_step_calculator import from_previous_time_step_calculator
from torax._src.torax_pydantic import config


class TimeStepCalculatorTest(parameterized.TestCase):

  @parameterized.parameters(
      (0.0, True, 2.0), (4.0, True, 1.0), (4.0, False, 2.0)
  )
  def test_next_dt_constant(self, t, exact_t_final, expected_dt):
    time_step_calculator_instance = (
        fixed_time_step_calculator.FixedTimeStepCalculator()
    )
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        't_initial': t,
        't_final': 5.0,
        'dt': 2.0,
        'exact_t_final': exact_t_final,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    sim_state, _, step_fn = run_simulation.prepare_simulation(torax_config)

    runtime_params_t = step_fn.runtime_params_provider(t=t)
    dt = time_step_calculator_instance.next_dt(
        runtime_params=runtime_params_t.time_step_calculator,
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
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        't_initial': t,
        't_final': 5.0,
        'dt': {
            0.0: 2.0,
            2.0: 1.0,
            4.0: 2.0,
        },
        'exact_t_final': exact_t_final,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    sim_state, _, step_fn = run_simulation.prepare_simulation(torax_config)

    runtime_params_t = step_fn.runtime_params_provider(t=t)
    dt = time_step_calculator_instance.next_dt(
        runtime_params=runtime_params_t.time_step_calculator,
        sim_state=sim_state,
    )
    self.assertEqual(dt, expected_dt)

  @parameterized.parameters(
      (2.0, 1.0, 0.375),
      (2.0, 0.1, 0.1),
  )
  def test_chi_time_step_calculator(
      self, chi_timestep_prefactor, max_dt, expected_dt
  ):
    calculator = chi_time_step_calculator.ChiTimeStepCalculator()
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'chi',
        'chi_timestep_prefactor': chi_timestep_prefactor,
        'max_dt': max_dt,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    sim_state, _, step_fn = run_simulation.prepare_simulation(torax_config)
    runtime_params_t = step_fn.runtime_params_provider(t=0.0)
    dt = calculator.next_dt(
        runtime_params=runtime_params_t.time_step_calculator,
        sim_state=sim_state,
    )
    self.assertAlmostEqual(float(dt), expected_dt)

  @parameterized.parameters(
      (None, 0.25),
      (0.2, 0.6),
      (0.5, 1.0),
  )
  def test_from_previous_time_step_calculator(self, prev_dt, expected_dt):
    calculator = (
        from_previous_time_step_calculator.FromPreviousTimeStepCalculator()
    )
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'from_previous_dt',
        't_initial': 0.0,
        't_final': 5.0,
        'initial_dt': 0.25,
        'max_dt': 1.0,
        'dt_backtracking_factor': 3.0,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    sim_state, _, step_fn = run_simulation.prepare_simulation(torax_config)
    if prev_dt is not None:
      sim_state = dataclasses.replace(sim_state, dt=jnp.asarray(prev_dt))
      sim_state = dataclasses.replace(
          sim_state,
          time_step_calculator_state=calculator.get_updated_state(sim_state),
      )
    runtime_params_t = step_fn.runtime_params_provider(t=0.0)
    dt = calculator.next_dt(
        runtime_params=runtime_params_t.time_step_calculator,
        sim_state=sim_state,
    )
    self.assertAlmostEqual(float(dt), expected_dt)

  def test_from_previous_time_step_calculator_sawtooth_crash(self):
    calculator = (
        from_previous_time_step_calculator.FromPreviousTimeStepCalculator()
    )
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'from_previous_dt',
        'initial_dt': 0.25,
        'max_dt': 1.0,
        'dt_backtracking_factor': 3.0,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    sim_state, _, step_fn = run_simulation.prepare_simulation(torax_config)
    runtime_params_t = step_fn.runtime_params_provider(t=0.0)

    # A sawtooth crash on the first step preserves `initial_dt` for the first
    # non-sawtooth step without multiplying by `dt_backtracking_factor`.
    sim_state = dataclasses.replace(
        sim_state,
        dt=jnp.asarray(1e-3),
        solver_numeric_outputs=dataclasses.replace(
            sim_state.solver_numeric_outputs, sawtooth_crash=True
        ),
    )
    sim_state = dataclasses.replace(
        sim_state,
        time_step_calculator_state=calculator.get_updated_state(sim_state),
    )
    dt_after_initial_crash = calculator.next_dt(
        runtime_params=runtime_params_t.time_step_calculator,
        sim_state=sim_state,
    )
    self.assertAlmostEqual(float(dt_after_initial_crash), 0.25)

    # After a non-sawtooth step completes, subsequent steps scale by
    # `dt_backtracking_factor`, and later sawtooth crashes preserve that dt.
    sim_state = dataclasses.replace(
        sim_state,
        dt=jnp.asarray(0.25),
        solver_numeric_outputs=dataclasses.replace(
            sim_state.solver_numeric_outputs, sawtooth_crash=False
        ),
    )
    sim_state = dataclasses.replace(
        sim_state,
        time_step_calculator_state=calculator.get_updated_state(sim_state),
    )
    dt_after_normal_step = calculator.next_dt(
        runtime_params=runtime_params_t.time_step_calculator,
        sim_state=sim_state,
    )
    self.assertAlmostEqual(float(dt_after_normal_step), 0.75)

    sim_state = dataclasses.replace(
        sim_state,
        dt=jnp.asarray(1e-3),
        solver_numeric_outputs=dataclasses.replace(
            sim_state.solver_numeric_outputs, sawtooth_crash=True
        ),
    )
    sim_state = dataclasses.replace(
        sim_state,
        time_step_calculator_state=calculator.get_updated_state(sim_state),
    )
    dt_after_later_crash = calculator.next_dt(
        runtime_params=runtime_params_t.time_step_calculator,
        sim_state=sim_state,
    )
    self.assertAlmostEqual(float(dt_after_later_crash), 0.75)


if __name__ == '__main__':
  absltest.main()
