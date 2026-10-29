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
import jax
import pydantic
from torax._src import jax_utils
from torax._src.test_utils import default_configs
from torax._src.time_step_calculator import chi_time_step_calculator
from torax._src.time_step_calculator import config as time_step_config
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.time_step_calculator import from_previous_time_step_calculator
from torax._src.torax_pydantic import config as torax_config_lib


class ConfigTest(parameterized.TestCase):

  def test_unknown_time_step_calculator_type_raises_error(self):
    config = default_configs.get_default_config_dict()
    config['time_step_calculator'] = {'calculator_type': 'x'}
    with self.assertRaises(pydantic.ValidationError):
      torax_config_lib.ToraxConfig.from_dict(config)

  @parameterized.parameters('chi', 'from_previous_dt')
  def test_min_dt_greater_than_max_dt_raises_error(self, calculator_type):
    config = default_configs.get_default_config_dict()
    config['time_step_calculator'] = {
        'calculator_type': calculator_type,
        'min_dt': 2.0,
        'max_dt': 1.0,
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        'max_dt must be greater than or equal to min_dt',
    ):
      torax_config_lib.ToraxConfig.from_dict(config)

  def test_initial_dt_greater_than_max_dt_raises_error(self):
    config = default_configs.get_default_config_dict()
    config['time_step_calculator'] = {
        'calculator_type': 'from_previous_dt',
        'initial_dt': 2.0,
        'max_dt': 1.0,
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        'max_dt must be greater than or equal to initial_dt',
    ):
      torax_config_lib.ToraxConfig.from_dict(config)

  def test_from_previous_dt_requires_allow_backtracking(self):
    config = default_configs.get_default_config_dict()
    config['time_step_calculator'] = {
        'calculator_type': 'from_previous_dt',
        'allow_backtracking': False,
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        'allow_backtracking must be True when using from_previous_dt',
    ):
      torax_config_lib.ToraxConfig.from_dict(config)

  @parameterized.parameters('fixed', 'chi', 'from_previous_dt')
  def test_t_initial_greater_than_t_final_raises_error(self, calculator_type):
    config = default_configs.get_default_config_dict()
    config['time_step_calculator'] = {
        'calculator_type': calculator_type,
        't_initial': 2.0,
        't_final': 1.0,
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        't_initial must be less than or equal to t_final',
    ):
      torax_config_lib.ToraxConfig.from_dict(config)

  @parameterized.named_parameters(
      dict(
          testcase_name='fixed',
          calculator_type='fixed',
          expected_config_type=time_step_config.FixedTimeStepCalculatorConfig,
          expected_type=fixed_time_step_calculator.FixedTimeStepCalculator,
      ),
      dict(
          testcase_name='chi',
          calculator_type='chi',
          expected_config_type=time_step_config.ChiTimeStepCalculatorConfig,
          expected_type=chi_time_step_calculator.ChiTimeStepCalculator,
      ),
      dict(
          testcase_name='from_previous_dt',
          calculator_type='from_previous_dt',
          expected_config_type=time_step_config.FromPreviousTimeStepCalculatorConfig,
          expected_type=from_previous_time_step_calculator.FromPreviousTimeStepCalculator,
      ),
  )
  def test_build_time_step_calculator_from_config(
      self, calculator_type, expected_config_type, expected_type
  ):
    """Builds a time step calculator from the config."""
    config = default_configs.get_default_config_dict()
    config['time_step_calculator'] = {'calculator_type': calculator_type}
    torax_config = torax_config_lib.ToraxConfig.from_dict(config)
    self.assertIsInstance(
        torax_config.time_step_calculator, expected_config_type
    )
    time_stepper = (
        torax_config.time_step_calculator.build_time_step_calculator()
    )
    self.assertIsInstance(time_stepper, expected_type)

  @parameterized.named_parameters(
      dict(
          testcase_name='fixed',
          calculator_type='fixed',
      ),
      dict(
          testcase_name='chi',
          calculator_type='chi',
      ),
      dict(
          testcase_name='from_previous_dt',
          calculator_type='from_previous_dt',
      ),
  )
  def test_time_step_calculator_under_jit(self, calculator_type):
    """Builds a time step calculator from the config."""
    config = default_configs.get_default_config_dict()
    config['time_step_calculator'] = {
        'calculator_type': calculator_type,
        'tolerance': 2e-5,
    }
    torax_config = torax_config_lib.ToraxConfig.from_dict(config)
    x = torax_config.time_step_calculator

    @jax.jit
    def f(time_stepper: time_step_config.TimeStepCalculatorConfig):
      return time_stepper.build_runtime_params(t=0.0)

    with self.subTest('first_jit_compiles_and_returns_expected_value'):
      output = f(x)
      self.assertEqual(output.tolerance, 2e-5)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    with self.subTest('second_jit_updates_value_without_recompile'):
      x._update_fields({'tolerance': 1e-5, 'min_dt': 1e-6})
      output = f(x)
      self.assertEqual(output.tolerance, 1e-5)
      self.assertEqual(output.min_dt, 1e-6)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    with self.subTest('static_field_update_recompiles'):
      x._update_fields({'exact_t_final': False})
      output = f(x)
      self.assertFalse(output.exact_t_final)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 2)

  @parameterized.parameters('chi', 'from_previous_dt')
  def test_non_fixed_max_steps_is_none(self, calculator_type):
    config = default_configs.get_default_config_dict()
    config['time_step_calculator'] = {'calculator_type': calculator_type}
    torax_config = torax_config_lib.ToraxConfig.from_dict(config)
    self.assertIsNone(torax_config.time_step_calculator.max_steps)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_backtracking',
          allow_backtracking=False,
          expected_max_steps=3,
      ),
      dict(
          testcase_name='with_backtracking',
          allow_backtracking=True,
          expected_max_steps=30,
      ),
  )
  def test_fixed_max_steps(self, allow_backtracking, expected_max_steps):
    config = default_configs.get_default_config_dict()
    config['time_step_calculator'] = {
        'calculator_type': 'fixed',
        't_initial': 0.0,
        't_final': 5.0,
        'dt': 2.0,
        'allow_backtracking': allow_backtracking,
    }
    torax_config = torax_config_lib.ToraxConfig.from_dict(config)
    self.assertEqual(
        torax_config.time_step_calculator.max_steps, expected_max_steps
    )

  def test_max_steps_is_abstract(self):
    self.assertIn(
        'max_steps',
        time_step_config.TimeStepCalculatorConfigBase.__abstractmethods__,
    )

    class IncompleteTimeStepCalculatorConfig(
        time_step_config.TimeStepCalculatorConfigBase
    ):

      def build_time_step_calculator(self):
        return fixed_time_step_calculator.FixedTimeStepCalculator()

    with self.assertRaises(TypeError):
      IncompleteTimeStepCalculatorConfig()  # pyrefly: ignore[bad-instantiation]


if __name__ == '__main__':
  absltest.main()
