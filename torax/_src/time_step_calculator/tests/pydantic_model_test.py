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
from torax._src.time_step_calculator import chi_time_step_calculator
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.time_step_calculator import pydantic_model as time_step_pydantic_model


class PydanticModelTest(parameterized.TestCase):

  def test_unknown_time_step_calculator_type_raises_error(self):
    with self.assertRaises(pydantic.ValidationError):
      time_step_pydantic_model.TimeStepCalculator.from_dict(
          dict(calculator_type='x')
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='fixed',
          calculator_type='fixed',
          expected_type=fixed_time_step_calculator.FixedTimeStepCalculator,
      ),
      dict(
          testcase_name='chi',
          calculator_type='chi',
          expected_type=chi_time_step_calculator.ChiTimeStepCalculator,
      ),
  )
  def test_build_time_step_calculator_from_config(
      self, calculator_type, expected_type
  ):
    """Builds a time step calculator from the config."""
    time_stepper = time_step_pydantic_model.TimeStepCalculator.from_dict(
        {'calculator_type': calculator_type}
    ).time_step_calculator
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
  )
  def test_time_step_calculator_under_jit(self, calculator_type):
    """Builds a time step calculator from the config."""
    x = time_step_pydantic_model.TimeStepCalculator.from_dict(
        {'calculator_type': calculator_type, 'tolerance': 2e-5}
    )

    @jax.jit
    def f(time_stepper: time_step_pydantic_model.TimeStepCalculator):
      return time_stepper.build_runtime_params()

    with self.subTest('first_jit_compiles_and_returns_expected_value'):
      output = f(x)
      self.assertEqual(output.tolerance, 2e-5)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    with self.subTest('second_jit_updates_value_without_recompile'):
      x._update_fields({'tolerance': 1e-5})
      output = f(x)
      self.assertEqual(output.tolerance, 1e-5)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)


if __name__ == '__main__':
  absltest.main()
