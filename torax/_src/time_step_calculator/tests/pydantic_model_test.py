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
import pydantic
from torax._src.time_step_calculator import chi_time_step_calculator
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.time_step_calculator import pydantic_model as time_step_pydantic_model


class PydanticModelTest(parameterized.TestCase):

  def test_unknown_solver_type_raises_error(self):
    with self.assertRaises(pydantic.ValidationError):
      time_step_pydantic_model.TimeStepCalculator.from_dict(
          {'calculator_type': 'foo'}
      )

  def test_missing_time_step_calculator_type_raises_error(self):
    with self.assertRaises(TypeError):
      time_step_pydantic_model.TimeStepCalculator({})

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


if __name__ == '__main__':
  absltest.main()
