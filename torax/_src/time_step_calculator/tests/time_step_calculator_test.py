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
from torax._src.time_step_calculator import chi_time_step_calculator
from torax._src.time_step_calculator import fixed_time_step_calculator


class TimeStepCalculatorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='fixed',
          calculator_type=fixed_time_step_calculator.FixedTimeStepCalculator,
      ),
      dict(
          testcase_name='chi',
          calculator_type=chi_time_step_calculator.ChiTimeStepCalculator,
      ),
  )
  def test_different_time_step_calculators_have_same_hash_and_equals(
      self,
      calculator_type,
  ):
    calculator1 = calculator_type()
    calculator2 = calculator_type()
    self.assertEqual(calculator1, calculator2)
    self.assertEqual(hash(calculator1), hash(calculator2))


if __name__ == '__main__':
  absltest.main()
