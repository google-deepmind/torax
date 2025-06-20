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

"""Tests that TORAX can be run with compilation disabled."""
from absl.testing import absltest
from absl.testing import parameterized
from torax._src.output_tools import output
from torax._src.test_utils import sim_test_case


_ALL_PROFILES = (
    output.T_I,
    output.T_E,
    output.PSI,
    output.Q,
    output.N_E,
)


class SimExperimentalCompileTest(sim_test_case.SimTestCase):

  @parameterized.named_parameters(
      # Using newton raphson non linear solver.
      (
          'test_iterhybrid_rampup',
          'test_iterhybrid_rampup.py',
      ),
      # Using linear solver.
      (
          'test_iterhybrid_predictor_corrector',
          'test_iterhybrid_predictor_corrector.py',
      ),
  )
  def test_run_simulation_experimental_compile(
      self,
      config_name: str,
  ):
    self._test_run_simulation(
        config_name,
        profiles=_ALL_PROFILES,
    )


if __name__ == '__main__':
  absltest.main()
