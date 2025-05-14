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
import os
from typing import Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from torax.output_tools import output
from torax.tests.test_lib import sim_test_case


_ALL_PROFILES = (
    output.T_I,
    output.T_E,
    output.PSI,
    output.Q,
    output.N_E,
)


class SimNoCompileTest(sim_test_case.SimTestCase):

  @parameterized.named_parameters(
      (
          'test_psi_and_heat',
          'test_psi_and_heat.py',
      ),
  )
  def test_run_simulation_no_compile(
      self,
      config_name: str,
      profiles: Sequence[str] = _ALL_PROFILES,
      rtol: float | None = None,
      atol: float | None = None,
  ):
    with mock.patch.dict(os.environ, {'TORAX_COMPILATION_ENABLED': 'False'}):
      self._test_run_simulation(
          config_name,
          profiles,
          rtol=rtol,
          atol=atol,
      )


if __name__ == '__main__':
  absltest.main()
