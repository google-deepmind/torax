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
from typing import Optional, Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from torax.tests.test_lib import sim_test_case


_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


class SimTest(sim_test_case.SimTestCase):
  """No-compilation integration tests for torax.sim."""

  @parameterized.named_parameters(
      (
          'test_implicit_optimizer_no_compile',
          'test_implicit_short_optimizer.py',
          _ALL_PROFILES,
          1e-5,
          None,
      ),
      # test_qlknnheat is the simplest test known to have had the no-compile
      # mode diverge from the compiled mode.
      (
          'test_qlknnheat',
          'test_qlknnheat.py',
          _ALL_PROFILES,
          0,
          1e-11,
      ),
  )
  def test_torax_sim(
      self,
      config_name: str,
      profiles: Sequence[str],
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
  ):
    """No-compilation version of integration tests."""
    with mock.patch.dict(os.environ, {'TORAX_COMPILATION_ENABLED': 'False'}):
      self._test_torax_sim(
          config_name,
          profiles,
          rtol=rtol,
          atol=atol,
      )


if __name__ == '__main__':
  absltest.main()
