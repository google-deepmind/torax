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

"""Tests that Torax can be run with compilation disabled."""

from typing import Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from torax import jax_utils
from torax.tests.test_lib import sim_test_case


_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


class SimTest(sim_test_case.SimTestCase):
  """No-compilation integration tests for torax.sim."""

  @parameterized.named_parameters(
      (
          'test2_optimizer_no_compile',
          'test2_short_optimizer.py',
          'test2_short',
          _ALL_PROFILES,
          1e-5,
          None,
          False,
      ),
      # test7 is the simplest test known to have had the no-compile mode
      # diverge from the compiled mode.
      (
          'test7',
          'test7.py',
          'test7',
          _ALL_PROFILES,
          0,
          1e-11,
          False,
      ),
  )
  def test_pyntegrated(
      self,
      config_name: str,
      ref_name: str,
      profiles: Sequence[str],
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
      use_ref_time: bool = False,
  ):
    """No-compilation version of integration tests."""
    assert not jax_utils.env_bool('TORAX_COMPILATION_ENABLED', True)

    self._test_pyntegrated(
        config_name,
        ref_name,
        profiles,
        rtol,
        atol,
        use_ref_time,
    )


if __name__ == '__main__':
  absltest.main()
