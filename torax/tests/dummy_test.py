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
import jax


class DummyTest(absltest.TestCase):

  def test_x64(self):
    jax_x64_flag = jax.config.read('jax_enable_x64')
    print(jax_x64_flag)
    if not jax_x64_flag:
      raise RuntimeError('jax_enable_x64 is not set')


if __name__ == '__main__':
  absltest.main()
