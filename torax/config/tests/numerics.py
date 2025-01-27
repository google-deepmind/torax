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

"""Unit tests for the `torax.config.numerics` module."""

from absl.testing import absltest
from absl.testing import parameterized
from torax.config import numerics


class NumericsTest(parameterized.TestCase):
  """Unit tests for the `torax.config.numerics` module."""

  def test_numerics_make_provider(self):
    nums = numerics.Numerics()
    nums.build_dynamic_params(t=0.0)


if __name__ == "__main__":
  absltest.main()
