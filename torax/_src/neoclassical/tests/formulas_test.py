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

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.geometry import geometry
from torax._src.neoclassical import formulas


class FormulasTest(parameterized.TestCase):

  def test_calculate_f_trap_positive_triangularity(self):
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        delta_face=np.array(0.2),
        epsilon_face=np.array(0.1),
    )
    result = formulas.calculate_f_trap(geo)
    expected = 0.4362384616678634
    np.testing.assert_allclose(result, expected)

  def test_calculate_f_trap_negative_triangularity(self):
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        delta_face=np.array(-0.2),
        epsilon_face=np.array(0.1),
    )
    result = formulas.calculate_f_trap(geo)
    expected = 0.45134158459680895
    np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
  absltest.main()
