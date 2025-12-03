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
from torax._src.geometry import chease
# pylint: disable=invalid-name


class CheaseGeometryTest(parameterized.TestCase):

  def test_build_geometry_from_chease(self):
    chease.CheaseConfig().build_geometry()

  def test_access_z_magnetic_axis_raises_error_for_chease_geometry(self):
    """Test that accessing z_magnetic_axis raises error for CHEASE geometry."""
    geo = chease.CheaseConfig().build_geometry()
    with self.assertRaisesRegex(ValueError, 'does not have a z magnetic axis'):
      geo.z_magnetic_axis()


if __name__ == '__main__':
  absltest.main()
