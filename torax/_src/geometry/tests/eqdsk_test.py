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
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src import array_typing
from torax._src.geometry import eqdsk

# pylint: disable=invalid-name


class EqdskGeometryTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(geometry_file='iterhybrid_cocos02.eqdsk', cocos=2),
      dict(geometry_file='iterhybrid_cocos11.eqdsk', cocos=11),
  ])
  def test_build_geometry_from_eqdsk(self, geometry_file, cocos):
    """Test that EQDSK geometries can be built."""
    config = eqdsk.EQDSKConfig(geometry_file=geometry_file, cocos=cocos)
    config.build_geometry()

  def test_eqdsk_cocos_conversion_is_consistent(self):
    """Tests that EQDSK geometries from different COCOS are identical after conversion."""
    geo_cocos2 = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos02.eqdsk', cocos=2
    ).build_geometry()
    geo_cocos11 = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos11.eqdsk', cocos=11
    ).build_geometry()
    for field in dataclasses.fields(geo_cocos2):
      name = field.name
      val1 = getattr(geo_cocos2, name)
      val2 = getattr(geo_cocos11, name)
      if isinstance(val1, array_typing.Array):
        np.testing.assert_allclose(
            val1, val2, err_msg=f'Field "{name}" mismatch.'
        )
      elif val1 is None:
        self.assertIsNone(val2, msg=f'Field "{name}" mismatch.')
      else:
        self.assertEqual(val1, val2, msg=f'Field "{name}" mismatch.')


if __name__ == '__main__':
  absltest.main()
