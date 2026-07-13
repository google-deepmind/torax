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
import eqdsk as eqdsk_lib
import numpy as np
from torax._src import array_typing
from torax._src.geometry import base
from torax._src.geometry import eqdsk
from torax._src.geometry import geometry_loader

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

  def test_build_geometry_from_eqdsk_object(self):
    """Test that EQDSK geometries can be built from an EQDSKInterface object."""
    geo_dir = geometry_loader.get_geometry_dir()
    file_name = 'iterhybrid_cocos02.eqdsk'
    file_path = f'{geo_dir}/{file_name}'

    eqdsk_obj = eqdsk_lib.EQDSKInterface.from_file(file_path, from_cocos=2)

    geo_file = eqdsk.EQDSKConfig(
        geometry_file=file_name, cocos=2
    ).build_geometry()
    geo_obj = eqdsk.EQDSKConfig(
        eqdsk_object=eqdsk_obj, cocos=2
    ).build_geometry()

    for field in dataclasses.fields(geo_file):
      name = field.name
      val1 = getattr(geo_file, name)
      val2 = getattr(geo_obj, name)
      if isinstance(val1, array_typing.Array):
        np.testing.assert_allclose(
            val1, val2, err_msg=f'Field "{name}" mismatch.'
        )
      elif val1 is None:
        self.assertIsNone(val2, msg=f'Field "{name}" mismatch.')
      else:
        self.assertEqual(val1, val2, msg=f'Field "{name}" mismatch.')

  def test_trapped_fraction_is_physically_sensible(self):
    """Tests that the exact trapped particle fraction is well-behaved."""
    geo = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos11.eqdsk',
        cocos=11,
        trapped_fraction_source=base.TrappedFractionSource.EXACT,
    ).build_geometry()
    trapped_fraction = geo.trapped_fraction_face
    self.assertIsNotNone(trapped_fraction)
    # No trapped particles on the magnetic axis, where B is uniform.
    self.assertAlmostEqual(float(trapped_fraction[0]), 0.0)
    # The trapped particle fraction is a fraction, so must lie in [0, 1].
    self.assertTrue(np.all(trapped_fraction >= 0.0))
    self.assertTrue(np.all(trapped_fraction <= 1.0))
    # Trapped fraction increases with normalized radius over most of the
    # profile (small deviations from strict monotonicity are possible near
    # the edge for diverted geometries, due to the X-point).
    self.assertGreater(
        np.mean(np.diff(trapped_fraction) >= -1e-6),
        0.8,
    )

  def test_trapped_fraction_source_file_not_supported(self):
    """Tests that FILE is rejected for EQDSK (no precomputed value)."""
    with self.assertRaisesRegex(ValueError, 'not supported for EQDSKConfig'):
      eqdsk.EQDSKConfig(
          geometry_file='iterhybrid_cocos11.eqdsk',
          cocos=11,
          trapped_fraction_source=base.TrappedFractionSource.FILE,
      )

  def test_trapped_fraction_geometry_consistent_with_sauter(self):
    """Tests that the exact and Sauter trapped fractions roughly agree."""
    geo_sauter = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos11.eqdsk',
        cocos=11,
        trapped_fraction_source=base.TrappedFractionSource.SAUTER,
    ).build_geometry()
    geo_geometry = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos11.eqdsk',
        cocos=11,
        trapped_fraction_source=base.TrappedFractionSource.EXACT,
    ).build_geometry()
    # Moderately coarse tolerance: Sauter is only an analytic approximation,
    # so it need not match the exact integral closely, but a large deviation
    # would indicate a bug rather than the expected model discrepancy.
    np.testing.assert_allclose(
        geo_geometry.trapped_fraction_face,
        geo_sauter.trapped_fraction_face,
        atol=0.05,
        rtol=0.15,
    )

  def test_eqdsk_serialization_round_trip(self):
    """Test that EQDSKConfig with eqdsk_object can be serialized and deserialized."""
    geo_dir = geometry_loader.get_geometry_dir()
    file_name = 'iterhybrid_cocos11.eqdsk'
    file_path = f'{geo_dir}/{file_name}'
    eqdsk_obj = eqdsk_lib.EQDSKInterface.from_file(file_path, from_cocos=11)

    config = eqdsk.EQDSKConfig(eqdsk_object=eqdsk_obj, cocos=11)
    dumped = config.model_dump()
    config_restored = eqdsk.EQDSKConfig.from_dict(dumped)
    self.assertIsNotNone(config_restored.eqdsk_object)
    self.assertIsInstance(
        config_restored.eqdsk_object, eqdsk_lib.EQDSKInterface
    )

    # Verify that the built geometries match
    geo_original = config.build_geometry()
    geo_restored = config_restored.build_geometry()

    for field in dataclasses.fields(geo_original):
      name = field.name
      val1 = getattr(geo_original, name)
      val2 = getattr(geo_restored, name)
      if isinstance(val1, array_typing.Array):
        np.testing.assert_allclose(
            val1, val2, err_msg=f'Field "{name}" mismatch (dict).'
        )
      elif val1 is None:
        self.assertIsNone(val2, msg=f'Field "{name}" mismatch (dict).')
      else:
        self.assertEqual(val1, val2, msg=f'Field "{name}" mismatch (dict).')


if __name__ == '__main__':
  absltest.main()
