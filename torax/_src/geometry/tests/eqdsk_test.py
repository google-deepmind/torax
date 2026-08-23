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
from torax._src.geometry import eqdsk
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.neoclassical.formulas import formulas

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

  def test_eqdsk_stores_surface_B_for_numerical_f_trap(self):
    """EQDSK keeps contour |B|(θ) so numerical f_t does not need Miller q."""
    geo = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos11.eqdsk', cocos=11
    ).build_geometry()
    self.assertIsNotNone(geo.B_surface_face)
    self.assertIsNotNone(geo.fsa_weight_face)
    n_face = geo.rho_face_norm.shape[0]
    self.assertEqual(
        geo.B_surface_face.shape, (n_face, geometry.N_THETA_SURFACE)
    )
    self.assertEqual(geo.fsa_weight_face.shape, geo.B_surface_face.shape)
    self.assertTrue(np.all(geo.B_surface_face > 0.0))
    self.assertTrue(np.all(geo.fsa_weight_face >= 0.0))
    f_t = formulas.calculate_f_trap(geo, f_trap_model='numerical')
    np.testing.assert_allclose(f_t[0], 0.0, atol=1e-5)
    self.assertTrue(np.all(f_t >= 0.0))
    self.assertTrue(np.all(f_t <= 1.0))
    self.assertGreater(float(f_t[-1]), float(f_t[1]))


if __name__ == '__main__':
  absltest.main()
