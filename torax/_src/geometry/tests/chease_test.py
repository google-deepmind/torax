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
import chex
import numpy as np
from torax._src.geometry import base
from torax._src.geometry import chease
from torax._src.geometry import standard_geometry

# pylint: disable=invalid-name


class CheaseGeometryTest(parameterized.TestCase):

  def test_chease_config_requires_geometry_file_or_data(self):
    """Tests that CheaseConfig requires geometry_file or chease_data."""
    with self.assertRaises(ValueError):
      chease.CheaseConfig()
    with self.assertRaises(ValueError):
      chease.CheaseConfig.from_dict({'model_name': 'chease'})

  def test_build_geometry_from_chease(self):
    _ = (
        chease.CheaseConfig(geometry_file='iterhybrid.mat2cols')
        .build_runtime_params(0.0)
        .geometry
    )

  def test_access_z_magnetic_axis_raises_error_for_chease_geometry(self):
    """Test that accessing z_magnetic_axis raises error for CHEASE geometry."""
    geo = (
        chease.CheaseConfig(geometry_file='iterhybrid.mat2cols')
        .build_runtime_params(0.0)
        .geometry
    )
    with self.assertRaisesRegex(ValueError, 'does not have a z magnetic axis'):
      geo.z_magnetic_axis()

  def test_trapped_fraction_source_exact_not_supported(self):
    """Tests that EXACT is rejected for CHEASE (no full 2D equilibrium)."""
    with self.assertRaisesRegex(ValueError, 'not supported for CheaseConfig'):
      chease.CheaseConfig(
          geometry_file='iterhybrid.mat2cols',
          trapped_fraction_source=base.TrappedFractionSource.EXACT,
      )

  def test_trapped_fraction_is_loaded_from_file(self):
    """Tests that the exact trapped particle fraction is loaded from CHEASE."""
    geo = (
        chease.CheaseConfig.from_dict({
            'geometry_file': 'iterhybrid.mat2cols',
            'trapped_fraction_source': base.TrappedFractionSource.FILE,
        })
        .build_runtime_params(0.0)
        .geometry
    )
    trapped_fraction = geo.trapped_fraction_face
    self.assertIsNotNone(trapped_fraction)

  def test_chease_json_round_trip(self):
    """Tests JSON serialization and deserialization."""
    config = chease.CheaseConfig(geometry_file='iterhybrid.mat2cols')
    json_str = config.model_dump_json()
    restored = chease.CheaseConfig.model_validate_json(json_str)
    self.assertIsInstance(config.chease_data, chease.CheaseData)
    assert isinstance(config.chease_data, chease.CheaseData)
    self.assertIsInstance(restored.chease_data, chease.CheaseData)
    assert isinstance(restored.chease_data, chease.CheaseData)
    np.testing.assert_allclose(
        restored.chease_data.psi_chease, config.chease_data.psi_chease
    )

  def test_chease_multi_slice(self):
    """Tests multi-slice CHEASE configuration."""
    base_config = chease.CheaseConfig(geometry_file='iterhybrid.mat2cols')
    self.assertIsInstance(base_config.chease_data, chease.CheaseData)
    multi_config = chease.CheaseConfig(
        chease_data={
            0.0: base_config.chease_data,
            1.0: base_config.chease_data,
        }
    )
    geo_0 = multi_config.build_runtime_params(0.0).geometry
    geo_1 = multi_config.build_runtime_params(1.0).geometry
    self.assertIsInstance(geo_0, standard_geometry.StandardGeometry)
    self.assertIsInstance(geo_1, standard_geometry.StandardGeometry)

  def test_chease_build_runtime_params(self):
    """Tests runtime params creation."""
    config = chease.CheaseConfig(geometry_file='iterhybrid.mat2cols')
    runtime_params = config.build_runtime_params(t=0.0)
    self.assertIsInstance(
        runtime_params.geometry, standard_geometry.StandardGeometry
    )
    chex.assert_trees_all_close(
        runtime_params.geometry.rho_face_norm, config.get_face_centers()
    )


if __name__ == '__main__':
  absltest.main()
