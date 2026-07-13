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
import numpy as np
from torax._src.geometry import base
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

  def test_trapped_fraction_is_physically_sensible(self):
    """Tests that the exact trapped particle fraction (FTRAP) is sensible."""
    geo = chease.CheaseConfig(
        trapped_fraction_source=base.TrappedFractionSource.FILE,
    ).build_geometry()
    trapped_fraction = geo.trapped_fraction_face
    self.assertTrue(np.all(trapped_fraction >= 0.0))
    self.assertTrue(np.all(trapped_fraction <= 1.0))
    self.assertGreater(
        np.mean(np.diff(trapped_fraction) >= -1e-6),
        0.8,
    )

  def test_trapped_fraction_source_exact_not_supported(self):
    """Tests that EXACT is rejected for CHEASE (no full 2D equilibrium)."""
    with self.assertRaisesRegex(ValueError, 'not supported for CheaseConfig'):
      chease.CheaseConfig(
          trapped_fraction_source=base.TrappedFractionSource.EXACT,
      )

  def test_trapped_fraction_geometry_consistent_with_sauter(self):
    """Tests that the exact and Sauter trapped fractions roughly agree."""
    geo_sauter = chease.CheaseConfig(
        trapped_fraction_source=base.TrappedFractionSource.SAUTER,
    ).build_geometry()
    geo_geometry = chease.CheaseConfig(
        trapped_fraction_source=base.TrappedFractionSource.FILE,
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


if __name__ == '__main__':
  absltest.main()
