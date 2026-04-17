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
from torax._src.geometry import chease
from torax._src.geometry import eqdsk
from torax._src.geometry import imas
from torax._src.geometry import standard_geometry

# pylint: disable=invalid-name


class IMASGeometryTest(parameterized.TestCase):

  def test_build_standard_geometry_from_IMAS(self):
    """Test that the default IMAS geometry can be built."""
    config = imas.IMASConfig(imas_filepath='ITERhybrid_COCOS17_IDS_ddv4.nc')
    config.build_geometry()

  def test_build_single_slice_geometry_from_IMAS(self):
    """Test that a single-slice IMAS geometry can be built."""
    config = imas.IMASConfig(
        imas_filepath='ITERhybrid_COCOS17_IDS_ddv4.nc',
        load_all_time_slices=False,
        slice_index=0,
    )
    geo = config.build_geometry()
    self.assertIsInstance(geo, standard_geometry.StandardGeometry)

  def test_single_slice_consistent_with_provider(self):
    """Test that single-slice geometry matches the all-slices provider at t=0."""
    single_slice_config = imas.IMASConfig(
        imas_filepath='ITERhybrid_COCOS17_IDS_ddv4.nc',
        load_all_time_slices=False,
        slice_index=0,
    )
    single_geo = single_slice_config.build_geometry()

    provider_config = imas.IMASConfig(
        imas_filepath='ITERhybrid_COCOS17_IDS_ddv4.nc',
    )
    provider = provider_config.build_geometry_provider(calcphibdot=False)
    provider_geo = provider(0.0)

    np.testing.assert_allclose(single_geo.gm4, provider_geo.gm4, rtol=1e-5)
    np.testing.assert_allclose(single_geo.gm5, provider_geo.gm5, rtol=1e-5)

  def test_gm4_gm5_terms(self):
    """Test that gm4 and gm5 are correctly computed."""
    eqdsk_geo = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos02.eqdsk',
        cocos=2,
    ).build_geometry()

    imas_geo = imas.IMASConfig(
        imas_filepath='ITERhybrid_COCOS17_IDS_ddv4.nc'
    ).build_geometry()

    chease_geo = chease.CheaseConfig(
        geometry_file='iterhybrid.mat2cols'
    ).build_geometry()

    np.testing.assert_allclose(eqdsk_geo.gm4, chease_geo.gm4, rtol=0.01)
    np.testing.assert_allclose(imas_geo.gm4, chease_geo.gm4, rtol=0.01)
    np.testing.assert_allclose(eqdsk_geo.gm5, chease_geo.gm5, rtol=0.02)
    np.testing.assert_allclose(imas_geo.gm5, chease_geo.gm5, rtol=0.01)


if __name__ == '__main__':
  absltest.main()
