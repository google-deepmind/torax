import dataclasses

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
import numpy as np
from torax.geometry import circular_geometry
from torax.geometry import geometry
from torax.geometry import geometry_provider


class GeometryProviderTest(absltest.TestCase):

  def test_constant_geometry_return_same_value(self):
    geo = circular_geometry.build_circular_geometry()
    provider = geometry_provider.ConstantGeometryProvider(geo)
    self.assertEqual(provider(0.0), geo)
    self.assertEqual(provider(1.0), geo)
    self.assertEqual(provider(2.0), geo)

  def test_time_dependent_geometry_return_different_values(self):
    geo_0 = circular_geometry.build_circular_geometry(
        Rmaj=6.2, Rmin=2.0, B0=5.3
    )
    geo_1 = circular_geometry.build_circular_geometry(
        Rmaj=7.4, Rmin=1.0, B0=6.5
    )
    provider = geometry_provider.TimeDependentGeometryProvider.create_provider(
        {0.0: geo_0, 10.0: geo_1}
    )
    geo = provider(5.0)
    np.testing.assert_allclose(geo.Rmaj, 6.8)
    np.testing.assert_allclose(geo.Rmin, 1.5)
    np.testing.assert_allclose(geo.B0, 5.9)

  def test_time_dependent_different_types(self):
    geo_0 = circular_geometry.build_circular_geometry()
    geo_1 = dataclasses.replace(geo_0, geometry_type=geometry.GeometryType.FBT)
    with self.assertRaisesRegex(
        ValueError, "All geometries must have the same geometry type."
    ):
      geometry_provider.TimeDependentGeometryProvider.create_provider(
          {0.0: geo_0, 10.0: geo_1}
      )

  def test_time_dependent_different_meshes(self):
    geo_0 = circular_geometry.build_circular_geometry(n_rho=25)
    geo_1 = circular_geometry.build_circular_geometry(n_rho=50)
    with self.assertRaisesRegex(
        ValueError, "All geometries must have the same mesh."
    ):
      geometry_provider.TimeDependentGeometryProvider.create_provider(
          {0.0: geo_0, 10.0: geo_1}
      )

  def test_none_z_magnetic_axis_stays_none_time_dependent(self):
    geo = circular_geometry.build_circular_geometry()
    geo = dataclasses.replace(geo, _z_magnetic_axis=None)
    provider = geometry_provider.TimeDependentGeometryProvider.create_provider(
        {0.0: geo, 10.0: geo}
    )
    self.assertIsNone(provider(0.0)._z_magnetic_axis)
    self.assertIsNone(provider(10.0)._z_magnetic_axis)


if __name__ == "__main__":
  absltest.main()
