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
import jax
import numpy as np
from torax.geometry import circular_geometry
from torax.geometry import geometry
from torax.geometry import geometry_provider


class CircularGeometryTest(absltest.TestCase):

  def test_build_geometry_provider_from_circular(self):
    """Test that the circular geometry provider can be built."""
    geo_0 = circular_geometry.build_circular_geometry(
        n_rho=25,
        elongation_LCFS=1.72,
        Rmaj=6.2,
        Rmin=2.0,
        B0=5.3,
        hires_fac=4,
    )
    geo_1 = circular_geometry.build_circular_geometry(
        n_rho=25,
        elongation_LCFS=1.72,
        Rmaj=7.2,
        Rmin=1.0,
        B0=5.3,
        hires_fac=4,
    )
    provider = geometry_provider.TimeDependentGeometryProvider.create_provider(
        {0.0: geo_0, 10.0: geo_1}
    )
    geo = provider(5.0)
    np.testing.assert_allclose(geo.Rmaj, 6.7)
    np.testing.assert_allclose(geo.Rmin, 1.5)

  def test_circular_geometry_can_be_input_to_jitted_function(self):

    @jax.jit
    def foo(geo: geometry.Geometry):
      return geo.Rmaj

    geo = circular_geometry.build_circular_geometry(
        n_rho=25,
        elongation_LCFS=1.72,
        Rmaj=6.2,
        Rmin=2.0,
        B0=5.3,
        hires_fac=4,
    )
    # Make sure you can call the function with geo as an arg.
    foo(geo)


if __name__ == "__main__":
  absltest.main()
