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

"""Unit tests for torax.geometry."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax import geometry


class GeometryTest(parameterized.TestCase):
  """Unit tests for the `geometry` module."""

  @parameterized.parameters([
      dict(n_rho=25, seed=20220930),
  ])
  def test_face_to_cell(self, n_rho, seed):
    """Compare `face_to_cell` to a PINT reference."""

    rng_state = jax.random.PRNGKey(seed)
    del seed  # Make sure seed isn't accidentally re-used

    # Generate face variables
    rng_use, _ = jax.random.split(rng_state)
    face = jax.random.normal(rng_use, (n_rho + 1,))
    del rng_use  # Make sure rng_use isn't accidentally re-used

    # Convert face values to cell values using jax code being tested
    cell_jax = geometry.face_to_cell(jnp.array(face))

    # Make ground truth
    cell_np = face_to_cell(n_rho, face)

    np.testing.assert_allclose(cell_jax, cell_np)

  def test_frozen(self):
    """Test that the Geometry class is frozen."""
    geo = geometry.build_circular_geometry()
    with self.assertRaises(dataclasses.FrozenInstanceError):
      geo.drho_norm = 0.1

  def test_circular_geometry_can_be_input_to_jitted_function(self):
    """Test that a circular geometry can be input to a jitted function."""

    def foo(geo: geometry.Geometry):
      _ = geo  # do nothing.

    foo_jitted = jax.jit(foo)
    geo = geometry.build_circular_geometry()
    # Make sure you can call the function with geo as an arg.
    foo_jitted(geo)

  def test_standard_geometry_can_be_input_to_jitted_function(self):
    """Test that a StandardGeometry can be input to a jitted function."""

    def foo(geo: geometry.Geometry):
      _ = geo  # do nothing.

    foo_jitted = jax.jit(foo)
    intermediate = geometry.StandardGeometryIntermediates(
        Ip_from_parameters=True,
        n_rho=25,
        Rmaj=6.2,
        Rmin=2.0,
        B=5.3,
        # Use the same dummy value for the rest.
        psi=np.arange(0, 1.0, 0.01),
        Ip_profile=np.arange(0, 1.0, 0.01),
        Phi=np.arange(0, 1.0, 0.01),
        Rin=np.arange(0, 1.0, 0.01),
        Rout=np.arange(0, 1.0, 0.01),
        F=np.arange(0, 1.0, 0.01),
        int_dl_over_Bp=np.arange(0, 1.0, 0.01),
        flux_surf_avg_1_over_R2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_Bp2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_RBp=np.arange(0, 1.0, 0.01),
        flux_surf_avg_R2Bp2=np.arange(0, 1.0, 0.01),
        delta_upper_face=np.arange(0, 1.0, 0.01),
        delta_lower_face=np.arange(0, 1.0, 0.01),
        vpr=np.arange(0, 1.0, 0.01),
        hires_fac=4,
    )
    geo = geometry.build_standard_geometry(intermediate)
    foo_jitted(geo)

  def test_build_geometry_from_chease(self):
    """Test that the default CHEASE geometry can be built."""
    intermediate = geometry.StandardGeometryIntermediates.from_chease()
    geometry.build_standard_geometry(intermediate)

  def test_build_geometry_provider(self):
    """Test that the default geometry provider can be built."""
    intermediate_0 = geometry.StandardGeometryIntermediates(
        Ip_from_parameters=True,
        n_rho=25,
        Rmaj=6.2,
        Rmin=2.0,
        B=5.3,
        # Use the same dummy value for the rest.
        psi=np.arange(0, 1.0, 0.01),
        Ip_profile=np.arange(0, 1.0, 0.01),
        Phi=np.arange(0, 1.0, 0.01),
        Rin=np.arange(0, 1.0, 0.01),
        Rout=np.arange(0, 1.0, 0.01),
        F=np.arange(0, 1.0, 0.01),
        int_dl_over_Bp=np.arange(0, 1.0, 0.01),
        flux_surf_avg_1_over_R2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_Bp2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_RBp=np.arange(0, 1.0, 0.01),
        flux_surf_avg_R2Bp2=np.arange(0, 1.0, 0.01),
        delta_upper_face=np.arange(0, 1.0, 0.01),
        delta_lower_face=np.arange(0, 1.0, 0.01),
        vpr=np.arange(0, 1.0, 0.01),
        hires_fac=4,
    )
    geo_0 = geometry.build_standard_geometry(intermediate_0)

    intermediate_1 = geometry.StandardGeometryIntermediates(
        Ip_from_parameters=True,
        n_rho=25,
        Rmaj=7.4,
        Rmin=1.0,
        B=6.5,
        # Use the same dummy value for the rest.
        psi=np.arange(0, 1.0, 0.01),
        Ip_profile=np.arange(0, 2.0, 0.02),
        Phi=np.arange(0, 1.0, 0.01),
        Rin=np.arange(0, 1.0, 0.01),
        Rout=np.arange(0, 1.0, 0.01),
        F=np.arange(0, 1.0, 0.01),
        int_dl_over_Bp=np.arange(0, 1.0, 0.01),
        flux_surf_avg_1_over_R2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_Bp2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_RBp=np.arange(0, 1.0, 0.01),
        flux_surf_avg_R2Bp2=np.arange(0, 1.0, 0.01),
        delta_upper_face=np.arange(0, 1.0, 0.01),
        delta_lower_face=np.arange(0, 1.0, 0.01),
        vpr=np.arange(0, 2.0, 0.02),
        hires_fac=4,
    )
    geo_1 = geometry.build_standard_geometry(intermediate_1)

    provider = geometry.StandardGeometryProvider.create_provider(
        {0.: geo_0, 10.: geo_1})
    geo = provider(5.)
    np.testing.assert_allclose(geo.Rmaj, 6.8)
    np.testing.assert_allclose(geo.Rmin, 1.5)
    np.testing.assert_allclose(geo.B0, 5.9)

  def test_build_geometry_provider_from_circular(self):
    """Test that the circular geometry provider can be built."""
    geo_0 = geometry.build_circular_geometry(
        n_rho=25,
        kappa=1.72,
        Rmaj=6.2,
        Rmin=2.0,
        B0=5.3,
        hires_fac=4,
    )
    geo_1 = geometry.build_circular_geometry(
        n_rho=25,
        kappa=1.72,
        Rmaj=7.2,
        Rmin=1.0,
        B0=5.3,
        hires_fac=4,
    )
    provider = geometry.CircularAnalyticalGeometryProvider.create_provider(
        {0.: geo_0, 10.: geo_1})
    geo = provider(5.)
    np.testing.assert_allclose(geo.Rmaj, 6.7)
    np.testing.assert_allclose(geo.Rmin, 1.5)


def face_to_cell(n_rho, face):
  cell = np.zeros(n_rho)
  cell[:] = 0.5 * (face[1:] + face[:-1])
  return cell


if __name__ == '__main__':
  absltest.main()
