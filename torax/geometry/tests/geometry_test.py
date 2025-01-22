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
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax.config import build_sim
from torax.geometry import geometry
from torax.geometry import geometry_loader

# Internal import.


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
        geometry_type=geometry.GeometryType.CIRCULAR,
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
        elongation=np.arange(0, 1.0, 0.01),
        vpr=np.arange(0, 1.0, 0.01),
        hires_fac=4,
        z_magnetic_axis=np.array(0.0),
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
        geometry_type=geometry.GeometryType.CIRCULAR,
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
        elongation=np.arange(0, 1.0, 0.01),
        vpr=np.arange(0, 1.0, 0.01),
        hires_fac=4,
        z_magnetic_axis=np.array(0.0),
    )
    geo_0 = geometry.build_standard_geometry(intermediate_0)

    intermediate_1 = geometry.StandardGeometryIntermediates(
        geometry_type=geometry.GeometryType.CIRCULAR,
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
        elongation=np.arange(0, 1.0, 0.01),
        vpr=np.arange(0, 2.0, 0.02),
        hires_fac=4,
        z_magnetic_axis=np.array(0.0),
    )
    geo_1 = geometry.build_standard_geometry(intermediate_1)

    provider = geometry.StandardGeometryProvider.create_provider(
        {0.0: geo_0, 10.0: geo_1}
    )
    geo = provider(5.0)
    np.testing.assert_allclose(geo.Rmaj, 6.8)
    np.testing.assert_allclose(geo.Rmin, 1.5)
    np.testing.assert_allclose(geo.B0, 5.9)

  def test_build_geometry_provider_from_circular(self):
    """Test that the circular geometry provider can be built."""
    geo_0 = geometry.build_circular_geometry(
        n_rho=25,
        elongation_LCFS=1.72,
        Rmaj=6.2,
        Rmin=2.0,
        B0=5.3,
        hires_fac=4,
    )
    geo_1 = geometry.build_circular_geometry(
        n_rho=25,
        elongation_LCFS=1.72,
        Rmaj=7.2,
        Rmin=1.0,
        B0=5.3,
        hires_fac=4,
    )
    provider = geometry.CircularAnalyticalGeometryProvider.create_provider(
        {0.0: geo_0, 10.0: geo_1}
    )
    geo = provider(5.0)
    np.testing.assert_allclose(geo.Rmaj, 6.7)
    np.testing.assert_allclose(geo.Rmin, 1.5)

  @parameterized.parameters([
      dict(invalid_key='rBt', invalid_shape=(2,)),
      dict(invalid_key='aminor', invalid_shape=(10, 3)),
      dict(invalid_key='rgeom', invalid_shape=(10, 2)),
      dict(invalid_key='TQ', invalid_shape=(20, 2)),
      dict(invalid_key='FB', invalid_shape=(2,)),
      dict(invalid_key='FA', invalid_shape=(2,)),
      dict(invalid_key='Q1Q', invalid_shape=(10, 3)),
      dict(invalid_key='Q2Q', invalid_shape=(10, 2)),
      dict(invalid_key='Q3Q', invalid_shape=(10, 3)),
      dict(invalid_key='Q4Q', invalid_shape=(10, 2)),
      dict(invalid_key='Q5Q', invalid_shape=(20, 2)),
      dict(invalid_key='ItQ', invalid_shape=(10, 3)),
      dict(invalid_key='deltau', invalid_shape=(10, 3)),
      dict(invalid_key='deltal', invalid_shape=(10, 3)),
      dict(invalid_key='kappa', invalid_shape=(10, 3)),
      dict(invalid_key='FtPQ', invalid_shape=(20, 2)),
      dict(invalid_key='zA', invalid_shape=(2,)),
      dict(invalid_key='t', invalid_shape=(2,)),
      dict(missing_key='rBt'),
      dict(missing_key='aminor'),
      dict(missing_key='rgeom'),
      dict(missing_key='TQ'),
      dict(missing_key='FB'),
      dict(missing_key='FA'),
      dict(missing_key='Q1Q'),
      dict(missing_key='Q2Q'),
      dict(missing_key='Q3Q'),
      dict(missing_key='Q4Q'),
      dict(missing_key='Q5Q'),
      dict(missing_key='ItQ'),
      dict(missing_key='deltau'),
      dict(missing_key='deltal'),
      dict(missing_key='kappa'),
      dict(missing_key='FtPQ'),
      dict(missing_key='zA'),
      dict(missing_key='t'),
      dict(len_pq=10),
  ])
  def test_validate_fbt_data(
      self,
      invalid_key=None,
      invalid_shape=None,
      missing_key=None,
      len_pq=None,
  ):
    """Tests _validate_fbt_data in geometry.py."""

    # create dummy LY and L data dictionaries.
    # len_times is initialized as 3 (corresponding to three time slices).
    # len_psinorm is initialized as 100.
    len_psinorm = 20
    len_times = 3
    # pylint: disable=invalid-name
    LY = {
        'rBt': np.zeros(len_times),
        'aminor': np.zeros((len_psinorm, len_times)),
        'rgeom': np.zeros((len_psinorm, len_times)),
        'TQ': np.zeros((len_psinorm, len_times)),
        'FB': np.zeros(len_times),
        'FA': np.zeros(len_times),
        'Q1Q': np.zeros((len_psinorm, len_times)),
        'Q2Q': np.zeros((len_psinorm, len_times)),
        'Q3Q': np.zeros((len_psinorm, len_times)),
        'Q4Q': np.zeros((len_psinorm, len_times)),
        'Q5Q': np.zeros((len_psinorm, len_times)),
        'ItQ': np.zeros((len_psinorm, len_times)),
        'deltau': np.zeros((len_psinorm, len_times)),
        'deltal': np.zeros((len_psinorm, len_times)),
        'kappa': np.zeros((len_psinorm, len_times)),
        'FtPQ': np.zeros((len_psinorm, len_times)),
        'zA': np.zeros(len_times),
        't': np.zeros(len_times),
    }
    L = {'pQ': np.zeros(len_psinorm)}
    # pylint: enable=invalid-name

    if invalid_key:
      LY[invalid_key] = np.zeros(invalid_shape)
    if missing_key:
      del LY[missing_key]
    if len_pq:
      L['pQ'] = np.zeros(len_pq)

    with self.assertRaises(ValueError):
      geometry._validate_fbt_data(LY, L)  # pylint: disable=protected-access

  @parameterized.parameters([
      dict(geometry_file='eqdsk_cocos02.eqdsk'),
      dict(geometry_file='EQDSK_ITERhybrid_COCOS02.eqdsk'),
  ])
  def test_build_geometry_from_eqdsk(self, geometry_file):
    """Test that EQDSK geometries can be built."""
    intermediate = geometry.StandardGeometryIntermediates.from_eqdsk(
        geometry_file=geometry_file
    )
    geometry.build_standard_geometry(intermediate)

  def test_geometry_objects_can_be_used_in_jax_jitted_functions(self):
    """Test public API of geometry objects can be used in jitted functions."""
    geo = geometry.build_circular_geometry()

    @jax.jit
    def f(geo: geometry.Geometry):
      for field in dir(geo):
        if not field.startswith('_'):
          getattr(geo, field)

    f(geo)

  def test_access_z_magnetic_axis_raises_error_for_chease_geometry(self):
    """Test that accessing z_magnetic_axis raises error for CHEASE geometry."""
    intermediate = geometry.StandardGeometryIntermediates.from_chease()
    geo = geometry.build_standard_geometry(intermediate)
    # Check that a runtime error is raised under both JIT and non-JIT.
    with self.assertRaisesRegex(
        RuntimeError, 'does not have a z magnetic axis'
    ):
      _ = geo.z_magnetic_axis
    with self.assertRaisesRegex(
        RuntimeError, 'does not have a z magnetic axis'
    ):

      def f():
        return geo.z_magnetic_axis

      _ = jax.jit(f)()


def face_to_cell(n_rho, face):
  cell = np.zeros(n_rho)
  cell[:] = 0.5 * (face[1:] + face[:-1])
  return cell


if __name__ == '__main__':
  absltest.main()
