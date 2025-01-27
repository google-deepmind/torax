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

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax.config import build_sim
from torax.geometry import geometry
from torax.geometry import geometry_loader
from torax.geometry import standard_geometry

# Internal import.

# pylint: disable=invalid-name


class GeometryTest(parameterized.TestCase):
  """Unit tests for the `geometry` module."""

  def test_standard_geometry_can_be_input_to_jitted_function(self):

    @jax.jit
    def foo(geo: geometry.Geometry):
      return geo.Rmaj

    intermediate = standard_geometry.StandardGeometryIntermediates(
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
    geo = standard_geometry.build_standard_geometry(intermediate)
    foo(geo)

  def test_build_geometry_from_chease(self):
    intermediate = standard_geometry.StandardGeometryIntermediates.from_chease()
    standard_geometry.build_standard_geometry(intermediate)

  def test_build_geometry_provider(self):
    intermediate_0 = standard_geometry.StandardGeometryIntermediates(
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
    geo_0 = standard_geometry.build_standard_geometry(intermediate_0)

    intermediate_1 = standard_geometry.StandardGeometryIntermediates(
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
    geo_1 = standard_geometry.build_standard_geometry(intermediate_1)

    provider = standard_geometry.StandardGeometryProvider.create_provider(
        {0.0: geo_0, 10.0: geo_1}
    )
    geo = provider(5.0)
    np.testing.assert_allclose(geo.Rmaj, 6.8)
    np.testing.assert_allclose(geo.Rmin, 1.5)
    np.testing.assert_allclose(geo.B0, 5.9)

  @parameterized.parameters([
      dict(geometry_file='eqdsk_cocos02.eqdsk'),
      dict(geometry_file='EQDSK_ITERhybrid_COCOS02.eqdsk'),
  ])
  def test_build_geometry_from_eqdsk(self, geometry_file):
    """Test that EQDSK geometries can be built."""
    intermediate = standard_geometry.StandardGeometryIntermediates.from_eqdsk(
        geometry_file=geometry_file
    )
    standard_geometry.build_standard_geometry(intermediate)

  def test_access_z_magnetic_axis_raises_error_for_chease_geometry(self):
    """Test that accessing z_magnetic_axis raises error for CHEASE geometry."""
    intermediate = standard_geometry.StandardGeometryIntermediates.from_chease()
    geo = standard_geometry.build_standard_geometry(intermediate)
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
  ])
  def test_validate_fbt_data_invalid_shape(self, invalid_key, invalid_shape):
    len_psinorm = 20
    len_times = 3

    L, LY = _get_example_L_LY_data(len_psinorm, len_times)

    LY[invalid_key] = np.zeros(invalid_shape)

    with self.assertRaisesRegex(ValueError, 'Incorrect shape'):
      standard_geometry._validate_fbt_data(LY, L)

  @parameterized.parameters(
      'rBt', 'aminor', 'rgeom', 'TQ', 'FB', 'FA', 'Q1Q', 'Q2Q', 'Q3Q', 'Q4Q',
      'Q5Q', 'ItQ', 'deltau', 'deltal', 'kappa', 'FtPQ', 'zA', 't',
  )
  def test_validate_fbt_data_missing_LY_key(self, missing_key):
    len_psinorm = 20
    len_times = 3

    L, LY = _get_example_L_LY_data(len_psinorm, len_times)
    del LY[missing_key]
    with self.assertRaisesRegex(ValueError, 'LY data is missing'):
      standard_geometry._validate_fbt_data(LY, L)

  def test_validate_fbt_data_missing_L_key(self):
    len_psinorm = 20
    len_times = 3
    L, LY = _get_example_L_LY_data(len_psinorm, len_times)
    del L['pQ']
    with self.assertRaisesRegex(ValueError, 'L data is missing'):
      standard_geometry._validate_fbt_data(LY, L)

  def test_validate_fbt_data_incorrect_L_pQ_shape(self):
    len_psinorm = 20
    len_times = 3
    L, LY = _get_example_L_LY_data(len_psinorm, len_times)
    L['pQ'] = np.zeros((len_psinorm + 1,))
    with self.assertRaisesRegex(ValueError, 'Incorrect shape'):
      standard_geometry._validate_fbt_data(LY, L)


def _get_example_L_LY_data(len_psinorm: int, len_times: int):
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
  return L, LY


if __name__ == '__main__':
  absltest.main()
