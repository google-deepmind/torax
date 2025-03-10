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
from torax.geometry import geometry
from torax.geometry import geometry_loader
from torax.geometry import pydantic_model as geometry_pydantic_model
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
        Rin=np.arange(1, 2, 0.01),
        Rout=np.arange(1, 2, 0.01),
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
    geometry_pydantic_model.CheaseConfig().build_geometry()

  @parameterized.parameters([
      dict(geometry_file='eqdsk_cocos02.eqdsk'),
      dict(geometry_file='EQDSK_ITERhybrid_COCOS02.eqdsk'),
  ])
  def test_build_geometry_from_eqdsk(self, geometry_file):
    """Test that EQDSK geometries can be built."""
    config = geometry_pydantic_model.EQDSKConfig(geometry_file=geometry_file)
    config.build_geometry()

  def test_access_z_magnetic_axis_raises_error_for_chease_geometry(self):
    """Test that accessing z_magnetic_axis raises error for CHEASE geometry."""
    geo = geometry_pydantic_model.CheaseConfig().build_geometry()
    with self.assertRaisesRegex(ValueError, 'does not have a z magnetic axis'):
      geo.z_magnetic_axis()

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
      'rBt',
      'aminor',
      'rgeom',
      'TQ',
      'FB',
      'FA',
      'Q1Q',
      'Q2Q',
      'Q3Q',
      'Q4Q',
      'Q5Q',
      'ItQ',
      'deltau',
      'deltal',
      'kappa',
      'FtPQ',
      'zA',
      't',
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

  def test_stack_geometries_standard_geometries(self):
    """Test stack_geometries for standard geometries."""
    # Create a few different geometries
    L, LY0 = _get_example_L_LY_data(10, 1, prefactor=1.0)
    _, LY1 = _get_example_L_LY_data(10, 1, prefactor=2.0)
    _, LY2 = _get_example_L_LY_data(10, 1, prefactor=3.0)
    geo0_intermediate = (
        standard_geometry.StandardGeometryIntermediates.from_fbt_single_slice(
            geometry_dir=None, LY_object=LY0, L_object=L
        )
    )
    geo1_intermediate = (
        standard_geometry.StandardGeometryIntermediates.from_fbt_single_slice(
            geometry_dir=None, LY_object=LY1, L_object=L
        )
    )
    geo2_intermediate = (
        standard_geometry.StandardGeometryIntermediates.from_fbt_single_slice(
            geometry_dir=None, LY_object=LY2, L_object=L
        )
    )
    geo0 = standard_geometry.build_standard_geometry(geo0_intermediate)
    geo1 = standard_geometry.build_standard_geometry(geo1_intermediate)
    geo2 = standard_geometry.build_standard_geometry(geo2_intermediate)

    # Stack them
    stacked_geo = geometry.stack_geometries([geo0, geo1, geo2])

    # Check that the stacked geometry has the correct type and mesh
    self.assertEqual(stacked_geo.geometry_type, geo0.geometry_type)
    self.assertEqual(stacked_geo.torax_mesh, geo0.torax_mesh)

    # Check some specific stacked values
    np.testing.assert_allclose(
        stacked_geo.g2[:, -1],
        np.array([geo0.g2[-1], geo1.g2[-1], geo2.g2[-1]]),
    )
    np.testing.assert_allclose(
        stacked_geo.Ip_profile_face[:, -1],
        np.array([
            geo0.Ip_profile_face[-1],
            geo1.Ip_profile_face[-1],
            geo2.Ip_profile_face[-1],
        ]),
    )

    # Check stacking of derived properties
    np.testing.assert_allclose(
        stacked_geo.rho_b, np.array([geo0.rho_b, geo1.rho_b, geo2.rho_b])
    )

    # Check a property that depends on a stacked property (rho depends on rho_b)
    # Note that rho_norm is the same for all geometries.
    np.testing.assert_allclose(
        stacked_geo.rho,
        np.array([
            geo0.rho_norm * geo0.rho_b,
            geo0.rho_norm * geo1.rho_b,
            geo0.rho_norm * geo2.rho_b,
        ]),
    )

    # Check properties with special handling for on-axis values.
    np.testing.assert_allclose(
        stacked_geo.g0_over_vpr_face[:, 0], 1 / stacked_geo.rho_b
    )
    np.testing.assert_allclose(
        stacked_geo.g1_over_vpr2_face[:, 0], 1 / stacked_geo.rho_b**2
    )


def _get_example_L_LY_data(
    len_psinorm: int, len_times: int, prefactor: float = 0.0
):
  LY = {  # Squeeze when intended for a single time slice.
      'rBt': np.full(len_times, prefactor).squeeze(),
      'aminor': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'rgeom': np.full((len_psinorm, len_times), 2.0 * prefactor).squeeze(),
      'TQ': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'FB': np.full(len_times, prefactor).squeeze(),
      'FA': np.full(len_times, prefactor).squeeze(),
      'Q1Q': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'Q2Q': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'Q3Q': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'Q4Q': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'Q5Q': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'ItQ': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'deltau': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'deltal': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'kappa': np.full((len_psinorm, len_times), prefactor).squeeze(),
      # When prefactor != 0 (i.e. intended to generate a standard geometry),
      # needs to be linspace to avoid drho_norm = 0.
      'FtPQ': np.array(
          [np.linspace(0, prefactor, len_psinorm) for _ in range(len_times)]
      ).squeeze(),
      'zA': np.zeros(len_times).squeeze(),
      't': np.zeros(len_times).squeeze(),
  }
  L = {'pQ': np.linspace(0, 1, len_psinorm)}
  return L, LY


if __name__ == '__main__':
  absltest.main()
