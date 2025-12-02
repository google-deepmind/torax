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
import numpy as np
from torax._src.geometry import fbt
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.geometry import get_example_L_LY_data
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.geometry import standard_geometry

# Internal import.

# pylint: disable=invalid-name


class FBTGeometryTest(parameterized.TestCase):

  def test_edge_geometry_params_are_propagated(self):
    """Tests that edge geometry parameters are propagated to StandardGeometry."""
    intermediate = standard_geometry.StandardGeometryIntermediates(
        geometry_type=geometry.GeometryType.FBT,
        Ip_from_parameters=True,
        n_rho=25,
        R_major=6.2,
        a_minor=2.0,
        B_0=5.3,
        psi=np.arange(0, 1.0, 0.01),
        Ip_profile=np.arange(0, 1.0, 0.01),
        Phi=np.arange(0, 1.0, 0.01),
        R_in=np.arange(1, 2, 0.01),
        R_out=np.arange(1, 2, 0.01),
        F=np.arange(0, 1.0, 0.01),
        int_dl_over_Bp=np.arange(0, 1.0, 0.01),
        flux_surf_avg_1_over_R=np.arange(0, 1.0, 0.01),
        flux_surf_avg_1_over_R2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_grad_psi2_over_R2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_grad_psi=np.arange(0, 1.0, 0.01),
        flux_surf_avg_grad_psi2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_B2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_1_over_B2=np.arange(0, 1.0, 0.01),
        delta_upper_face=np.arange(0, 1.0, 0.01),
        delta_lower_face=np.arange(0, 1.0, 0.01),
        elongation=np.arange(0, 1.0, 0.01),
        vpr=np.arange(0, 1.0, 0.01),
        hires_factor=4,
        z_magnetic_axis=np.array(0.0),
        diverted=True,
        connection_length_target=np.array(10.0),
        connection_length_divertor=np.array(5.0),
        angle_of_incidence_target=np.array(3.0),
        R_OMP=np.array(8.2),
        R_target=np.array(7.0),
        B_pol_OMP=np.array(0.5),
    )
    geo = standard_geometry.build_standard_geometry(intermediate)
    self.assertTrue(geo.diverted)
    self.assertEqual(geo.connection_length_target, 10.0)
    self.assertEqual(geo.connection_length_divertor, 5.0)
    self.assertEqual(geo.angle_of_incidence_target, 3.0)
    self.assertEqual(geo.R_OMP, 8.2)
    self.assertEqual(geo.R_target, 7.0)
    self.assertEqual(geo.B_pol_OMP, 0.5)

  @parameterized.parameters([
      dict(invalid_key='rBt', invalid_shape=(2,)),
      dict(invalid_key='aminor', invalid_shape=(10, 3)),
      dict(invalid_key='rgeom', invalid_shape=(10, 2)),
      dict(invalid_key='epsilon', invalid_shape=(20, 2)),
      dict(invalid_key='TQ', invalid_shape=(20, 2)),
      dict(invalid_key='FB', invalid_shape=(2,)),
      dict(invalid_key='FA', invalid_shape=(2,)),
      dict(invalid_key='Q0Q', invalid_shape=(10, 3)),
      dict(invalid_key='Q1Q', invalid_shape=(10, 3)),
      dict(invalid_key='Q2Q', invalid_shape=(10, 2)),
      dict(invalid_key='Q3Q', invalid_shape=(10, 3)),
      dict(invalid_key='Q4Q', invalid_shape=(10, 2)),
      dict(invalid_key='Q5Q', invalid_shape=(20, 2)),
      dict(invalid_key='ItQ', invalid_shape=(10, 3)),
      dict(invalid_key='deltau', invalid_shape=(10, 3)),
      dict(invalid_key='deltal', invalid_shape=(10, 3)),
      dict(invalid_key='kappa', invalid_shape=(10, 3)),
      dict(invalid_key='FtPVQ', invalid_shape=(20, 2)),
      dict(invalid_key='zA', invalid_shape=(2,)),
      dict(invalid_key='t', invalid_shape=(2,)),
  ])
  def test_validate_fbt_data_invalid_shape(self, invalid_key, invalid_shape):
    len_psinorm = 20
    len_times = 3

    L, LY = get_example_L_LY_data.get_example_L_LY_data(len_psinorm, len_times)

    LY[invalid_key] = np.zeros(invalid_shape)

    with self.assertRaisesRegex(ValueError, 'Incorrect shape'):
      fbt._validate_fbt_data(LY, L)

  @parameterized.parameters(
      'rBt',
      'aminor',
      'rgeom',
      'epsilon',
      'TQ',
      'FB',
      'FA',
      'Q0Q',
      'Q1Q',
      'Q2Q',
      'Q3Q',
      'Q4Q',
      'Q5Q',
      'ItQ',
      'deltau',
      'deltal',
      'kappa',
      'FtPQ',  # TODO(b/412965439)  remove support for LY files w/o FtPVQ.
      'zA',
      't',
      'lX',
  )
  def test_validate_fbt_data_missing_LY_key(self, missing_key):
    len_psinorm = 20
    len_times = 3
    L, LY = get_example_L_LY_data.get_example_L_LY_data(len_psinorm, len_times)
    del LY[missing_key]

    if missing_key == 'FtPQ':
      errmsg = 'LY data is missing a toroidal flux-related key'
    else:
      errmsg = 'LY data is missing the'

    with self.assertRaisesRegex(ValueError, errmsg):
      fbt._validate_fbt_data(LY, L)

  def test_validate_fbt_data_missing_L_key(self):
    len_psinorm = 20
    len_times = 3
    L, LY = get_example_L_LY_data.get_example_L_LY_data(len_psinorm, len_times)
    del L['pQ']
    with self.assertRaisesRegex(ValueError, 'L data is missing'):
      fbt._validate_fbt_data(LY, L)

  def test_validate_fbt_data_incorrect_L_pQ_shape(self):
    len_psinorm = 20
    len_times = 3
    L, LY = get_example_L_LY_data.get_example_L_LY_data(len_psinorm, len_times)
    L['pQ'] = np.zeros((len_psinorm + 1,))
    with self.assertRaisesRegex(ValueError, 'Incorrect shape'):
      fbt._validate_fbt_data(LY, L)

  @parameterized.named_parameters(
      (
          'lower_null',
          fbt.DivertorDomain.LOWER_NULL,
          np.array([10.0, 20.0]),
          np.array([1.0, 2.0]),
          np.array([3.0, 4.0]),
          np.array([6.0, 7.0]),
          np.array([5.0, 6.0]),
          np.array([0.5, 0.7]),
      ),
      (
          'upper_null',
          fbt.DivertorDomain.UPPER_NULL,
          np.array([15.0, 25.0]),
          np.array([1.5, 2.5]),
          np.array([3.5, 4.5]),
          np.array([6.5, 7.5]),
          np.array([5.5, 6.5]),
          np.array([0.6, 0.8]),
      ),
  )
  def test_fbt_edge_parameters(
      self,
      divertor_domain,
      expected_connection_length_target,
      expected_connection_length_divertor,
      expected_angle_of_incidence_target,
      expected_r_omp,
      expected_r_target,
      expected_bp_omp,
  ):
    len_psinorm = 20
    len_times = 2
    L, LY = get_example_L_LY_data.get_example_L_LY_data(len_psinorm, len_times)

    # Add edge parameters to LY: (n_domains, n_times)
    # Rows: domains, Cols: time.
    # Domain 0 (Lower): [10, 20] over time for Lpar_target.
    # Domain 1 (Upper): [15, 25] over time for Lpar_target.
    LY['Lpar_target'] = np.array([[10.0, 20.0], [15.0, 25.0]])
    LY['Lpar_div'] = np.array([[1.0, 2.0], [1.5, 2.5]])
    LY['alpha_target'] = np.array([[3.0, 4.0], [3.5, 4.5]])
    LY['r_OMP'] = np.array([[6.0, 7.0], [6.5, 7.5]])
    LY['r_target'] = np.array([[5.0, 6.0], [5.5, 6.5]])
    LY['Bp_OMP'] = np.array([[0.5, 0.7], [0.6, 0.8]])

    # z_div to distinguish nulls.
    # Index 0: lower null (<0), Index 1: upper null (>0) for ALL times.
    LY['z_div'] = np.array([[-1.0, -1.0], [1.2, 1.2]])

    # Set diverted flag to true to trigger the domain selection logic.
    LY['lX'] = np.ones(len_times, dtype=int)

    geo_intermediates = fbt._from_fbt_bundle(
        geometry_directory=None,
        LY_bundle_object=LY,
        LY_to_torax_times=np.array([0.0, 1.0]),
        L_object=L,
        divertor_domain=divertor_domain,
    )

    # from_fbt_bundle returns a dictionary of StandardGeometryIntermediates
    # objects, so we need to extract the values for each time slice.
    intermediates_list = list(geo_intermediates.values())

    np.testing.assert_allclose(
        [i.connection_length_target for i in intermediates_list],
        expected_connection_length_target,
    )
    np.testing.assert_allclose(
        [i.connection_length_divertor for i in intermediates_list],
        expected_connection_length_divertor,
    )
    np.testing.assert_allclose(
        [i.angle_of_incidence_target for i in intermediates_list],
        expected_angle_of_incidence_target,
    )
    np.testing.assert_allclose(
        [i.R_OMP for i in intermediates_list], expected_r_omp
    )
    np.testing.assert_allclose(
        [i.R_target for i in intermediates_list], expected_r_target
    )
    np.testing.assert_allclose(
        [i.B_pol_OMP for i in intermediates_list], expected_bp_omp
    )

  def test_fbt_edge_parameters_missing_edge_parameters(self):
    len_psinorm = 20
    len_times = 2
    L, LY = get_example_L_LY_data.get_example_L_LY_data(len_psinorm, len_times)
    geo_intermediates_no_z = fbt._from_fbt_bundle(
        geometry_directory=None,
        LY_bundle_object=LY,
        LY_to_torax_times=np.array([0.0, 1.0]),
        L_object=L,
    )
    for intermediate in geo_intermediates_no_z.values():
      self.assertIsNone(intermediate.connection_length_target)
      self.assertIsNone(intermediate.connection_length_divertor)
      self.assertIsNone(intermediate.angle_of_incidence_target)
      self.assertIsNone(intermediate.R_OMP)
      self.assertIsNone(intermediate.R_target)
      self.assertIsNone(intermediate.B_pol_OMP)

  def test_fbt_edge_parameters_bad_domain_request(self):
    len_psinorm = 20
    len_times = 2
    L, LY = get_example_L_LY_data.get_example_L_LY_data(len_psinorm, len_times)
    LY['z_div'] = np.array([[1.0, 1.2], [2.0, 2.2]])  # All upper null.
    LY['lX'] = np.ones(len_times, dtype=int).squeeze()

    with self.assertRaisesRegex(
        ValueError, 'not present in edge geometry data'
    ):
      fbt._from_fbt_bundle(
          geometry_directory=None,
          LY_bundle_object=LY,
          LY_to_torax_times=np.array([0.0, 1.0]),
          L_object=L,
          divertor_domain=fbt.DivertorDomain.LOWER_NULL,
      )


if __name__ == '__main__':
  absltest.main()
