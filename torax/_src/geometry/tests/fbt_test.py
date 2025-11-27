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
        target_angle_of_incidence=np.array(3.0),
        R_OMP=np.array(8.2),
        R_target=np.array(7.0),
        B_pol_OMP=np.array(0.5),
    )
    geo = standard_geometry.build_standard_geometry(intermediate)
    self.assertTrue(geo.diverted)
    self.assertEqual(geo.connection_length_target, 10.0)
    self.assertEqual(geo.connection_length_divertor, 5.0)
    self.assertEqual(geo.target_angle_of_incidence, 3.0)
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

    L, LY = _get_example_L_LY_data(len_psinorm, len_times)

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
    L, LY = _get_example_L_LY_data(len_psinorm, len_times)
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
    L, LY = _get_example_L_LY_data(len_psinorm, len_times)
    del L['pQ']
    with self.assertRaisesRegex(ValueError, 'L data is missing'):
      fbt ._validate_fbt_data(LY, L)

  def test_validate_fbt_data_incorrect_L_pQ_shape(self):
    len_psinorm = 20
    len_times = 3
    L, LY = _get_example_L_LY_data(len_psinorm, len_times)
    L['pQ'] = np.zeros((len_psinorm + 1,))
    with self.assertRaisesRegex(ValueError, 'Incorrect shape'):
      fbt._validate_fbt_data(LY, L)


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
      'Q0Q': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'Q1Q': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'Q2Q': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'Q3Q': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'Q4Q': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'Q5Q': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'ItQ': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'deltau': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'deltal': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'kappa': np.full((len_psinorm, len_times), prefactor).squeeze(),
      'epsilon': np.full((len_psinorm, len_times), prefactor).squeeze(),
      # When prefactor != 0 (i.e. intended to generate a standard geometry),
      # needs to be linspace to avoid drho_norm = 0.
      'FtPQ': (
          np.array(
              [np.linspace(0, prefactor, len_psinorm) for _ in range(len_times)]
          ).squeeze()
      ),
      'zA': np.zeros(len_times).squeeze(),
      't': np.zeros(len_times).squeeze(),
      'lX': np.zeros(len_times, dtype=int).squeeze(),
  }
  L = {'pQ': np.linspace(0, 1, len_psinorm)}
  return L, LY


if __name__ == '__main__':
  absltest.main()
