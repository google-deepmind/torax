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
from torax.config import build_runtime_params
from torax.config import runtime_params as runtime_params_lib
from torax.geometry import geometry_provider
from torax.geometry import pydantic_model
from torax.geometry import standard_geometry
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.transport_model import runtime_params as transport_model_params


class PydanticModelTest(parameterized.TestCase):

  def test_missing_geometry_type_raises_error(self):
    with self.assertRaisesRegex(
        ValueError, 'geometry_type must be set in the input config'
    ):
      pydantic_model.Geometry.from_dict({})

  def test_build_circular_geometry(self):
    geo_provider = pydantic_model.Geometry.from_dict({
        'geometry_type': 'circular',
        'n_rho': 5,  # override a default.
    }).build_provider

    self.assertIsInstance(
        geo_provider, geometry_provider.ConstantGeometryProvider
    )
    geo = geo_provider(t=0)
    np.testing.assert_array_equal(geo_provider.torax_mesh.nx, 5)
    np.testing.assert_array_equal(geo.B0, 5.3)  # test a default.

  def test_build_geometry_from_chease(self):
    geo_provider = pydantic_model.Geometry.from_dict(
        {
            'geometry_type': 'chease',
            'n_rho': 5,  # override a default.
        },
    ).build_provider
    self.assertIsInstance(
        geo_provider, geometry_provider.ConstantGeometryProvider
    )
    self.assertIsInstance(geo_provider(t=0), standard_geometry.StandardGeometry)
    np.testing.assert_array_equal(geo_provider.torax_mesh.nx, 5)

  def test_build_time_dependent_geometry_from_chease(self):
    """Tests correctness of config constraints with time-dependent geometry."""

    base_config = {
        'geometry_type': 'chease',
        'Ip_from_parameters': True,
        'n_rho': 10,  # overrides the default
        'geometry_configs': {
            0.0: {
                'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
                'Rmaj': 6.2,
                'Rmin': 2.0,
                'B0': 5.3,
            },
            1.0: {
                'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
                'Rmaj': 6.2,
                'Rmin': 2.0,
                'B0': 5.3,
            },
        },
    }

    # Test valid config
    geo_provider = pydantic_model.Geometry.from_dict(base_config).build_provider
    self.assertIsInstance(
        geo_provider, standard_geometry.StandardGeometryProvider
    )
    self.assertIsInstance(geo_provider(t=0), standard_geometry.StandardGeometry)
    np.testing.assert_array_equal(geo_provider.torax_mesh.nx, 10)

  @parameterized.parameters([
      dict(param='n_rho', value=5),
      dict(param='Ip_from_parameters', value=True),
      dict(param='geometry_dir', value='.'),
  ])
  def test_build_time_dependent_geometry_from_chease_failure(
      self, param, value
  ):

    base_config = {
        'geometry_type': 'chease',
        'Ip_from_parameters': True,
        'n_rho': 10,  # overrides the default
        'geometry_configs': {
            0.0: {
                'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
                'Rmaj': 6.2,
                'Rmin': 2.0,
                'B0': 5.3,
            },
            1.0: {
                'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
                'Rmaj': 6.2,
                'Rmin': 2.0,
                'B0': 5.3,
            },
        },
    }

    # Test invalid configs:
    for time_key in [0.0, 1.0]:
      invalid_config = base_config.copy()
      invalid_config['geometry_configs'][time_key][param] = value
      with self.assertRaisesRegex(
          ValueError, 'following parameters cannot be set per geometry_config'
      ):
        pydantic_model.Geometry.from_dict(invalid_config)

  # pylint: disable=invalid-name
  def test_chease_geometry_updates_Ip(self):
    """Tests that the Ip is updated when using chease geometry."""
    runtime_params = runtime_params_lib.GeneralRuntimeParams()
    original_Ip_tot = runtime_params.profile_conditions.Ip_tot
    geo_provider = pydantic_model.Geometry.from_dict({
        'geometry_type': 'chease',
        'Ip_from_parameters': (
            False
        ),  # this will force update runtime_params.Ip_tot
    }).build_provider
    runtime_params_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            transport=transport_model_params.RuntimeParams(),
            sources={},
            stepper=stepper_pydantic_model.Stepper(),
            torax_mesh=geo_provider.torax_mesh,
        )
    )
    dynamic_slice, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=0,
            dynamic_runtime_params_slice_provider=runtime_params_provider,
            geometry_provider=geo_provider,
        )
    )
    self.assertIsInstance(geo, standard_geometry.StandardGeometry)
    self.assertIsNotNone(dynamic_slice)
    self.assertNotEqual(
        dynamic_slice.profile_conditions.Ip_tot, original_Ip_tot
    )
    # pylint: enable=invalid-name

  @parameterized.parameters([
      dict(config=pydantic_model.CheaseConfig),
      dict(config=pydantic_model.CircularConfig),
  ])
  def test_rmin_rmax_ordering(self, config):

    with self.subTest('rmin_greater_than_rmaj'):
      with self.assertRaisesRegex(
          ValueError, 'Rmin must be less than or equal to Rmaj'
      ):
        config(Rmaj=1.0, Rmin=2.0)

    with self.subTest('negative_values'):
      with self.assertRaises(ValueError):
        config(Rmaj=-1.0, Rmin=-2.0)

  def test_failed_test(self):
    config = {
        'geometry_type': 'eqdsk',
        'geometry_file': 'EQDSK_ITERhybrid_COCOS02.eqdsk',
        'Ip_from_parameters': True,
        'last_surface_factor': 0.99,
        'n_surfaces': 100,
    }
    pydantic_model.Geometry.from_dict(config)


if __name__ == '__main__':
  absltest.main()
