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
from torax._src.config import build_runtime_params
from torax._src.geometry import chease
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry_provider
from torax._src.geometry import pydantic_model
from torax._src.geometry import standard_geometry
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


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
    np.testing.assert_array_equal(geo.B_0, 5.3)  # test a default.

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
                'geometry_file': 'iterhybrid.mat2cols',
                'R_major': 6.2,
                'a_minor': 2.0,
                'B_0': 5.3,
            },
            1.0: {
                'geometry_file': 'iterhybrid.mat2cols',
                'R_major': 6.2,
                'a_minor': 2.0,
                'B_0': 5.3,
            },
        },
    }

    # Test valid config
    geo_pydantic = pydantic_model.Geometry.from_dict(base_config)
    geo_provider = geo_pydantic.build_provider
    self.assertIsInstance(
        geo_provider, standard_geometry.StandardGeometryProvider
    )
    self.assertIsInstance(
        geo_provider(t=0.0), standard_geometry.StandardGeometry
    )
    np.testing.assert_array_equal(geo_provider.torax_mesh.nx, 10)

    np.testing.assert_array_equal(
        geo_pydantic.get_face_centers(),
        geo_provider.torax_mesh.face_centers)

  @parameterized.parameters([
      dict(param='n_rho', value=5),
      dict(param='Ip_from_parameters', value=True),
      dict(param='geometry_directory', value='.'),
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
                'geometry_file': 'iterhybrid.mat2cols',
                'R_major': 6.2,
                'a_minor': 2.0,
                'B_0': 5.3,
            },
            1.0: {
                'geometry_file': 'iterhybrid.mat2cols',
                'R_major': 6.2,
                'a_minor': 2.0,
                'B_0': 5.3,
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
    config = default_configs.get_default_config_dict()
    config['geometry'] = {
        'geometry_type': 'chease',
        'Ip_from_parameters': False,
        'n_rho': 4,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    runtime_params_provider = (
        build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
    )
    runtime_params, geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=0,
            runtime_params_provider=runtime_params_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    original_Ip = torax_config.profile_conditions.Ip
    self.assertIsInstance(geo, standard_geometry.StandardGeometry)
    self.assertIsNotNone(runtime_params)
    self.assertNotEqual(
        runtime_params.profile_conditions.Ip, original_Ip.value[0]
    )
    # pylint: enable=invalid-name

  @parameterized.parameters([
      dict(config=chease.CheaseConfig),
      dict(config=circular_geometry.CircularConfig),
  ])
  def test_rmin_rmax_ordering(self, config):

    with self.subTest('rmin_greater_than_rmaj'):
      with self.assertRaisesRegex(
          ValueError, 'a_minor must be less than or equal to R_major'
      ):
        config(R_major=1.0, a_minor=2.0)

    with self.subTest('negative_values'):
      with self.assertRaises(ValueError):
        config(R_major=-1.0, a_minor=-2.0)

  def test_failed_test(self):
    config = {
        'geometry_type': 'eqdsk',
        'geometry_file': 'iterhybrid_cocos02.eqdsk',
        'Ip_from_parameters': True,
        'last_surface_factor': 0.99,
        'n_surfaces': 100,
        'cocos': 2,
    }
    pydantic_model.Geometry.from_dict(config)


if __name__ == '__main__':
  absltest.main()
