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
import pydantic
from torax._src.config import build_runtime_params
from torax._src.geometry import chease
from torax._src.geometry import circular_geometry
from torax._src.geometry import config as geometry_config
from torax._src.geometry import standard_geometry
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config


_adapter = pydantic.TypeAdapter(geometry_config.GeometryConfig)


class GeometryConfigTest(parameterized.TestCase):

  def test_missing_model_name_raises_error(self):
    with self.assertRaises(pydantic.ValidationError):
      _adapter.validate_python({})

  def test_build_circular_geometry(self):
    cfg = _adapter.validate_python({
        'model_name': 'circular',
        'n_rho': 5,  # override a default.
    })
    geo = cfg.build_runtime_params(0.0).geometry
    np.testing.assert_array_equal(geo.n_rho, 5)
    np.testing.assert_array_equal(geo.B_0, 5.3)  # test a default.

  def test_build_geometry_from_chease(self):
    cfg = _adapter.validate_python(
        {
            'model_name': 'chease',
            'geometry_file': 'iterhybrid.mat2cols',
            'n_rho': 5,  # override a default.
        },
    )
    geo = cfg.build_runtime_params(0.0).geometry
    self.assertIsInstance(geo, standard_geometry.StandardGeometry)
    np.testing.assert_array_equal(geo.n_rho, 5)

  def test_build_time_dependent_geometry_from_chease(self):
    """Tests correctness of config constraints with time-dependent geometry."""

    base_config_dict = {
        'model_name': 'chease',
        'n_rho': 10,  # overrides the default
        'R_major': 6.2,
        'a_minor': 2.0,
        'B_0': 5.3,
        'geometry_file': {
            0.0: 'iterhybrid.mat2cols',
            1.0: 'iterhybrid.mat2cols',
        },
    }

    # Test valid config
    geo_pydantic = _adapter.validate_python(base_config_dict)
    runtime_params = geo_pydantic.build_runtime_params(t=0.0)
    geo = runtime_params.geometry
    self.assertIsInstance(geo, standard_geometry.StandardGeometry)
    np.testing.assert_array_equal(geo.n_rho, 10)

    np.testing.assert_array_equal(
        geo_pydantic.get_face_centers(), geo.rho_face_norm
    )

  # pylint: disable=invalid-name
  def test_chease_geometry_updates_I_p(self):
    """Tests that the I_p is updated when using chease geometry."""
    config_dict = default_configs.get_default_config_dict()
    config_dict['profile_conditions']['I_p_source'] = 'geometry'
    config_dict['geometry'] = {
        'model_name': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
        'n_rho': 4,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    runtime_params_provider = (
        build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
    )
    runtime_params, geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=0,
            runtime_params_provider=runtime_params_provider,
            geometry_model=torax_config.geometry.build_geometry_model(),
            is_initialization=True,
        )
    )
    original_I_p = torax_config.profile_conditions.I_p
    assert original_I_p is not None
    self.assertIsInstance(geo, standard_geometry.StandardGeometry)
    self.assertIsNotNone(runtime_params)
    self.assertNotEqual(
        runtime_params.profile_conditions.I_p, original_I_p.value[0]
    )
    # pylint: enable=invalid-name

  @parameterized.parameters([
      dict(config_cls=chease.CheaseConfig),
      dict(config_cls=circular_geometry.CircularConfig),
  ])
  def test_rmin_rmax_ordering(self, config_cls):
    kwargs = {}
    if config_cls is chease.CheaseConfig:
      kwargs['geometry_file'] = 'iterhybrid.mat2cols'

    with self.subTest('rmin_greater_than_rmaj'):
      with self.assertRaisesRegex(
          ValueError, 'a_minor must be less than or equal to R_major'
      ):
        config_cls(R_major=1.0, a_minor=2.0, **kwargs)

    with self.subTest('negative_values'):
      with self.assertRaises(ValueError):
        config_cls(R_major=-1.0, a_minor=-2.0, **kwargs)

  def test_failed_test(self):
    config_dict = {
        'model_name': 'eqdsk',
        'geometry_file': 'iterhybrid_cocos02.eqdsk',
        'last_surface_factor': 0.99,
        'n_surfaces': 100,
        'cocos': 2,
    }
    _adapter.validate_python(config_dict)


if __name__ == '__main__':
  absltest.main()
