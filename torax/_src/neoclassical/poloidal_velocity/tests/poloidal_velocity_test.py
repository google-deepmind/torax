# Copyright 2026 DeepMind Technologies Limited
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
"""Tests for neoclassical poloidal velocity models."""
from absl.testing import absltest
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.neoclassical.poloidal_velocity import kim
from torax._src.neoclassical.poloidal_velocity import zeros
from torax._src.torax_pydantic import model_config


class PoloidalVelocityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torax_config = model_config.ToraxConfig.from_dict({
        'profile_conditions': {
            'Ip': 15e6,
            'current_profile_nu': 3,
            'n_e_nbar_is_fGW': True,
            'normalize_n_e_to_nbar': True,
            'nbar': 0.85,
            'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
        },
        'numerics': {},
        'plasma_composition': {
            'Z_eff': 2.0,
        },
        'geometry': {
            'geometry_type': 'chease',
            'Ip_from_parameters': False,
            'n_rho': 10,
        },
        'transport': {},
        'solver': {},
        'pedestal': {},
        'sources': {},
    })
    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    self.runtime_params, self.geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=params_provider,
            geometry_provider=torax_config.geometry.build_provider,
            is_initialization=True,
        )
    )
    models = torax_config.build_models()
    self.core_profiles = initialization.initial_core_profiles(
        self.runtime_params,
        self.geo,
        source_models=models.source_models,
        neoclassical_models=models.neoclassical_models,
    )

  def test_kim_model_produces_expected_shapes_and_non_zero_velocity(self):
    model = kim.KimModel()
    output = model.calculate_poloidal_velocity(
        self.runtime_params, self.geo, self.core_profiles
    )
    self.assertEqual(output.v_pol.value.shape, self.geo.rho_norm.shape)
    self.assertEqual(
        output.v_pol.face_value().shape, self.geo.rho_face_norm.shape
    )
    self.assertTrue(np.any(np.abs(output.v_pol.face_value()) > 0.0))

  def test_zeros_model_returns_zero_poloidal_velocity(self):
    model = zeros.ZerosModel()
    output = model.calculate_poloidal_velocity(
        self.runtime_params, self.geo, self.core_profiles
    )
    np.testing.assert_allclose(
        output.v_pol.value, np.zeros_like(self.geo.rho_norm)
    )
    np.testing.assert_allclose(
        output.v_pol.face_value(), np.zeros_like(self.geo.rho_face_norm)
    )


if __name__ == '__main__':
  absltest.main()
