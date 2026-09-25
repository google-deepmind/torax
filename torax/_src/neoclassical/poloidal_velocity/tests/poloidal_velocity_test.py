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
from torax._src.neoclassical.poloidal_velocity import runtime_params as poloidal_velocity_runtime_params
from torax._src.neoclassical.poloidal_velocity import zeros
from torax._src.torax_pydantic import model_config

_A_TOL = 1e-6
_R_TOL = 1e-6

# Reference values from running test code in a notebook.
_POLOIDAL_VELOCITY_EXPECTED = np.array([
    -1485.871716,
    -2507.496827,
    -3933.755809,
    -4537.621566,
    -4854.858931,
    -5031.592012,
    -5073.608117,
    -4858.248803,
    -3559.941551,
    3265.428187,
    18579.094079,
])


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
        neoclassical_model=models.neoclassical_model,
    )

  def test_calculate_poloidal_velocity_values_are_correct(self):
    poloidal_velocity = kim.calculate_poloidal_velocity(
        T_i=self.core_profiles.T_i,
        n_i=self.core_profiles.n_i.face_value(),
        q=self.core_profiles.q_face,
        Z_eff=self.core_profiles.Z_eff_face,
        Z_i=self.core_profiles.Z_i_face,
        B_tor=np.ones_like(self.geo.rho_face_norm),
        B_total_squared=np.ones_like(self.geo.rho_face_norm),
        geo=self.geo,
    )
    np.testing.assert_allclose(
        _POLOIDAL_VELOCITY_EXPECTED,
        poloidal_velocity.face_value(),
        atol=_A_TOL,
        rtol=_R_TOL,
    )

  def test_kim_model_produces_expected_shapes_and_non_zero_velocity(self):
    model = kim.KimModel()
    output = model.calculate_poloidal_velocity(
        poloidal_velocity_runtime_params.RuntimeParams(),
        self.geo,
        self.core_profiles,
    )
    self.assertEqual(output.v_pol.value.shape, self.geo.rho_norm.shape)
    self.assertEqual(output.v_pol_face.shape, self.geo.rho_face_norm.shape)
    self.assertTrue(np.any(np.abs(output.v_pol_face) > 0.0))

  def test_zeros_model_returns_zero_poloidal_velocity(self):
    model = zeros.ZerosModel()
    output = model.calculate_poloidal_velocity(
        poloidal_velocity_runtime_params.RuntimeParams(),
        self.geo,
        self.core_profiles,
    )
    np.testing.assert_allclose(
        output.v_pol.value, np.zeros_like(self.geo.rho_norm)
    )
    np.testing.assert_allclose(
        output.v_pol_face, np.zeros_like(self.geo.rho_face_norm)
    )


if __name__ == '__main__':
  absltest.main()
