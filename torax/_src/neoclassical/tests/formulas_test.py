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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.neoclassical import formulas
from torax._src.physics import collisions
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name

_N_RHO = 10
_A_TOL = 1e-6
_R_TOL = 1e-6


class FormulasTest(parameterized.TestCase):

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
            'n_rho': _N_RHO,
        },
        'transport': {},
        'solver': {},
        'pedestal': {},
        'sources': {},
    })

    dynamic_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.numerics.t_initial,
            dynamic_runtime_params_slice_provider=dynamic_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    self.core_profiles = initialization.initial_core_profiles(
        static_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    log_lambda_ei = collisions.calculate_log_lambda_ei(
        self.core_profiles.T_e.face_value(), self.core_profiles.n_e.face_value()
    )
    self.nu_e_star = formulas.calculate_nu_e_star(
        q=self.core_profiles.q_face,
        geo=geo,
        n_e=self.core_profiles.n_e.face_value(),
        T_e=self.core_profiles.T_e.face_value(),
        Z_eff=self.core_profiles.Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )

    self.f_trap = formulas.calculate_f_trap(geo)

  def test_calculate_f_trap_positive_triangularity(self):
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        delta_face=np.array(0.2),
        epsilon_face=np.array(0.1),
    )
    result = formulas.calculate_f_trap(geo)
    expected = 0.4362384616678634
    np.testing.assert_allclose(result, expected)

  def test_calculate_f_trap_negative_triangularity(self):
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        delta_face=np.array(-0.2),
        epsilon_face=np.array(0.1),
    )
    result = formulas.calculate_f_trap(geo)
    expected = 0.45134158459680895
    np.testing.assert_allclose(result, expected)

  def test_L31_values_are_correct(self):
    L31 = formulas.calculate_L31(
        self.f_trap, self.nu_e_star, self.core_profiles.Z_eff_face
    )
    np.testing.assert_allclose(L31, _L31_EXPECTED, atol=_A_TOL, rtol=_R_TOL)

  def test_L32_values_are_correct(self):
    L32 = formulas.calculate_L32(
        self.f_trap, self.nu_e_star, self.core_profiles.Z_eff_face
    )
    np.testing.assert_allclose(L32, _L32_EXPECTED, atol=_A_TOL, rtol=_R_TOL)


# Reference values from running test code in a notebook.
# The test thus does not directly test the implementation, but rather
# guards against unexpected modifications.
_L31_EXPECTED = np.array([
    0.0,
    0.26066935,
    0.39309629,
    0.48129704,
    0.53726614,
    0.57173028,
    0.59029346,
    0.59193504,
    0.56909422,
    0.50068221,
    0.33686107,
])
_L32_EXPECTED = np.array([
    0.0,
    -0.03629958,
    -0.07623998,
    -0.09369294,
    -0.09266533,
    -0.08148761,
    -0.06304672,
    -0.03471382,
    0.01076693,
    0.08515409,
    0.16296786,
])

if __name__ == '__main__':
  absltest.main()
