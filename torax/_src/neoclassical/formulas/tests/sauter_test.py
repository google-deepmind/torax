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
from torax._src.core_profiles import initialization
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.formulas import sauter as sauter_formulas
from torax._src.physics import collisions
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name

_N_RHO = 10
_A_TOL = 1e-6
_R_TOL = 1e-6


class SauterFormulasTest(parameterized.TestCase):

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

    params_provider = (
        build_runtime_params.RuntimeParamsProvider.from_config(
            torax_config
        )
    )
    runtime_params, self.geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=params_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    self.core_profiles = initialization.initial_core_profiles(
        runtime_params,
        self.geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    log_lambda_ei = collisions.calculate_log_lambda_ei(
        self.core_profiles.T_e.face_value(), self.core_profiles.n_e.face_value()
    )
    self.nu_e_star = formulas.calculate_nu_e_star(
        q=self.core_profiles.q_face,
        geo=self.geo,
        n_e=self.core_profiles.n_e.face_value(),
        T_e=self.core_profiles.T_e.face_value(),
        Z_eff=self.core_profiles.Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )

    self.f_trap = formulas.calculate_f_trap(self.geo)

  def test_L31_values_are_correct(self):
    L31 = sauter_formulas.calculate_L31(
        self.f_trap, self.nu_e_star, self.core_profiles.Z_eff_face
    )
    np.testing.assert_allclose(L31, _L31_EXPECTED, atol=_A_TOL, rtol=_R_TOL)

  def test_L32_values_are_correct(self):
    L32 = sauter_formulas.calculate_L32(
        self.f_trap, self.nu_e_star, self.core_profiles.Z_eff_face
    )
    np.testing.assert_allclose(L32, _L32_EXPECTED, atol=_A_TOL, rtol=_R_TOL)

_L31_EXPECTED = np.array([
    0.0,
    0.25942749,
    0.39198664,
    0.48032915,
    0.53634519,
    0.57082292,
    0.5894148,
    0.59111759,
    0.56839259,
    0.5001917,
    0.33682819,
])
_L32_EXPECTED = np.array([
    0.0,
    -0.03501634,
    -0.07505174,
    -0.09268982,
    -0.09172278,
    -0.08055448,
    -0.06213122,
    -0.03385067,
    0.01149523,
    0.08557197,
    0.16296924,
])

if __name__ == '__main__':
  absltest.main()
