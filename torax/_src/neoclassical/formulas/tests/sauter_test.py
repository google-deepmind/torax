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
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.formulas import sauter as sauter_formulas
from torax._src.physics import collisions
from torax._src.test_utils import core_profile_helpers
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
            'initial_psi_source': 'geometry',
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

    self.core_profiles, _, self.geo = core_profile_helpers.make_initial_inputs(
        torax_config
    )

    log_lambda_ei = collisions.calculate_log_lambda_ei(
        self.core_profiles.T_e.face_value(), self.core_profiles.n_e.face_value()
    )
    self.nu_e_star = formulas._calculate_nu_e_star(
        q=self.core_profiles.q_face,
        geo=self.geo,
        n_e=self.core_profiles.n_e.face_value(),
        T_e=self.core_profiles.T_e.face_value(),
        Z_eff=self.core_profiles.Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )

    self.f_trap = self.geo.trapped_fraction_face

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
    0.2593337181519866,
    0.39153447076596765,
    0.4807690237233787,
    0.5365556078960572,
    0.5709167460950202,
    0.5894855323075728,
    0.5911775420554187,
    0.5685738182248189,
    0.49992595504440956,
    0.33682819436163186,
])
_L32_EXPECTED = np.array([
    0.0,
    -0.03493750269284397,
    -0.07456799540957554,
    -0.09314769202185386,
    -0.09193939560405384,
    -0.08065105782253396,
    -0.06220535082859979,
    -0.03391458098337086,
    0.011304831914444868,
    0.08579156757246964,
    0.16296924403319402,
])

if __name__ == '__main__':
  absltest.main()
