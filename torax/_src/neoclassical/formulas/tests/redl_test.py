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
from torax._src.neoclassical.formulas import redl as redl_formulas
from torax._src.physics import collisions
from torax._src.test_utils import core_profile_helpers
from torax._src.torax_pydantic import config

# pylint: disable=invalid-name

_A_TOL = 1e-6
_R_TOL = 1e-6


class RedlFormulasTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torax_config = config.ToraxConfig.from_dict({
        'profile_conditions': {
            'I_p': 15e6,
            'I_p_source': 'geometry',
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
            'n_rho': 10,
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
        self.core_profiles.kinetic.T_e.face_value(),
        self.core_profiles.kinetic.n_e.face_value(),
    )
    self.nu_e_star = formulas._calculate_nu_e_star(
        q=self.core_profiles.flux.q_face,
        geo=self.geo,
        n_e=self.core_profiles.kinetic.n_e.face_value(),
        T_e=self.core_profiles.kinetic.T_e.face_value(),
        Z_eff=self.core_profiles.ions.Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )

    self.f_trap = self.geo.trapped_fraction_face

  def test_L31_values_are_correct(self):
    L31 = redl_formulas.calculate_L31(
        self.f_trap, self.nu_e_star, self.core_profiles.ions.Z_eff_face
    )
    np.testing.assert_allclose(L31, _L31_EXPECTED, atol=_A_TOL, rtol=_R_TOL)

  def test_L32_values_are_correct(self):
    L32 = redl_formulas.calculate_L32(
        self.f_trap, self.nu_e_star, self.core_profiles.ions.Z_eff_face
    )
    np.testing.assert_allclose(L32, _L32_EXPECTED, atol=_A_TOL, rtol=_R_TOL)

  def test_L33_values_are_correct(self):
    L33 = redl_formulas.calculate_L33(
        self.f_trap, self.nu_e_star, self.core_profiles.ions.Z_eff_face
    )
    self.assertEqual(L33.shape, self.f_trap.shape)
    self.assertAlmostEqual(float(L33[0]), 1.0, places=6)
    self.assertTrue(np.all((L33 > 0.0) & (L33 <= 1.0)))


_L31_EXPECTED = np.array([
    0.0,
    0.24294991651356793,
    0.3618703785919998,
    0.4450928995658736,
    0.5037594760697106,
    0.5456727404132087,
    0.5744101483558093,
    0.5885157027432865,
    0.5789918247992369,
    0.5130779776729055,
    0.28646594502191125,
])
_L32_EXPECTED = np.array([
    0.0,
    -0.040906737822690414,
    -0.08508237929507831,
    -0.10155531054828773,
    -0.10064277811760239,
    -0.09172274233796579,
    -0.07645202444151344,
    -0.05126287383933026,
    -0.006815121535202695,
    0.07609215354012955,
    0.18210763398539093,
])

if __name__ == '__main__':
  absltest.main()
