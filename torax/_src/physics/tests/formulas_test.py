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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src import jax_utils
from torax._src import math_utils
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.physics import formulas
from torax._src.test_utils import core_profile_helpers


# pylint: disable=invalid-name
class FormulasTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)
    self.geo = circular_geometry.CircularConfig(
        n_rho=10, a_minor=1.0
    ).build_geometry()
    self.core_profiles = core_profile_helpers.make_zero_core_profiles(self.geo)
    self.core_profiles = dataclasses.replace(
        self.core_profiles,
        T_i=core_profile_helpers.make_constant_core_profile(self.geo, 1.0),
        T_e=core_profile_helpers.make_constant_core_profile(self.geo, 2.0),
        n_e=core_profile_helpers.make_constant_core_profile(self.geo, 3.0e20),
        n_i=core_profile_helpers.make_constant_core_profile(self.geo, 2.5e20),
        n_impurity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.25e20
        ),
        Ip_profile_face=[np.pi * 1e6],  # pyrefly: ignore[bad-argument-type]
    )

  # TODO(b/377225415): generalize to arbitrary number of ions.
  @parameterized.parameters([
      dict(Z_i=1.0, Z_impurity=10.0, Z_eff=1.0, expected=1.0),
      dict(Z_i=1.0, Z_impurity=5.0, Z_eff=1.0, expected=1.0),
      dict(Z_i=2.0, Z_impurity=10.0, Z_eff=2.0, expected=0.5),
      dict(Z_i=2.0, Z_impurity=5.0, Z_eff=2.0, expected=0.5),
      dict(Z_i=1.0, Z_impurity=10.0, Z_eff=1.9, expected=0.9),
      dict(Z_i=2.0, Z_impurity=10.0, Z_eff=3.6, expected=0.4),
  ])
  def test_calculate_main_ion_dilution_factor(
      self, Z_i, Z_impurity, Z_eff, expected
  ):
    """Unit test of `calculate_main_ion_dilution_factor`."""
    np.testing.assert_allclose(
        formulas.calculate_main_ion_dilution_factor(Z_i, Z_impurity, Z_eff),
        expected,
    )

  def test_calculate_stored_thermal_energy(self):
    """Test that stored thermal energy is computed correctly."""
    p_el = core_profile_helpers.make_constant_core_profile(self.geo, 1.0)
    p_ion = core_profile_helpers.make_constant_core_profile(self.geo, 2.0)
    p_tot = core_profile_helpers.make_constant_core_profile(self.geo, 3.0)
    wth_el, wth_ion, wth_tot = formulas.calculate_stored_thermal_energy(
        p_el, p_ion, p_tot, self.geo
    )

    volume = math_utils.volume_integration(np.array([1.0]), self.geo)

    np.testing.assert_allclose(wth_el, 1.5 * p_el.value[0] * volume)  # pyrefly: ignore[bad-index]
    np.testing.assert_allclose(wth_ion, 1.5 * p_ion.value[0] * volume)  # pyrefly: ignore[bad-index]
    np.testing.assert_allclose(wth_tot, 1.5 * p_tot.value[0] * volume)  # pyrefly: ignore[bad-index]

  def test_calculate_greenwald_fraction(self):
    """Test that Greenwald fraction is calculated correctly."""
    n_e_avg = 1.0e20

    fgw_n_e_volume_avg_calculated = formulas.calculate_greenwald_fraction(
        n_e_avg, self.core_profiles, self.geo
    )

    fgw_n_e_volume_avg_expected = 1.0

    np.testing.assert_allclose(
        fgw_n_e_volume_avg_calculated, fgw_n_e_volume_avg_expected
    )

  def test_calculate_betas(self):
    """Test that betas are calculated correctly."""

    beta_tor, beta_pol, beta_N = formulas.calculate_betas(  # pyrefly: ignore[not-iterable]
        self.core_profiles, self.geo
    )
    beta_tor_expected = 0.012530022
    beta_pol_expected = 1.5334594
    beta_N_expected = 2.113868
    np.testing.assert_allclose(beta_tor, beta_tor_expected)
    np.testing.assert_allclose(beta_pol, beta_pol_expected)
    np.testing.assert_allclose(beta_N, beta_N_expected)

  def test_calc_dvar_dpsi_against_analytical_solution(self):
    geo = circular_geometry.CircularConfig(
        n_rho=50, a_minor=1.0, R_major=10.0, B_0=2.0
    ).build_geometry()
    psi_0 = 1.0
    Delta_psi = 2.5
    rho_norm_face = geo.rho_face / geo.rho_face[-1]
    rho_norm_cell = geo.rho / geo.rho_face[-1]

    psi_face = psi_0 + Delta_psi * (rho_norm_face**2)
    psi_cell = psi_0 + Delta_psi * (rho_norm_cell**2)
    psi_var = cell_variable.CellVariable(
        value=psi_cell,
        face_centers=geo.rho_face,
        left_face_constraint=psi_face[0],
        right_face_constraint=psi_face[-1],
        left_face_grad_constraint=None,
        right_face_grad_constraint=None,
    )

    c0, c1 = 3.0, -1.5
    psi_N_face = rho_norm_face**2
    psi_N_cell = rho_norm_cell**2
    var_face = c0 + c1 * psi_N_face
    var_cell = c0 + c1 * psi_N_cell

    var = cell_variable.CellVariable(
        value=var_cell,
        face_centers=geo.rho_face,
        left_face_constraint=var_face[0],
        right_face_constraint=var_face[-1],
        left_face_grad_constraint=None,
        right_face_grad_constraint=None,
    )

    expected_dvar_dpsi_norm = c1 * np.ones_like(rho_norm_face)
    expected_dvar_dpsi = expected_dvar_dpsi_norm / Delta_psi

    with self.subTest('unnormalized'):
      dvar_dpsi = formulas.calc_dvar_dpsi(var, psi_var, normalized=False)
      np.testing.assert_allclose(dvar_dpsi, expected_dvar_dpsi, rtol=1e-6)

    with self.subTest('normalized'):
      dvar_dpsi_norm = formulas.calc_dvar_dpsi(var, psi_var, normalized=True)
      np.testing.assert_allclose(
          dvar_dpsi_norm, expected_dvar_dpsi_norm, rtol=1e-6
      )


if __name__ == '__main__':
  absltest.main()
