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
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry
from torax._src.physics import formulas
from torax._src.physics import psi_calculations
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

  def test_calculate_beta_pol_profile_and_derivative_analytical_circular(self):
    geo = circular_geometry.CircularConfig(
        n_rho=50, a_minor=1.0, R_major=10.0, B_0=2.0
    ).build_geometry()
    Ip = 1e6
    mu0 = constants.CONSTANTS.mu_0
    R0 = geo.R_major

    psi_face = 0.5 * mu0 * Ip * R0 * (geo.rho_face / geo.rho_face[-1]) ** 2
    psi_cell = 0.5 * mu0 * Ip * R0 * (geo.rho / geo.rho_face[-1]) ** 2
    psi_var = cell_variable.CellVariable(
        value=psi_cell,
        face_centers=geo.rho_face,
        left_face_constraint=psi_face[0],
        right_face_constraint=psi_face[-1],
        left_face_grad_constraint=None,
        right_face_grad_constraint=None,
    )

    # Compute magnetic pressure denominator: <Bp^2> / (2 * mu0)
    bpol2_face = psi_calculations.calc_bpol_squared(geo, psi_var)
    bpol2_cell = geometry.face_to_cell(bpol2_face)
    denom_cell = bpol2_cell / (2.0 * mu0)
    denom_right = bpol2_face[-1] / (2.0 * mu0)

    # Define a linear profile in normalized psi space:
    # beta_pol_target = beta_0 + beta_prime * (1 - psi_N)
    # Then d(beta_pol) / d(psi_N) = -beta_prime everywhere.
    psi_range = psi_var.right_face_value - psi_var.left_face_value
    psi_N_cell = (psi_var.value - psi_var.left_face_value) / psi_range
    beta_0 = 0.5
    beta_prime = 1.2
    target_beta_pol_cell = beta_0 + beta_prime * (1.0 - psi_N_cell)

    P_tot_cell = target_beta_pol_cell * denom_cell
    P_tot_right = beta_0 * denom_right

    # Construct T_e and n_e so that pressure_total equals P_tot:
    # With n_i = 0, n_impurity = 0, fast_ions = ():
    # P_tot = P_el = n_e * T_e * keV_to_J.
    # Set n_e = 1.0 / keV_to_J, and T_e = P_tot.
    n_e_val = cell_variable.CellVariable(
        value=np.ones_like(geo.rho) / constants.CONSTANTS.keV_to_J,
        face_centers=geo.rho_face,
        right_face_constraint=1.0 / constants.CONSTANTS.keV_to_J,
        right_face_grad_constraint=None,
    )
    T_e_val = cell_variable.CellVariable(
        value=P_tot_cell,
        face_centers=geo.rho_face,
        left_face_constraint=None,
        left_face_grad_constraint=0.0,
        right_face_constraint=P_tot_right,
        right_face_grad_constraint=None,
    )

    core_profiles = core_profile_helpers.make_zero_core_profiles(geo)
    core_profiles = dataclasses.replace(
        core_profiles,
        psi=psi_var,
        n_e=n_e_val,
        T_e=T_e_val,
    )

    with self.subTest('beta_pol_profile'):
      beta_pol_prof = formulas.calculate_beta_pol_profile(core_profiles, geo)
      np.testing.assert_allclose(
          beta_pol_prof.value, target_beta_pol_cell, rtol=1e-6
      )
      self.assertIsNotNone(beta_pol_prof.right_face_constraint)
      np.testing.assert_allclose(
          beta_pol_prof.right_face_constraint, beta_0, rtol=1e-6
      )

    with self.subTest('beta_pol_prime'):
      beta_pol_prime = formulas.calculate_beta_pol_prime(core_profiles, geo)
      # Check inner faces (faces 1 through N-1) and right face
      np.testing.assert_allclose(beta_pol_prime[1:], beta_prime, rtol=1e-4)

if __name__ == '__main__':
  absltest.main()
