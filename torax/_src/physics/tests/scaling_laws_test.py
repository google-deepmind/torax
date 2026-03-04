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
from jax import numpy as jnp
import numpy as np
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.physics import scaling_laws


# pylint: disable=invalid-name
class ScalingLawsTest(parameterized.TestCase):

  def test_calculate_plh_scaling_factor(self):
    geo = circular_geometry.CircularConfig(
        n_rho=25,
        elongation_LCFS=1.0,
        hires_factor=4,
        R_major=6.0,
        a_minor=2.0,
        B_0=5.0,
    ).build_geometry()

    # Using mock.ANY instead of mock.create_autospec to maintain the Ip_total
    # property needed in calculate_plh_scaling_factor.
    core_profiles = state.CoreProfiles(
        n_e=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 2e20,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(2.0e20),
            face_centers=geo.rho_face_norm,
        ),
        n_i=mock.ANY,
        n_impurity=mock.ANY,
        impurity_fractions=mock.ANY,
        main_ion_fractions=mock.ANY,
        T_i=mock.ANY,
        T_e=mock.ANY,
        psi=mock.ANY,
        psidot=mock.ANY,
        v_loop_lcfs=mock.ANY,
        q_face=mock.ANY,
        s_face=mock.ANY,
        Z_i=mock.ANY,
        Z_i_face=mock.ANY,
        A_i=3.0,
        Z_impurity=mock.ANY,
        Z_impurity_face=mock.ANY,
        A_impurity=mock.ANY,
        A_impurity_face=mock.ANY,
        Z_eff=mock.ANY,
        Z_eff_face=mock.ANY,
        sigma=mock.ANY,
        sigma_face=mock.ANY,
        j_total=mock.ANY,
        j_total_face=mock.ANY,
        Ip_profile_face=jnp.ones_like(geo.rho_face_norm) * 10e6,
        toroidal_angular_velocity=mock.ANY,
        charge_state_info=mock.ANY,
        charge_state_info_face=mock.ANY,
        fast_ions=mock.ANY,
    )
    P_LH_hi_dens, P_LH_min, P_LH, n_e_min_P_LH = (
        scaling_laws.calculate_plh_scaling_factor(geo, core_profiles)
    )
    expected_PLH_hi_dens = (
        2.15 * 2**0.782 * 5**0.772 * 2**0.975 * 6**0.999 * (2.0141 / 3)
    )
    expected_PLH_min = (
        0.36 * 10**0.27 * 5**1.25 * 6**1.23 * 3**0.08 * (2.0141 / 3)
    )
    expected_n_e_min_P_LH = (
        0.7 * 10**0.34 * 5**0.62 * 2.0**-0.95 * 3**0.4 * 1e19
    )
    np.testing.assert_allclose(
        P_LH_hi_dens / 1e6, expected_PLH_hi_dens, rtol=1e-6
    )
    np.testing.assert_allclose(P_LH_min / 1e6, expected_PLH_min, rtol=1e-6)
    np.testing.assert_allclose(n_e_min_P_LH, expected_n_e_min_P_LH, rtol=1e-6)
    np.testing.assert_allclose(P_LH, P_LH_hi_dens, rtol=1e-6)

  @parameterized.parameters([
      dict(elongation_LCFS=1.0),
      dict(elongation_LCFS=1.5),
  ])
  def test_calculate_scaling_law_confinement_time(self, elongation_LCFS):
    geo = circular_geometry.CircularConfig(
        n_rho=25,
        elongation_LCFS=elongation_LCFS,
        hires_factor=4,
        R_major=6.0,
        a_minor=2.0,
        B_0=5.0,
    ).build_geometry()
    # Using mock.ANY instead of mock.create_autospec to maintain the Ip_total
    # property needed in calculate_plh_scaling_factor.
    core_profiles = state.CoreProfiles(
        n_e=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 2e20,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(2.0e20),
            face_centers=geo.rho_face_norm,
        ),
        n_i=mock.ANY,
        n_impurity=mock.ANY,
        impurity_fractions=mock.ANY,
        main_ion_fractions=mock.ANY,
        T_i=mock.ANY,
        T_e=mock.ANY,
        psi=mock.ANY,
        psidot=mock.ANY,
        v_loop_lcfs=mock.ANY,
        q_face=mock.ANY,
        s_face=mock.ANY,
        Z_i=mock.ANY,
        Z_i_face=mock.ANY,
        A_i=3.0,
        Z_impurity=mock.ANY,
        Z_impurity_face=mock.ANY,
        A_impurity=mock.ANY,
        A_impurity_face=mock.ANY,
        Z_eff=mock.ANY,
        Z_eff_face=mock.ANY,
        sigma=mock.ANY,
        sigma_face=mock.ANY,
        j_total=mock.ANY,
        j_total_face=mock.ANY,
        Ip_profile_face=jnp.ones_like(geo.rho_face_norm) * 10e6,
        toroidal_angular_velocity=mock.ANY,
        charge_state_info=mock.ANY,
        charge_state_info_face=mock.ANY,
        fast_ions=mock.ANY,
    )
    Ploss = jnp.array(50e6)

    H89P = scaling_laws.calculate_scaling_law_confinement_time(
        geo, core_profiles, Ploss, 'H89P'
    )
    H98 = scaling_laws.calculate_scaling_law_confinement_time(
        geo, core_profiles, Ploss, 'H98'
    )
    H97L = scaling_laws.calculate_scaling_law_confinement_time(
        geo, core_profiles, Ploss, 'H97L'
    )
    H20 = scaling_laws.calculate_scaling_law_confinement_time(
        geo, core_profiles, Ploss, 'H20'
    )

    expected_H89P = (
        0.038128
        * 10**0.85
        * 5**0.2
        * 20**0.1
        * 50**-0.5
        * 6**1.5
        * (1 / 3) ** 0.3
        * 3**0.50
        * elongation_LCFS**0.50
    )

    expected_H98 = (
        0.0562
        * 10**0.93
        * 5**0.15
        * 20**0.41
        * 50**-0.69
        * 6**1.97
        * (1 / 3) ** 0.58
        * 3**0.19
        * elongation_LCFS**0.78
    )

    expected_H97L = (
        0.023
        * 10**0.96
        * 5**0.03
        * 20**0.4
        * 50**-0.73
        * 6**1.83
        * (1 / 3) ** -0.06
        * 3**0.20
        * elongation_LCFS**0.64
    )

    expected_H20 = (
        0.053
        * 10**0.98
        * 5**0.22
        * 20**0.24
        * 50**-0.669
        * 6**1.71
        * (1 / 3) ** 0.35
        * 3**0.20
        * elongation_LCFS**0.80
    )
    np.testing.assert_allclose(H89P, expected_H89P, rtol=1e-6)
    np.testing.assert_allclose(H98, expected_H98, rtol=1e-6)
    np.testing.assert_allclose(H97L, expected_H97L, rtol=1e-6)
    np.testing.assert_allclose(H20, expected_H20, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
