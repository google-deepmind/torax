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

"""Unit tests for fast_ion_utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src import constants
from torax._src.geometry import circular_geometry
from torax._src.physics import fast_ion_utils


# pylint: disable=invalid-name
class FastIonUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(
        n_rho=10, a_minor=1.0
    ).build_geometry()
    self.T_e_val = 5.0  # keV
    self.n_e_val = 1.0e20  # m^-3
    self.T_i_val = 4.0  # keV
    self.n_i_val = 0.9e20  # m^-3
    self.n_impurity_val = 0.05e20  # m^-3

    self.T_e = jnp.full(len(self.geo.rho), self.T_e_val)
    self.n_e = jnp.full(len(self.geo.rho), self.n_e_val)
    self.T_i = jnp.full(len(self.geo.rho), self.T_i_val)
    self.n_i = jnp.full(len(self.geo.rho), self.n_i_val)
    self.n_impurity = jnp.full(len(self.geo.rho), self.n_impurity_val)

  def _call_bimaxwellian_split(self, **kwargs):
    defaults = dict(
        power_deposition=jnp.ones(len(self.geo.rho)),
        T_e=self.T_e,
        n_e=self.n_e,
        T_i=self.T_i,
        n_i=self.n_i,
        minority_concentration=0.05,
        P_total_W=1.0e6,
        charge_number=2,
        mass_number=3.016,
        bulk_ion_mass=2.014,
        Z_i=1.0,
        n_impurity=self.n_impurity,
        Z_impurity=10.0,
        A_impurity=20.18,
    )
    defaults.update(kwargs)
    return fast_ion_utils.bimaxwellian_split(**defaults)

  def test_bimaxwellian_split_basic(self):
    """Test basic functionality of bimaxwellian_split."""

    minority_concentration = 0.05
    n_tail, T_tail = self._call_bimaxwellian_split(
        minority_concentration=minority_concentration,
    )

    self.assertEqual(n_tail.shape, (len(self.geo.rho),))
    self.assertEqual(T_tail.shape, (len(self.geo.rho),))

    self.assertTrue(jnp.all(T_tail > self.T_e_val))

    n_total = self.n_e_val * minority_concentration

    self.assertTrue(jnp.all(n_tail >= 0))

    self.assertTrue(jnp.all(n_tail < n_total))

  def test_bimaxwellian_split_zero_power(self):
    """Test with zero power, should result in no tail."""

    n_tail, T_tail = self._call_bimaxwellian_split(
        power_deposition=jnp.zeros(len(self.geo.rho)),
        P_total_W=0.0,
    )

    np.testing.assert_allclose(n_tail, 0.0, atol=1e-9)
    np.testing.assert_allclose(T_tail, self.T_i_val)

  def test_bimaxwellian_split_high_power(self):
    """Test with high power to trigger constraints or higher tail fraction."""

    n_tail, _ = self._call_bimaxwellian_split(
        minority_concentration=0.03,
        P_total_W=10.0e6,
    )

    self.assertTrue(jnp.all(n_tail > 0))

  def test_bimaxwellian_split_regression(self):
    """Test specific numerical values to catch regressions."""

    n_tail, T_tail = self._call_bimaxwellian_split()

    np.testing.assert_allclose(n_tail[0], 1.0e13, rtol=5e-2)
    np.testing.assert_allclose(T_tail[0], 46.4197, rtol=1e-4)

  def test_bimaxwellian_energy_consistency(self):
    minority_concentration = 0.05
    n_total = self.n_e * minority_concentration

    n_tail, T_tail = self._call_bimaxwellian_split(
        minority_concentration=minority_concentration,
    )

    n_bulk = n_total - n_tail
    T_eff = (n_bulk * self.T_i + n_tail * T_tail) / n_total

    self.assertTrue(jnp.all(T_eff > self.T_i))
    self.assertTrue(jnp.all(T_eff < T_tail))

  def test_compute_T_tail_he3_regression(self):
    """Regression test for _compute_T_tail with He3 ICRH parameters."""
    T_e = self.T_e
    n_e = self.n_e
    minority_concentration = 0.05
    n_total = n_e * minority_concentration
    P_total_W = 1.0e6
    power_deposition = jnp.ones(len(self.geo.rho))
    P_density_W = power_deposition * P_total_W

    T_tail = fast_ion_utils._compute_T_tail(
        P_density_W=P_density_W,
        T_e=T_e,
        n_e=n_e,
        n_total=n_total,
        charge_number=2,
        mass_number=3.016,
    )

    np.testing.assert_allclose(T_tail[0], 46.4197, rtol=1e-4)

    self.assertTrue(jnp.all(T_tail > T_e))

  def test_nu_epsilon_equal_temperatures(self):
    """Test that nu_epsilon/n matches the simplified NRL formula.

    For T_a = T_b = T in the ion-electron case (m_e << m_i), the NRL energy
    exchange rate per unit density simplifies to
    (NRL Plasma Formulary, p. 34):
      nu_epsilon / n = 3.2e-9 * Z^2 * ln_lambda / (mu * T[K]^(3/2))
    where mu = m_i / m_proton and T is in Kelvin.
    """
    m_ion = 2.014  # Deuterium mass in amu
    Z_ion = 1.0
    m_e = (
        constants.CONSTANTS.m_e / constants.CONSTANTS.m_amu
    )  # electron mass in amu
    Z_e = 1.0
    T_keV = jnp.array([5.0])
    n_e_m3 = jnp.array([1.0e20])
    ln_lambda = jnp.array([15.0])

    nu = fast_ion_utils._nu_epsilon(
        m_a_amu=m_ion,
        Z_a=Z_ion,
        T_a_keV=T_keV,
        m_b_amu=m_e,
        Z_b=Z_e,
        n_b_m3=n_e_m3,
        T_b_keV=T_keV,
        ln_lambda=ln_lambda,
    )

    T_ev = float(T_keV[0]) * 1000.0
    n_e_cm3 = float(n_e_m3[0]) / 1.0e6
    mu = m_ion / 1.007276  # mu = m_ion / m_proton

    expected_nu_over_n = (
        3.2e-9 * Z_ion**2 * Z_e**2 * float(ln_lambda[0]) / (mu * T_ev**1.5)
    )

    np.testing.assert_allclose(
        float(nu[0]) / float(n_e_cm3), float(expected_nu_over_n), rtol=2e-2
    )


if __name__ == '__main__':
  absltest.main()
