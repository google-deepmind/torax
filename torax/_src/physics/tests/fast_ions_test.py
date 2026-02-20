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

"""Unit tests for fast_ions.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src.geometry import circular_geometry
from torax._src.physics import fast_ions


# pylint: disable=invalid-name
class FastIonsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(
        n_rho=10, a_minor=1.0
    ).build_geometry()
    self.T_e_val = 5.0  # keV
    self.n_e_val = 1.0e20  # m^-3
    self.T_i_val = 4.0  # keV

    self.T_e = jnp.full(len(self.geo.rho), self.T_e_val)
    self.n_e = jnp.full(len(self.geo.rho), self.n_e_val)
    self.T_i = jnp.full(len(self.geo.rho), self.T_i_val)

    self.charge_number = 2
    self.mass_number = 3.016

  def test_bimaxwellian_split_basic(self):
    """Test basic functionality of bimaxwellian_split."""

    power_deposition = jnp.ones(len(self.geo.rho))
    minority_concentration = 0.05
    P_total_W = 1.0e6

    n_tail, T_tail = fast_ions.bimaxwellian_split(
        power_deposition=power_deposition,
        T_e=self.T_e,
        n_e=self.n_e,
        T_i=self.T_i,
        minority_concentration=minority_concentration,
        P_total_W=P_total_W,
        charge_number=self.charge_number,
        mass_number=self.mass_number,
    )

    self.assertEqual(n_tail.shape, (len(self.geo.rho),))
    self.assertEqual(T_tail.shape, (len(self.geo.rho),))

    self.assertTrue(jnp.all(T_tail > self.T_e_val))

    n_total = self.n_e_val * minority_concentration

    self.assertTrue(jnp.all(n_tail >= 0))

    self.assertTrue(jnp.all(n_tail < n_total))

  def test_bimaxwellian_split_zero_power(self):
    """Test with zero power, should result in no tail or minimal tail/thermal."""

    power_deposition = jnp.zeros(len(self.geo.rho))
    minority_concentration = 0.05
    P_total_W = 0.0

    n_tail, T_tail = fast_ions.bimaxwellian_split(
        power_deposition=power_deposition,
        T_e=self.T_e,
        n_e=self.n_e,
        T_i=self.T_i,
        minority_concentration=minority_concentration,
        P_total_W=P_total_W,
        charge_number=self.charge_number,
        mass_number=self.mass_number,
    )

    np.testing.assert_allclose(n_tail, 0.0, atol=1e-9)
    np.testing.assert_allclose(T_tail, self.T_e_val)

  def test_bimaxwellian_split_high_power(self):
    """Test with high power to trigger constraints or higher tail fraction."""

    power_deposition = jnp.ones(len(self.geo.rho))
    minority_concentration = 0.03
    n_tail, _ = fast_ions.bimaxwellian_split(
        power_deposition=power_deposition,
        T_e=self.T_e,
        n_e=self.n_e,
        T_i=self.T_i,
        minority_concentration=minority_concentration,
        P_total_W=10.0e6,
        charge_number=self.charge_number,
        mass_number=self.mass_number,
    )

    self.assertTrue(jnp.all(n_tail > 0))

if __name__ == '__main__':
  absltest.main()
