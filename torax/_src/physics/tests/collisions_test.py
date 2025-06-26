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
from torax._src.physics import collisions


# pylint: disable=invalid-name
class CollisionsTest(parameterized.TestCase):

  def test_fast_ion_fractional_heating_formula_ion_heating_limit(self):
    # Inertial energy small compared to critical energy.
    birth_energy = 1e-3
    T_e = jnp.array(0.1, dtype=jnp.float32)
    fast_ion_mass = 1
    frac_i = collisions.fast_ion_fractional_heating_formula(
        birth_energy, T_e, fast_ion_mass
    )
    np.testing.assert_allclose(frac_i, 1.0, atol=1e-3)

  def test_fast_ion_fractional_heating_formula_electron_heating_limit(self):
    # Inertial energy large compared to critical energy.
    birth_energy = 1e10
    T_e = jnp.array(0.1, dtype=jnp.float32)
    fast_ion_mass = 1
    frac_i = collisions.fast_ion_fractional_heating_formula(
        birth_energy, T_e, fast_ion_mass
    )
    np.testing.assert_allclose(frac_i, 0.0, atol=1e-9)

  @parameterized.parameters([
      dict(T_e_ev=1.0, n_e=1.0, expected=31.3),
      dict(T_e_ev=np.exp(8), n_e=np.exp(42), expected=18.3),
  ])
  def test_calculate_lambda_ei(self, T_e_ev, n_e, expected):
    T_e_kev = jnp.array(T_e_ev / 1e3)
    n_e = jnp.array(n_e)
    result = collisions.calculate_lambda_ei(T_e_kev, n_e)
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters([
      dict(T_i_ev=1.0, n_i=1.0, Z_i=1.0, expected=30.0),
      dict(T_i_ev=np.exp(8), n_i=np.exp(42), Z_i=np.exp(1), expected=18.0),
  ])
  def test_calculate_lambda_ii(self, T_i_ev, n_i, Z_i, expected):
    T_i_kev = jnp.array(T_i_ev / 1e3)
    n_i = jnp.array(n_i)
    Z_i = jnp.array(Z_i)
    result = collisions.calculate_lambda_ii(T_i_kev, n_i, Z_i)
    np.testing.assert_allclose(result, expected, atol=1e-6)

  # TODO(b/377225415): generalize to arbitrary number of ions.
  @parameterized.parameters([
      dict(
          A_impurity=20.0,
          Z_impurity=10.0,
          Z_i=1.0,
          A_i=1.0,
          n_i=1.0,
          expected=1.0,
      ),
      dict(
          A_impurity=20.0,
          Z_impurity=10.0,
          Z_i=1.0,
          A_i=2.0,
          n_i=1.0,
          expected=0.5,
      ),
      dict(
          A_impurity=20.0,
          Z_impurity=10.0,
          Z_i=2.0,
          A_i=4.0,
          n_i=0.5,
          expected=0.5,
      ),
      dict(
          A_impurity=20.0,
          Z_impurity=10.0,
          Z_i=1.0,
          A_i=2.0,
          n_i=0.9,
          expected=0.5,
      ),
      dict(
          A_impurity=40.0,
          Z_impurity=20.0,
          Z_i=1.0,
          A_i=2.0,
          n_i=0.92,
          expected=0.5,
      ),
  ])
  def test_calculate_weighted_Z_eff(
      self, A_impurity, Z_impurity, Z_i, A_i, n_i, expected
  ):
    """Compare `_calculate_weighted_Z_eff` to a reference value."""
    n_e = 1.0
    n_impurity = (n_e - n_i * Z_i) / Z_impurity
    core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        n_e=cell_variable.CellVariable(
            value=jnp.array(n_e),
            dr=jnp.array(1.0),
        ),
        n_i=cell_variable.CellVariable(
            value=jnp.array(n_i),
            dr=jnp.array(1.0),
        ),
        n_impurity=cell_variable.CellVariable(
            value=jnp.array(n_impurity),
            dr=jnp.array(1.0),
        ),
        Z_i=Z_i,
        A_i=A_i,
        Z_impurity=Z_impurity,
        A_impurity=A_impurity,
    )
    np.testing.assert_allclose(
        collisions._calculate_weighted_Z_eff(core_profiles), expected
    )


if __name__ == '__main__':
  absltest.main()
