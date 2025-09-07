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
from torax._src import constants
from torax._src.physics import charge_states


# pylint: disable=invalid-name
class ChargeStatesTest(parameterized.TestCase):

  @parameterized.product(
      ion_symbol=[
          'C',
          'N',
          'O',
          'Ne',
          'Ar',
          'Kr',
          'Xe',
          'W',
      ],
      temperature=[[0.1], [1.0], [10.0, 20.0], [90.0]],
  )
  def test_calculate_average_charge_state_with_impurity(
      self, ion_symbol, temperature
  ):
    """Test with valid ions and within temperature range."""
    T_e = np.array(temperature)
    Z_calculated = charge_states.calculate_average_charge_state_single_species(
        T_e, ion_symbol
    )
    np.testing.assert_equal(
        Z_calculated.shape,
        T_e.shape,
        err_msg=(
            f'Z and T shapes not equal for {ion_symbol} at temperature {T_e}. Z'
            f' = {Z_calculated}, Z.shape = {Z_calculated.shape}, T_e.shape ='
            f' {T_e.shape}.'
        ),
    )
    # Physical sanity checking
    np.testing.assert_array_less(
        Z_calculated,
        np.ones_like(Z_calculated) * constants.ION_PROPERTIES_DICT[ion_symbol].Z
        + 1e-6,
        err_msg=(
            f'Z is not less than Z_max for {ion_symbol} at temperature {T_e}. Z'
            f' = {Z_calculated}, Z_max ='
            f' {constants.ION_PROPERTIES_DICT[ion_symbol].Z}.'
        ),
    )

    np.testing.assert_array_less(
        0.0,
        Z_calculated,
        err_msg=(
            f'Unphysical negative Z for {ion_symbol} at temperature {T_e}. '
            f'Z = {Z_calculated}.'
        ),
    )

  @parameterized.product(
      ion_symbol=[
          'H',
          'D',
          'T',
          'He3',
          'He4',
      ],
      temperature=[0.1, 1.0, 10.0],
  )
  def test_calculate_average_charge_state_with_low_Z_ion(
      self, ion_symbol, temperature
  ):
    """Test with valid ions and within temperature range."""
    T_e = np.array([temperature])
    Z_calculated = charge_states.calculate_average_charge_state_single_species(
        T_e, ion_symbol
    )
    Z_expected = np.ones_like(T_e) * constants.ION_PROPERTIES_DICT[ion_symbol].Z
    np.testing.assert_allclose(
        Z_calculated,
        Z_expected,
        err_msg=(
            f'Low-Z full ionization not as expected for {ion_symbol} for'
            f' T_e={T_e}, Z_calculated = {Z_calculated},'
            f' Z_expected={Z_expected}'
        ),
    )

  @parameterized.named_parameters(
      ('T_e_low_input', 0.05, 0.1),
      ('T_e_high_input', 150.0, 100.0),
  )
  def test_temperature_clipping(self, T_e_input, T_e_clipped):
    """Test with valid ions and within temperature range."""
    ion_symbol = 'W'
    Z_calculated = charge_states.calculate_average_charge_state_single_species(
        np.array([T_e_input]), ion_symbol
    )
    Z_expected = charge_states.calculate_average_charge_state_single_species(
        np.array([T_e_clipped]),
        ion_symbol,
    )

    np.testing.assert_allclose(
        Z_calculated,
        Z_expected,
        err_msg=(
            f'T_e clipping not working as expected for T_e_input={T_e_input},'
            f' Z_calculated = {Z_calculated}, Z_expected={Z_expected}'
        ),
    )

  def test_calculate_average_charge_state_invalid_ion(self):
    """Test that an invalid ion symbol raises an error."""
    with self.assertRaisesRegex(ValueError, 'Invalid ion symbol'):
      charge_states.calculate_average_charge_state_single_species(
          np.array([1.0]), 'Xx'
      )

  @parameterized.named_parameters(
      ('Carbon', {'C': 1.0}, [0.1, 2, 10], [5.2862, 6.0, 6.0]),
      ('Nitrogen', {'N': 1.0}, [0.1, 2, 10], [5.32042, 7.0, 7.0]),
      ('Oxygen', {'O': 1.0}, [0.1, 2, 10], [6.0341, 8.0, 8.0]),
      ('Neon', {'Ne': 1.0}, [0.1, 2, 10], [7.9777, 9.98704209, 10.0]),
      ('Argon', {'Ar': 1.0}, [0.1, 2, 10], [8.6427, 16.606264, 17.9163]),
      ('Krypton', {'Kr': 1.0}, [0.1, 2, 10], [12.66, 27.6349, 33.8548]),
      ('Xenon', {'Xe': 1.0}, [0.1, 2, 10], [12.0, 35.382961, 46.965]),
      ('Tungsten', {'W': 1.0}, [0.1, 2, 10], [13.453, 33.65923, 54.447]),
      (
          'Mixture',
          {
              'C': 0.2,
              'N': 0.2,
              'O': 0.1,
              'Ne': 0.1,
              'Ar': 0.1,
              'Kr': 0.1,
              'Xe': 0.1,
              'W': 0.1,
          },
          [0.1, 2, 10],
          # Calculated by calculating (in a Colab) <Z^2>/<Z> based on the
          # non-mixture expected values, and provided mixture fractions
          [9.42307536, 23.895752, 35.35459338],
      ),
  )
  def test_get_average_charge_state(
      self,
      species,
      T_e,
      expected_Z,
  ):
    """Test the get_average_charge_state function, where expected_Z references are pre-calculated."""
    T_e = np.array(T_e)
    expected_Z = np.array(expected_Z)
    ion_symbols = tuple(species.keys())
    fractions = np.array(tuple(species.values()))
    Z_calculated = charge_states.get_average_charge_state(
        ion_symbols,
        T_e,
        fractions,
        Z_override=None,
    ).Z_mixture

    np.testing.assert_allclose(Z_calculated, expected_Z, rtol=1e-5)

  def test_Z_override_in_get_average_charge_state(self):
    """Test Z_override logic."""
    species = {'W': 1.0}
    T_e = np.array([0.1, 2, 10])
    Z_override = np.array([50.0, 50.0, 50.0])
    ion_symbols = tuple(species.keys())
    Z_calculated = charge_states.get_average_charge_state(
        ion_symbols,
        T_e,
        np.array(tuple(species.values())),
        np.array([50.0, 50.0, 50.0]),
    ).Z_mixture
    np.testing.assert_allclose(Z_calculated, Z_override)


if __name__ == '__main__':
  absltest.main()
