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

"""Unit tests for module torax.charge_states."""

from absl.testing import parameterized
import numpy as np
from torax import charge_states
from torax import constants
from torax.config import plasma_composition


# pylint: disable=invalid-name
class ChargeStatesTest(parameterized.TestCase):
  """Tests for impurity charge states."""

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
      temperature=[0.1, 1.0, [10.0, 20.0], 90.0],
  )
  def test_calculate_average_charge_state_with_impurity(
      self, ion_symbol, temperature
  ):
    """Test with valid ions and within temperature range."""
    Te = np.array(temperature)
    Z_calculated = charge_states.calculate_average_charge_state_single_species(
        Te, ion_symbol
    )
    np.testing.assert_equal(
        Z_calculated.shape,
        Te.shape,
        err_msg=(
            f'Z and T shapes not equal for {ion_symbol} at temperature {Te}. Z'
            f' = {Z_calculated}, Z.shape = {Z_calculated.shape}, Te.shape ='
            f' {Te.shape}.'
        ),
    )
    # Physical sanity checking
    np.testing.assert_array_less(
        Z_calculated,
        np.ones_like(Z_calculated) * constants.ION_PROPERTIES_DICT[ion_symbol].Z
        + 1e-6,
        err_msg=(
            f'Z is not less than Z_max for {ion_symbol} at temperature {Te}. Z'
            f' = {Z_calculated}, Z_max ='
            f' {constants.ION_PROPERTIES_DICT[ion_symbol].Z}.'
        ),
    )

    np.testing.assert_array_less(
        0.0,
        Z_calculated,
        err_msg=(
            f'Unphysical negative Z for {ion_symbol} at temperature {Te}. '
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
    Te = np.array(temperature)
    Z_calcuated = charge_states.calculate_average_charge_state_single_species(
        Te, ion_symbol
    )
    Z_expected = np.ones_like(Te) * constants.ION_PROPERTIES_DICT[ion_symbol].Z
    np.testing.assert_allclose(
        Z_calcuated,
        Z_expected,
        err_msg=(
            f'Low-Z full ionization not as expected for {ion_symbol} for'
            f' Te={Te}, Z_calcualted = {Z_calcuated}, Z_expected={Z_expected}'
        ),
    )

  @parameterized.named_parameters(
      ('Te_low_input', 0.05, 0.1),
      ('Te_high_input', 150.0, 100.0),
  )
  def test_temperature_clipping(self, Te_input, Te_clipped):
    """Test with valid ions and within temperature range."""
    ion_symbol = 'W'
    Z_calculated = charge_states.calculate_average_charge_state_single_species(
        Te_input, ion_symbol
    )
    Z_expected = charge_states.calculate_average_charge_state_single_species(
        Te_clipped,
        ion_symbol,
    )

    np.testing.assert_allclose(
        Z_calculated,
        Z_expected,
        err_msg=(
            f'Te clipping not working as expected for Te_input={Te_input},'
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
          [8.198074, 15.72704044, 19.718317],
      ),
  )
  def test_get_average_charge_state(
      self,
      species,
      Te,
      expected_Z,
  ):
    """Test the get_average_charge_state function, where expected_Z references are pre-calculated."""
    Te = np.array(Te)
    expected_Z = np.array(expected_Z)
    avg_A = 2.0  # arbitrary, not used.
    ion_symbols = tuple(species.keys())
    fractions = np.array(tuple(species.values()))
    ion_mixture = plasma_composition.DynamicIonMixture(
        fractions=fractions,
        avg_A=avg_A,
    )
    Z_calculated = charge_states.get_average_charge_state(
        ion_symbols, ion_mixture, Te
    )

    np.testing.assert_allclose(Z_calculated, expected_Z, rtol=1e-5)

  def test_Z_override_in_get_average_charge_state(self):
    """Test Z_override logic."""
    species = {'W': 1.0}
    Te = np.array([0.1, 2, 10])
    Z_override = np.array([50.0, 50.0, 50.0])
    ion_symbols = tuple(species.keys())
    ion_mixture = plasma_composition.DynamicIonMixture(
        fractions=np.array(tuple(species.values())),
        avg_A=2.0,  # arbitrary, not used.
        Z_override=np.array([50.0, 50.0, 50.0]),
    )
    Z_calculated = charge_states.get_average_charge_state(
        ion_symbols, ion_mixture, Te
    )
    np.testing.assert_allclose(Z_calculated, Z_override)
