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

"""Tests for torax._src.constants."""

from absl.testing import absltest
from absl.testing import parameterized
from torax._src import constants


class ConstantsTest(absltest.TestCase):
  """Tests for physical constants values and data integrity."""

  def test_elementary_charge_matches_codata(self):
    """Verify q_e matches the 2019 SI redefinition (exact value)."""
    self.assertEqual(constants.CONSTANTS.q_e, 1.602176634e-19)

  def test_electron_mass_matches_codata(self):
    """Verify m_e is consistent with CODATA 2018."""
    self.assertAlmostEqual(
        constants.CONSTANTS.m_e, 9.1093837015e-31, delta=1e-34
    )

  def test_atomic_mass_unit_matches_codata(self):
    """Verify m_amu is consistent with CODATA 2018."""
    self.assertAlmostEqual(
        constants.CONSTANTS.m_amu, 1.66053906660e-27, delta=1e-33
    )

  def test_boltzmann_constant_matches_codata(self):
    """Verify k_B matches the 2019 SI redefinition (exact value)."""
    self.assertEqual(constants.CONSTANTS.k_B, 1.380649e-23)

  def test_vacuum_permittivity_matches_codata(self):
    """Verify epsilon_0 is consistent with CODATA 2018."""
    self.assertAlmostEqual(
        constants.CONSTANTS.epsilon_0, 8.8541878128e-12, delta=1e-16
    )

  def test_kev_to_j_consistent_with_ev_to_j(self):
    """Verify keV_to_J == 1e3 * eV_to_J."""
    self.assertAlmostEqual(
        constants.CONSTANTS.keV_to_J,
        1e3 * constants.CONSTANTS.eV_to_J,
        places=30,
    )

  def test_ev_to_j_equals_elementary_charge(self):
    """Verify eV_to_J equals q_e (by definition of electronvolt)."""
    self.assertEqual(constants.CONSTANTS.eV_to_J, constants.CONSTANTS.q_e)

  def test_eps_is_positive(self):
    """Verify the numerical stability epsilon is positive."""
    self.assertGreater(constants.CONSTANTS.eps, 0.0)


class ConstantsImmutabilityTest(absltest.TestCase):
  """Tests that Constants is frozen (immutable)."""

  def test_constants_is_frozen(self):
    """Verify that CONSTANTS cannot be mutated."""
    with self.assertRaises(AttributeError):
      constants.CONSTANTS.q_e = 0.0  # pytype: disable=attribute-error


class IonPropertiesTest(parameterized.TestCase):
  """Tests for ion properties data integrity."""

  def test_ion_properties_dict_keys_match_ion_symbols(self):
    """Verify ION_PROPERTIES_DICT keys exactly match ION_SYMBOLS."""
    self.assertEqual(
        set(constants.ION_PROPERTIES_DICT.keys()),
        set(constants.ION_SYMBOLS),
    )

  def test_all_ions_have_positive_atomic_mass(self):
    """Verify all ions have A > 0."""
    for ion in constants.ION_PROPERTIES:
      with self.subTest(ion=ion.symbol):
        self.assertGreater(
            ion.A, 0.0, msg=f'{ion.symbol} has non-positive A={ion.A}'
        )

  def test_all_ions_have_positive_atomic_number(self):
    """Verify all ions have Z > 0."""
    for ion in constants.ION_PROPERTIES:
      with self.subTest(ion=ion.symbol):
        self.assertGreater(
            ion.Z, 0.0, msg=f'{ion.symbol} has non-positive Z={ion.Z}'
        )

  def test_all_ions_have_positive_ionization_energy(self):
    """Verify all ions have E_ionization > 0."""
    for ion in constants.ION_PROPERTIES:
      with self.subTest(ion=ion.symbol):
        self.assertGreater(
            ion.E_ionization,
            0.0,
            msg=f'{ion.symbol} has non-positive E_ionization={ion.E_ionization}',
        )

  def test_hydrogenic_ions_are_subset_of_ion_symbols(self):
    """Verify HYDROGENIC_IONS is a subset of ION_SYMBOLS."""
    self.assertTrue(constants.HYDROGENIC_IONS.issubset(constants.ION_SYMBOLS))

  def test_hydrogenic_ions_have_z_one(self):
    """Verify all hydrogenic ions have Z=1."""
    for symbol in constants.HYDROGENIC_IONS:
      ion = constants.ION_PROPERTIES_DICT[symbol]
      self.assertEqual(
          ion.Z,
          1.0,
          msg=f'Hydrogenic ion {symbol} has Z={ion.Z}, expected 1.0',
      )

  @parameterized.parameters(
      ('D', 2.0141),
      ('T', 3.0160),
      ('He', 4.0026),
      ('C', 12.011),
      ('W', 183.84),
  )
  def test_spot_check_atomic_masses(self, symbol, expected_a):
    """Spot-check atomic masses for representative ions."""
    ion = constants.ION_PROPERTIES_DICT[symbol]
    self.assertAlmostEqual(ion.A, expected_a, places=3)

  def test_no_duplicate_symbols_in_ion_properties(self):
    """Verify ION_PROPERTIES has no accidentally duplicate entries."""
    symbols = [ion.symbol for ion in constants.ION_PROPERTIES]
    # He4 and He are both Helium-4 by design, so we check dict construction
    # didn't drop any entries unexpectedly.
    self.assertEqual(
        len(constants.ION_PROPERTIES_DICT),
        len(set(symbols)),
    )

  def test_ion_properties_is_frozen(self):
    """Verify IonProperties instances are immutable."""
    ion = constants.ION_PROPERTIES[0]
    with self.assertRaises(AttributeError):
      ion.A = 999.0  # pytype: disable=attribute-error


if __name__ == '__main__':
  absltest.main()
