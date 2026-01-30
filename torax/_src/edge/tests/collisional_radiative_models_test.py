# Copyright 2025 DeepMind Technologies Limited
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

"""Tests for the Mavrin 2017 collisional radiative fits valid for edge."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.edge import collisional_radiative_models
from torax._src.edge import mavrin_2017_charge_states_data
from torax._src.physics import charge_states as charge_states_core

# pylint: disable=invalid-name


class CollisionalRadiativeModelsTest(parameterized.TestCase):

  def test_temperature_clipping(self):
    """Tests that T_e is correctly clipped to the model's validity range."""
    ion_symbol = 'Ar'
    ne_tau = 5e16
    # Get the valid temperature range for Argon from the model
    min_temp_keV = (
        mavrin_2017_charge_states_data.MIN_TEMPERATURES[ion_symbol] / 1e3
    )
    max_temp_keV = (
        mavrin_2017_charge_states_data.MAX_TEMPERATURES[ion_symbol] / 1e3
    )

    # Test lower bound clipping
    t_e_low = np.array([min_temp_keV / 2.0])
    val_low = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            t_e_low,
            ne_tau,
            ion_symbol,
        )
    )
    val_min_ref = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            np.array([min_temp_keV]),
            ne_tau,
            ion_symbol,
        )
    )
    np.testing.assert_allclose(val_low, val_min_ref)

    # Test upper bound clipping
    t_e_high = np.array([max_temp_keV * 2.0])
    val_high = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            t_e_high,
            ne_tau,
            ion_symbol,
        )
    )
    val_max_ref = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            np.array([max_temp_keV]),
            ne_tau,
            ion_symbol,
        )
    )
    np.testing.assert_allclose(val_high, val_max_ref)

  def test_ne_tau_clipping(self):
    """Tests that ne_tau is correctly capped at the coronal limit."""
    ion_symbol = 'C'
    t_e = np.array([1.0])  # 1 keV
    ne_tau_high = 2e19
    ne_tau_limit = collisional_radiative_models._NE_TAU_CORONAL_LIMIT  # pylint: disable=protected-access

    val_high = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            t_e,
            ne_tau_high,
            ion_symbol,
        )
    )
    val_limit = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            t_e,
            ne_tau_limit,
            ion_symbol,
        )
    )
    np.testing.assert_allclose(val_high, val_limit)

  @parameterized.named_parameters(
      dict(testcase_name='Carbon', ion_symbol='C'),
      dict(testcase_name='Nitrogen', ion_symbol='N'),
      dict(testcase_name='Oxygen', ion_symbol='O'),
      dict(testcase_name='Neon', ion_symbol='Ne'),
      dict(testcase_name='Argon', ion_symbol='Ar'),
  )
  def test_coronal_limit_vs_2018_model(self, ion_symbol):
    """Compares to 2018 model in coronal limit for overlapping ions and T_e range."""
    ne_tau = collisional_radiative_models._NE_TAU_CORONAL_LIMIT  # pylint: disable=protected-access
    t_e_keV = np.array([0.1, 0.2, 0.5, 0.9, 1.5, 9.0])

    z_edge_model = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            t_e_keV,
            ne_tau,
            ion_symbol,
        )
    )
    z_core_model = (
        charge_states_core.calculate_average_charge_state_single_species(
            t_e_keV, ion_symbol
        )
    )

    # The models are based on different fits, so we expect some minor deviation.
    np.testing.assert_allclose(
        z_edge_model,
        z_core_model,
        rtol=3e-2,
        err_msg=f'Mismatch for {ion_symbol}',
    )

  def test_helium_isotope_equivalence(self):
    """Tests that He3 and He4 produce identical results to He."""
    t_e_keV = np.array([0.005, 0.01, 0.1])
    ne_tau = 1e17
    val_he = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            t_e_keV, ne_tau, 'He'
        )
    )
    val_he3 = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            t_e_keV, ne_tau, 'He3'
        )
    )
    val_he4 = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            t_e_keV, ne_tau, 'He4'
        )
    )

    np.testing.assert_allclose(
        val_he3, val_he, err_msg='He3 != He'
    )
    np.testing.assert_allclose(
        val_he4, val_he, err_msg='He4 != He'
    )

  @parameterized.named_parameters(
      # These values are for Z (charge states)
      # The expected values were sanity-checked by manual inspection against the
      # paper plots.
      dict(
          testcase_name='Helium',
          ion_symbol='He',
          expected_z_values=np.array([
              8.324989e-04,
              9.753990e-01,
              1.872604e00,
              1.988668e00,
              1.997732e00,
              1.996799e00,
              1.997318e00,
              1.996017e00,
              1.997188e00,
              1.995265e00,
          ]),
      ),
      dict(
          testcase_name='Lithium',
          ion_symbol='Li',
          expected_z_values=np.array([
              1.095361,
              0.968953,
              1.783344,
              2.780949,
              2.968104,
              2.985503,
              2.995818,
              2.987124,
              2.989245,
              2.984395,
          ]),
      ),
      dict(
          testcase_name='Beryllium',
          ion_symbol='Be',
          expected_z_values=np.array([
              1.267926,
              1.994181,
              2.017934,
              2.904112,
              3.704038,
              3.923132,
              3.963963,
              3.966999,
              3.966429,
              3.939733,
          ]),
      ),
      dict(
          testcase_name='Carbon',
          ion_symbol='C',
          expected_z_values=np.array([
              0.951205,
              2.092229,
              3.702464,
              3.922132,
              4.299599,
              5.172242,
              5.712655,
              5.828535,
              5.882577,
              5.869976,
          ]),
      ),
      dict(
          testcase_name='Nitrogen',
          ion_symbol='N',
          expected_z_values=np.array([
              0.658709,
              1.791184,
              3.956604,
              4.917214,
              5.014153,
              5.632296,
              6.45795,
              6.651093,
              6.748325,
              6.790535,
          ]),
      ),
      dict(
          testcase_name='Oxygen',
          ion_symbol='O',
          expected_z_values=np.array([
              0.357494,
              1.545742,
              3.618413,
              5.550328,
              5.928382,
              6.236587,
              7.067014,
              7.371705,
              7.593031,
              7.69121,
          ]),
      ),
      dict(
          testcase_name='Neon',
          ion_symbol='Ne',
          expected_z_values=np.array([
              0.017778,
              1.552201,
              3.319961,
              5.796926,
              7.299495,
              7.718165,
              8.501275,
              8.906104,
              9.153739,
              9.284849,
          ]),
      ),
      dict(
          testcase_name='Argon',
          ion_symbol='Ar',
          expected_z_values=np.array([
              0.120285,
              2.045676,
              4.446644,
              7.253697,
              8.129242,
              10.443711,
              13.497581,
              14.484606,
              14.90374,
              15.590451,
          ]),
      ),
  )
  def test_Z_hardcoded_values(self, ion_symbol, expected_z_values):
    """Compares model output against hard-coded values for regression testing."""

    # Covers all intervals in the model.
    t_e_keV = np.array([
        0.0015,
        0.005,
        0.015,
        0.04,
        0.09,
        0.2,
        0.5,
        0.9,
        1.5,
        9.0,
    ])
    ne_tau = 1e17  # A representative non-coronal value

    calculated_z = (
        collisional_radiative_models.calculate_mavrin_noncoronal_charge_state(
            t_e_keV,
            ne_tau,
            ion_symbol,
        )
    )

    np.testing.assert_allclose(
        calculated_z,
        expected_z_values,
        err_msg=f'Mismatch for {ion_symbol}',
        atol=1e-5,
        rtol=1e-5,
    )

  def test_calculate_L_INT(self):
    """Tests the _calculate_L_INT function against a reference value."""
    start_temp = 0.01  # keV
    stop_temp = 0.02  # keV
    ne_tau = 0.5e17
    ion_symbol = 'N'

    # Reference value taken from running the same routine in
    # https://github.com/cfs-energy/extended-lengyel
    expected_L_INT = 2.144797293548036e-30

    calculated_L_INT = collisional_radiative_models._calculate_L_INT(
        start_temp=start_temp,
        stop_temp=stop_temp,
        ne_tau=ne_tau,
        ion_symbol=ion_symbol,
    )

    np.testing.assert_allclose(
        calculated_L_INT,
        expected_L_INT,
        rtol=1e-2,
        err_msg='Lint calculation does not match the reference value.',
    )

  def test_calculate_weighted_L_INT(self):
    """Tests the calculate_weighted_L_INT function against a reference value."""
    start_temp = 6.167578954082415e-3  # keV
    stop_temp = 55.02789988290978e-3  # keV
    ne_tau = 0.5e17
    impurity_map = {'N': 1.0, 'Ar': 0.05}

    # Reference value taken from running the same routine in
    # https://github.com/cfs-energy/extended-lengyel
    expected_weighted_L_INT = 7.09255e-30

    calculated_weighted_L_INT = (
        collisional_radiative_models.calculate_weighted_L_INT(
            impurity_map=impurity_map,
            start_temp=start_temp,
            stop_temp=stop_temp,
            ne_tau=ne_tau,
        )
    )

    np.testing.assert_allclose(
        calculated_weighted_L_INT,
        expected_weighted_L_INT,
        rtol=1e-5,
        err_msg=(
            'Weighted L_INT calculation does not match the reference value.'
        ),
    )


if __name__ == '__main__':
  absltest.main()
