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

"""Tests for the 2017 Mavrin charge state model valid for the tokamak edge."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.physics import charge_states
from torax._src.physics import charge_states_edge

# pylint: disable=invalid-name


class ChargeStatesEdgeTest(parameterized.TestCase):

  def test_temperature_clipping(self):
    """Tests that T_e is correctly clipped to the model's validity range."""
    ion_symbol = 'Ar'
    ne_tau = 5e16
    # Get the valid temperature range for Tungsten from the model
    min_temp_keV = charge_states_edge._MIN_TEMPERATURES[ion_symbol] / 1e3
    max_temp_keV = charge_states_edge._MAX_TEMPERATURES[ion_symbol] / 1e3

    # Test lower bound clipping
    t_e_low = np.array([min_temp_keV / 2.0])
    z_low = charge_states_edge.calculate_average_charge_single_species(
        t_e_low, ne_tau, ion_symbol
    )
    z_min_ref = charge_states_edge.calculate_average_charge_single_species(
        np.array([min_temp_keV]), ne_tau, ion_symbol
    )
    np.testing.assert_allclose(z_low, z_min_ref)

    # Test upper bound clipping
    t_e_high = np.array([max_temp_keV * 2.0])
    z_high = charge_states_edge.calculate_average_charge_single_species(
        t_e_high, ne_tau, ion_symbol
    )
    z_max_ref = charge_states_edge.calculate_average_charge_single_species(
        np.array([max_temp_keV]), ne_tau, ion_symbol
    )
    np.testing.assert_allclose(z_high, z_max_ref)

  def test_ne_tau_clipping(self):
    """Tests that ne_tau is correctly capped at the coronal limit."""
    ion_symbol = 'C'
    t_e = np.array([1.0])  # 1 keV
    ne_tau_high = 2e19
    ne_tau_limit = 1e19

    z_high = charge_states_edge.calculate_average_charge_single_species(
        t_e, ne_tau_high, ion_symbol
    )
    z_limit = charge_states_edge.calculate_average_charge_single_species(
        t_e, ne_tau_limit, ion_symbol
    )
    np.testing.assert_allclose(z_high, z_limit)

  @parameterized.named_parameters(
      dict(testcase_name='Carbon', ion_symbol='C'),
      dict(testcase_name='Nitrogen', ion_symbol='N'),
      dict(testcase_name='Oxygen', ion_symbol='O'),
      dict(testcase_name='Neon', ion_symbol='Ne'),
      dict(testcase_name='Argon', ion_symbol='Ar'),
  )
  def test_coronal_limit_vs_2018_model(self, ion_symbol):
    """Compares to 2018 model in coronal limit for overlapping ions and T_e range."""
    ne_tau = 1e19  # Coronal limit
    t_e_keV = np.array([0.1, 0.2, 0.5, 0.9, 1.5, 9.0])

    z_2017 = charge_states_edge.calculate_average_charge_single_species(
        t_e_keV, ne_tau, ion_symbol
    )
    z_2018 = charge_states.calculate_average_charge_state_single_species(
        t_e_keV, ion_symbol
    )

    # The models are based on different fits, so we expect some minor deviation.
    np.testing.assert_allclose(
        z_2017, z_2018, rtol=3e-2, err_msg=f'Mismatch for {ion_symbol}'
    )

  @parameterized.named_parameters(
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
  def test_hardcoded_values(self, ion_symbol, expected_z_values):
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

    calculated_z = charge_states_edge.calculate_average_charge_single_species(
        t_e_keV, ne_tau, ion_symbol
    )

    # The expected values were also sanity-checked by manual inspection against
    # the paper plots.
    np.testing.assert_allclose(
        calculated_z,
        expected_z_values,
        err_msg=f'Mismatch for {ion_symbol}',
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()
