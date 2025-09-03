# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the 2017 Mavrin radiative cooling rate model valid for the tokamak edge."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit_edge

# pylint: disable=invalid-name


class ImpurityRadiationMavrinFitEdgeTest(parameterized.TestCase):
  """Tests for the impurity_radiation_mavrin_fit_edge module."""

  def test_temperature_clipping(self):
    """Tests that T_e is correctly clipped to the model's validity range."""
    ion_symbol = 'Ar'
    ne_tau = 5e16
    # Get the valid temperature range for Argon from the model
    min_temp_keV = (
        impurity_radiation_mavrin_fit_edge._MIN_TEMPERATURES[ion_symbol] / 1e3
    )
    max_temp_keV = (
        impurity_radiation_mavrin_fit_edge._MAX_TEMPERATURES[ion_symbol] / 1e3
    )

    # Test lower bound clipping
    t_e_low = np.array([min_temp_keV / 2.0])
    lz_low = impurity_radiation_mavrin_fit_edge.calculate_radiative_cooling_rate_single_species(
        t_e_low, ne_tau, ion_symbol
    )
    lz_min_ref = impurity_radiation_mavrin_fit_edge.calculate_radiative_cooling_rate_single_species(
        np.array([min_temp_keV]), ne_tau, ion_symbol
    )
    np.testing.assert_allclose(lz_low, lz_min_ref)

    # Test upper bound clipping
    t_e_high = np.array([max_temp_keV * 2.0])
    lz_high = impurity_radiation_mavrin_fit_edge.calculate_radiative_cooling_rate_single_species(
        t_e_high, ne_tau, ion_symbol
    )
    lz_max_ref = impurity_radiation_mavrin_fit_edge.calculate_radiative_cooling_rate_single_species(
        np.array([max_temp_keV]), ne_tau, ion_symbol
    )
    np.testing.assert_allclose(lz_high, lz_max_ref)

  def test_ne_tau_clipping(self):
    """Tests that ne_tau is correctly capped at the coronal limit."""
    ion_symbol = 'C'
    t_e = np.array([1.0])
    ne_tau_high = 2e19
    ne_tau_limit = 1e19

    lz_high = impurity_radiation_mavrin_fit_edge.calculate_radiative_cooling_rate_single_species(
        t_e, ne_tau_high, ion_symbol
    )
    lz_limit = impurity_radiation_mavrin_fit_edge.calculate_radiative_cooling_rate_single_species(
        t_e, ne_tau_limit, ion_symbol
    )
    np.testing.assert_allclose(lz_high, lz_limit)

  def test_Ar_coronal_limit_vs_2018_model(self):
    """Compares to 2018 model in coronal limit for overlapping ions and T_e range."""
    ion_symbol = 'Ar'
    ne_tau = 1e19  # Coronal limit
    t_e_keV = np.array([0.1, 0.2, 0.5, 0.9, 1.5, 9.0])

    lz_2017 = impurity_radiation_mavrin_fit_edge.calculate_radiative_cooling_rate_single_species(
        t_e_keV, ne_tau, ion_symbol
    )
    lz_2018 = impurity_radiation_mavrin_fit.calculate_total_impurity_radiation(
        (ion_symbol,), np.array([1.0]), t_e_keV
    )

    # The 2017 model Lz "coronal limit" is only valid for low temperatures.
    # However for Ar it begins to approach the true coronal limit also at higher
    # temperatures so we can compare specifically for Argon.
    np.testing.assert_allclose(lz_2017, lz_2018, rtol=1e-1)

  @parameterized.named_parameters(
      dict(
          testcase_name='Helium',
          ion_symbol='He',
          expected_lz_values=np.array([
              7.164384e-38,
              5.618427e-35,
              3.851217e-34,
              1.705402e-34,
              1.114522e-34,
              8.765788e-35,
              7.916563e-35,
              7.952525e-35,
              8.229452e-35,
              9.990870e-35,
          ]),
      ),
      dict(
          testcase_name='Lithium',
          ion_symbol='Li',
          expected_lz_values=np.array([
              1.116119e-35,
              3.059060e-36,
              2.951917e-34,
              8.614428e-34,
              5.089874e-34,
              3.424938e-34,
              2.770282e-34,
              2.645015e-34,
              2.593414e-34,
              2.624211e-34,
          ]),
      ),
      dict(
          testcase_name='Beryllium',
          ion_symbol='Be',
          expected_lz_values=np.array([
              1.618171e-32,
              7.385397e-34,
              2.325928e-34,
              1.037398e-33,
              1.442003e-33,
              9.639416e-34,
              7.215475e-34,
              6.751719e-34,
              6.718236e-34,
              7.636536e-34,
          ]),
      ),
      dict(
          testcase_name='Carbon',
          ion_symbol='C',
          expected_lz_values=np.array([
              3.981443e-34,
              2.795115e-32,
              1.896847e-32,
              2.307938e-33,
              2.598795e-33,
              3.290844e-33,
              2.534101e-33,
              2.065501e-33,
              1.793180e-33,
              1.328436e-33,
          ]),
      ),
      dict(
          testcase_name='Nitrogen',
          ion_symbol='N',
          expected_lz_values=np.array([
              7.448264e-35,
              1.229151e-32,
              5.084176e-32,
              5.970419e-33,
              2.890315e-33,
              4.438263e-33,
              4.453883e-33,
              3.610470e-33,
              3.099060e-33,
              2.173703e-33,
          ]),
      ),
      dict(
          testcase_name='Oxygen',
          ion_symbol='O',
          expected_lz_values=np.array([
              1.167357e-35,
              5.926321e-33,
              4.840825e-32,
              2.159402e-32,
              5.305097e-33,
              5.252147e-33,
              6.042805e-33,
              5.204022e-33,
              4.632967e-33,
              3.280800e-33,
          ]),
      ),
      dict(
          testcase_name='Neon',
          ion_symbol='Ne',
          expected_lz_values=np.array([
              1.898906e-37,
              6.353051e-34,
              2.240690e-32,
              5.370902e-32,
              2.028828e-32,
              8.839085e-33,
              7.999183e-33,
              7.344319e-33,
              6.866852e-33,
              7.244497e-33,
          ]),
      ),
      dict(
          testcase_name='Argon',
          ion_symbol='Ar',
          expected_lz_values=np.array([
              1.856651e-35,
              1.380534e-32,
              2.081830e-31,
              7.461065e-32,
              3.026853e-32,
              4.361645e-32,
              2.588634e-32,
              1.595603e-32,
              1.216467e-32,
              6.767691e-33,
          ]),
      ),
  )
  def test_hardcoded_values(self, ion_symbol, expected_lz_values):
    """Compares model output against hard-coded values for regression testing."""
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

    calculated_lz = impurity_radiation_mavrin_fit_edge.calculate_radiative_cooling_rate_single_species(
        t_e_keV, ne_tau, ion_symbol
    )

    # The expected values were also sanity-checked by manual inspection against
    # the paper plots.
    np.testing.assert_allclose(
        calculated_lz,
        expected_lz_values,
        err_msg=f'Mismatch for {ion_symbol}',
        rtol=1e-4,
    )


if __name__ == '__main__':
  absltest.main()
