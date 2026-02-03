# Copyright 2026 DeepMind Technologies Limited
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
from torax._src.physics.radiation import mavrin_coronal_cooling_rate
from torax._src.physics.radiation import mavrin_noncoronal_cooling_rate
from torax._src.physics.radiation import radiation


# pylint: disable=invalid-name


class RadiationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_coronal',
          model_type=radiation.MavrinModelType.CORONAL,
          data_module=mavrin_coronal_cooling_rate,
      ),
      dict(
          testcase_name='_noncoronal',
          model_type=radiation.MavrinModelType.NONCORONAL,
          data_module=mavrin_noncoronal_cooling_rate,
      ),
  )
  def test_temperature_clipping(self, model_type, data_module):
    """Tests that T_e is correctly clipped to the model's validity range."""
    ion_symbol = 'Ar'
    ne_tau = 5e16
    T_e_min = data_module.MIN_TEMPERATURES[ion_symbol]
    T_e_max = data_module.MAX_TEMPERATURES[ion_symbol]

    # Test lower bound clipping
    T_e_low = np.array([T_e_min / 2.0])
    val_low = radiation.calculate_mavrin_cooling_rate(
        T_e_low, ion_symbol, model_type, ne_tau=ne_tau
    )
    val_min_ref = radiation.calculate_mavrin_cooling_rate(
        np.array([T_e_min]), ion_symbol, model_type, ne_tau=ne_tau
    )
    np.testing.assert_allclose(val_low, val_min_ref)

    # Test upper bound clipping
    T_e_high = np.array([T_e_max * 2.0])
    val_high = radiation.calculate_mavrin_cooling_rate(
        T_e_high,
        ion_symbol,
        model_type,
        ne_tau=ne_tau,
    )
    val_max_ref = radiation.calculate_mavrin_cooling_rate(
        np.array([T_e_max]),
        ion_symbol,
        model_type,
        ne_tau=ne_tau,
    )
    np.testing.assert_allclose(val_high, val_max_ref)

  def test_ne_tau_clipping(self):
    """Tests that ne_tau is correctly capped at the coronal limit."""
    ion_symbol = 'C'
    T_e = np.array([1.0])
    ne_tau_limit = radiation._NE_TAU_CORONAL_LIMIT  # pylint: disable=protected-access
    ne_tau_high = 2 * ne_tau_limit

    val_high = radiation.calculate_mavrin_cooling_rate(
        T_e,
        ion_symbol,
        radiation.MavrinModelType.NONCORONAL,
        ne_tau=ne_tau_high,
    )
    val_limit = radiation.calculate_mavrin_cooling_rate(
        T_e,
        ion_symbol,
        radiation.MavrinModelType.NONCORONAL,
        ne_tau=ne_tau_limit,
    )
    np.testing.assert_allclose(val_high, val_limit)

  def test_noncoronal_coronal_limit_vs_coronal_model(self):
    """Compares noncoronal model in coronal limit to coronal model for Ar."""
    ion_symbol = 'Ar'
    ne_tau = radiation._NE_TAU_CORONAL_LIMIT  # pylint: disable=protected-access
    T_e = np.array([0.1, 0.2, 0.5, 0.9, 1.5, 9.0])

    lz_noncoronal_limit = radiation.calculate_mavrin_cooling_rate(
        T_e,
        ion_symbol,
        radiation.MavrinModelType.NONCORONAL,
        ne_tau=ne_tau,
    )
    lz_coronal = radiation.calculate_mavrin_cooling_rate(
        T_e,
        ion_symbol,
        radiation.MavrinModelType.CORONAL,
    )

    # The 2017 model Lz "coronal limit" is only valid for low temperatures.
    # However for Ar it begins to approach the true coronal limit also at higher
    # temperatures so we can compare specifically for Argon.
    np.testing.assert_allclose(lz_noncoronal_limit, lz_coronal, rtol=1e-1)

  @parameterized.named_parameters(
      dict(
          testcase_name='_coronal',
          model_type=radiation.MavrinModelType.CORONAL,
      ),
      dict(
          testcase_name='_noncoronal',
          model_type=radiation.MavrinModelType.NONCORONAL,
      ),
  )
  def test_helium_isotope_equivalence(self, model_type):
    """Tests that He3 and He4 produce identical results to He."""
    T_e = np.array([0.005, 0.01, 0.1])
    ne_tau = 1e17

    val_He = radiation.calculate_mavrin_cooling_rate(
        T_e, 'He', model_type, ne_tau=ne_tau
    )
    val_He3 = radiation.calculate_mavrin_cooling_rate(
        T_e, 'He3', model_type, ne_tau=ne_tau
    )
    val_He4 = radiation.calculate_mavrin_cooling_rate(
        T_e, 'He4', model_type, ne_tau=ne_tau
    )

    np.testing.assert_allclose(
        val_He3, val_He, err_msg=f'He3 != He for {model_type}'
    )
    np.testing.assert_allclose(
        val_He4, val_He, err_msg=f'He4 != He for {model_type}'
    )

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
  def test_noncoronal_cooling_rate_hardcoded_values(
      self, ion_symbol, expected_lz_values
  ):
    """Compares model output against hard-coded values for regression testing."""
    T_e = np.array([
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

    calculated_lz = radiation.calculate_mavrin_cooling_rate(
        T_e,
        ion_symbol,
        radiation.MavrinModelType.NONCORONAL,
        ne_tau=ne_tau,
    )

    np.testing.assert_allclose(
        calculated_lz,
        expected_lz_values,
        err_msg=f'Mismatch for {ion_symbol}',
        rtol=1e-4,
    )

  @parameterized.named_parameters(
      (
          'Helium-3',
          'He3',
          [2.26267051e-36, 3.55291080e-36, 6.25952387e-36],
      ),
      (
          'Helium-4',
          'He4',
          [2.26267051e-36, 3.55291080e-36, 6.25952387e-36],
      ),
      (
          'Lithium',
          'Li',
          [1.37075024e-35, 9.16765402e-36, 1.60346076e-35],
      ),
      (
          'Beryllium',
          'Be',
          [6.86895406e-35, 1.88578938e-35, 3.04614535e-35],
      ),
      (
          'Carbon',
          'C',
          [6.74683566e-34, 5.89332177e-35, 7.94786067e-35],
      ),
      (
          'Nitrogen',
          'N',
          [6.97912189e-34, 9.68950644e-35, 1.15250226e-34],
      ),
      (
          'Oxygen',
          'O',
          [4.10676301e-34, 1.57469152e-34, 1.58599054e-34],
      ),
      (
          'Neon',
          'Ne',
          [1.19151664e-33, 3.27468464e-34, 2.82416557e-34],
      ),
      (
          'Argon',
          'Ar',
          [1.92265224e-32, 4.02388371e-33, 1.53295491e-33],
      ),
      (
          'Krypton',
          'Kr',
          [6.57654706e-32, 3.23512795e-32, 7.53285680e-33],
      ),
      (
          'Xenon',
          'Xe',
          [2.89734288e-31, 8.96916315e-32, 2.87740863e-32],
      ),
      (
          'Tungsten',
          'W',
          [1.66636258e-31, 4.46651033e-31, 1.31222935e-31],
      ),
  )
  def test_coronal_cooling_rate_hardcoded_values(
      self,
      ion_symbol,
      expected_LZ,
  ):
    T_e = np.array([0.1, 2, 10])
    calculated_LZ = radiation.calculate_mavrin_cooling_rate(
        T_e,
        ion_symbol,
        radiation.MavrinModelType.CORONAL,
    )
    np.testing.assert_allclose(calculated_LZ, expected_LZ, rtol=1e-5)

  def test_cooling_rate_below_t_min_noncoronal_is_extrapolated(self):
    ion_symbol = 'Ar'
    ne_tau = 1e17
    T_e_min_noncoronal = mavrin_noncoronal_cooling_rate.MIN_TEMPERATURES[
        ion_symbol
    ]
    rate_at_min = radiation.calculate_cooling_rate(
        np.array([T_e_min_noncoronal]), ion_symbol, ne_tau=ne_tau
    )
    rate_below_min = radiation.calculate_cooling_rate(
        np.array([T_e_min_noncoronal / 2.0]), ion_symbol, ne_tau=ne_tau
    )
    self.assertGreater(rate_at_min[0], rate_below_min[0])
    # w is approx 0 in this regime, so cooling rate should be approx
    # noncoronal cooling rate, which is linearly extrapolated.
    np.testing.assert_allclose(
        rate_below_min[0], rate_at_min[0] / 2.0, rtol=1e-4
    )

  def test_cooling_rate_in_upper_coronal_regime_matches_coronal_model(self):
    ion_symbol = 'Ar'
    ne_tau = 1e17
    T_e_min_coronal = mavrin_coronal_cooling_rate.MIN_TEMPERATURES[ion_symbol]
    # At 5 * T_e_min_coronal, sigmoid weight w should be ~1.
    T_e = np.array([T_e_min_coronal * 5])
    cooling_rate = radiation.calculate_cooling_rate(
        T_e, ion_symbol, ne_tau=ne_tau
    )
    coronal_cooling_rate = radiation.calculate_mavrin_cooling_rate(
        T_e, ion_symbol, radiation.MavrinModelType.CORONAL, ne_tau=ne_tau
    )
    np.testing.assert_allclose(cooling_rate, coronal_cooling_rate, rtol=1e-2)

  def test_cooling_rate_in_lower_noncoronal_regime_matches_noncoronal_model(
      self,
  ):
    ion_symbol = 'Ar'
    ne_tau = 1e17
    T_e_min_coronal = mavrin_coronal_cooling_rate.MIN_TEMPERATURES[ion_symbol]
    T_e_min_noncoronal = mavrin_noncoronal_cooling_rate.MIN_TEMPERATURES[
        ion_symbol
    ]
    # At T_e_min_coronal / 4, sigmoid weight w should be ~0.
    # This temperature must be >= T_e_min_noncoronal for non-coronal model
    # to be in fit range (not extrapolated range).
    T_e = np.array([T_e_min_coronal / 4.0])
    self.assertGreaterEqual(T_e[0], T_e_min_noncoronal)

    cooling_rate = radiation.calculate_cooling_rate(
        T_e, ion_symbol, ne_tau=ne_tau
    )
    noncoronal_cooling_rate = radiation.calculate_mavrin_cooling_rate(
        T_e, ion_symbol, radiation.MavrinModelType.NONCORONAL, ne_tau=ne_tau
    )
    np.testing.assert_allclose(cooling_rate, noncoronal_cooling_rate, rtol=1e-2)


if __name__ == '__main__':
  absltest.main()
