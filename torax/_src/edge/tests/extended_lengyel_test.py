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
import numpy as np
from torax._src.edge import divertor_sol_1d
from torax._src.edge import extended_lengyel
from torax._src.edge import extended_lengyel_defaults

# pylint: disable=invalid-name


class ExtendedLengyelTest(absltest.TestCase):

  def test_solve_for_c_z(self):
    """Test _solve_for_c_z against reference values."""

    # The plasma state is based on the second loop of
    # the reference case in https://github.com/cfs-energy/extended-lengyel

    sol_state = divertor_sol_1d.DivertorSOL1D(
        q_parallel=5.06193577e8,
        divertor_Z_eff=2.291360670810858,
        target_electron_temp=2.34,
        SOL_conduction_fraction=extended_lengyel_defaults.SOL_CONDUCTION_FRACTION,
        divertor_broadening_factor=extended_lengyel_defaults.DIVERTOR_BROADENING_FACTOR,
        divertor_parallel_length=5.0,
        parallel_connection_length=20.0,
        separatrix_mach_number=extended_lengyel_defaults.SEPARATRIX_MACH_NUMBER,
        separatrix_electron_density=3.3e19,
        separatrix_ratio_of_ion_to_electron_temp=extended_lengyel_defaults.SEPARATRIX_RATIO_ION_TO_ELECTRON_TEMP,
        separatrix_ratio_of_electron_to_ion_density=extended_lengyel_defaults.SEPARATRIX_RATIO_ELECTRON_TO_ION_DENSITY,
        average_ion_mass=2.0,
        sheath_heat_transmission_factor=extended_lengyel_defaults.SHEATH_HEAT_TRANSMISSION_FACTOR,
        target_mach_number=extended_lengyel_defaults.TARGET_MACH_NUMBER,
        target_ratio_of_ion_to_electron_temp=extended_lengyel_defaults.TARGET_RATIO_ION_TO_ELECTRON_TEMP,
        target_ratio_of_electron_to_ion_density=extended_lengyel_defaults.TARGET_RATIO_ELECTRON_TO_ION_DENSITY,
        toroidal_flux_expansion=extended_lengyel_defaults.TOROIDAL_FLUX_EXPANSION,
    )

    calculated_c_z, status = extended_lengyel._solve_for_c_z(
        divertor_sol_1d=sol_state,
        seed_impurity_weights={'N': 1.0, 'Ar': 0.05},
        fixed_impurity_concentrations={'He': 0.01},
        ne_tau=extended_lengyel_defaults.NE_TAU,
    )

    expected_c_z = 0.06323862137705387

    self.assertEqual(status, extended_lengyel.SolveCzStatus.SUCCESS)
    np.testing.assert_allclose(
        calculated_c_z,
        expected_c_z,
        rtol=1e-4,
    )

  def test_run_extended_lengyel_model_inverse_mode(self):
    """Integration test for the full extended_lengyel model in inverse mode."""
    # Input parameters for the test case. Rest are kept as defaults.
    _RTOL = 5e-4
    inputs = {
        'target_electron_temp': 2.34,
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'seed_impurity_weights': {'N': 1.0, 'Ar': 0.05},
        'fixed_impurity_concentrations': {'He': 0.01},
        'magnetic_field_on_axis': 2.5,
        'plasma_current': 1.0e6,
        'parallel_connection_length': 20.0,
        'divertor_parallel_length': 5.0,
        'major_radius': 1.65,
        'minor_radius': 0.5,
        'elongation_psi95': 1.6,
        'triangularity_psi95': 0.3,
        'average_ion_mass': 2.0,
        'computation_mode': extended_lengyel.ComputationMode.INVERSE,
    }

    # --- Expected output values ---
    # Reference values from running the reference case in:
    # https://github.com/cfs-energy/extended-lengyel
    expected_outputs = {
        'neutral_pressure_in_divertor': 1.737773924511501,
        'alpha_t': 0.35908862950459736,
        'q_parallel': 3.64822996e8,
        'heat_flux_perp_to_target': 7.92853e5,
        'separatrix_electron_temp': 0.1028445648,  # in keV
        'separatrix_Z_eff': 1.8621973566614212,
        'impurity_concentrations': {
            'N': 0.038397305226362526,
            'Ar': 0.0019198652613181264,
        },
    }

    # Run the model
    outputs = extended_lengyel.run_extended_lengyel_model(**inputs)

    # --- Assertions ---
    np.testing.assert_allclose(
        outputs.neutral_pressure_in_divertor,
        expected_outputs['neutral_pressure_in_divertor'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.alpha_t,
        expected_outputs['alpha_t'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.q_parallel,
        expected_outputs['q_parallel'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.heat_flux_perp_to_target,
        expected_outputs['heat_flux_perp_to_target'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.separatrix_electron_temp,
        expected_outputs['separatrix_electron_temp'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.separatrix_Z_eff,
        expected_outputs['separatrix_Z_eff'],
        rtol=_RTOL,
    )
    for impurity, conc in expected_outputs['impurity_concentrations'].items():
      self.assertIn(impurity, outputs.impurity_concentrations)
      np.testing.assert_allclose(
          outputs.impurity_concentrations[impurity],
          conc,
          rtol=_RTOL,
          err_msg=f'Impurity concentration for {impurity} does not match.',
      )

  def test_validate_inputs_for_computation_mode(self):
    # Test valid FORWARD mode
    extended_lengyel._validate_inputs_for_computation_mode(
        computation_mode=extended_lengyel.ComputationMode.FORWARD,
        target_electron_temp=None,
        seed_impurity_weights=None,
    )
    # Test invalid FORWARD mode
    with self.assertRaisesRegex(
        ValueError,
        'Target electron temperature must not be provided for forward'
        ' computation.',
    ):
      extended_lengyel._validate_inputs_for_computation_mode(
          computation_mode=extended_lengyel.ComputationMode.FORWARD,
          target_electron_temp=10.0,
          seed_impurity_weights=None,
      )
    with self.assertRaisesRegex(
        ValueError,
        'Seed impurity weights must not be provided for forward computation.',
    ):
      extended_lengyel._validate_inputs_for_computation_mode(
          computation_mode=extended_lengyel.ComputationMode.FORWARD,
          target_electron_temp=None,
          seed_impurity_weights={'N': 1.0},
      )
    # Test valid INVERSE mode
    extended_lengyel._validate_inputs_for_computation_mode(
        computation_mode=extended_lengyel.ComputationMode.INVERSE,
        target_electron_temp=10.0,
        seed_impurity_weights={'N': 1.0},
    )
    # Test invalid INVERSE mode
    with self.assertRaisesRegex(
        ValueError,
        'Target electron temperature must be provided for inverse computation.',
    ):
      extended_lengyel._validate_inputs_for_computation_mode(
          computation_mode=extended_lengyel.ComputationMode.INVERSE,
          target_electron_temp=None,
          seed_impurity_weights={'N': 1.0},
      )
    with self.assertRaisesRegex(
        ValueError,
        'Seed impurity weights must be provided for inverse computation.',
    ):
      extended_lengyel._validate_inputs_for_computation_mode(
          computation_mode=extended_lengyel.ComputationMode.INVERSE,
          target_electron_temp=10.0,
          seed_impurity_weights=None,
      )


if __name__ == '__main__':
  absltest.main()
