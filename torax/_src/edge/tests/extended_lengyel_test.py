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

    params = divertor_sol_1d.ExtendedLengyelParameters(
        # Dummy values for unused parameters in these specific tests.
        major_radius=1.0,
        minor_radius=1.0,
        separatrix_average_poloidal_field=1.0,
        fieldline_pitch_at_omp=1.0,
        cylindrical_safety_factor=1.0,
        power_crossing_separatrix=1.0,
        ratio_of_upstream_to_average_poloidal_field=(
            extended_lengyel_defaults.RATIO_UPSTREAM_TO_AVG_BPOL
        ),
        fraction_of_P_SOL_to_divertor=(
            extended_lengyel_defaults.FRACTION_OF_PSOL_TO_DIVERTOR
        ),
        target_angle_of_incidence=(
            extended_lengyel_defaults.TARGET_ANGLE_OF_INCIDENCE
        ),
        wall_temperature=extended_lengyel_defaults.WALL_TEMPERATURE,
        ratio_of_molecular_to_ion_mass=(
            extended_lengyel_defaults.RATIO_MOLECULAR_TO_ION_MASS
        ),
        # Parameters from reference case
        seed_impurity_weights={'N': 1.0, 'Ar': 0.05},
        fixed_impurity_concentrations={'He': 0.01},
        main_ion_charge=1.0,
        ne_tau=extended_lengyel_defaults.NE_TAU,
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
    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=3.39611622588553e8,
        c_z_prefactor=0.059314229517142096,
        kappa_e=1751.6010938527386,
        alpha_t=0.0,
        target_electron_temp=2.34,
    )
    sol_model = divertor_sol_1d.DivertorSOL1D(
        params=params,
        state=state,
    )

    calculated_c_z, status = extended_lengyel._solve_for_c_z_prefactor(
        sol_model=sol_model,
    )
    expected_c_z = 0.03487637336277587

    self.assertEqual(status, extended_lengyel.SolveStatus.SUCCESS)
    np.testing.assert_allclose(
        calculated_c_z,
        expected_c_z,
        rtol=5e-4,
    )

  def test_solve_for_qcc(self):
    """Test _solve_for_qcc."""

    # The plasma state is based on the first loop of the forward model reference
    # case in https://github.com/cfs-energy/extended-lengyel.

    params = divertor_sol_1d.ExtendedLengyelParameters(
        # Dummy values for unused parameters in these specific tests.
        major_radius=1.0,
        minor_radius=1.0,
        separatrix_average_poloidal_field=1.0,
        fieldline_pitch_at_omp=1.0,
        cylindrical_safety_factor=1.0,
        power_crossing_separatrix=1.0,
        ratio_of_upstream_to_average_poloidal_field=(
            extended_lengyel_defaults.RATIO_UPSTREAM_TO_AVG_BPOL
        ),
        fraction_of_P_SOL_to_divertor=(
            extended_lengyel_defaults.FRACTION_OF_PSOL_TO_DIVERTOR
        ),
        target_angle_of_incidence=(
            extended_lengyel_defaults.TARGET_ANGLE_OF_INCIDENCE
        ),
        wall_temperature=extended_lengyel_defaults.WALL_TEMPERATURE,
        ratio_of_molecular_to_ion_mass=(
            extended_lengyel_defaults.RATIO_MOLECULAR_TO_ION_MASS
        ),
        # Parameters from reference case
        seed_impurity_weights={},
        fixed_impurity_concentrations={
            'He': 0.01,
            'N': 0.038397305226362526,
            'Ar': 0.0019198652613181264,
        },
        main_ion_charge=1.0,
        ne_tau=extended_lengyel_defaults.NE_TAU,
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
    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=5.061935771095335e8,
        c_z_prefactor=0.0,
        kappa_e=1931.8277173925928,
        alpha_t=0.0,
        target_electron_temp=2.34,
    )
    sol_model = divertor_sol_1d.DivertorSOL1D(
        params=params,
        state=state,
    )

    calculated_qcc, status = extended_lengyel._solve_for_qcc(
        sol_model=sol_model,
    )
    expected_qcc = 1.11662e08
    self.assertEqual(status, extended_lengyel.SolveStatus.SUCCESS)
    np.testing.assert_allclose(calculated_qcc, expected_qcc, rtol=5e-4)

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
        'seed_impurity_concentrations': {
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
    for impurity, conc in expected_outputs[
        'seed_impurity_concentrations'
    ].items():
      self.assertIn(impurity, outputs.seed_impurity_concentrations)
      np.testing.assert_allclose(
          outputs.seed_impurity_concentrations[impurity],
          conc,
          rtol=_RTOL,
          err_msg=f'Impurity concentration for {impurity} does not match.',
      )

  def test_run_extended_lengyel_model_forward_mode(self):
    """Integration test for the full extended_lengyel model in forward mode."""
    # Input parameters for the test case. Rest are kept as defaults.
    _RTOL = 2e-3
    inputs = {
        'target_electron_temp': None,
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'seed_impurity_weights': {},
        'fixed_impurity_concentrations': {
            'He': 0.01,
            'N': 0.038397305226362526,
            'Ar': 0.0019198652613181264,
        },
        'magnetic_field_on_axis': 2.5,
        'plasma_current': 1.0e6,
        'parallel_connection_length': 20.0,
        'divertor_parallel_length': 5.0,
        'major_radius': 1.65,
        'minor_radius': 0.5,
        'elongation_psi95': 1.6,
        'triangularity_psi95': 0.3,
        'average_ion_mass': 2.0,
        'computation_mode': extended_lengyel.ComputationMode.FORWARD,
        'iterations': 100,
    }

    # --- Expected output values ---
    # Reference values from running the inverse mode reference case in:
    # https://github.com/cfs-energy/extended-lengyel

    # The rtol is lower here since we are comparing the forward mode to the
    # inverse mode reference case.
    expected_outputs = {
        'neutral_pressure_in_divertor': 1.737773924511501,
        'alpha_t': 0.35908862950459736,
        'q_parallel': 3.64822996e8,
        'heat_flux_perp_to_target': 7.92853e5,
        'separatrix_electron_temp': 0.1028445648,  # in keV
        'separatrix_Z_eff': 1.8621973566614212,
        'target_electron_temp': 2.34,  # in eV
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
    np.testing.assert_allclose(
        outputs.target_electron_temp,
        expected_outputs['target_electron_temp'],
        rtol=_RTOL,
    )

  def test_validate_inputs_for_computation_mode(self):
    # Test valid FORWARD mode
    extended_lengyel._validate_inputs_for_computation_mode(
        computation_mode=extended_lengyel.ComputationMode.FORWARD,
        target_electron_temp=None,
        seed_impurity_weights={},
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
          seed_impurity_weights={},
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
          seed_impurity_weights={},
      )


if __name__ == '__main__':
  absltest.main()
