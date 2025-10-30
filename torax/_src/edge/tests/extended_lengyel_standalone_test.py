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

from unittest import mock
from absl.testing import absltest
import numpy as np
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_solvers
from torax._src.edge import extended_lengyel_standalone

# pylint: disable=invalid-name


class ExtendedLengyelTest(absltest.TestCase):

  def test_run_extended_lengyel_model_inverse_mode_fixed_step(self):
    """Integration test for the full extended_lengyel model in inverse mode."""
    # Input parameters for the test case. Rest are kept as defaults.
    _RTOL = 5e-4
    inputs = {
        'target_electron_temp': 2.34,
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
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
        'computation_mode': extended_lengyel_enums.ComputationMode.INVERSE,
        'solver_mode': extended_lengyel_enums.SolverMode.FIXED_STEP,
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
    outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
        **inputs
    )

    # --- Assertions ---
    self.assertEqual(
        outputs.solver_status.physics_outcome,
        extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
    )
    self.assertEqual(
        outputs.solver_status.numerics_outcome,
        extended_lengyel_solvers.FixedStepOutcome.SUCCESS,
    )
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

  def test_run_extended_lengyel_model_forward_mode_fixed_step(self):
    # Input parameters for the test case. Rest are kept as defaults.
    _RTOL = 2e-3
    inputs = {
        'target_electron_temp': None,
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
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
        'computation_mode': extended_lengyel_enums.ComputationMode.FORWARD,
        'solver_mode': extended_lengyel_enums.SolverMode.FIXED_STEP,
        'fixed_step_iterations': 100,
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
    outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
        **inputs
    )

    # --- Assertions ---
    self.assertEqual(
        outputs.solver_status.physics_outcome,
        extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
    )
    self.assertEqual(
        outputs.solver_status.numerics_outcome,
        extended_lengyel_solvers.FixedStepOutcome.SUCCESS,
    )
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

  def test_run_extended_lengyel_model_inverse_mode_newton_raphson(self):
    # Input parameters for the test case. Rest are kept as defaults.
    _RTOL = 2e-3
    inputs = {
        'target_electron_temp': 2.34,
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
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
        'computation_mode': extended_lengyel_enums.ComputationMode.INVERSE,
        'solver_mode': extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
        'newton_raphson_iterations': 30,
        'newton_raphson_tol': 1e-5,
    }

    # --- Expected output values ---
    # Reference values from running the inverse mode reference case in:
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
    outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
        **inputs
    )

    # --- Assertions ---
    self.assertEqual(
        outputs.solver_status.physics_outcome,
        extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
    )
    self.assertEqual(
        outputs.solver_status.numerics_outcome.error,
        0,
    )
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

  def test_run_extended_lengyel_model_forward_mode_newton_raphson(self):
    # Input parameters for the test case. Rest are kept as defaults.
    _RTOL = 2e-3
    inputs = {
        'target_electron_temp': None,
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
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
        'computation_mode': extended_lengyel_enums.ComputationMode.FORWARD,
        'solver_mode': extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
        'newton_raphson_iterations': 30,
        'newton_raphson_tol': 1e-5,
    }

    # --- Expected output values ---
    # Reference values from running the inverse mode reference case in:
    # https://github.com/cfs-energy/extended-lengyel

    # Same outputs as fixed step solver.
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
    outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
        **inputs
    )

    # --- Assertions ---
    self.assertEqual(
        outputs.solver_status.physics_outcome,
        extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
    )
    self.assertEqual(
        outputs.solver_status.numerics_outcome.error,
        0,
    )
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

  def test_default_fixed_step_iterations(self):
    # Minimal inputs to run the function
    inputs = {
        'target_electron_temp': 2.34,
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
        'seed_impurity_weights': {'N': 1.0},
        'fixed_impurity_concentrations': {},
        'magnetic_field_on_axis': 2.5,
        'plasma_current': 1.0e6,
        'parallel_connection_length': 20.0,
        'divertor_parallel_length': 5.0,
        'major_radius': 1.65,
        'minor_radius': 0.5,
        'elongation_psi95': 1.6,
        'triangularity_psi95': 0.3,
        'average_ion_mass': 2.0,
        'computation_mode': extended_lengyel_enums.ComputationMode.INVERSE,
    }

    # Mock the solver functions
    mock_fixed_step = self.enter_context(
        mock.patch.object(
            extended_lengyel_solvers,
            'inverse_mode_fixed_step_solver',
            autospec=True,
        )
    )
    mock_hybrid = self.enter_context(
        mock.patch.object(
            extended_lengyel_solvers,
            'inverse_mode_hybrid_solver',
            autospec=True,
        )
    )
    # Set a return value that can be unpacked.
    mock_fixed_step.return_value = (mock.MagicMock(), mock.MagicMock())
    mock_hybrid.return_value = (mock.MagicMock(), mock.MagicMock())
    # Mock the post-processing function to avoid errors from the dummy
    # sol_model returned by the solver mocks.
    mock_post_process = self.enter_context(
        mock.patch.object(
            extended_lengyel_standalone,
            '_calc_post_processed_outputs',
            autospec=True,
        )
    )
    mock_post_process.return_value = (0.0, 0.0)

    # Get the original, non-JITted function to avoid issues with mocking.
    run_standalone_nojit = (
        extended_lengyel_standalone.run_extended_lengyel_standalone.__wrapped__
    )

    # Case 1: FIXED_STEP, default iterations
    run_standalone_nojit(
        **inputs, solver_mode=extended_lengyel_enums.SolverMode.FIXED_STEP
    )
    mock_fixed_step.assert_called_once()
    self.assertEqual(
        mock_fixed_step.call_args.kwargs['iterations'],
        extended_lengyel_defaults.FIXED_STEP_ITERATIONS,
    )
    mock_fixed_step.reset_mock()

    # Case 2: HYBRID, default iterations
    run_standalone_nojit(
        **inputs, solver_mode=extended_lengyel_enums.SolverMode.HYBRID
    )
    mock_hybrid.assert_called_once()
    self.assertEqual(
        mock_hybrid.call_args.kwargs['fixed_step_iterations'],
        extended_lengyel_defaults.HYBRID_FIXED_STEP_ITERATIONS,
    )
    mock_hybrid.reset_mock()

    # Case 3: FIXED_STEP, user-provided iterations
    run_standalone_nojit(
        **inputs,
        solver_mode=extended_lengyel_enums.SolverMode.FIXED_STEP,
        fixed_step_iterations=123,
    )
    mock_fixed_step.assert_called_once()
    self.assertEqual(mock_fixed_step.call_args.kwargs['iterations'], 123)
    mock_fixed_step.reset_mock()

  def test_validate_inputs_for_computation_mode(self):
    # Test valid FORWARD mode
    extended_lengyel_standalone._validate_inputs_for_computation_mode(
        computation_mode=extended_lengyel_enums.ComputationMode.FORWARD,
        target_electron_temp=None,
        seed_impurity_weights={},
    )
    # Test invalid FORWARD mode
    with self.assertRaisesRegex(
        ValueError,
        'Target electron temperature must not be provided for forward'
        ' computation.',
    ):
      extended_lengyel_standalone._validate_inputs_for_computation_mode(
          computation_mode=extended_lengyel_enums.ComputationMode.FORWARD,
          target_electron_temp=10.0,
          seed_impurity_weights={},
      )
    with self.assertRaisesRegex(
        ValueError,
        'Seed impurity weights must not be provided for forward computation.',
    ):
      extended_lengyel_standalone._validate_inputs_for_computation_mode(
          computation_mode=extended_lengyel_enums.ComputationMode.FORWARD,
          target_electron_temp=None,
          seed_impurity_weights={'N': 1.0},
      )
    # Test valid INVERSE mode
    extended_lengyel_standalone._validate_inputs_for_computation_mode(
        computation_mode=extended_lengyel_enums.ComputationMode.INVERSE,
        target_electron_temp=10.0,
        seed_impurity_weights={'N': 1.0},
    )
    # Test invalid INVERSE mode
    with self.assertRaisesRegex(
        ValueError,
        'Target electron temperature must be provided for inverse computation.',
    ):
      extended_lengyel_standalone._validate_inputs_for_computation_mode(
          computation_mode=extended_lengyel_enums.ComputationMode.INVERSE,
          target_electron_temp=None,
          seed_impurity_weights={'N': 1.0},
      )
    with self.assertRaisesRegex(
        ValueError,
        'Seed impurity weights must be provided for inverse computation.',
    ):
      extended_lengyel_standalone._validate_inputs_for_computation_mode(
          computation_mode=extended_lengyel_enums.ComputationMode.INVERSE,
          target_electron_temp=10.0,
          seed_impurity_weights={},
      )


if __name__ == '__main__':
  absltest.main()
