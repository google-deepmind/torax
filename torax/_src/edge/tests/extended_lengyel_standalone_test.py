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
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src.edge import divertor_sol_1d
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_solvers
from torax._src.edge import extended_lengyel_standalone
from torax._src.solver import jax_root_finding

# pylint: disable=invalid-name


class ExtendedLengyelTest(parameterized.TestCase):

  def test_run_extended_lengyel_model_inverse_mode_fixed_point(self):
    """Integration test for the full extended_lengyel model in inverse mode."""
    # Input parameters for the test case. Rest are kept as defaults.
    _RTOL = 5e-4
    inputs = {
        'T_e_target': 2.34,
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
        'seed_impurity_weights': {'N': 1.0, 'Ar': 0.05},
        'fixed_impurity_concentrations': {'He': 0.01},
        'magnetic_field_on_axis': 2.5,
        'plasma_current': 1.0e6,
        'connection_length_target': 20.0,
        'connection_length_divertor': 5.0,
        'major_radius': 1.65,
        'minor_radius': 0.5,
        'elongation_psi95': 1.6,
        'triangularity_psi95': 0.3,
        'average_ion_mass': 2.0,
        'computation_mode': extended_lengyel_enums.ComputationMode.INVERSE,
        'solver_mode': extended_lengyel_enums.SolverMode.FIXED_POINT,
    }

    # --- Expected output values ---
    # Reference values from running the reference case in:
    # https://github.com/cfs-energy/extended-lengyel
    expected_outputs = {
        'pressure_neutral_divertor': 1.737773924511501,
        'alpha_t': 0.35908862950459736,
        'q_parallel': 3.64822996e8,
        'q_perpendicular_target': 7.92853e5,
        'T_e_separatrix': 0.1028445648,  # in keV
        'Z_eff_separatrix': 1.8621973566614212,
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
        extended_lengyel_solvers.FixedPointOutcome.SUCCESS,
    )
    np.testing.assert_allclose(
        outputs.pressure_neutral_divertor,
        expected_outputs['pressure_neutral_divertor'],
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
        outputs.q_perpendicular_target,
        expected_outputs['q_perpendicular_target'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.T_e_separatrix,
        expected_outputs['T_e_separatrix'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.Z_eff_separatrix,
        expected_outputs['Z_eff_separatrix'],
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

  def test_run_extended_lengyel_model_forward_mode_fixed_point(self):
    # Input parameters for the test case. Rest are kept as defaults.
    _RTOL = 2e-3
    inputs = {
        'T_e_target': None,
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
        'connection_length_target': 20.0,
        'connection_length_divertor': 5.0,
        'major_radius': 1.65,
        'minor_radius': 0.5,
        'elongation_psi95': 1.6,
        'triangularity_psi95': 0.3,
        'average_ion_mass': 2.0,
        'computation_mode': extended_lengyel_enums.ComputationMode.FORWARD,
        'solver_mode': extended_lengyel_enums.SolverMode.FIXED_POINT,
        'fixed_point_iterations': 100,
    }

    # --- Expected output values ---
    # Reference values from running the inverse mode reference case in:
    # https://github.com/cfs-energy/extended-lengyel

    # The rtol is lower here since we are comparing the forward mode to the
    # inverse mode reference case.
    expected_outputs = {
        'pressure_neutral_divertor': 1.737773924511501,
        'alpha_t': 0.35908862950459736,
        'q_parallel': 3.64822996e8,
        'q_perpendicular_target': 7.92853e5,
        'T_e_separatrix': 0.1028445648,  # in keV
        'Z_eff_separatrix': 1.8621973566614212,
        'T_e_target': 2.34,  # in eV
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
        extended_lengyel_solvers.FixedPointOutcome.SUCCESS,
    )
    np.testing.assert_allclose(
        outputs.pressure_neutral_divertor,
        expected_outputs['pressure_neutral_divertor'],
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
        outputs.q_perpendicular_target,
        expected_outputs['q_perpendicular_target'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.T_e_separatrix,
        expected_outputs['T_e_separatrix'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.Z_eff_separatrix,
        expected_outputs['Z_eff_separatrix'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.T_e_target,
        expected_outputs['T_e_target'],
        rtol=_RTOL,
    )

  def test_run_extended_lengyel_model_inverse_mode_newton_raphson(self):
    # Input parameters for the test case. Rest are kept as defaults.
    _RTOL = 2e-3
    inputs = {
        'T_e_target': 2.34,
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
        'seed_impurity_weights': {'N': 1.0, 'Ar': 0.05},
        'fixed_impurity_concentrations': {'He': 0.01},
        'magnetic_field_on_axis': 2.5,
        'plasma_current': 1.0e6,
        'connection_length_target': 20.0,
        'connection_length_divertor': 5.0,
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
        'pressure_neutral_divertor': 1.737773924511501,
        'alpha_t': 0.35908862950459736,
        'q_parallel': 3.64822996e8,
        'q_perpendicular_target': 7.92853e5,
        'T_e_separatrix': 0.1028445648,  # in keV
        'Z_eff_separatrix': 1.8621973566614212,
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
        outputs.pressure_neutral_divertor,
        expected_outputs['pressure_neutral_divertor'],
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
        outputs.q_perpendicular_target,
        expected_outputs['q_perpendicular_target'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.T_e_separatrix,
        expected_outputs['T_e_separatrix'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.Z_eff_separatrix,
        expected_outputs['Z_eff_separatrix'],
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
        'T_e_target': None,
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
        'connection_length_target': 20.0,
        'connection_length_divertor': 5.0,
        'major_radius': 1.65,
        'minor_radius': 0.5,
        'elongation_psi95': 1.6,
        'triangularity_psi95': 0.3,
        'average_ion_mass': 2.0,
        'computation_mode': extended_lengyel_enums.ComputationMode.FORWARD,
        'solver_mode': extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
        'newton_raphson_iterations': 30,
        'newton_raphson_tol': 1e-5,
        'initial_guess': divertor_sol_1d.ForwardInitialGuess(
            alpha_t=0.1,
            kappa_e=2.0e3,
            T_e_separatrix=2e2,
            T_e_target=1e2,
        ),
    }

    # --- Expected output values ---
    # Reference values from running the inverse mode reference case in:
    # https://github.com/cfs-energy/extended-lengyel

    # Same outputs as fixed step solver.
    expected_outputs = {
        'pressure_neutral_divertor': 1.737773924511501,
        'alpha_t': 0.35908862950459736,
        'q_parallel': 3.64822996e8,
        'q_perpendicular_target': 7.92853e5,
        'T_e_separatrix': 0.1028445648,  # in keV
        'Z_eff_separatrix': 1.8621973566614212,
        'T_e_target': 2.34,  # in eV
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
        outputs.pressure_neutral_divertor,
        expected_outputs['pressure_neutral_divertor'],
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
        outputs.q_perpendicular_target,
        expected_outputs['q_perpendicular_target'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.T_e_separatrix,
        expected_outputs['T_e_separatrix'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.Z_eff_separatrix,
        expected_outputs['Z_eff_separatrix'],
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        outputs.T_e_target,
        expected_outputs['T_e_target'],
        rtol=_RTOL,
    )

  def test_default_fixed_point_iterations(self):
    # Minimal inputs to run the function
    inputs = {
        'T_e_target': 2.34,
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
        'seed_impurity_weights': {'N': 1.0},
        'fixed_impurity_concentrations': {},
        'magnetic_field_on_axis': 2.5,
        'plasma_current': 1.0e6,
        'connection_length_target': 20.0,
        'connection_length_divertor': 5.0,
        'major_radius': 1.65,
        'minor_radius': 0.5,
        'elongation_psi95': 1.6,
        'triangularity_psi95': 0.3,
        'average_ion_mass': 2.0,
        'computation_mode': extended_lengyel_enums.ComputationMode.INVERSE,
    }

    # Mock the solver functions
    mock_fixed_point = self.enter_context(
        mock.patch.object(
            extended_lengyel_solvers,
            'inverse_mode_fixed_point_solver',
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
    mock_fixed_point.return_value = (mock.MagicMock(), mock.MagicMock())
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

    # Case 1: FIXED_POINT, default iterations
    run_standalone_nojit(
        **inputs, solver_mode=extended_lengyel_enums.SolverMode.FIXED_POINT
    )
    mock_fixed_point.assert_called_once()
    self.assertEqual(
        mock_fixed_point.call_args.kwargs['iterations'],
        extended_lengyel_defaults.FIXED_POINT_ITERATIONS,
    )
    mock_fixed_point.reset_mock()

    # Case 2: HYBRID, default iterations
    run_standalone_nojit(
        **inputs, solver_mode=extended_lengyel_enums.SolverMode.HYBRID
    )
    mock_hybrid.assert_called_once()
    self.assertEqual(
        mock_hybrid.call_args.kwargs['fixed_point_iterations'],
        extended_lengyel_defaults.HYBRID_FIXED_POINT_ITERATIONS,
    )
    mock_hybrid.reset_mock()

    # Case 3: FIXED_POINT, user-provided iterations
    run_standalone_nojit(
        **inputs,
        solver_mode=extended_lengyel_enums.SolverMode.FIXED_POINT,
        fixed_point_iterations=123,
    )
    mock_fixed_point.assert_called_once()
    self.assertEqual(mock_fixed_point.call_args.kwargs['iterations'], 123)
    mock_fixed_point.reset_mock()

  def test_validate_inputs_for_computation_mode(self):
    # Test valid FORWARD mode
    extended_lengyel_standalone._validate_inputs_for_computation_mode(
        computation_mode=extended_lengyel_enums.ComputationMode.FORWARD,
        T_e_target=None,
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
          T_e_target=10.0,
          seed_impurity_weights={},
      )
    with self.assertRaisesRegex(
        ValueError,
        'Seed impurity weights must not be provided for forward computation.',
    ):
      extended_lengyel_standalone._validate_inputs_for_computation_mode(
          computation_mode=extended_lengyel_enums.ComputationMode.FORWARD,
          T_e_target=None,
          seed_impurity_weights={'N': 1.0},
      )
    # Test valid INVERSE mode
    extended_lengyel_standalone._validate_inputs_for_computation_mode(
        computation_mode=extended_lengyel_enums.ComputationMode.INVERSE,
        T_e_target=10.0,
        seed_impurity_weights={'N': 1.0},
    )
    # Test invalid INVERSE mode
    with self.assertRaisesRegex(
        ValueError,
        'Target electron temperature must be provided for inverse computation.',
    ):
      extended_lengyel_standalone._validate_inputs_for_computation_mode(
          computation_mode=extended_lengyel_enums.ComputationMode.INVERSE,
          T_e_target=None,
          seed_impurity_weights={'N': 1.0},
      )
    with self.assertRaisesRegex(
        ValueError,
        'Seed impurity weights must be provided for inverse computation.',
    ):
      extended_lengyel_standalone._validate_inputs_for_computation_mode(
          computation_mode=extended_lengyel_enums.ComputationMode.INVERSE,
          T_e_target=10.0,
          seed_impurity_weights={},
      )

  @parameterized.named_parameters(
      ('low_ip', {'plasma_current': 2.0e6, 'power_crossing_separatrix': 10e6}),
      (
          'low_power',
          {'plasma_current': 15.0e6, 'power_crossing_separatrix': 1.0e6},
      ),
  )
  def test_underpowered_scenario(self, inputs_update):
    """Test scenario where input power is too low to reach target temperature.

    This uses unrealistically low inputs for ITER-like scenarios (Ip or P_SOL)
    which results in required impurity concentration being negative
    (physically impossible).
    The solver should report this via physics_outcome and a non-zero residual.

    Args:
      inputs_update: Dictionary of input parameters to override defaults.
    """
    inputs = {
        'T_e_target': 5.0,
        'power_crossing_separatrix': 10e6,
        'separatrix_electron_density': 3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
        'fixed_impurity_concentrations': {},
        'magnetic_field_on_axis': 5.3,
        'plasma_current': 15.0e6,
        'connection_length_target': 50.0,
        'connection_length_divertor': 10.0,
        'major_radius': 6.2,
        'minor_radius': 2.0,
        'elongation_psi95': 1.7,
        'triangularity_psi95': 0.33,
        'average_ion_mass': 2.0,
        'computation_mode': extended_lengyel_enums.ComputationMode.INVERSE,
        'solver_mode': extended_lengyel_enums.SolverMode.HYBRID,
        'seed_impurity_weights': {'Ne': 1.0},
    }
    inputs.update(inputs_update)

    outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
        **inputs
    )

    numerics = outputs.solver_status.numerics_outcome

    # 1. Assert no NaNs in output.
    self.assertFalse(np.any(np.isnan(numerics.residual)))
    self.assertFalse(np.isnan(outputs.Z_eff_separatrix))
    self.assertFalse(np.isnan(outputs.alpha_t))

    # 2. Physics outcome should flag the issue.
    self.assertEqual(
        outputs.solver_status.physics_outcome.item(),
        extended_lengyel_solvers.PhysicsOutcome.C_Z_PREFACTOR_NEGATIVE,
    )

    # 3. Impurities should be clamped to 0.
    np.testing.assert_allclose(
        outputs.seed_impurity_concentrations['Ne'], 0.0, atol=1e-5
    )

  def test_initial_guess_is_respected(self):
    """Tests that providing an initial guess changes solver behavior.

    We set the max iterations to 1 for the newton solver. If the initial guess
    is close to the solution, the error should be small. If not, error large.
    """

    # 1. Find the solution with plenty of iterations
    inputs = {
        'T_e_target': 2.34,
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
        'seed_impurity_weights': {'N': 1.0, 'Ar': 0.05},
        'fixed_impurity_concentrations': {'He': 0.01},
        'magnetic_field_on_axis': 2.5,
        'plasma_current': 1.0e6,
        'connection_length_target': 20.0,
        'connection_length_divertor': 5.0,
        'major_radius': 1.65,
        'minor_radius': 0.5,
        'elongation_psi95': 1.6,
        'triangularity_psi95': 0.3,
        'average_ion_mass': 2.0,
        'computation_mode': extended_lengyel_enums.ComputationMode.INVERSE,
        'solver_mode': extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
        'newton_raphson_iterations': 50,
    }

    outputs_converged = (
        extended_lengyel_standalone.run_extended_lengyel_standalone(**inputs)
    )

    # 2. Run with 1 iteration and no initial guess (using defaults).
    # Should result in large error/residual.
    inputs['newton_raphson_iterations'] = 1
    outputs_no_guess = (
        extended_lengyel_standalone.run_extended_lengyel_standalone(**inputs)
    )

    # 3. Run with 1 iteration and initial guess set to the converged solution.
    # Should result in small error/residual.
    initial_guess = divertor_sol_1d.InverseInitialGuess(
        alpha_t=outputs_converged.alpha_t,
        kappa_e=extended_lengyel_defaults.KAPPA_E_0,
        T_e_separatrix=outputs_converged.T_e_separatrix
        * 1000.0,  # convert keV to eV
        c_z_prefactor=outputs_converged.seed_impurity_concentrations[
            'N'
        ],  # Approx, since N weight is 1.0
    )

    outputs_with_guess = (
        extended_lengyel_standalone.run_extended_lengyel_standalone(
            initial_guess=initial_guess, **inputs
        )
    )

    # Check residuals
    # Access residual scalar. Note: numerics_outcome is RootMetadata
    residual_no_guess = np.mean(
        np.abs(outputs_no_guess.solver_status.numerics_outcome.residual)
    )
    residual_with_guess = np.mean(
        np.abs(outputs_with_guess.solver_status.numerics_outcome.residual)
    )

    self.assertLess(
        residual_with_guess,
        residual_no_guess,
        'Initial guess should reduce residual compared to default start with'
        ' few iterations.',
    )

<<<<<<< HEAD

class ExtendedLengyelUniqueRootsTest(parameterized.TestCase):

  def test_get_unique_roots_non_stacked(self):
    # Create fake roots array (shape: [num_guesses, ...])

    roots_solver_status = extended_lengyel_solvers.ExtendedLengyelSolverStatus(
        physics_outcome=jnp.array([  # pytype: disable=wrong-arg-types
            extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
        ]),
        numerics_outcome=extended_lengyel_solvers.FixedPointOutcome.SUCCESS,
    )

    # 3 roots: 0 and 1 are valid and distinct, 2 is duplicate of 1
    # T_e roots: [5.0, 10.0, 10.000001]
    # We expect 2 unique roots.
    roots = extended_lengyel_standalone.ExtendedLengyelOutputs(
        T_e_target=jnp.array([5.0, 10.0, 10.000001]),
        pressure_neutral_divertor=jnp.array([1.0, 2.0, 2.0]),
        alpha_t=jnp.array([0.1, 0.2, 0.2]),
        q_parallel=jnp.array([1e6, 2e6, 2e6]),
        q_perpendicular_target=jnp.array([1e5, 2e5, 2e5]),
        T_e_separatrix=jnp.array([100.0, 200.0, 200.0]),
        Z_eff_separatrix=jnp.array([1.5, 1.6, 1.6]),
        seed_impurity_concentrations={'N': jnp.array([0.01, 0.02, 0.02])},
        solver_status=roots_solver_status,
        calculated_enrichment={},
        kappa_e=jnp.array([0.0]),
        c_z_prefactor=jnp.array([0.0]),
    )

    # Wrap in ExtendedLengyelOutputs as the container
    outputs = extended_lengyel_standalone.ExtendedLengyelOutputs(
        roots=roots,
        multiple_roots_found=jnp.array([True]),
        # Dummy values for required fields
        T_e_target=jnp.array([0.0]),
        pressure_neutral_divertor=jnp.array([0.0]),
        alpha_t=jnp.array([0.0]),
        q_parallel=jnp.array([0.0]),
        q_perpendicular_target=jnp.array([0.0]),
        T_e_separatrix=jnp.array([0.0]),
        Z_eff_separatrix=jnp.array([0.0]),
        seed_impurity_concentrations={},
        solver_status=roots_solver_status,
        calculated_enrichment={},
        kappa_e=jnp.array([0.0]),
        c_z_prefactor=jnp.array([0.0]),
    )

    unique = outputs.get_unique_roots()

    self.assertIsNotNone(unique)
    self.assertIsNone(unique.roots)

    self.assertEqual(unique.T_e_target.shape, (2,))
    np.testing.assert_allclose(unique.T_e_target[:2], [5.0, 10.0], atol=1e-4)

    self.assertIsNotNone(unique.solver_status.physics_outcome)
    physics_outcome = unique.solver_status.physics_outcome
    self.assertEqual(physics_outcome.shape, (2,))  # pytype: disable=attribute-error

  def test_get_unique_roots_time_dependent(self):
    # Shape: [time, num_guesses]
    # Time steps: 2
    # Guesses: 3

    Te_data = [[5.0, 10.0, 10.000001], [20.0, 20.0, 20.000001]]

    roots_solver_status = extended_lengyel_solvers.ExtendedLengyelSolverStatus(
        physics_outcome=jnp.zeros((2, 3), dtype=jnp.int32),  # pytype: disable=wrong-arg-types
        numerics_outcome=extended_lengyel_solvers.FixedPointOutcome.SUCCESS,
    )

    # t=0: [5, 10, 10] -> 2 unique roots
    # t=1: [20, 20, 20] -> 1 unique root
    roots = extended_lengyel_standalone.ExtendedLengyelOutputs(
        T_e_target=jnp.array(Te_data),
        pressure_neutral_divertor=jnp.array([[1.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
        alpha_t=jnp.zeros((2, 3)),
        q_parallel=jnp.zeros((2, 3)),
        q_perpendicular_target=jnp.zeros((2, 3)),
        T_e_separatrix=jnp.zeros((2, 3)),
        Z_eff_separatrix=jnp.zeros((2, 3)),
        seed_impurity_concentrations={'N': jnp.zeros((2, 3))},
        solver_status=roots_solver_status,
        # Required fields default to None or can be mocked if needed
        calculated_enrichment={},
        kappa_e=jnp.zeros((2, 3)),
        c_z_prefactor=jnp.zeros((2, 3)),
    )

    outputs = extended_lengyel_standalone.ExtendedLengyelOutputs(
        roots=roots,
        # Dummy values for required fields
        T_e_target=jnp.zeros((2, 3)),
        pressure_neutral_divertor=jnp.zeros((2, 3)),
        alpha_t=jnp.zeros((2, 3)),
        q_parallel=jnp.zeros((2, 3)),
        q_perpendicular_target=jnp.zeros((2, 3)),
        T_e_separatrix=jnp.zeros((2, 3)),
        Z_eff_separatrix=jnp.zeros((2, 3)),
        seed_impurity_concentrations={},
        solver_status=roots_solver_status,
        calculated_enrichment={},
        kappa_e=jnp.zeros((2, 3)),
        c_z_prefactor=jnp.zeros((2, 3)),
    )

    unique = outputs.get_unique_roots()

    self.assertIsNotNone(unique)

    self.assertEqual(unique.T_e_target.shape, (2, 2))

    np.testing.assert_allclose(unique.T_e_target[0, :2], [5.0, 10.0], atol=1e-4)

    np.testing.assert_allclose(unique.T_e_target[1, :1], [20.0], atol=1e-4)
    self.assertTrue(jnp.isnan(unique.T_e_target[1, 1]))

    self.assertIsNotNone(unique)

    physics_outcome = unique.solver_status.physics_outcome
    self.assertEqual(physics_outcome.shape, (2, 2))  # pytype: disable=attribute-error
    self.assertEqual(physics_outcome[1, 1], -1)  # pytype: disable=unsupported-operands

  def test_get_unique_roots_keeps_error_roots(self):
    """Test that roots with solver error=1 are kept in deduplicated output."""
    root_metadata = jax_root_finding.RootMetadata(
        iterations=jnp.array([5, 10, 3], dtype=jnp.int32),
        residual=jnp.array([[0.1, 0.2], [0.3, 0.4], [0.01, 0.02]]),
        error=jnp.array([0, 1, 0], dtype=jnp.int32),
        last_tau=jnp.array([0.5, 0.5, 0.5]),
    )
    roots_solver_status = extended_lengyel_solvers.ExtendedLengyelSolverStatus(
        physics_outcome=jnp.array([  # pytype: disable=wrong-arg-types
            extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
        ]),
        numerics_outcome=root_metadata,
    )

    roots = extended_lengyel_standalone.ExtendedLengyelOutputs(
        T_e_target=jnp.array([5.0, 15.0, 25.0]),
        pressure_neutral_divertor=jnp.array([1.0, 2.0, 3.0]),
        alpha_t=jnp.zeros(3),
        q_parallel=jnp.zeros(3),
        q_perpendicular_target=jnp.zeros(3),
        T_e_separatrix=jnp.zeros(3),
        Z_eff_separatrix=jnp.zeros(3),
        seed_impurity_concentrations={},
        solver_status=roots_solver_status,
        calculated_enrichment={},
        kappa_e=jnp.zeros(3),
        c_z_prefactor=jnp.zeros(3),
    )

    outputs = extended_lengyel_standalone.ExtendedLengyelOutputs(
        roots=roots,
        T_e_target=jnp.array([0.0]),
        pressure_neutral_divertor=jnp.array([0.0]),
        alpha_t=jnp.array([0.0]),
        q_parallel=jnp.array([0.0]),
        q_perpendicular_target=jnp.array([0.0]),
        T_e_separatrix=jnp.array([0.0]),
        Z_eff_separatrix=jnp.array([0.0]),
        seed_impurity_concentrations={},
        solver_status=roots_solver_status,
        calculated_enrichment={},
        kappa_e=jnp.array([0.0]),
        c_z_prefactor=jnp.array([0.0]),
    )

    unique = outputs.get_unique_roots()
    self.assertIsNotNone(unique)
    self.assertEqual(unique.T_e_target.shape, (3,))
    np.testing.assert_allclose(unique.T_e_target, [5.0, 15.0, 25.0], atol=1e-4)
    numerics = unique.solver_status.numerics_outcome
    np.testing.assert_array_equal(numerics.error, [0, 1, 0])  # pytype: disable=attribute-error

=======
>>>>>>> 85479b3d (Fix IMAS required terms check and improve Extended Lengyel edge model)

if __name__ == '__main__':
  absltest.main()
