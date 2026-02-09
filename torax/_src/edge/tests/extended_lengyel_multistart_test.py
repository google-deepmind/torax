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

import dataclasses
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from torax._src.edge import divertor_sol_1d
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_solvers
from torax._src.edge import extended_lengyel_standalone
from torax._src.solver import jax_root_finding


class ExtendedLengyelMultistartTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    extended_lengyel_standalone.run_extended_lengyel_standalone.clear_cache()
    # Basic valid inputs for the model
    self.inputs = {
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
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
        'computation_mode': extended_lengyel_enums.ComputationMode.FORWARD,
    }

  def test_multistart_fallback_selects_closest_valid(self):
    """Tests that fallback logic picks the closest valid root to nominal."""
    with mock.patch.object(
        extended_lengyel_solvers, 'forward_mode_newton_solver', autospec=True
    ) as mock_solver:

      def solver_side_effect(initial_sol_model, **_):
        temp_target_guess = initial_sol_model.state.T_e_target
        out_state = initial_sol_model.state

        # Default success status
        status = extended_lengyel_solvers.ExtendedLengyelSolverStatus(
            physics_outcome=extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            numerics_outcome=jax_root_finding.RootMetadata(
                iterations=jnp.array(5),
                error=jnp.array(0),  # Valid
                residual=jnp.zeros(2),
                last_tau=jnp.array(1.0),
            ),
        )

        # Fail nominal (50.0) with numerical error
        fail_cond = jnp.isclose(temp_target_guess, 50.0)
        high_cond = temp_target_guess > 400.0

        error_val = jnp.where(fail_cond, 1, 0)
        new_numerics = dataclasses.replace(
            status.numerics_outcome, error=error_val
        )
        status = dataclasses.replace(status, numerics_outcome=new_numerics)

        # Roots:
        # Nominal (failed) -> 999.0
        # High guess -> 123.0 (closer to 50.0 log-wise)
        # Low guess -> 10.0
        temp_target_out = jnp.where(
            fail_cond, 999.0, jnp.where(high_cond, 123.0, 10.0)
        )
        out_state = dataclasses.replace(out_state, T_e_target=temp_target_out)

        return (
            divertor_sol_1d.DivertorSOL1D(
                params=initial_sol_model.params, state=out_state
            ),
            status,
        )

      mock_solver.side_effect = solver_side_effect

      initial_guess = divertor_sol_1d.ForwardInitialGuess(
          alpha_t=0.1, kappa_e=2000.0, T_e_separatrix=200.0, T_e_target=50.0
      )

      outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
          **self.inputs,
          solver_mode=extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
          initial_guess=initial_guess,
      )

      # Fallback should pick 123.0 as it is closer to 50.0 than 10.0 is.
      # log10(50)~1.7, log10(123)~2.09 (diff 0.39), log10(10)=1.0 (diff 0.7)
      self.assertAlmostEqual(outputs.T_e_target, 123.0, delta=1.0)

  def test_multistart_filters_numerical_errors(self):
    """Tests that validity depends purely on numerical error status."""
    with mock.patch.object(
        extended_lengyel_solvers, 'forward_mode_newton_solver', autospec=True
    ) as mock_solver:

      def solver_side_effect(initial_sol_model, **_):
        temp_target_guess = initial_sol_model.state.T_e_target
        out_state = initial_sol_model.state

        # Setup scenarios based on guess
        # Guess ~ 50.0 (Nominal): Error=1 (Invalid), Physics=Success
        # Guess < 10.0: Error=0 (Valid), Physics=Failure (e.g. Q_CC_NEG)
        # Guess > 100.0: Error=0 (Valid), Physics=Success (Normal)

        is_nominal = jnp.isclose(temp_target_guess, 50.0)
        is_low = temp_target_guess < 10.0

        # Error: 1 if nominal, else 0
        error_val = jnp.where(is_nominal, 1, 0)

        # Physics: Failure if low, else Success
        phys_outcome = jnp.where(
            is_low,
            extended_lengyel_solvers.PhysicsOutcome.Q_CC_SQUARED_NEGATIVE,
            extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
        )

        status = extended_lengyel_solvers.ExtendedLengyelSolverStatus(
            physics_outcome=phys_outcome,
            numerics_outcome=jax_root_finding.RootMetadata(
                iterations=jnp.array(5),
                error=error_val,
                residual=jnp.zeros(2),
                last_tau=jnp.array(1.0),
            ),
        )

        # Roots:
        # Nominal -> 50.0 (But Invalid)
        # Low -> 5.0 (Valid despite physics fail)
        # High -> 200.0 (Valid)
        temp_target_out = jnp.where(
            is_nominal, 50.0, jnp.where(is_low, 5.0, 200.0)
        )
        out_state = dataclasses.replace(out_state, T_e_target=temp_target_out)

        return (
            divertor_sol_1d.DivertorSOL1D(
                params=initial_sol_model.params, state=out_state
            ),
            status,
        )

      mock_solver.side_effect = solver_side_effect

      initial_guess = divertor_sol_1d.ForwardInitialGuess(
          alpha_t=0.1, kappa_e=2000.0, T_e_separatrix=200.0, T_e_target=50.0
      )

      outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
          **self.inputs,
          solver_mode=extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
          initial_guess=initial_guess,
      )

      # 1. Nominal (50.0) was marked Invalid (error=1).
      # Even though it was closest (exact match), it should NOT be selected.
      # 2. Low (5.0) was Valid (error=0), even though Physics=Failure.
      # 3. High (200.0) was Valid (error=0).

      # Distances:
      # log10(50) = 1.7
      # log10(5) = 0.7 (diff 1.0)
      # log10(200) = 2.3 (diff 0.6)

      # Should pick 200.0 as it is the closest VALID root.
      self.assertAlmostEqual(outputs.T_e_target, 200.0, delta=1.0)

  def test_multiple_roots_found_flag(self):
    """Tests that multiple_roots_found is set when distinct roots exist."""

    with mock.patch.object(
        extended_lengyel_solvers, 'forward_mode_newton_solver', autospec=True
    ) as mock_solver:

      def solver_side_effect(initial_sol_model, **_):
        temp_target_guess = initial_sol_model.state.T_e_target
        status = extended_lengyel_solvers.ExtendedLengyelSolverStatus(
            physics_outcome=extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            numerics_outcome=jax_root_finding.RootMetadata(
                iterations=jnp.array(5),
                error=jnp.array(0.0),
                residual=jnp.zeros(2),
                last_tau=jnp.array(1.0),
            ),
        )
        out_state = initial_sol_model.state

        # Determine root based on guess
        # Half guesses go to 10.0, Half go to 100.0 (distinct > tolerance)
        temp_target_out = jnp.where(temp_target_guess < 50.0, 10.0, 100.0)
        out_state = dataclasses.replace(out_state, T_e_target=temp_target_out)

        return (
            divertor_sol_1d.DivertorSOL1D(
                params=initial_sol_model.params, state=out_state
            ),
            status,
        )

      mock_solver.side_effect = solver_side_effect

      outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
          **self.inputs,
          solver_mode=extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
      )

      self.assertTrue(outputs.multiple_roots_found)

      # Verify roots has both 10 and 100
      self.assertTrue(jnp.any(jnp.isclose(outputs.roots.T_e_target, 10.0)))
      self.assertTrue(jnp.any(jnp.isclose(outputs.roots.T_e_target, 100.0)))

  def test_configurable_num_guesses(self):
    """Tests that the number of guesses can be configured and JIT-compiled."""

    # Use a specific number of guesses to verify it's used (and triggers JIT)
    n_guesses = 20

    outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
        **self.inputs,
        solver_mode=extended_lengyel_enums.SolverMode.NEWTON_RAPHSON,
        multistart_num_guesses=n_guesses,
    )

    self.assertIsNotNone(outputs.roots)
    self.assertEqual(outputs.roots.T_e_target.shape[0], n_guesses)


if __name__ == '__main__':
  absltest.main()
