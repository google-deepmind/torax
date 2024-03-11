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

"""TORAX integration tests.

These are full integration tests that run the simulation and compare to a
previously executed TORAX reference:
"""

from typing import Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np
import torax
from torax import sim as sim_lib
from torax import state as state_lib
from torax.spectators import spectator as spectator_lib
from torax.stepper import linear_theta_method
from torax.tests.test_lib import explicit_stepper
from torax.tests.test_lib import sim_test_case
from torax.time_step_calculator import chi_time_step_calculator


_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


class SimTest(sim_test_case.SimTestCase):
  """Integration tests for torax.sim.

  The numbering is legacy from when a subset of these tests were compared to
  PINT runs. This numbering is kept to maintain backwards compatibility for now.
  """

  @parameterized.named_parameters(
      # Tests explicit solver
      (
          'test1',
          'test1.py',
          'test1',
          _ALL_PROFILES,
          0,
      ),
      # Tests implicit solver with theta=0.5 (Crank-Nicholson)
      (
          'test2_cn',
          'test2_cn.py',
          'test2',
          ('temp_ion', 'temp_el'),
          2e-1,
      ),
      # Tests implicit solver with theta=1.0 (backwards Euler)
      (
          'test2',
          'test2.py',
          'test2',
          _ALL_PROFILES,
          0,
      ),
      # Tests that optimizer gets the same result as the linear solver
      # when coefficients are frozen.
      (
          'test2_optimizer',
          'test2_optimizer.py',
          'test2',
          _ALL_PROFILES,
          1e-5,
      ),
      # Tests that Newton-Raphson gets the same result as the linear solver
      # when the coefficient matrix is frozen
      (
          'test2_newton_raphson',
          'test2.py',
          'test2',
          _ALL_PROFILES,
          1e-6,
      ),
      # Test ion-electron heat exchange at low density
      (
          'test3',
          'test3.py',
          'test3',
          _ALL_PROFILES,
          0,
      ),
      # Tests sim.ArrayTimeStepCalculator
      (
          'test3_ref',
          'test3.py',
          'test3',
          _ALL_PROFILES,
          0,
          True,
      ),
      # Tests ion-electron heat exchange at high density
      (
          'test4',
          'test4.py',
          'test4',
          _ALL_PROFILES,
          0,
      ),
      # Tests pedestal internal boundary condition
      (
          'test5',
          'test5.py',
          'test5',
          _ALL_PROFILES,
          0,
      ),
      # Tests CGM model heat transport only
      (
          'test6',
          'test6.py',
          'test6',
          _ALL_PROFILES,
          0,
      ),
      # Tests QLKNN model, heat transport only
      (
          'test7',
          'test7.py',
          'test7',
          _ALL_PROFILES,
          0,
          1e-11,
          False,
      ),
      # Tests fixed_dt timestep
      (
          'test7_fixed_dt',
          'test7_fixed_dt.py',
          'test7_fixed_dt',
          _ALL_PROFILES,
          0,
          1e-11,
          False,
      ),
      # Tests current diffusion
      (
          'test8',
          'test8.py',
          'test8',
          _ALL_PROFILES,
          0,
      ),
      # Tests combined current diffusion + heat transport with QLKNN
      (
          'test9',
          'test9.py',
          'test9',
          _ALL_PROFILES,
          0,
      ),
      # Tests that optimizer gets the same result as the linear solver
      # when using linear initial guess and 0 iterations.
      # Making sure to use a test involving Pereverzev-Corrigan for this,
      # since we do want it in the linear initial guess.
      (
          'test9_optimizer',
          'test9_optimizer.py',
          'test9',
          _ALL_PROFILES,
          0,
      ),
      # Tests that Newton-Raphson gets the same result as the linear solver
      # when using linear initial guess and 0 iterations
      # Making sure to use a test involving Pereverzev-Corrigan for this,
      # since we do want it in the linear initial guess.
      (
          'test9_newton_raphson',
          'test9_newton_raphson.py',
          'test9',
          _ALL_PROFILES,
          0,
      ),
      # Tests bootstrap current with heat+current-diffusion. CGM model
      (
          'test10',
          'test10.py',
          'test10',
          _ALL_PROFILES,
          0,
      ),
      # Tests bootstrap current with heat+current-diffusion. QLKNN model
      (
          'test11',
          'test11.py',
          'test11',
          _ALL_PROFILES,
          0,
      ),
      # Tests heat+current-diffusion+particle transport with constant transport
      (
          'test12',
          'test12.py',
          'test12',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle sources with constant transport. No NBI source
      (
          'test13',
          'test13.py',
          'test13',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle sources with CGM transport. No NBI source
      (
          'test14',
          'test14.py',
          'test14',
          _ALL_PROFILES,
          0,
      ),
      # Tests all particle sources with CGM transport
      (
          'test15',
          'test15.py',
          'test15',
          _ALL_PROFILES,
          0,
      ),
      # Tests density transport with QLKNN. De scaled from chi_e model
      (
          'test16',
          'test16.py',
          'test16',
          _ALL_PROFILES,
          1e-3,
          5e-4,
      ),
      # Tests density transport with QLKNN. Deff+Veff model
      (
          'test17',
          'test17.py',
          'test17',
          _ALL_PROFILES,
          1e-5,
          2e-6,
      ),
      # Tests fusion power. CGM transport, heat+particle+psi transport
      (
          'test18',
          'test18.py',
          'test18',
          _ALL_PROFILES,
          0,
      ),
      # Tests fusion power. QLKNN transport, heat+particle+psi transport.
      (
          'test19',
          'test19.py',
          'test19',
          _ALL_PROFILES,
          7e-5,
          5e-4,
      ),
      # Tests explicit solver. Ti only. CHEASE geometry.
      (
          'test20',
          'test20.py',
          'test20',
          _ALL_PROFILES,
          0,
      ),
      # Tests implicit solver. Heat transport only. CHEASE geometry.
      (
          'test21',
          'test21.py',
          'test21',
          _ALL_PROFILES,
          0,
      ),
      # Tests ion-electron heat exchange test at low density. CHEASE geometry.
      (
          'test22',
          'test22.py',
          'test22',
          _ALL_PROFILES,
          0,
      ),
      # Tests Ohmic electron heat source. CHEASE geometry.
      (
          'test22_pohm',
          'test22_pohm.py',
          'test22_pohm',
          _ALL_PROFILES,
          0,
      ),
      # Tests ion-electron heat exchange test at high density. CHEASE geometry.
      (
          'test23',
          'test23.py',
          'test23',
          _ALL_PROFILES,
          0,
      ),
      # Tests pedestal internal boundary condition. CHEASE geometry.
      (
          'test24',
          'test24.py',
          'test24',
          _ALL_PROFILES,
          0,
      ),
      # Tests CGM transport model. Heat transport only. CHEASE geometry.
      (
          'test25',
          'test25.py',
          'test25',
          _ALL_PROFILES,
          0,
      ),
      # Tests QLKNN transport model. Heat transport only. CHEASE geometry.
      (
          'test26',
          'test26.py',
          'test26',
          _ALL_PROFILES,
          1e-10,
          1e-10,
      ),
      # Tests current diffusion. CHEASE geometry. Ip from parameters.
      (
          'test27',
          'test27.py',
          'test27',
          _ALL_PROFILES,
          0,
      ),
      # Tests current diffusion. CHEASE geometry. Ip from CHEASE.
      (
          'test28',
          'test28.py',
          'test28',
          _ALL_PROFILES,
          0,
      ),
      # Tests combined heat+current-diffusion. CHEASE geometry. QLKNN.
      (
          'test29',
          'test29.py',
          'test29',
          _ALL_PROFILES,
          0,
      ),
      # Tests time-dependent pedestal, Ptot, Ip. CHEASE geometry. QLKNN.
      (
          'test29_timedependent',
          'test29_timedependent.py',
          'test29_timedependent',
          _ALL_PROFILES,
          0,
      ),
      # Tests bootstrap current. CHEASE geometry. Heat+current-diffusion. CGM.
      (
          'test30',
          'test30.py',
          'test30',
          _ALL_PROFILES,
          0,
      ),
      # Tests combined heat, particle, current-diffusion.
      # CHEASE geometry. Constant transport
      (
          'test31',
          'test31.py',
          'test31',
          _ALL_PROFILES,
          0,
      ),
      # Tests combined heat, particle, current-diffusion and pedestal.
      # CHEASE geometry. Constant transport
      (
          'test32',
          'test32.py',
          'test32',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle sources. CHEASE geometry. No NBI.
      (
          'test33',
          'test33.py',
          'test33',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle sources. CHEASE geometry. No NBI. CGM + pedestal.
      (
          'test34',
          'test34.py',
          'test34',
          _ALL_PROFILES,
          0,
      ),
      # Tests all particle sources. CHEASE geometry. CGM + pedestal.
      (
          'test35',
          'test35.py',
          'test35',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle transport with QLKNN. De scaled from chie.
      (
          'test36',
          'test36.py',
          'test36',
          _ALL_PROFILES,
          1e-3,
          6e-5,
      ),
      # Tests particle transport with QLKNN. Deff+Veff model.
      (
          'test37',
          'test37.py',
          'test37',
          _ALL_PROFILES,
          1e-4,
          2e-6,
      ),
      # Tests Crank-Nicholson with particle transport and QLKNN. Deff+Veff
      (
          'test37_theta05',
          'test37_theta05.py',
          'test37_theta05',
          _ALL_PROFILES,
          0,
      ),
      # Tests fusion power. CHEASE geometry. Current, heat, particle transport.
      # CGM transport model.
      (
          'test38',
          'test38.py',
          'test38',
          _ALL_PROFILES,
          0,
      ),
      # Tests Pereverzev-Corrigan method for density. CHEASE geometry. QLKNN.
      # De scaled from chie.
      (
          'test39',
          'test39.py',
          'test39',
          _ALL_PROFILES,
          7e-5,
          5e-5,
      ),
      # Tests full integration for ITER-baseline-like config.
      (
          'test40',
          'test40.py',
          'test40',
          _ALL_PROFILES,
          7e-5,
          5e-5,
      ),
      # Tests full integration for ITER-baseline-like config. Linear solver.
      (
          'test41',
          'test41.py',
          'test41',
          _ALL_PROFILES,
          7e-5,
          5e-5,
      ),
      # Tests full integration for ITER-hybrid-like config. Linear solver.
      (
          'test42',
          'test42.py',
          'test42',
          _ALL_PROFILES,
          7e-5,
          5e-5,
      ),
      # Tests full integration for ITER-hybrid-like config.
      # Predictor-corrector solver.
      (
          'test42_predictor_corrector',
          'test42_predictor_corrector.py',
          'test42_predictor_corrector',
          _ALL_PROFILES,
          7e-5,
          5e-5,
      ),
      # Tests TORAX regression of test42
      (
          'test42_torax',
          'test42.py',
          'test42_torax',
          _ALL_PROFILES,
          1e-11,
          1e-11,
      ),
      # Tests Newton-Raphson nonlinear solver for ITER-hybrid-like-config
      (
          'test42_nl_Hmode',
          'test42_nl_Hmode.py',
          'test42_nl_Hmode',
          _ALL_PROFILES,
          1e-6,
          1e-6,
      ),
  )
  def test_torax_sim(
      self,
      config_name: str,
      ref_name: str,
      profiles: Sequence[str],
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
      use_ref_time: bool = False,
  ):
    """Integration test comparing to reference output from TORAX."""
    # The @parameterized decorator removes the `test_torax_sim` method,
    # so we separate the actual functionality into a helper method that will
    # not be removed.
    self._test_torax_sim(
        config_name,
        ref_name,
        profiles,
        rtol,
        atol,
        use_ref_time,
    )

  def test_fail(self):
    """Test that the integration tests can actually fail."""

    # Run test3 but pass in the reference result from test2
    with self.assertRaises(AssertionError):
      self._test_torax_sim(
          'test3.py',
          'test2',
          ('temp_ion', 'temp_el'),
      )

  # TODO(b/323504363): Re-enable this test once we can force the optimizer
  # to fail an error check.
  # def test_error_context(self):
  #   """Test that equinox errors are raised in optimizer but not in newton.

  #   Inputs chosen such that error_if in residual is raised (if enabled)
  #   for the final x_new state in the simulation, triggering the error for
  #   optimizer but not for newton-raphson
  #   """
  #   config = torax.config.Config()
  #   config.nbar = 5e-7
  #   config.set_fGW = False
  #   config.ne_bound_right = 5e-7
  #   config.npeak = 1
  #   config.t_final = 0.05
  #   config.set_pedestal = False
  #   config.Ptot = 0
  #   config.dens_eq = True
  #   config.S_pellet_tot = 0
  #   config.S_puff_tot = 0
  #   config.S_nbi_tot = 0
  #   config.transport.Ve_const = 3

  #   # Run default sim but with low density that triggers equinox error in
  #   # residual, but no NaN in x_new outputs such that Newton-Raphson proceeds
  #   time_step_calculator = chi_time_step_calculator.ChiTimeStepCalculator()

  #   # Equinox should raise here a XlaRuntimeError
  #   with self.assertRaises(jax.interpreters.xla.xe.XlaRuntimeError):
  #     sim_lib.build_sim_from_config(
  #         config,
  #         nonlinear_theta_method.OptimizerThetaMethod,
  #         time_step_calculator,
  #     ).run()

  #   # Should pass without errors
  #   sim_lib.build_sim_from_config(
  #       config,
  #       nonlinear_theta_method.NewtonRaphsonThetaMethod,
  #       time_step_calculator,
  #   ).run()

  def test_no_op(self):
    """Tests that running the stepper with all equations off is a no-op."""

    config = torax.config.Config()
    config.ion_heat_eq = False
    config.el_heat_eq = False
    config.current_eq = False

    time_step_calculator = chi_time_step_calculator.ChiTimeStepCalculator()
    geo = torax.build_circular_geometry(config)

    sim = sim_lib.build_sim_from_config(
        config, geo, linear_theta_method.LinearThetaMethod, time_step_calculator
    )

    torax_outputs = sim.run()
    state_history = state_lib.build_state_history_from_outputs(torax_outputs)
    t = state_lib.build_time_history_from_outputs(torax_outputs)

    chex.assert_rank(t, 1)
    history_length = state_history.temp_ion.value.shape[0]
    self.assertEqual(history_length, t.shape[0])
    self.assertGreater(t[-1], config.t_final)

    for torax_profile in _ALL_PROFILES:
      profile_history = state_history[torax_profile]
      # This is needed for CellVariable but not face variables
      if hasattr(profile_history, 'value'):
        profile_history = profile_history.value
      first_profile = profile_history[0]
      if not all(
          [np.all(profile == first_profile) for profile in profile_history]
      ):
        for i in range(1, len(profile_history)):
          # Most profiles should be == but jtot, q_face, and s_face can be
          # merely allclose because they are recalculated on each step.
          if not np.allclose(profile_history[i], first_profile):
            msg = (
                'Profile changed over time despite all equations being '
                'disabled.\n'
                f'Profile name: {torax_profile}\n'
                f'Initial value: {first_profile}\n'
                f'Failing time index: {i}\n'
                f'Failing value: {profile_history[i]}\n'
                f'Equality mask: {profile_history[i] == first_profile}\n'
                f'Diff: {profile_history[i] - first_profile}\n'
            )
            raise AssertionError(msg)

  @parameterized.named_parameters(
      (
          'implicit_update',
          linear_theta_method.LinearThetaMethod,
      ),
      (
          'explicit_update',
          explicit_stepper.ExplicitStepper,
      ),
  )
  def test_observers_update_during_runs(self, stepper):
    """Verify that the observer's state is updated after the simulation run."""
    # Load config structure.
    config_module = self._get_config_module('test1.py')
    config = config_module.get_config()
    geo = config_module.get_geometry(config)

    time_step_calculator = chi_time_step_calculator.ChiTimeStepCalculator()
    spectator = spectator_lib.InMemoryJaxArraySpectator()
    sim = sim_lib.build_sim_from_config(
        config, geo, stepper, time_step_calculator
    )
    sim.run(
        spectator=spectator,
    )
    self.assertNotEmpty(spectator.arrays)


if __name__ == '__main__':
  absltest.main()
