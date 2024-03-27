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
  """Integration tests for torax.sim."""

  @parameterized.named_parameters(
      # Tests explicit solver
      (
          'test_explicit',
          'test_explicit.py',
          'test_explicit',
          _ALL_PROFILES,
          0,
      ),
      # Tests implicit solver with theta=0.5 (Crank-Nicolson)
      # Low tolerance since solver parameters are different
      (
          'test_crank_nicolson',
          'test_crank_nicolson.py',
          'test_implicit',
          ('temp_ion', 'temp_el'),
          2e-1,
      ),
      # Tests implicit solver with theta=1.0 (backwards Euler)
      (
          'test_implicit',
          'test_implicit.py',
          'test_implicit',
          _ALL_PROFILES,
          0,
      ),
      # Test ion-electron heat exchange at low density
      (
          'test_qei',
          'test_qei.py',
          'test_qei',
          _ALL_PROFILES,
          0,
      ),
      # Tests sim.ArrayTimeStepCalculator
      (
          'test_arraytimestepcalculator',
          'test_qei.py',
          'test_qei',
          _ALL_PROFILES,
          0,
          True,
      ),
      # Tests pedestal internal boundary condition
      (
          'test_pedestal',
          'test_pedestal.py',
          'test_pedestal',
          _ALL_PROFILES,
          0,
      ),
      # Tests CGM model heat transport only
      (
          'test_cgmheat',
          'test_cgmheat.py',
          'test_cgmheat',
          _ALL_PROFILES,
          0,
      ),
      # Test that we are able to reproduce FiPy's behavior in a case where
      # FiPy is unstable
      (
          'test_semiimplicit_convection',
          'test_semiimplicit_convection.py',
          'test_semiimplicit_convection',
          _ALL_PROFILES,
          0,
      ),
      # Tests QLKNN model, heat transport only
      (
          'test_qlknnheat',
          'test_qlknnheat.py',
          'test_qlknnheat',
          _ALL_PROFILES,
          0,
          1e-11,
          False,
      ),
      # Tests fixed_dt timestep
      (
          'test_fixed_dt',
          'test_fixed_dt.py',
          'test_fixed_dt',
          _ALL_PROFILES,
          0,
          1e-11,
          False,
      ),
      # Tests current diffusion
      (
          'test_psiequation',
          'test_psiequation.py',
          'test_psiequation',
          _ALL_PROFILES,
          0,
      ),
      # Tests combined current diffusion + heat transport with QLKNN
      (
          'test_psi_and_heat',
          'test_psi_and_heat.py',
          'test_psi_and_heat',
          _ALL_PROFILES,
          0,
      ),
      # Tests implementation of use_absolute_jext
      (
          'test_absolute_jext',
          'test_absolute_jext.py',
          'test_psi_and_heat',
          _ALL_PROFILES,
          0,
      ),

      # Tests that Newton-Raphson gets the same result as the linear solver
      # when using linear initial guess and 0 iterations
      # Making sure to use a test involving Pereverzev-Corrigan for this,
      # since we do want it in the linear initial guess.
      (
          'test_newton_raphson_zeroiter',
          'test_newton_raphson_zeroiter.py',
          'test_psi_and_heat',
          _ALL_PROFILES,
          0,
      ),
      # Tests bootstrap current with heat+current-diffusion. QLKNN model
      (
          'test_bootstrap',
          'test_bootstrap.py',
          'test_bootstrap',
          _ALL_PROFILES,
          0,
      ),
      # Tests heat+current-diffusion+particle transport with constant transport
      (
          'test_psi_heat_dens',
          'test_psi_heat_dens.py',
          'test_psi_heat_dens',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle sources with constant transport. No NBI source
      (
          'test_particle_sources_constant',
          'test_particle_sources_constant.py',
          'test_particle_sources_constant',
          _ALL_PROFILES,
          0,
      ),
      # Tests all particle sources including NBI, with CGM transport
      (
          'test_particle_sources_cgm',
          'test_particle_sources_cgm.py',
          'test_particle_sources_cgm',
          _ALL_PROFILES,
          0,
      ),
      # Tests fusion power. CGM transport, heat+particle+psi transport
      (
          'test_fusion_power',
          'test_fusion_power.py',
          'test_fusion_power',
          _ALL_PROFILES,
          0,
      ),
      # Tests fusion power. QLKNN transport, heat+particle+psi transport.
      (
          'test_all_transport_fusion_qlknn',
          'test_all_transport_fusion_qlknn.py',
          'test_all_transport_fusion_qlknn',
          _ALL_PROFILES,
          0,
      ),
      # Tests CHEASE geometry. Implicit solver. Heat transport only.
      (
          'test_chease',
          'test_chease.py',
          'test_chease',
          _ALL_PROFILES,
          0,
      ),
      # Tests Ohmic electron heat source. CHEASE geometry.
      (
          'test_ohmic_power',
          'test_ohmic_power.py',
          'test_ohmic_power',
          _ALL_PROFILES,
          0,
      ),
      # Tests ion-electron heat exchange test at high density. CHEASE geometry.
      (
          'test_qei_chease_highdens',
          'test_qei_chease_highdens.py',
          'test_qei_chease_highdens',
          _ALL_PROFILES,
          0,
      ),
      # Tests current diffusion with Ip from parameters. CHEASE geometry.
      (
          'test_psichease_ip_parameters',
          'test_psichease_ip_parameters.py',
          'test_psichease_ip_parameters',
          _ALL_PROFILES,
          0,
      ),
      # Tests current diffusion with Ip from CHEASE.
      (
          'test_psichease_ip_chease',
          'test_psichease_ip_chease.py',
          'test_psichease_ip_chease',
          _ALL_PROFILES,
          0,
      ),
      # Tests time-dependent pedestal, Ptot, Ip. CHEASE geometry. QLKNN.
      (
          'test_timedependence',
          'test_timedependence.py',
          'test_timedependence',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle transport with QLKNN. De scaled from chie.
      # CHEASE geometry
      (
          'test_ne_qlknn_defromchie',
          'test_ne_qlknn_defromchie.py',
          'test_ne_qlknn_defromchie',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle transport with QLKNN. Deff+Veff model. CHEASE geometry.
      (
          'test_ne_qlknn_deff_veff',
          'test_ne_qlknn_deff_veff.py',
          'test_ne_qlknn_deff_veff',
          _ALL_PROFILES,
          0,
      ),
      # Tests Crank-Nicholson with particle transport and QLKNN. Deff+Veff
      (
          'test_all_transport_crank_nicolson',
          'test_all_transport_crank_nicolson.py',
          'test_all_transport_crank_nicolson',
          _ALL_PROFILES,
          0,
      ),
      # Tests Pereverzev-Corrigan method for density. CHEASE geometry. QLKNN.
      # De scaled from chie.
      (
          'test_pc_method_ne',
          'test_pc_method_ne.py',
          'test_pc_method_ne',
          _ALL_PROFILES,
          0,
      ),
      # Tests full integration for ITER-baseline-like config. Linear solver.
      (
          'test_iterbaseline_mockup',
          'test_iterbaseline_mockup.py',
          'test_iterbaseline_mockup',
          _ALL_PROFILES,
          1e-10,
      ),
      # Tests full integration for ITER-hybrid-like config. Linear solver.
      (
          'test_iterhybrid_mockup',
          'test_iterhybrid_mockup.py',
          'test_iterhybrid_mockup',
          _ALL_PROFILES,
          0,
      ),
      # Tests full integration for ITER-hybrid-like config.
      # Predictor-corrector solver.
      (
          'test_iterhybrid_predictor_corrector',
          'test_iterhybrid_predictor_corrector.py',
          'test_iterhybrid_predictor_corrector',
          _ALL_PROFILES,
          0,
      ),
      # Tests Newton-Raphson nonlinear solver for ITER-hybrid-like-config
      (
          'test_iterhybrid_newton',
          'test_iterhybrid_newton.py',
          'test_iterhybrid_newton',
          _ALL_PROFILES,
          0,
      ),
      # Tests current and density rampup for for ITER-hybrid-like-config
      # using Newton-Raphson. Only case which reverts to coarse_tol for several
      # timesteps (with negligible impact on results compared to full tol).
      (
          'test_iterhybrid_rampup',
          'test_iterhybrid_rampup.py',
          'test_iterhybrid_rampup',
          _ALL_PROFILES,
          0,
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
          'test_qei.py',
          'test_implicit',
          ('temp_ion', 'temp_el'),
      )

  def test_no_op(self):
    """Tests that running the stepper with all equations off is a no-op."""

    config = torax.config.Config(
        t_final=0.1,
    )
    config.ion_heat_eq = False
    config.el_heat_eq = False
    config.current_eq = False

    time_step_calculator = chi_time_step_calculator.ChiTimeStepCalculator()
    geo = torax.build_circular_geometry(config)

    sim = sim_lib.build_sim_from_config(
        config, geo, linear_theta_method.LinearThetaMethod, time_step_calculator
    )

    torax_outputs = sim.run()
    state_history, _, _ = state_lib.build_history_from_states(torax_outputs)
    t = state_lib.build_time_history_from_states(torax_outputs)

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
    config_module = self._get_config_module('test_explicit.py')
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
