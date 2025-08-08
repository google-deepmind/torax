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

"""Tests that TORAX can be run with compilation disabled."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from torax._src.orchestration import jit_run_loop
from torax._src.orchestration import run_loop
from torax._src.output_tools import output
from torax._src.test_utils import sim_test_case

_ALL_PROFILES = (
    output.T_I,
    output.T_E,
    output.PSI,
    output.Q,
    output.N_E,
)


class SimExperimentalCompileTest(sim_test_case.SimTestCase):

  @parameterized.named_parameters(
      # Tests implicit solver with theta=1.0 (backwards Euler)
      (
          'test_implicit',
          'test_implicit.py',
      ),
      # Tests prescribed transport
      (
          'test_prescribed_transport',
          'test_prescribed_transport.py',
      ),
      # Tests BgB model heat transport only
      (
          'test_bohmgyrobohm_all',
          'test_bohmgyrobohm_all.py',
      ),
      # Tests combined transport model
      (
          'test_combined_transport',
          'test_combined_transport.py',
      ),
      # Test that we are able to reproduce FiPy's behavior in a case where
      # FiPy is unstable
      (
          'test_semiimplicit_convection',
          'test_semiimplicit_convection.py',
      ),
      # Tests fixed_dt timestep
      (
          'test_fixed_dt',
          'test_fixed_dt.py',
      ),
      # Tests combined current diffusion + heat transport with QLKNN
      (
          'test_psi_and_heat',
          'test_psi_and_heat.py',
      ),
      # Tests heat+current-diffusion+particle transport with constant transport
      (
          'test_psi_heat_dens',
          'test_psi_heat_dens.py',
      ),
      # Tests all particle sources including particle source, with CGM transport
      (
          'test_particle_sources_cgm',
          'test_particle_sources_cgm.py',
      ),
      # Tests specifying a prescribed time-varying arbitrary current profile
      (
          'test_prescribed_generic_current_source',
          'test_prescribed_generic_current_source.py',
      ),
      # Tests fusion power. QLKNN transport, heat+particle+psi transport.
      (
          'test_all_transport_fusion_qlknn',
          'test_all_transport_fusion_qlknn.py',
      ),
      # Tests CHEASE geometry. Implicit solver. Heat transport only.
      (
          'test_chease',
          'test_chease.py',
      ),
      # Tests EQDSK geometry. QLKNN, predictor-corrector, all transport.
      (
          'test_eqdsk',
          'test_eqdsk.py',
      ),
      # Tests Bremsstrahlung heat sink with time dependent Zimp and Z_eff.
      # CHEASE
      (
          'test_bremsstrahlung_time_dependent_Zimp',
          'test_bremsstrahlung_time_dependent_Zimp.py',
      ),
      # Tests CHEASE geometry with psi initialized from prescribed jtot.
      (
          'test_psichease_prescribed_jtot',
          'test_psichease_prescribed_jtot.py',
      ),
      # Tests CHEASE geometry with psi initialized from prescribed johm.
      (
          'test_psichease_prescribed_johm',
          'test_psichease_prescribed_johm.py',
      ),
      # Tests time-dependent pedestal, P_total, Ip. CHEASE geometry. QLKNN.
      (
          'test_timedependence',
          'test_timedependence.py',
      ),
      # Tests prescribed time-dependent n_e (tied to GW frac with evolving Ip).
      (
          'test_prescribed_timedependent_ne',
          'test_prescribed_timedependent_ne.py',
      ),
      # Tests particle transport with QLKNN. De scaled from chie.
      # CHEASE geometry
      (
          'test_ne_qlknn_defromchie',
          'test_ne_qlknn_defromchie.py',
      ),
      # Tests particle transport with QLKNN. Deff+Veff model. CHEASE geometry.
      (
          'test_ne_qlknn_deff_veff',
          'test_ne_qlknn_deff_veff.py',
      ),
      # Tests full integration for ITER-baseline-like config. Linear solver.
      (
          'test_iterbaseline_mockup',
          'test_iterbaseline_mockup.py',
      ),
      # Tests full integration for ITER-hybrid-like config. Linear solver.
      (
          'test_iterhybrid_mockup',
          'test_iterhybrid_mockup.py',
      ),
      # Tests full integration for ITER-hybrid-like config.
      # Predictor-corrector solver.
      (
          'test_iterhybrid_predictor_corrector',
          'test_iterhybrid_predictor_corrector.py',
      ),
      # ITERhybrid_predictor_corrector with EQDSK geometry.
      # See https://github.com/google-deepmind/torax/pull/482 for a plot
      # of the CHEASE vs EQDSK sim test comparison.
      (
          'test_iterhybrid_predictor_corrector_eqdsk',
          'test_iterhybrid_predictor_corrector_eqdsk.py',
      ),
      # Predictor-corrector solver with clipped QLKNN inputs.
      (
          'test_iterhybrid_predictor_corrector_clip_inputs',
          'test_iterhybrid_predictor_corrector_clip_inputs.py',
      ),
      # Predictor-corrector solver with non-constant Z_eff profile.
      (
          'test_iterhybrid_predictor_corrector_zeffprofile',
          'test_iterhybrid_predictor_corrector_zeffprofile.py',
      ),
      # Predictor-corrector solver with a time-dependent isotope mix.
      (
          'test_iterhybrid_predictor_corrector_timedependent_isotopes',
          'test_iterhybrid_predictor_corrector_timedependent_isotopes.py',
      ),
      # Predictor-corrector solver with tungsten.
      (
          'test_iterhybrid_predictor_corrector_tungsten',
          'test_iterhybrid_predictor_corrector_tungsten.py',
      ),
      # Predictor-corrector solver with ECCD Lin Liu model.
      (
          'test_iterhybrid_predictor_corrector_ec_linliu',
          'test_iterhybrid_predictor_corrector_ec_linliu.py',
      ),
      # Predictor-corrector solver with constant fraction of Pin radiation
      (
          'test_iterhybrid_predictor_corrector_constant_fraction_impurity_radiation',
          'test_iterhybrid_predictor_corrector_constant_fraction_impurity_radiation.py',
      ),
      # Predictor-corrector solver with Mavrin polynomial model for radiation.
      (
          'test_iterhybrid_predictor_corrector_mavrin_impurity_radiation',
          'test_iterhybrid_predictor_corrector_mavrin_impurity_radiation.py',
      ),
      # Predictor-corrector solver with constant pressure pedestal model.
      (
          'test_iterhybrid_predictor_corrector_set_pped_tpedratio_nped',
          'test_iterhybrid_predictor_corrector_set_pped_tpedratio_nped.py',
      ),
      # Predictor-corrector solver with cyclotron radiation heat sink
      (
          'test_iterhybrid_predictor_corrector_cyclotron',
          'test_iterhybrid_predictor_corrector_cyclotron.py',
      ),
      # Predictor-corrector solver with neoclassical transport
      (
          'test_iterhybrid_predictor_corrector_neoclassical',
          'test_iterhybrid_predictor_corrector_neoclassical.py',
      ),
      # Tests current and density rampup for ITER-hybrid-like-config
      # using Newton-Raphson. Only case which reverts to coarse_tol for several
      # timesteps (with negligible impact on results compared to full tol).
      (
          'test_iterhybrid_rampup',
          'test_iterhybrid_rampup.py',
      ),
      # Modified version of test_iterhybrid_rampup with sawtooth model.
      # Has an initial peaked current density, no heating, no current drive,
      # and resistivity is artificially increased to help induce more sawteeth.
      (
          'test_iterhybrid_rampup_sawtooth',
          'test_iterhybrid_rampup_sawtooth.py',
      ),
      # Tests used for testing changing configs without recompiling.
      # Based on test_iterhybrid_predictor_corrector
      (
          'test_changing_config_before',
          'test_changing_config_before.py',
      ),
      # Tests used for testing changing configs without recompiling.
      # Based on test_iterhybrid_predictor_corrector
      (
          'test_changing_config_after',
          'test_changing_config_after.py',
      ),
      # Tests current diffusion with vloop BC.
      # Initial Ip from parameters and psi from CHEASE, varying vloop BC.
      (
          'test_psichease_ip_parameters_vloop_varying',
          'test_psichease_ip_parameters_vloop_varying.py',
      ),
      # Tests current diffusion with vloop BC.
      # Initial Ip and psi from CHEASE.
      (
          'test_psichease_ip_chease_vloop',
          'test_psichease_ip_chease_vloop.py',
      ),
      # Tests current diffusion with vloop BC.
      # Initial Ip from parameters and psi from current_profile_nu formula.
      (
          'test_psichease_prescribed_jtot_vloop',
          'test_psichease_prescribed_jtot_vloop.py',
      ),
      (
          'test_implicit_short_optimizer',
          'test_implicit_short_optimizer.py',
      ),
      # Tests full integration for ITER-hybrid-like config with IMAS geometry.
      (
          'test_iterhybrid_predictor_corrector_imas',
          'test_iterhybrid_predictor_corrector_imas.py',
      ),
  )
  def test_run_simulation_experimental_compile(
      self,
      config_name: str,
  ):
    mock_run_loop = mock.MagicMock(side_effect=jit_run_loop.run_loop)
    with mock.patch.object(run_loop, 'run_loop', mock_run_loop):
      self._test_run_simulation(
          config_name,
          profiles=_ALL_PROFILES,
      )
    # Check the mock run loop was actually called.
    mock_run_loop.assert_called_once()


if __name__ == '__main__':
  absltest.main()
