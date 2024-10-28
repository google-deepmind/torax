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
import jax
import numpy as np
import torax
from torax import output
from torax import sim as sim_lib
from torax import state
from torax.config import build_sim as build_sim_lib
from torax.config import numerics as numerics_lib
from torax.sources import source_models as source_models_lib
from torax.spectators import spectator as spectator_lib
from torax.stepper import linear_theta_method
from torax.tests.test_lib import explicit_stepper
from torax.tests.test_lib import sim_test_case
from torax.time_step_calculator import chi_time_step_calculator
from torax.transport_model import constant as constant_transport_model


_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


# Sim tests. Note that if any test data file does not have the same name as the
# test name, then the nonstandard mapping must be added to _REF_MAP_OVERRIDES in
# tests/test_lib/__init__.py
class SimTest(sim_test_case.SimTestCase):
  """Integration tests for torax.sim."""

  @parameterized.named_parameters(
      # Tests explicit solver
      (
          'test_explicit',
          'test_explicit.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests implicit solver with theta=0.5 (Crank-Nicolson)
      # Low tolerance since solver parameters are different
      (
          'test_crank_nicolson',
          'test_crank_nicolson.py',
          ('temp_ion', 'temp_el'),
          2e-1,
      ),
      # Tests implicit solver with theta=1.0 (backwards Euler)
      (
          'test_implicit',
          'test_implicit.py',
          _ALL_PROFILES,
          0,
      ),
      # Test ion-electron heat exchange at low density
      (
          'test_qei',
          'test_qei.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests sim.ArrayTimeStepCalculator
      (
          'test_arraytimestepcalculator',
          'test_qei.py',
          _ALL_PROFILES,
          0,
          True,
      ),
      # Tests pedestal internal boundary condition
      (
          'test_pedestal',
          'test_pedestal.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests CGM model heat transport only
      (
          'test_cgmheat',
          'test_cgmheat.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests BgB model heat transport only
      (
          'test_bohmgyrobohm_all',
          'test_bohmgyrobohm_all.py',
          _ALL_PROFILES,
          0,
      ),
      # Test that we are able to reproduce FiPy's behavior in a case where
      # FiPy is unstable
      (
          'test_semiimplicit_convection',
          'test_semiimplicit_convection.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests QLKNN model, heat transport only
      (
          'test_qlknnheat',
          'test_qlknnheat.py',
          _ALL_PROFILES,
          0,
          1e-11,
          False,
      ),
      # Tests fixed_dt timestep
      (
          'test_fixed_dt',
          'test_fixed_dt.py',
          _ALL_PROFILES,
          0,
          1e-11,
          False,
      ),
      # Tests current diffusion
      (
          'test_psiequation',
          'test_psiequation.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests combined current diffusion + heat transport with QLKNN
      (
          'test_psi_and_heat',
          'test_psi_and_heat.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests implementation of use_absolute_current
      (
          'test_absolute_generic_current_source',
          'test_absolute_generic_current_source.py',
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
          _ALL_PROFILES,
          0,
      ),
      # Tests bootstrap current with heat+current-diffusion. QLKNN model
      (
          'test_bootstrap',
          'test_bootstrap.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests heat+current-diffusion+particle transport with constant transport
      (
          'test_psi_heat_dens',
          'test_psi_heat_dens.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle sources with constant transport. No particle source
      (
          'test_particle_sources_constant',
          'test_particle_sources_constant.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests all particle sources including particle source, with CGM transport
      (
          'test_particle_sources_cgm',
          'test_particle_sources_cgm.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests specifying a prescribed time-varying arbitrary current profile
      (
          'test_prescribed_generic_current_source',
          'test_prescribed_generic_current_source.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests fusion power. CGM transport, heat+particle+psi transport
      (
          'test_fusion_power',
          'test_fusion_power.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests fusion power. QLKNN transport, heat+particle+psi transport.
      (
          'test_all_transport_fusion_qlknn',
          'test_all_transport_fusion_qlknn.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests CHEASE geometry. Implicit solver. Heat transport only.
      (
          'test_chease',
          'test_chease.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests EQDSK geometry. QLKNN, predictor-corrector, all transport.
      (
          'test_eqdsk',
          'test_eqdsk.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests Ohmic electron heat source. CHEASE geometry.
      (
          'test_ohmic_power',
          'test_ohmic_power.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests Bremsstrahlung heat sink. CHEASE geometry.
      (
          'test_bremsstrahlung',
          'test_bremsstrahlung.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests ion-electron heat exchange test at high density. CHEASE geometry.
      (
          'test_qei_chease_highdens',
          'test_qei_chease_highdens.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests current diffusion with Ip from parameters. CHEASE geometry.
      (
          'test_psichease_ip_parameters',
          'test_psichease_ip_parameters.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests current diffusion with Ip from CHEASE.
      (
          'test_psichease_ip_chease',
          'test_psichease_ip_chease.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests CHEASE geometry with psi initialized from prescribed jtot.
      (
          'test_psichease_prescribed_jtot',
          'test_psichease_prescribed_jtot.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests CHEASE geometry with psi initialized from prescribed johm.
      (
          'test_psichease_prescribed_johm',
          'test_psichease_prescribed_johm.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests time-dependent pedestal, Ptot, Ip. CHEASE geometry. QLKNN.
      (
          'test_timedependence',
          'test_timedependence.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests prescribed time-dependent ne (tied to GW frac with evolving Ip).
      (
          'test_prescribed_timedependent_ne',
          'test_prescribed_timedependent_ne.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests particle transport with QLKNN. De scaled from chie.
      # CHEASE geometry
      (
          'test_ne_qlknn_defromchie',
          'test_ne_qlknn_defromchie.py',
          _ALL_PROFILES,
          1e-8,
      ),
      # Tests particle transport with QLKNN. Deff+Veff model. CHEASE geometry.
      (
          'test_ne_qlknn_deff_veff',
          'test_ne_qlknn_deff_veff.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests Crank-Nicholson with particle transport and QLKNN. Deff+Veff
      (
          'test_all_transport_crank_nicolson',
          'test_all_transport_crank_nicolson.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests Pereverzev-Corrigan method for density. CHEASE geometry. QLKNN.
      # De scaled from chie.
      (
          'test_pc_method_ne',
          'test_pc_method_ne.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests full integration for ITER-baseline-like config. Linear solver.
      (
          'test_iterbaseline_mockup',
          'test_iterbaseline_mockup.py',
          _ALL_PROFILES,
          1e-10,
      ),
      # Tests full integration for ITER-hybrid-like config. Linear solver.
      (
          'test_iterhybrid_mockup',
          'test_iterhybrid_mockup.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests full integration for ITER-hybrid-like config.
      # Predictor-corrector solver.
      (
          'test_iterhybrid_predictor_corrector',
          'test_iterhybrid_predictor_corrector.py',
          _ALL_PROFILES,
          0,
      ),
      # Predictor-corrector solver with clipped QLKNN inputs.
      (
          'test_iterhybrid_predictor_corrector_clip_inputs',
          'test_iterhybrid_predictor_corrector_clip_inputs.py',
          _ALL_PROFILES,
          0,
      ),
      # Predictor-corrector solver with non-constant Zeff profile.
      (
          'test_iterhybrid_predictor_corrector_zeffprofile',
          'test_iterhybrid_predictor_corrector_zeffprofile.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests Newton-Raphson nonlinear solver for ITER-hybrid-like-config
      (
          'test_iterhybrid_newton',
          'test_iterhybrid_newton.py',
          _ALL_PROFILES,
          5e-7,
      ),
      # Tests current and density rampup for for ITER-hybrid-like-config
      # using Newton-Raphson. Only case which reverts to coarse_tol for several
      # timesteps (with negligible impact on results compared to full tol).
      (
          'test_iterhybrid_rampup',
          'test_iterhybrid_rampup.py',
          _ALL_PROFILES,
          0,
          1e-6,
      ),
      # Tests time-dependent circular geometry.
      (
          'test_time_dependent_circular_geo',
          'test_time_dependent_circular_geo.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests used for testing changing configs without recompiling.
      # Based on test_iterhybrid_predictor_corrector
      (
          'test_changing_config_before',
          'test_changing_config_before.py',
          _ALL_PROFILES,
          0,
      ),
      # Tests used for testing changing configs without recompiling.
      # Based on test_iterhybrid_predictor_corrector
      (
          'test_changing_config_after',
          'test_changing_config_after.py',
          _ALL_PROFILES,
          0,
      ),
  )
  def test_torax_sim(
      self,
      config_name: str,
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
        profiles,
        rtol=rtol,
        atol=atol,
        use_ref_time=use_ref_time,
    )

  def test_fail(self):
    """Test that the integration tests can actually fail."""

    # Run test_qei but pass in the reference result from test_implicit.
    with self.assertRaises(AssertionError):
      self._test_torax_sim(
          'test_qei.py',
          ('temp_ion', 'temp_el'),
          ref_name='test_implicit.nc',
          write_output=False,
      )

  def test_no_op(self):
    """Tests that running the stepper with all equations off is a no-op."""

    runtime_params = torax.general_runtime_params.GeneralRuntimeParams(
        numerics=numerics_lib.Numerics(
            t_final=0.1,
            ion_heat_eq=False,
            el_heat_eq=False,
            current_eq=False,
        ),
    )

    time_step_calculator = chi_time_step_calculator.ChiTimeStepCalculator()
    geo_provider = torax.ConstantGeometryProvider(
        torax.build_circular_geometry()
    )

    sim = sim_lib.build_sim_object(
        runtime_params=runtime_params,
        geometry_provider=geo_provider,
        stepper_builder=linear_theta_method.LinearThetaMethodBuilder(),
        transport_model_builder=constant_transport_model.ConstantTransportModelBuilder(),
        source_models_builder=source_models_lib.SourceModelsBuilder(),
        time_step_calculator=time_step_calculator,
    )

    sim_outputs = sim.run()
    history = output.StateHistory(sim_outputs, sim.source_models)

    history_length = history.core_profiles.temp_ion.value.shape[0]
    self.assertEqual(history_length, history.times.shape[0])
    self.assertGreater(history.times[-1], runtime_params.numerics.t_final)

    for torax_profile in _ALL_PROFILES:
      profile_history = history.core_profiles[torax_profile]
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
          linear_theta_method.LinearThetaMethodBuilder,
      ),
      (
          'explicit_update',
          explicit_stepper.ExplicitStepperBuilder,
      ),
  )
  def test_observers_update_during_runs(self, stepper_builder_constructor):
    """Verify that the observer's state is updated after the simulation run."""
    stepper_builder = stepper_builder_constructor()
    # Load config structure.
    config_module = self._get_config_module('test_explicit.py')
    runtime_params = config_module.get_runtime_params()
    geo_provider = config_module.get_geometry_provider()

    time_step_calculator = chi_time_step_calculator.ChiTimeStepCalculator()
    spectator = spectator_lib.InMemoryJaxArraySpectator()
    sim = sim_lib.build_sim_object(
        runtime_params=runtime_params,
        geometry_provider=geo_provider,
        stepper_builder=stepper_builder,
        transport_model_builder=config_module.get_transport_model_builder(),
        source_models_builder=config_module.get_sources_builder(),
        time_step_calculator=time_step_calculator,
    )
    sim.run(
        spectator=spectator,
    )
    self.assertNotEmpty(spectator.arrays)

  # pylint: disable=invalid-name
  @parameterized.product(
      test_config=(
          'test_psi_heat_dens',
          'test_psichease_ip_parameters',
          'test_psichease_ip_chease',
          'test_psichease_prescribed_jtot',
          'test_psichease_prescribed_johm',
          'test_iterhybrid_rampup',
      ),
      halfway=(False, True),
  )
  def test_core_profiles_are_recomputable(self, test_config, halfway):
    """Tests that core profiles from a previous run are recomputable.

    In this test we:
    - Load up a reference file and build a sim from its config.
    - Get profile values from either halfway or final time of the sim.
    - Override the dynamic runtime params slice with values from the reference.
    - Check that the initial core_profiles are equal.
    - In the case of loading from halfway, run the sim to the end and also check
    against reference.

    Args:
      test_config: the config id under test.
      halfway: Whether to load from halfway (or the end if not) to test in case
      there is different behaviour for the final step.
    """
    profiles = [
        output.TEMP_ION,
        output.TEMP_ION_RIGHT_BC,
        output.TEMP_EL,
        output.TEMP_EL_RIGHT_BC,
        output.NE,
        output.NI,
        output.NE_RIGHT_BC,
        output.NI_RIGHT_BC,
        output.PSI,
        output.PSIDOT,
        output.IP,
        output.NREF,
        output.Q_FACE,
        output.S_FACE,
        output.J_BOOTSTRAP,
        output.J_BOOTSTRAP_FACE,
        output.JOHM,
        output.CORE_PROFILES_GENERIC_CURRENT,
        output.JTOT,
        output.JTOT_FACE,
        output.I_BOOTSTRAP,
        output.SIGMA,
    ]
    ref_profiles, ref_time = self._get_refs(test_config + '.nc', profiles)
    if halfway:
      index = len(ref_time) // 2
    else:
      index = -1
    loading_time = ref_time[index]

    # Build the sim and runtime params at t=`loading_time`.
    sim = self._get_sim(test_config + '.py')
    geo = sim.geometry_provider(t=loading_time)
    dynamic_runtime_params_slice = sim.dynamic_runtime_params_slice_provider(
        t=loading_time,
    )
    source_models = sim.source_models_builder()

    # Load in the reference core profiles.
    Ip = ref_profiles[output.IP][index]
    temp_el = ref_profiles[output.TEMP_EL][index, :]
    temp_el_bc = ref_profiles[output.TEMP_EL_RIGHT_BC][index]
    temp_ion = ref_profiles[output.TEMP_ION][index, :]
    temp_ion_bc = ref_profiles[output.TEMP_ION_RIGHT_BC][index]
    ne = ref_profiles[output.NE][index, :]
    ne_bound_right = ref_profiles[output.NE_RIGHT_BC][index]
    psi = ref_profiles[output.PSI][index, :]

    # Override the dynamic runtime params with the loaded values.
    dynamic_runtime_params_slice.profile_conditions.Ip = Ip
    dynamic_runtime_params_slice.profile_conditions.Te = temp_el
    dynamic_runtime_params_slice.profile_conditions.Te_bound_right = temp_el_bc
    dynamic_runtime_params_slice.profile_conditions.Ti = temp_ion
    dynamic_runtime_params_slice.profile_conditions.Ti_bound_right = temp_ion_bc
    dynamic_runtime_params_slice.profile_conditions.ne = ne
    dynamic_runtime_params_slice.profile_conditions.ne_bound_right = (
        ne_bound_right
    )
    dynamic_runtime_params_slice.profile_conditions.psi = psi
    # When loading from file we want ne not to have transformations.
    # Both ne and the boundary condition are given in absolute values (not fGW).
    dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_fGW = (
        False
    )
    dynamic_runtime_params_slice.profile_conditions.ne_is_fGW = False
    dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_absolute = (
        True
    )
    # Additionally we want to avoid normalizing to nbar.
    dynamic_runtime_params_slice.profile_conditions.normalize_to_nbar = False

    # Get initial core profiles for the overriden dynamic runtime params.
    initial_state = sim_lib.get_initial_state(
        sim.static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
        sim.time_step_calculator,
        sim.step_fn,
    )

    # Check for agreement with the reference core profiles.
    verify_core_profiles(ref_profiles, index, initial_state.core_profiles)

    if halfway:
      # Run sim till the end and check that final core profiles match reference.
      initial_state.t = ref_time[index]
      step_fn = sim_lib.SimulationStepFn(
          stepper=sim.stepper,
          time_step_calculator=sim.time_step_calculator,
          transport_model=sim.transport_model,
      )
      sim_outputs = sim_lib.run_simulation(
          static_runtime_params_slice=sim.static_runtime_params_slice,
          dynamic_runtime_params_slice_provider=sim.dynamic_runtime_params_slice_provider,
          geometry_provider=sim.geometry_provider,
          initial_state=initial_state,
          time_step_calculator=sim.time_step_calculator,
          step_fn=step_fn,
      )
      final_core_profiles = sim_outputs.sim_history[-1].core_profiles
      verify_core_profiles(ref_profiles, -1, final_core_profiles)
    # pylint: enable=invalid-name

  def test_nans_trigger_error(self):
    """Verify that NaNs in profile evolution triggers early stopping and an error."""

    config_module = self._get_config_module('test_iterhybrid_makenans.py')
    sim = build_sim_lib.build_sim_from_config(config_module.CONFIG)
    sim_outputs = sim.run()

    state_history = output.StateHistory(sim_outputs, sim.source_models)
    self.assertEqual(state_history.sim_error, state.SimError.NAN_DETECTED)
    assert (
        state_history.times[-1]
        < config_module.CONFIG['runtime_params']['numerics']['t_final']
    )

  def test_restart_sim_from_file(self):
    test_config_state_file = 'test_iterhybrid_rampup.nc'
    restart_config = 'test_iterhybrid_rampup_restart.py'
    sim = self._get_sim(restart_config)
    profiles = [
        output.TEMP_ION,
        output.TEMP_ION_RIGHT_BC,
        output.TEMP_EL,
        output.TEMP_EL_RIGHT_BC,
        output.NE,
        output.NI,
        output.NE_RIGHT_BC,
        output.NI_RIGHT_BC,
        output.PSI,
        output.PSIDOT,
        output.IP,
        output.NREF,
        output.Q_FACE,
        output.S_FACE,
        output.J_BOOTSTRAP,
        output.JOHM,
        output.CORE_PROFILES_GENERIC_CURRENT,
        output.JTOT,
        output.JTOT_FACE,
        output.I_BOOTSTRAP,
        output.J_BOOTSTRAP_FACE,
        output.SIGMA,
    ]
    ref_profiles, ref_times = self._get_refs(
        test_config_state_file, profiles=profiles
    )

    sim_outputs = sim.run()
    history = output.StateHistory(sim_outputs, sim.source_models)
    ref_idx_offset = np.where(ref_times == history.times[0])

    for i in range(len(history.times)):
      core_profile_t = jax.tree.map(
          lambda x, idx=i: x[idx], history.core_profiles
      )
      verify_core_profiles(
          ref_profiles,
          i + ref_idx_offset[0][0],
          core_profile_t,
      )


def verify_core_profiles(ref_profiles, index, core_profiles):
  """Verify core profiles matches a reference at given index."""
  np.testing.assert_allclose(
      core_profiles.temp_el.value, ref_profiles[output.TEMP_EL][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.temp_ion.value, ref_profiles[output.TEMP_ION][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ne.value, ref_profiles[output.NE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ne.right_face_constraint,
      ref_profiles[output.NE_RIGHT_BC][index],
  )
  np.testing.assert_allclose(
      core_profiles.psi.value, ref_profiles[output.PSI][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.psidot.value, ref_profiles[output.PSIDOT][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ni.value, ref_profiles[output.NI][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ni.right_face_constraint,
      ref_profiles[output.NI_RIGHT_BC][index],
  )

  np.testing.assert_allclose(
      core_profiles.q_face, ref_profiles[output.Q_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.s_face, ref_profiles[output.S_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.nref, ref_profiles[output.NREF][index]
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_bootstrap,
      ref_profiles[output.J_BOOTSTRAP][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.jtot, ref_profiles[output.JTOT][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.jtot_face, ref_profiles[output.JTOT_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_bootstrap_face,
      ref_profiles[output.J_BOOTSTRAP_FACE][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.generic_current_source,
      ref_profiles[output.CORE_PROFILES_GENERIC_CURRENT][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.johm, ref_profiles[output.JOHM][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.I_bootstrap,
      ref_profiles[output.I_BOOTSTRAP][index],
  )
  np.testing.assert_allclose(
      core_profiles.currents.Ip, ref_profiles[output.IP][index]
  )


if __name__ == '__main__':
  absltest.main()
