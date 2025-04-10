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
import copy
from typing import Sequence, Final
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax import tree
import numpy as np
from torax import output
from torax import state
from torax.orchestration import initial_state
from torax.orchestration import run_simulation
from torax.tests.test_lib import sim_test_case
from torax.torax_pydantic import model_config

_ALL_PROFILES: Final[Sequence[str]] = (
    'temp_ion',
    'temp_el',
    'psi',
    'q_face',
    's_face',
    'ne',
)


# Sim tests. Note that if any test data file does not have the same name as the
# test name, then the nonstandard mapping must be added to _REF_MAP_OVERRIDES in
# tests/test_lib/__init__.py
class SimTest(sim_test_case.SimTestCase):
  """Integration tests for torax.sim."""

  @parameterized.named_parameters(
      # Tests implicit solver with theta=0.5 (Crank-Nicolson)
      # Low tolerance since solver parameters are different
      (
          'test_crank_nicolson',
          'test_crank_nicolson.py',
          ('temp_ion', 'temp_el'),
          2e-1,
          1e-10,
          'test_implicit.nc',
      ),
      # Tests implicit solver with theta=1.0 (backwards Euler)
      (
          'test_implicit',
          'test_implicit.py',
      ),
      # Tests BgB model heat transport only
      (
          'test_bohmgyrobohm_all',
          'test_bohmgyrobohm_all.py',
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
          _ALL_PROFILES,
          0,
          1e-8,
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
      # Tests Bremsstrahlung heat sink with time dependent Zimp and Zeff. CHEASE
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
      # Tests time-dependent pedestal, Ptot, Ip. CHEASE geometry. QLKNN.
      (
          'test_timedependence',
          'test_timedependence.py',
      ),
      # Tests prescribed time-dependent ne (tied to GW frac with evolving Ip).
      (
          'test_prescribed_timedependent_ne',
          'test_prescribed_timedependent_ne.py',
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
      # Predictor-corrector solver with non-constant Zeff profile.
      (
          'test_iterhybrid_predictor_corrector_zeffprofile',
          'test_iterhybrid_predictor_corrector_zeffprofile.py',
          _ALL_PROFILES,
          0,
          1e-8,
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
          _ALL_PROFILES,
          0,
          1e-8,
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
      # Modified version of test_iterhybrid_rampup with sawtooth model.
      # Has an initial peaked current density, no heating, no current drive,
      # and resistivity is artificially increased to help induce more sawteeth.
      (
          'test_iterhybrid_rampup_sawtooth',
          'test_iterhybrid_rampup_sawtooth.py',
          _ALL_PROFILES,
          0,
          1e-6,
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
      # Initial Ip from parameters and psi from nu formula.
      (
          'test_psichease_prescribed_jtot_vloop',
          'test_psichease_prescribed_jtot_vloop.py',
      ),
  )
  def test_run_simulation(
      self,
      config_name: str,
      profiles: Sequence[str] = _ALL_PROFILES,
      rtol: float | None = 0.,
      atol: float | None = None,
      ref_name: str | None = None,
  ):
    """Integration test comparing to reference output from TORAX."""
    # The @parameterized decorator removes the `test_torax_sim` method,
    # so we separate the actual functionality into a helper method that will
    # not be removed.
    self._test_run_simulation(
        config_name,
        profiles,
        rtol=rtol,
        atol=atol,
        ref_name=ref_name,
    )

  def test_fail(self):
    """Test that the integration tests can actually fail."""

    # Run test_qei but pass in the reference result from test_implicit.
    with self.assertRaises(AssertionError):
      self._test_run_simulation(
          'test_qei.py',
          ('temp_ion', 'temp_el'),
          ref_name='test_implicit.nc',
          write_output=False,
      )

  def test_no_op(self):
    """Tests that running the stepper with all equations off is a no-op."""
    torax_config = self._get_torax_config('test_iterhybrid_rampup.py')
    torax_config.update_fields({
        'runtime_params.numerics.t_final': 0.1,  # Modify final step.
        'runtime_params.numerics.ion_heat_eq': False,
        'runtime_params.numerics.el_heat_eq': False,
        'runtime_params.numerics.current_eq': False,
        'runtime_params.numerics.dens_eq': False,
        # Keep profiles fixed.
        'runtime_params.profile_conditions.Ip_tot': 3.0,
        'runtime_params.profile_conditions.ne': 1.0,
        'runtime_params.profile_conditions.ne_bound_right': 1.0,
        'runtime_params.profile_conditions.ne_is_fGW': False,
        'runtime_params.profile_conditions.Ti': 6.0,
        'runtime_params.profile_conditions.Te': 6.0,
    })

    history = run_simulation.run_simulation(torax_config, progress_bar=False)

    history_length = history.core_profiles.temp_ion.value.shape[0]
    self.assertEqual(history_length, history.times.shape[0])
    self.assertGreater(
        history.times[-1], torax_config.runtime_params.numerics.t_final
    )

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

  # pylint: disable=invalid-name
  @parameterized.parameters(
      'test_psi_heat_dens',
      'test_psichease_prescribed_jtot',
      'test_psichease_prescribed_johm',
      'test_iterhybrid_rampup',
  )
  def test_core_profiles_are_recomputable(self, test_config):
    """Tests that core profiles from a previous run are recomputable.

    In this test we:
    - Load up a reference file and build a sim from its config.
    - Get profile values from either halfway or final time of the sim.
    - Override the dynamic runtime params slice with values from the reference.
    - Run the sim to the end and check core profiles against reference.

    Args:
      test_config: the config id under test.
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
        output.IP_PROFILE_FACE,
        output.NREF,
        output.Q_FACE,
        output.S_FACE,
        output.J_BOOTSTRAP,
        output.J_BOOTSTRAP_FACE,
        output.JOHM,
        output.EXTERNAL_CURRENT,
        output.JTOT,
        output.JTOT_FACE,
        output.I_BOOTSTRAP,
        output.SIGMA,
    ]
    ref_profiles, ref_time = self._get_refs(test_config + '.nc', profiles)
    index = len(ref_time) // 2
    loading_time = ref_time[index]

    # Build the sim and runtime params at t=`loading_time`.
    config = self._get_config_dict(test_config + '.py')
    config['runtime_params']['numerics']['t_initial'] = loading_time
    torax_config = model_config.ToraxConfig.from_dict(config)

    original_get_initial_state = initial_state._get_initial_state

    def wrapped_get_initial_state(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        step_fn,
    ):
      # Load in the reference core profiles.
      Ip_total = ref_profiles[output.IP_PROFILE_FACE][index, -1] / 1e6
      temp_el = ref_profiles[output.TEMP_EL][index, :]
      temp_el_bc = ref_profiles[output.TEMP_EL_RIGHT_BC][index]
      temp_ion = ref_profiles[output.TEMP_ION][index, :]
      temp_ion_bc = ref_profiles[output.TEMP_ION_RIGHT_BC][index]
      ne = ref_profiles[output.NE][index, :]
      ne_bound_right = ref_profiles[output.NE_RIGHT_BC][index]
      psi = ref_profiles[output.PSI][index, :]

      # Override the dynamic runtime params with the loaded values.
      dynamic_runtime_params_slice.profile_conditions.Ip_tot = Ip_total
      dynamic_runtime_params_slice.profile_conditions.Te = temp_el
      dynamic_runtime_params_slice.profile_conditions.Te_bound_right = (
          temp_el_bc
      )
      dynamic_runtime_params_slice.profile_conditions.Ti = temp_ion
      dynamic_runtime_params_slice.profile_conditions.Ti_bound_right = (
          temp_ion_bc
      )
      dynamic_runtime_params_slice.profile_conditions.ne = ne
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right = (
          ne_bound_right
      )
      dynamic_runtime_params_slice.profile_conditions.psi = psi
      # When loading from file we want ne not to have transformations.
      # Both ne and the boundary condition are given in absolute values
      # (not fGW).
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_fGW = (
          False
      )
      dynamic_runtime_params_slice.profile_conditions.ne_is_fGW = False
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_absolute = (
          True
      )
      # Additionally we want to avoid normalizing to nbar.
      dynamic_runtime_params_slice.profile_conditions.normalize_to_nbar = False
      return original_get_initial_state(
          static_runtime_params_slice,
          dynamic_runtime_params_slice,
          geo,
          step_fn,
      )

    with mock.patch.object(
        initial_state, '_get_initial_state', wraps=wrapped_get_initial_state
    ):
      sim_outputs = run_simulation.run_simulation(
          torax_config, progress_bar=False)

    initial_core_profiles = tree.map(
        lambda x: x[0] if x is not None else None, sim_outputs.core_profiles
    )
    verify_core_profiles(ref_profiles, index, initial_core_profiles)

    final_core_profiles = tree.map(
        lambda x: x[-1] if x is not None else None, sim_outputs.core_profiles
    )
    verify_core_profiles(ref_profiles, -1, final_core_profiles)
    # pylint: enable=invalid-name

  def test_ip_bc_vloop_bc_equivalence(self):
    """Tests the equivalence of the Ip BC and the VLoop BC.

    In this test we:
    - Run a sim with the Ip BC.
    - Get from the output the vloop_lcfs
    - Run a second sim with the Vloop BC using vloop_lcfs from the first sim
    - Compare core profiles between the two sims. Exact equivalence is not
    expected since the boundary condition numerics are different, but should be
    close. This is a strong test that the VLoop BC is working as expected.
    """
    test_config = 'test_timedependence'
    profiles = [
        output.TEMP_ION,
        output.TEMP_EL,
        output.NE,
        output.PSI,
    ]

    # Run the first sim
    config_ip_bc = self._get_config_dict(test_config + '.py')
    torax_config = model_config.ToraxConfig.from_dict(config_ip_bc)
    sim_outputs_ip_bc = run_simulation.run_simulation(torax_config)
    middle_index = len(sim_outputs_ip_bc.times) // 2
    times = sim_outputs_ip_bc.times

    # Run the second sim
    config_vloop_bc = copy.deepcopy(config_ip_bc)
    config_vloop_bc['runtime_params']['profile_conditions'][
        'use_vloop_lcfs_boundary_condition'
    ] = True
    config_vloop_bc['runtime_params']['profile_conditions']['vloop_lcfs'] = (
        times,
        sim_outputs_ip_bc.core_profiles.vloop_lcfs,
    )
    torax_config = model_config.ToraxConfig.from_dict(config_vloop_bc)
    sim_outputs_vloop_bc = run_simulation.run_simulation(torax_config)

    for profile in profiles:
      np.testing.assert_allclose(
          sim_outputs_ip_bc.core_profiles[profile].value[middle_index, :],
          sim_outputs_vloop_bc.core_profiles[profile].value[middle_index, :],
          rtol=1e-3,
      )
      np.testing.assert_allclose(
          sim_outputs_ip_bc.core_profiles[profile].value[-1, :],
          sim_outputs_vloop_bc.core_profiles[profile].value[-1, :],
          rtol=1e-3,
      )

    # pylint: enable=invalid-name

  def test_nans_trigger_error(self):
    """Verify that NaNs in profile evolution triggers early stopping and an error."""
    torax_config = self._get_torax_config('test_iterhybrid_makenans.py')
    state_history = run_simulation.run_simulation(torax_config)

    self.assertEqual(state_history.sim_error, state.SimError.NAN_DETECTED)
    self.assertLess(
        state_history.times[-1],
        torax_config.runtime_params.numerics.t_final,
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
      core_profiles.currents.external_current_source,
      ref_profiles[output.EXTERNAL_CURRENT][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.johm, ref_profiles[output.JOHM][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.I_bootstrap,
      ref_profiles[output.I_BOOTSTRAP][index],
  )
  np.testing.assert_allclose(
      core_profiles.currents.Ip_profile_face,
      ref_profiles[output.IP_PROFILE_FACE][index, :],
  )


if __name__ == '__main__':
  absltest.main()
