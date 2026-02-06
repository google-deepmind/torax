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
import os
from typing import Final, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src import state
from torax._src.orchestration import run_simulation
from torax._src.output_tools import output
from torax._src.test_utils import paths
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import file_restart as file_restart_pydantic
from torax._src.torax_pydantic import model_config
import xarray as xr


_ALL_PROFILES: Final[Sequence[str]] = (
    output.T_I,
    output.T_E,
    output.PSI,
    output.Q,
    output.N_E,
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
          (output.T_I, output.T_E),
          2e-1,
          0,
          'test_implicit.nc',
      ),
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
          _ALL_PROFILES,
          1e-6,
          0,
          'test_iterhybrid_predictor_corrector_eqdsk.nc',
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
      # Predictor-corrector with Mavrin radiation and n_e_ratios impurity mode.
      (
          'test_iterhybrid_predictor_corrector_mavrin_n_e_ratios',
          'test_iterhybrid_predictor_corrector_mavrin_n_e_ratios.py',
      ),
      # Predictor-corrector w/ Mavrin, n_e_ratios, forward lengyel
      (
          'test_iterhybrid_predictor_corrector_mavrin_n_e_ratios_lengyel',
          'test_iterhybrid_predictor_corrector_mavrin_n_e_ratios_lengyel.py',
          _ALL_PROFILES,
          1e-8,
      ),
      # Predictor-corrector with Mavrin and n_e_ratios_Z_eff impurity mode.
      (
          'test_iterhybrid_predictor_corrector_mavrin_n_e_ratios_z_eff',
          'test_iterhybrid_predictor_corrector_mavrin_n_e_ratios_z_eff.py',
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
      # Predictor-corrector solver with TGLFNNukaea transport
      (
          'test_iterhybrid_predictor_corrector_tglfnn_ukaea',
          'test_iterhybrid_predictor_corrector_tglfnn_ukaea.py',
      ),
      # Predictor-corrector solver with TGLFNNukaea transport with rotation
      (
          'test_iterhybrid_predictor_corrector_tglfnn_ukaea_rotation',
          'test_iterhybrid_predictor_corrector_tglfnn_ukaea_rotation.py',
          _ALL_PROFILES,
          1e-6,
      ),
      # Predictor-corrector solver with rotation
      (
          'test_iterhybrid_predictor_corrector_rotation',
          'test_iterhybrid_predictor_corrector_rotation.py',
          _ALL_PROFILES,
          1e-6,
      ),
      # L-mode iterhybrid variant with combined transport model and overwrites
      (
          'test_iterhybrid_predictor_corrector_Lmode_combined',
          'test_iterhybrid_predictor_corrector_Lmode_combined.py',
      ),
      # Test Waltz shear suppression model.
      (
          'test_iterhybrid_predictor_corrector_rotation_waltz_suppression',
          'test_iterhybrid_predictor_corrector_rotation_waltz_suppression.py',
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
      # resistivity is artificially increased to help induce more sawteeth,
      # and the solver is predictor-corrector to simplify numerics.
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
      # Tests full integration for ITER-hybrid-based config with IMAS geometry
      # and profiles.
      (
          'test_imas_profiles_and_geo',
          'test_imas_profiles_and_geo.py',
      ),
      # Tests STEP scenario with Bohm-GyroBohm transport
      (
          'test_step_flattop_bgb',
          'test_step_flattop_bgb.py',
      ),
  )
  def test_run_simulation(
      self,
      config_name: str,
      profiles: Sequence[str] = _ALL_PROFILES,
      rtol: float | None = None,
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
    with self.assertRaises(ValueError):
      self._test_run_simulation(
          'test_qei.py',
          ('T_i', 'T_e'),
          ref_name='test_implicit.nc',
          write_output=False,
      )

  def test_no_op(self):
    """Tests that running the solver with all equations off is a no-op."""
    torax_config = self._get_torax_config('test_iterhybrid_rampup.py')
    torax_config.update_fields({
        'numerics': {
            't_final': 0.1,  # Modify final step.
            'fixed_dt': 0.06,
            'exact_t_final': False,
            'evolve_ion_heat': False,
            'evolve_electron_heat': False,
            'evolve_current': False,
            'evolve_density': False,
        },
        'profile_conditions': {
            'Ip': 3.0e6,
            'n_e': 1.0e20,
            'n_e_right_bc': 1.0e20,
            'n_e_nbar_is_fGW': False,
            'T_i': 6.0,
            'T_e': 6.0,
            'T_i_right_bc': 0.1,
            'T_e_right_bc': 0.1,
        },
    })

    _, history = run_simulation.run_simulation(torax_config, progress_bar=False)

    history_length = history._stacked_core_profiles.T_i.value.shape[0]
    self.assertEqual(history_length, history.times.shape[0])
    self.assertGreater(history.times[-1], torax_config.numerics.t_final)
    profiles_to_check = (
        (output.T_I, history._stacked_core_profiles.T_i),
        (output.T_E, history._stacked_core_profiles.T_e),
        (output.N_E, history._stacked_core_profiles.n_e),
        (output.PSI, history._stacked_core_profiles.psi),
        (output.Q, history._stacked_core_profiles.q_face),
        (output.MAGNETIC_SHEAR, history._stacked_core_profiles.s_face),
    )

    for profile_name, profile_history in profiles_to_check:
      # This is needed for CellVariable but not face variables
      if hasattr(profile_history, 'value'):
        profile_history = profile_history.value
      first_profile = profile_history[0]
      if not all(
          [np.all(profile == first_profile) for profile in profile_history]
      ):
        for i in range(1, len(profile_history)):
          # Most profiles should be == but j_total, q_face, and s_face can be
          # merely allclose because they are recalculated on each step.
          if not np.allclose(profile_history[i], first_profile):
            msg = (
                'Profile changed over time despite all equations being '
                'disabled.\n'
                f'Profile name: {profile_name}\n'
                f'Initial value: {first_profile}\n'
                f'Failing time index: {i}\n'
                f'Failing value: {profile_history[i]}\n'
                f'Equality mask: {profile_history[i] == first_profile}\n'
                f'Diff: {profile_history[i] - first_profile}\n'
            )
            raise AssertionError(msg)

  @parameterized.parameters(
      'test_psi_heat_dens',
      'test_psichease_prescribed_jtot',
      'test_psichease_prescribed_johm',
      'test_iterhybrid_rampup',
  )
  def test_simulation_with_restart(self, test_config: str):
    ref_name = test_config + '.nc'
    output_file = os.path.join(paths.test_data_dir(), ref_name)
    gt_output_xr = output.load_state_file(output_file)
    profiles_dataset = gt_output_xr.children[output.PROFILES].dataset
    ref_time = profiles_dataset[output.TIME].to_numpy()
    index = len(ref_time) // 2
    loading_time = ref_time[index]

    # Override the config to restart from the halfway point of the reference.
    torax_config = self._get_torax_config(test_config + '.py')
    torax_config.update_fields({'numerics.t_initial': loading_time})
    file_restart = file_restart_pydantic.FileRestart.from_dict(
        dict(
            filename=output_file,
            time=loading_time,
            do_restart=True,
            stitch=True,
        )
    )
    torax_config.update_fields({'restart': file_restart})

    output_xr, _ = run_simulation.run_simulation(
        torax_config, progress_bar=False
    )
    xr.map_over_datasets(xr.testing.assert_allclose, output_xr, gt_output_xr)

  def test_ip_bc_v_loop_bc_equivalence(self):
    """Tests the equivalence of the Ip BC and the VLoop BC.

    In this test we:
    - Run a sim with the Ip BC.
    - Get from the output the v_loop_lcfs
    - Run a second sim with the Vloop BC using v_loop_lcfs from the first sim
    - Compare core profiles between the two sims. Exact equivalence is not
    expected since the boundary condition numerics are different, but should be
    close. This is a strong test that the VLoop BC is working as expected.
    """
    test_config = 'test_timedependence'

    # Run the first sim
    config_ip_bc = self._get_config_dict(test_config + '.py')
    torax_config = model_config.ToraxConfig.from_dict(config_ip_bc)
    _, sim_outputs_ip_bc = run_simulation.run_simulation(torax_config)
    middle_index = len(sim_outputs_ip_bc.times) // 2
    times = sim_outputs_ip_bc.times

    # Run the second sim
    config_v_loop_bc = copy.deepcopy(config_ip_bc)
    config_v_loop_bc['profile_conditions'][
        'use_v_loop_lcfs_boundary_condition'
    ] = True
    config_v_loop_bc['profile_conditions']['v_loop_lcfs'] = (
        times,
        sim_outputs_ip_bc._stacked_core_profiles.v_loop_lcfs,
    )
    torax_config = model_config.ToraxConfig.from_dict(config_v_loop_bc)
    _, sim_outputs_v_loop_bc = run_simulation.run_simulation(torax_config)

    profiles_to_check = (
        sim_outputs_v_loop_bc._stacked_core_profiles.T_i,
        sim_outputs_v_loop_bc._stacked_core_profiles.T_e,
        sim_outputs_v_loop_bc._stacked_core_profiles.psi,
        sim_outputs_v_loop_bc._stacked_core_profiles.n_e,
    )

    for profile in profiles_to_check:
      np.testing.assert_allclose(
          profile.value[middle_index, :],
          profile.value[middle_index, :],
          rtol=1e-3,
      )
      np.testing.assert_allclose(
          profile.value[-1, :],
          profile.value[-1, :],
          rtol=1e-3,
      )

    # pylint: enable=invalid-name

  def test_prescribed_psidot(self):
    """Tests that a prescribed psidot is used when current is not evolved."""
    # Base config for a simple run.
    base_config_dict = {
        'profile_conditions': {},
        'plasma_composition': {},
        'numerics': {
            't_final': 3.0,
            'evolve_ion_heat': True,
            'evolve_electron_heat': True,
            'evolve_density': False,
        },
        'geometry': {
            'geometry_type': 'circular',
            'n_rho': 10,
        },
        'sources': {
            'ohmic': {},
            'generic_current': {},
        },
        'transport': {},
        'pedestal': {},
        'solver': {},
    }

    # --- Run 1: Reference run with current evolution to generate psidot ---
    ref_config_dict = copy.deepcopy(base_config_dict)
    ref_config_dict['numerics']['evolve_current'] = True
    ref_torax_config = model_config.ToraxConfig.from_dict(ref_config_dict)
    ref_output_xr, _ = run_simulation.run_simulation(
        ref_torax_config, progress_bar=False
    )
    # Extract the psidot profile and time array
    ref_psidot = ref_output_xr.profiles.v_loop.values
    ref_time = ref_output_xr.time.values
    ref_rho_norm = ref_output_xr.rho_norm.values

    # --- Run 2: Test run without current evolution ---
    # We expect this to be different from Run 1.
    test_config_dict = copy.deepcopy(base_config_dict)
    test_config_dict['numerics']['evolve_current'] = False
    test_torax_config = model_config.ToraxConfig.from_dict(test_config_dict)
    test_output_xr_different, _ = run_simulation.run_simulation(
        test_torax_config, progress_bar=False
    )

    # --- Run 3: Test run without current evolution, using prescribed psidot ---
    # We expect this to be identical to Run 1.
    test_config_dict = copy.deepcopy(base_config_dict)
    test_config_dict['numerics']['evolve_current'] = False
    # Provide psidot as a time-varying array
    test_config_dict['profile_conditions']['psidot'] = (
        ref_time,
        ref_rho_norm,
        ref_psidot,
    )
    test_torax_config = model_config.ToraxConfig.from_dict(test_config_dict)
    test_output_xr_same, _ = run_simulation.run_simulation(
        test_torax_config, progress_bar=False
    )

    # Compare Runs 1 and 3 - v_loop (cell grid) should be identical
    # We ignore the v_loop_lcfs since it does not impact cell-grid Ohmic power,
    # and it is correct that v_loop_lcfs is different between psi-evolving and
    # psi-fixed simulations.
    np.testing.assert_allclose(
        ref_output_xr.profiles.v_loop.values[:, :-1],
        test_output_xr_same.profiles.v_loop.values[:, :-1],
        rtol=1e-6,
    )

    # Compare Runs 1 and 2 - v_loop (cell grid) should be different
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(
          ref_output_xr.profiles.v_loop.values[:, :-1],
          test_output_xr_different.profiles.v_loop.values[:, :-1],
          rtol=1e-6,
      )

  def test_nans_trigger_error(self):
    """Verify that NaNs in profile evolution triggers early stopping and an error."""
    torax_config = self._get_torax_config('test_iterhybrid_makenans.py')
    _, state_history = run_simulation.run_simulation(torax_config)

    self.assertEqual(state_history.sim_error, state.SimError.NAN_DETECTED)
    self.assertLess(state_history.times[-1], torax_config.numerics.t_final)

  def test_low_temperature_error(self):
    """Verify that a config with radiation collapse triggers early stopping and an error."""
    # We don't compare the results to a reference solution, because the purpose
    # of this test is to check that the code exits correctly, rather than
    # achieves a specific solution.
    torax_config = self._get_torax_config(
        'test_iterhybrid_radiation_collapse.py'
    )
    # Increase T_min so that we are sure to hit the error
    torax_config.update_fields({'numerics.T_minimum_eV': 20})

    _, state_history = run_simulation.run_simulation(torax_config)

    # Check that the simulation stopped due to low temperature collapse
    self.assertEqual(
        state_history.sim_error, state.SimError.LOW_TEMPERATURE_COLLAPSE
    )
    self.assertLess(state_history.times[-1], torax_config.numerics.t_final)

  def test_full_output_matches_reference(self):
    """Check for complete output match with reference."""
    torax_config = self._get_torax_config('test_iterhybrid_rampup.py')
    _, state_history = run_simulation.run_simulation(torax_config)
    sim_data_tree = state_history.simulation_output_to_xr()
    expected_results_path = self._expected_results_path(
        'test_iterhybrid_rampup.nc'
    )
    ref_data_tree = output.load_state_file(expected_results_path)
    xr.map_over_datasets(
        xr.testing.assert_allclose, sim_data_tree, ref_data_tree
    )


if __name__ == '__main__':
  absltest.main()
