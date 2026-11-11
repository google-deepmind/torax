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
import copy
import dataclasses
import functools
import os
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import (
    runtime_params as core_profile_runtime_params,
)
from torax._src.orchestration import initial_state
from torax._src.orchestration import run_simulation
from torax._src.output_tools import output
from torax._src.output_tools import output_keys
from torax._src.pedestal_model import pedestal_state
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import config
import xarray as xr

# pylint: disable=invalid-name


class InitialStateTest(sim_test_case.SimTestCase):

  def test_initial_state_has_dt(self):
    torax_config = config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    step_fn = run_simulation.make_step_fn(torax_config)
    sim_state, _ = initial_state.get_initial_state_and_post_processed_outputs(
        step_fn=step_fn,
    )
    self.assertEqual(sim_state.dt, 0.0)

  def test_from_file_restart(self):
    torax_config = self._get_torax_config('test_iterhybrid_rampup_restart.py')
    step_fn = run_simulation.make_step_fn(torax_config)
    non_restart, _ = initial_state.get_initial_state_and_post_processed_outputs(
        step_fn=step_fn,
    )

    assert torax_config.restart is not None
    result, post_processed = (
        initial_state.get_initial_state_and_post_processed_outputs_from_file(
            file_restart=torax_config.restart,
            step_fn=step_fn,
        )
    )

    self.assertNotEqual(post_processed.E_fusion, 0.0)
    self.assertNotEqual(post_processed.E_aux_total, 0.0)
    self.assertNotEqual(post_processed.E_ohmic_e, 0.0)
    self.assertNotEqual(post_processed.E_external_injected, 0.0)
    self.assertNotEqual(post_processed.E_external_total, 0.0)

    with self.assertRaises(AssertionError):
      chex.assert_trees_all_equal(result, non_restart)
    self.assertNotEqual(result.t, non_restart.t)
    assert torax_config.restart is not None
    self.assertEqual(result.t, torax_config.restart.time)
    self.assertIsInstance(
        result.core_profiles.v_loop_lcfs, jax.Array
    )
    np.testing.assert_allclose(
        result.core_profiles.v_loop_lcfs,
        result.core_profiles.psidot.face_value()[-1],
        rtol=1e-6,
    )
    assert result.core_profiles.psidot.right_face_grad_constraint is not None
    self.assertNotEqual(
        float(result.core_profiles.psidot.right_face_grad_constraint), 0.0
    )
    chex.assert_trees_all_close(
        result.model_state.geometry.Phi_b, result.geometry.Phi_b
    )

  def test_from_file_restart_without_v_loop_dataset(self):
    """Verifies file restart under I_p BC when V_LOOP is omitted."""
    torax_config = self._get_torax_config('test_iterhybrid_rampup_restart.py')
    assert torax_config.restart is not None
    data_tree = output.load_state_file(str(torax_config.restart.filename))
    data_tree[output_keys.PROFILES] = (
        data_tree[output_keys.PROFILES].to_dataset().drop_vars(
            output_keys.V_LOOP
        )
    )

    temp_dir = self.create_tempdir().full_path
    no_v_loop_nc = os.path.join(temp_dir, 'no_v_loop_restart.nc')
    data_tree.to_netcdf(no_v_loop_nc)

    restart_config = torax_config.restart.model_copy(
        update={'filename': no_v_loop_nc}
    )
    step_fn = run_simulation.make_step_fn(torax_config)
    result, _ = (
        initial_state.get_initial_state_and_post_processed_outputs_from_file(
            file_restart=restart_config,
            step_fn=step_fn,
        )
    )
    assert result.core_profiles.psidot.right_face_grad_constraint is not None
    np.testing.assert_allclose(
        result.core_profiles.psidot.right_face_grad_constraint, 0.0
    )
    self.assertIsInstance(result.core_profiles.v_loop_lcfs, jax.Array)
    np.testing.assert_allclose(
        result.core_profiles.v_loop_lcfs,
        result.core_profiles.psidot.face_value()[-1],
        rtol=1e-6,
    )

  def test_restart_uses_new_geometry_phi_b(self):
    torax_config = self._get_torax_config('test_iterhybrid_rampup_restart.py')
    step_fn = run_simulation.make_step_fn(torax_config)
    result, _ = (
        initial_state.get_initial_state_and_post_processed_outputs_from_file(
            file_restart=torax_config.restart,  # pyrefly: ignore[bad-argument-type]
            step_fn=step_fn,
        )
    )
    chex.assert_trees_all_close(
        result.model_state.geometry.Phi_b, result.geometry.Phi_b
    )

  @parameterized.parameters(
      'test_psi_heat_dens',
      'test_psichease_prescribed_jtot',
      'test_psichease_prescribed_johm',
      'test_iterhybrid_rampup',
  )
  def test_core_profile_final_step(self, test_config):
    profiles = [
        output_keys.T_I,
        output_keys.T_E,
        output_keys.N_E,
        output_keys.N_I,
        output_keys.PSI,
        output_keys.V_LOOP,
        output_keys.I_P_PROFILE,
        output_keys.Q,
        output_keys.MAGNETIC_SHEAR,
        output_keys.J_TOROIDAL_BOOTSTRAP,
        output_keys.J_TOROIDAL_OHMIC,
        output_keys.J_TOROIDAL_EXTERNAL,
        output_keys.J_TOROIDAL_TOTAL,
        output_keys.SIGMA_PARALLEL,
    ]
    index = -1
    ref_profiles, ref_time = self._get_refs(test_config + '.nc', profiles)
    t = int(ref_time[index])

    config_dict = self._get_config_dict(test_config + '.py')
    config_dict.setdefault('time_step_calculator', {})['t_initial'] = t
    torax_config = config.ToraxConfig.from_dict(config_dict)

    step_fn = run_simulation.make_step_fn(torax_config)
    runtime_params, geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.time_step_calculator.t_initial,
            runtime_params_provider=step_fn.runtime_params_provider,
            geometry_model=step_fn.solver.models.geometry_model,
            is_initialization=True,
        )
    )

    # Load in the reference core profiles.
    I_p_total = ref_profiles[output_keys.I_P_PROFILE][index, -1]
    # All profiles are on a grid with [left_face, cell_grid, right_face]
    T_e = ref_profiles[output_keys.T_E][index, 1:-1]
    T_e_bc = ref_profiles[output_keys.T_E][index, -1]
    T_i = ref_profiles[output_keys.T_I][index, 1:-1]
    T_i_bc = ref_profiles[output_keys.T_I][index, -1]
    n_e = ref_profiles[output_keys.N_E][index, 1:-1]
    n_e_right_bc = ref_profiles[output_keys.N_E][index, -1]
    psi = ref_profiles[output_keys.PSI][index, 1:-1]

    # When loading from file we want ne not to have transformations.
    # Both ne and the boundary condition are given in absolute values (not
    # f_GW).
    # Additionally we want to avoid normalizing to n_e_bar.
    runtime_params = dataclasses.replace(
        runtime_params,
        profile_conditions=dataclasses.replace(
            runtime_params.profile_conditions,
            I_p=I_p_total,
            T_e=T_e,
            T_e_right_bc=T_e_bc,
            T_i=T_i,
            T_i_right_bc=T_i_bc,
            n_e=n_e,
            n_e_right_bc=n_e_right_bc,
            psi=psi,
            psi_source=(
                core_profile_runtime_params.PsiSource.PROFILE_CONDITIONS
            ),
            n_e_right_bc_is_f_GW=False,
            n_e_bar_is_f_GW=False,
            normalize_n_e=False,
            n_e_right_bc_is_absolute=True,
        ),
    )

    result = initial_state._get_initial_state(
        runtime_params=runtime_params,
        geo=geo,
        step_fn=step_fn,
    )
    core_profile_helpers.verify_core_profiles(
        ref_profiles, index, result.core_profiles
    )

  def test_restart_coupled_extended_lengyel_forward(self):
    config_dict = self._get_config_dict(
        'test_iterhybrid_predictor_corrector_mavrin_n_e_ratios_lengyel.py'
    )
    config_dict['geometry']['n_rho'] = 10
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        'dt': 0.2,
        't_initial': 0.0,
        't_final': 0.8,
        'exact_t_final': True,
    }
    config_dict['edge']['update_T_e'] = True
    config_dict['edge']['update_T_i'] = True
    config_dict['edge']['use_enrichment_model'] = True
    config_dict['edge']['impurity_sot'] = 'core'
    config_dict['edge']['initial_guess'] = {
        'use_previous_step_as_guess': True,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    continuous_res = run_simulation.run_simulation(
        torax_config, progress_bar=False
    )
    continuous_xr = continuous_res.simulation_output_to_xr()

    temp_dir = self.create_tempdir().full_path
    nc_path = os.path.join(temp_dir, 'continuous_forward.nc')
    continuous_xr.to_netcdf(nc_path)

    edge_state_t0 = (
        continuous_xr.children[output_keys.MODEL_STATE]
        .children[output_keys.EDGE]
        .sel(time=0.0, method='nearest')
        .dataset
    )
    edge_state_t_restart = (
        continuous_xr.children[output_keys.MODEL_STATE]
        .children[output_keys.EDGE]
        .sel(time=0.4, method='nearest')
        .dataset
    )
    self.assertFalse(
        np.allclose(
            edge_state_t_restart[output_keys.PRESSURE_NEUTRAL_DIVERTOR].values,
            edge_state_t0[output_keys.PRESSURE_NEUTRAL_DIVERTOR].values,
        )
    )
    self.assertFalse(
        np.allclose(
            edge_state_t_restart[output_keys.WARM_START_ALPHA_T].values,
            edge_state_t0[output_keys.WARM_START_ALPHA_T].values,
        )
    )

    restart_config_dict = copy.deepcopy(config_dict)
    restart_config_dict['time_step_calculator']['t_initial'] = 0.4
    restart_config_dict['restart'] = {
        'filename': nc_path,
        'time': 0.4,
        'do_restart': True,
        'stitch': True,
    }
    restarted_config = config.ToraxConfig.from_dict(restart_config_dict)
    restarted_res = run_simulation.run_simulation(
        restarted_config, progress_bar=False
    )
    restarted_xr = restarted_res.simulation_output_to_xr()

    assert_allclose_fn = functools.partial(
        xr.testing.assert_allclose, atol=1e-6
    )
    xr.map_over_datasets(assert_allclose_fn, restarted_xr, continuous_xr)

  def test_restart_coupled_extended_lengyel_inverse(self):
    config_dict = self._get_config_dict(
        'test_iterhybrid_predictor_corrector_mavrin_n_e_ratios_lengyel.py'
    )
    config_dict['geometry']['n_rho'] = 10
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        'dt': 0.2,
        't_initial': 0.0,
        't_final': 0.8,
        'exact_t_final': True,
    }
    config_dict['edge']['computation_mode'] = 'inverse'
    config_dict['edge']['fixed_impurity_concentrations'] = {'W': 1e-5}
    config_dict['edge']['seed_impurity_weights'] = {'Ne': 1.0}
    config_dict['edge']['T_e_target'] = 10.0
    config_dict['edge']['update_T_e'] = True
    config_dict['edge']['update_T_i'] = True
    config_dict['edge']['update_n_impurities'] = True
    config_dict['edge']['use_enrichment_model'] = True
    config_dict['edge']['impurity_sot'] = 'core'
    config_dict['edge']['initial_guess'] = {
        'use_previous_step_as_guess': True,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    continuous_res = run_simulation.run_simulation(
        torax_config, progress_bar=False
    )
    continuous_xr = continuous_res.simulation_output_to_xr()

    temp_dir = self.create_tempdir().full_path
    nc_path = os.path.join(temp_dir, 'continuous_inverse.nc')
    continuous_xr.to_netcdf(nc_path)

    restart_config_dict = copy.deepcopy(config_dict)
    restart_config_dict['time_step_calculator']['t_initial'] = 0.4
    restart_config_dict['restart'] = {
        'filename': nc_path,
        'time': 0.4,
        'do_restart': True,
        'stitch': True,
    }
    restarted_config = config.ToraxConfig.from_dict(restart_config_dict)
    restarted_res = run_simulation.run_simulation(
        restarted_config, progress_bar=False
    )
    restarted_xr = restarted_res.simulation_output_to_xr()

    cont_prof = continuous_xr.children[output_keys.PROFILES].dataset
    rest_prof = restarted_xr.children[output_keys.PROFILES].dataset
    np.testing.assert_allclose(
        rest_prof[output_keys.Z_EFF].values,
        cont_prof[output_keys.Z_EFF].values,
        atol=1e-6,
    )
    if output_keys.N_IMPURITY_SPECIES in cont_prof.data_vars:
      np.testing.assert_allclose(
          rest_prof[output_keys.N_IMPURITY_SPECIES].values,
          cont_prof[output_keys.N_IMPURITY_SPECIES].values,
          atol=1e-6,
      )

    assert_allclose_fn = functools.partial(
        xr.testing.assert_allclose, atol=1e-6
    )
    xr.map_over_datasets(assert_allclose_fn, restarted_xr, continuous_xr)

  def test_restart_ibc_pedestal_mid_transition(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {'model_name': 'circular', 'n_rho': 10}
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        'dt': 0.05,
        't_initial': 0.0,
        't_final': 0.25,
        'exact_t_final': True,
    }
    config_dict['pedestal'] = {
        'mode': 'internal_boundary_condition',
        'structure_model': {'model_name': 'set_T_top_n_top'},
        'transition_time_width': 0.3,
        'formation_model': {
            'model_name': 'prescribed',
            'pedestal_active': ({0.0: False, 0.05: True}, 'step'),
        },
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    continuous_res = run_simulation.run_simulation(
        torax_config, progress_bar=False
    )
    continuous_xr = continuous_res.simulation_output_to_xr()

    temp_dir = self.create_tempdir().full_path
    nc_path = os.path.join(temp_dir, 'continuous_ibc.nc')
    continuous_xr.to_netcdf(nc_path)

    pedestal_state_t = (
        continuous_xr.children[output_keys.MODEL_STATE]
        .children[output_keys.PEDESTAL]
        .sel(time=0.15, method='nearest')
        .dataset
    )
    self.assertEqual(
        int(pedestal_state_t[output_keys.CONFINEMENT_MODE].values),
        int(pedestal_runtime_params.ConfinementMode.TRANSITIONING_TO_H_MODE),
    )
    np.testing.assert_allclose(
        float(pedestal_state_t[output_keys.TRANSITION_START_TIME].values),
        0.05,
        atol=1e-6,
    )

    restart_config_dict = copy.deepcopy(config_dict)
    restart_config_dict['time_step_calculator']['t_initial'] = 0.15
    restart_config_dict['restart'] = {
        'filename': nc_path,
        'time': 0.15,
        'do_restart': True,
        'stitch': True,
    }
    restarted_config = config.ToraxConfig.from_dict(restart_config_dict)
    assert restarted_config.restart is not None
    initial_sim_state, _ = (
        initial_state.get_initial_state_and_post_processed_outputs_from_file(
            file_restart=restarted_config.restart,
            step_fn=run_simulation.make_step_fn(restarted_config),
        )
    )
    self.assertIsInstance(
        initial_sim_state.model_state.pedestal,
        pedestal_state.IBCPedestalState,
    )
    assert isinstance(
        initial_sim_state.model_state.pedestal,
        pedestal_state.IBCPedestalState,
    )
    self.assertEqual(
        int(initial_sim_state.model_state.pedestal.confinement_mode),
        int(pedestal_runtime_params.ConfinementMode.TRANSITIONING_TO_H_MODE),
    )
    np.testing.assert_allclose(
        float(initial_sim_state.model_state.pedestal.transition_start_time),
        0.05,
        atol=1e-6,
    )
    restarted_res = run_simulation.run_simulation(
        restarted_config, progress_bar=False
    )
    restarted_xr = restarted_res.simulation_output_to_xr()

    assert_allclose_fn = functools.partial(
        xr.testing.assert_allclose, atol=1e-6
    )
    xr.map_over_datasets(assert_allclose_fn, restarted_xr, continuous_xr)

  @parameterized.parameters(
      'internal_boundary_condition',
      'adaptive_transport',
  )
  def test_restart_power_scaling_formation_with_dw_dt(self, mode: str):
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {'model_name': 'circular', 'n_rho': 10}
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        'dt': 0.05,
        't_initial': 0.0,
        't_final': 0.25,
        'exact_t_final': True,
    }
    config_dict['pedestal'] = {
        'mode': mode,
        'structure_model': {'model_name': 'set_T_top_n_top'},
        'transition_time_width': 0.1,
        'formation_model': {
            'model_name': 'martin_scaling',
            'include_dW_dt_in_P_SOL': True,
            'P_LH_hysteresis_factor': 0.7,
        },
    }
    config_dict['sources'] = {
        'generic_heat': {
            'gaussian_location': 0.15,
            'gaussian_width': 0.1,
            'P_total': {0.0: 2.0e6, 0.1: 35.0e6, 0.2: 2.0e6},
            'electron_heat_fraction': 0.8,
        }
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    continuous_res = run_simulation.run_simulation(
        torax_config, progress_bar=False
    )
    continuous_xr = continuous_res.simulation_output_to_xr()

    temp_dir = self.create_tempdir().full_path
    nc_path = os.path.join(temp_dir, f'continuous_{mode}.nc')
    continuous_xr.to_netcdf(nc_path)

    scalars_t = (
        continuous_xr.children[output_keys.SCALARS]
        .sel(time=0.15, method='nearest')
        .dataset
    )
    self.assertNotEqual(
        float(scalars_t['dW_thermal_dt_smoothed'].values),
        0.0,
    )

    restart_config_dict = copy.deepcopy(config_dict)
    restart_config_dict['time_step_calculator']['t_initial'] = 0.15
    restart_config_dict['restart'] = {
        'filename': nc_path,
        'time': 0.15,
        'do_restart': True,
        'stitch': True,
    }
    restarted_config = config.ToraxConfig.from_dict(restart_config_dict)
    restarted_res = run_simulation.run_simulation(
        restarted_config, progress_bar=False
    )
    restarted_xr = restarted_res.simulation_output_to_xr()

    assert_allclose_fn = functools.partial(
        xr.testing.assert_allclose, atol=1e-6
    )
    xr.map_over_datasets(assert_allclose_fn, restarted_xr, continuous_xr)

  def test_restart_ip_boundary_condition_uses_file_ip(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {'model_name': 'circular', 'n_rho': 10}
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        'dt': 0.05,
        't_initial': 0.0,
        't_final': 0.2,
        'exact_t_final': True,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    continuous_res = run_simulation.run_simulation(
        torax_config, progress_bar=False
    )
    continuous_xr = continuous_res.simulation_output_to_xr()

    temp_dir = self.create_tempdir().full_path
    nc_path = os.path.join(temp_dir, 'continuous_ip.nc')
    continuous_xr.to_netcdf(nc_path)

    restart_config_dict = copy.deepcopy(config_dict)
    restart_config_dict['time_step_calculator']['t_initial'] = 0.1
    restart_config_dict['restart'] = {
        'filename': nc_path,
        'time': 0.1,
        'do_restart': True,
        'stitch': True,
    }
    # Modify profile_conditions.I_p in the new config.
    restart_config_dict['profile_conditions']['I_p'] = 5.0e6
    restarted_config = config.ToraxConfig.from_dict(restart_config_dict)
    assert restarted_config.restart is not None

    step_fn = run_simulation.make_step_fn(restarted_config)
    initial_sim_state, _ = (
        initial_state.get_initial_state_and_post_processed_outputs_from_file(
            file_restart=restarted_config.restart,
            step_fn=step_fn,
        )
    )

    t_slice = continuous_xr.sel(time=0.1, method='nearest')
    saved_profiles = t_slice.children[output_keys.PROFILES].dataset
    saved_scalars = t_slice.children[output_keys.SCALARS].dataset
    saved_ip = float(saved_profiles[output_keys.I_P_PROFILE].values[-1])

    # Reference initial state from restart keeping the original I_p:
    ref_config_dict = copy.deepcopy(config_dict)
    ref_config_dict['time_step_calculator']['t_initial'] = 0.1
    ref_config_dict['restart'] = {
        'filename': nc_path,
        'time': 0.1,
        'do_restart': True,
        'stitch': True,
    }
    ref_config = config.ToraxConfig.from_dict(ref_config_dict)
    assert ref_config.restart is not None
    ref_step_fn = run_simulation.make_step_fn(ref_config)
    ref_initial_state, _ = (
        initial_state.get_initial_state_and_post_processed_outputs_from_file(
            file_restart=ref_config.restart,
            step_fn=ref_step_fn,
        )
    )

    # Initial state psi gradient constraint should match the saved file I_p
    # rather than being perturbed by the new config's I_p (5.0e6).
    init_flux = initial_sim_state.core_profiles.poloidal_flux
    ref_flux = ref_initial_state.core_profiles.poloidal_flux
    assert init_flux.psi.right_face_grad_constraint is not None
    assert ref_flux.psi.right_face_grad_constraint is not None
    np.testing.assert_allclose(
        init_flux.psi.right_face_grad_constraint,
        ref_flux.psi.right_face_grad_constraint,
        atol=1e-6,
    )
    # Confirm the new config I_p differs significantly from saved I_p:
    expected_ratio = (
        restart_config_dict['profile_conditions']['I_p'] / saved_ip
    )
    self.assertFalse(
        np.isclose(
            float(init_flux.psi.right_face_grad_constraint),
            float(ref_flux.psi.right_face_grad_constraint) * expected_ratio,
        )
    )
    np.testing.assert_allclose(
        init_flux.j_toroidal_total,
        saved_profiles[output_keys.J_TOROIDAL_TOTAL].values[1:-1],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        init_flux.j_toroidal_total_face[-1],
        saved_profiles[output_keys.J_TOROIDAL_TOTAL].values[-1],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        initial_sim_state.core_profiles.v_loop_lcfs,
        float(saved_scalars[output_keys.V_LOOP_LCFS].values),
        atol=1e-6,
    )

    restarted_res = run_simulation.run_simulation(
        restarted_config, progress_bar=False
    )
    self.assertIsNotNone(restarted_res)

  def test_restart_unstitched_file_preserves_initial_edge_bcs(self):
    config_dict = self._get_config_dict(
        'test_iterhybrid_predictor_corrector_mavrin_n_e_ratios_lengyel.py'
    )
    config_dict['geometry']['n_rho'] = 10
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        'dt': 0.2,
        't_initial': 0.0,
        't_final': 0.4,
        'exact_t_final': True,
    }
    config_dict['edge']['update_T_e'] = True
    config_dict['edge']['update_T_i'] = True
    config_dict['edge']['use_enrichment_model'] = True
    config_dict['edge']['impurity_sot'] = 'core'
    config_dict['edge']['initial_guess'] = {
        'use_previous_step_as_guess': True,
    }
    torax_config_1 = config.ToraxConfig.from_dict(config_dict)
    res_1 = run_simulation.run_simulation(torax_config_1, progress_bar=False)
    xr_1 = res_1.simulation_output_to_xr()

    temp_dir = self.create_tempdir().full_path
    nc1_path = os.path.join(temp_dir, 'run1.nc')
    xr_1.to_netcdf(nc1_path)

    # Run 2: Unstitched restart producing File 2.
    cfg2_dict = copy.deepcopy(config_dict)
    cfg2_dict['time_step_calculator']['t_initial'] = 0.2
    cfg2_dict['time_step_calculator']['t_final'] = 0.4
    cfg2_dict['restart'] = {
        'filename': nc1_path,
        'time': 0.2,
        'do_restart': True,
        'stitch': False,
    }
    torax_config_2 = config.ToraxConfig.from_dict(cfg2_dict)
    res_2 = run_simulation.run_simulation(torax_config_2, progress_bar=False)
    xr_2 = res_2.simulation_output_to_xr()
    nc2_path = os.path.join(temp_dir, 'run2_unstitched.nc')
    xr_2.to_netcdf(nc2_path)

    # Run 3: Restart from time[0] of unstitched File 2.
    cfg3_dict = copy.deepcopy(config_dict)
    cfg3_dict['time_step_calculator']['t_initial'] = 0.2
    cfg3_dict['time_step_calculator']['t_final'] = 0.4
    cfg3_dict['restart'] = {
        'filename': nc2_path,
        'time': 0.2,
        'do_restart': True,
        'stitch': False,
    }
    torax_config_3 = config.ToraxConfig.from_dict(cfg3_dict)
    assert torax_config_3.restart is not None
    step_fn_3 = run_simulation.make_step_fn(torax_config_3)
    initial_sim_state_3, _ = (
        initial_state.get_initial_state_and_post_processed_outputs_from_file(
            file_restart=torax_config_3.restart,
            step_fn=step_fn_3,
        )
    )

    file2_t0_profiles = xr_2.children[output_keys.PROFILES].sel(
        time=0.2, method='nearest'
    ).dataset
    init_kinetic = initial_sim_state_3.core_profiles.kinetic
    assert init_kinetic.T_e.right_face_constraint is not None
    assert init_kinetic.T_i.right_face_constraint is not None
    np.testing.assert_allclose(
        init_kinetic.T_e.right_face_constraint,
        file2_t0_profiles[output_keys.T_E].values[-1],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        init_kinetic.T_i.right_face_constraint,
        file2_t0_profiles[output_keys.T_I].values[-1],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        initial_sim_state_3.core_profiles.ions.Z_eff,
        file2_t0_profiles[output_keys.Z_EFF].values[1:-1],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        initial_sim_state_3.core_profiles.ions.Z_eff_face[-1],
        file2_t0_profiles[output_keys.Z_EFF].values[-1],
        atol=1e-5,
    )

    res_3 = run_simulation.run_simulation(torax_config_3, progress_bar=False)
    xr_3 = res_3.simulation_output_to_xr()
    assert_allclose_fn = functools.partial(
        xr.testing.assert_allclose, atol=1e-6
    )
    xr.map_over_datasets(assert_allclose_fn, xr_3, xr_2)

  def test_runtime_params_dataclasses_are_frozen(self):
    torax_config = config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    with self.assertRaises(dataclasses.FrozenInstanceError):
      runtime_params.profile_conditions.I_p = 0.0  # pyrefly: ignore[read-only]
    plasma_comp = runtime_params.plasma_composition
    with self.assertRaises(dataclasses.FrozenInstanceError):
      plasma_comp.main_ion_names = ()  # pyrefly: ignore[read-only]


if __name__ == '__main__':
  absltest.main()
