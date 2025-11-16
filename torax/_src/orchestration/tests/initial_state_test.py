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
from absl.testing import absltest
from absl.testing import parameterized
import chex
from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state
from torax._src.orchestration import step_function
from torax._src.output_tools import output
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class InitialStateTest(sim_test_case.SimTestCase):

  def test_from_file_restart(self):
    torax_config = self._get_torax_config('test_iterhybrid_rampup_restart.py')
    runtime_params_provider, geo_provider, step_fn = _get_providers_and_step_fn(
        torax_config
    )
    non_restart, _ = initial_state.get_initial_state_and_post_processed_outputs(
        t=torax_config.numerics.t_initial,
        runtime_params_provider=runtime_params_provider,
        geometry_provider=geo_provider,
        step_fn=step_fn,
    )

    result, post_processed = (
        initial_state.get_initial_state_and_post_processed_outputs_from_file(
            t_initial=torax_config.numerics.t_initial,
            file_restart=torax_config.restart,
            runtime_params_provider=runtime_params_provider,
            geometry_provider=geo_provider,
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

  @parameterized.parameters(
      'test_psi_heat_dens',
      'test_psichease_prescribed_jtot',
      'test_psichease_prescribed_johm',
      'test_iterhybrid_rampup',
  )
  def test_core_profile_final_step(self, test_config):
    profiles = [
        output.T_I,
        output.T_E,
        output.N_E,
        output.N_I,
        output.PSI,
        output.V_LOOP,
        output.IP_PROFILE,
        output.Q,
        output.MAGNETIC_SHEAR,
        output.J_BOOTSTRAP,
        output.J_OHMIC,
        output.J_EXTERNAL,
        output.J_TOTAL,
        output.SIGMA_PARALLEL,
    ]
    index = -1
    ref_profiles, ref_time = self._get_refs(test_config + '.nc', profiles)
    t = int(ref_time[index])

    config = self._get_config_dict(test_config + '.py')
    config['numerics']['t_initial'] = t
    torax_config = model_config.ToraxConfig.from_dict(config)

    runtime_params_provider, geo_provider, step_fn = _get_providers_and_step_fn(
        torax_config
    )
    runtime_params, geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=runtime_params_provider,
            geometry_provider=geo_provider,
        )
    )

    # Load in the reference core profiles.
    Ip_total = ref_profiles[output.IP_PROFILE][index, -1]
    # All profiles are on a grid with [left_face, cell_grid, right_face]
    T_e = ref_profiles[output.T_E][index, 1:-1]
    T_e_bc = ref_profiles[output.T_E][index, -1]
    T_i = ref_profiles[output.T_I][index, 1:-1]
    T_i_bc = ref_profiles[output.T_I][index, -1]
    n_e = ref_profiles[output.N_E][index, 1:-1]
    n_e_right_bc = ref_profiles[output.N_E][index, -1]
    psi = ref_profiles[output.PSI][index, 1:-1]

    # Override the runtime params with the loaded values.
    runtime_params.profile_conditions.Ip = Ip_total
    runtime_params.profile_conditions.T_e = T_e
    runtime_params.profile_conditions.T_e_right_bc = T_e_bc
    runtime_params.profile_conditions.T_i = T_i
    runtime_params.profile_conditions.T_i_right_bc = T_i_bc
    runtime_params.profile_conditions.n_e = n_e
    runtime_params.profile_conditions.n_e_right_bc = n_e_right_bc
    runtime_params.profile_conditions.psi = psi
    # When loading from file we want ne not to have transformations.
    # Both ne and the boundary condition are given in absolute values (not fGW).
    # Additionally we want to avoid normalizing to nbar.
    runtime_params.profile_conditions.n_e_right_bc_is_fGW = False
    runtime_params.profile_conditions.n_e_nbar_is_fGW = False
    runtime_params.profile_conditions.normalize_n_e_to_nbar = False
    runtime_params.profile_conditions.n_e_right_bc_is_absolute = True

    result = initial_state._get_initial_state(runtime_params, geo, step_fn)
    core_profile_helpers.verify_core_profiles(
        ref_profiles, index, result.core_profiles
    )


def _get_providers_and_step_fn(torax_config):
  runtime_params_provider = (
      build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
  )
  geo_provider = torax_config.geometry.build_provider
  solver = torax_config.solver.build_solver(
      physics_models=torax_config.build_physics_models(),
  )
  step_fn = step_function.SimulationStepFn(
      solver=solver,
      time_step_calculator=torax_config.time_step_calculator.time_step_calculator,
      geometry_provider=geo_provider,
      runtime_params_provider=runtime_params_provider,
  )
  return runtime_params_provider, geo_provider, step_fn


if __name__ == '__main__':
  absltest.main()
