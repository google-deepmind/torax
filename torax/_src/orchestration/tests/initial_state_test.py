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
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import chex
from torax._src.config import build_runtime_params
from torax._src.core_profiles import runtime_params as core_profile_runtime_params
from torax._src.orchestration import initial_state
from torax._src.orchestration import run_simulation
from torax._src.output_tools import output_keys
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import config

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

    result, post_processed = (
        initial_state.get_initial_state_and_post_processed_outputs_from_file(
            file_restart=torax_config.restart,  # pyrefly: ignore[bad-argument-type]
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
            geometry_provider=step_fn.geometry_provider,
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

    result = initial_state._get_initial_state(runtime_params, geo, step_fn)
    core_profile_helpers.verify_core_profiles(
        ref_profiles, index, result.core_profiles
    )

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
