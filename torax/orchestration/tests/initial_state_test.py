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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np
from torax import constants
from torax.config import build_runtime_params
from torax.orchestration import initial_state
from torax.orchestration import step_function
from torax.output_tools import output
from torax.sources import source_models as source_models_lib
from torax.tests.test_lib import sim_test_case
from torax.torax_pydantic import model_config

# pylint: disable=invalid-name


class InitialStateTest(sim_test_case.SimTestCase):

  def test_from_file_restart(self):
    torax_config = self._get_torax_config('test_iterhybrid_rampup_restart.py')

    step_fn = _get_step_fn(torax_config)

    static, dynamic, geo = _get_geo_and_runtime_params_providers(torax_config)

    non_restart, _ = initial_state.get_initial_state_and_post_processed_outputs(
        t=torax_config.numerics.t_initial,
        static_runtime_params_slice=static,
        dynamic_runtime_params_slice_provider=dynamic,
        geometry_provider=geo,
        step_fn=step_fn,
    )

    result, post_processed = (
        initial_state.get_initial_state_and_post_processed_outputs_from_file(
            t_initial=torax_config.numerics.t_initial,
            file_restart=torax_config.restart,
            static_runtime_params_slice=static,
            dynamic_runtime_params_slice_provider=dynamic,
            geometry_provider=geo,
            step_fn=step_fn,
        )
    )

    self.assertNotEqual(post_processed.E_fusion, 0.0)
    self.assertNotEqual(post_processed.E_aux, 0.0)

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
        output.TEMPERATURE_ION,
        output.TEMPERATURE_ELECTRON,
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

    static, dynamic, geo = _get_geo_and_runtime_params_slice(torax_config)
    step_fn = _get_step_fn(torax_config)

    # Load in the reference core profiles.
    Ip_total = ref_profiles[output.IP_PROFILE][index, -1]
    # All profiles are on a grid with [left_face, cell_grid, right_face]
    T_e = ref_profiles[output.TEMPERATURE_ELECTRON][index, 1:-1]
    T_e_bc = ref_profiles[output.TEMPERATURE_ELECTRON][index, -1]
    T_i = ref_profiles[output.TEMPERATURE_ION][index, 1:-1]
    T_i_bc = ref_profiles[output.TEMPERATURE_ION][index, -1]
    n_e = ref_profiles[output.N_E][index, 1:-1]
    n_e_right_bc = ref_profiles[output.N_E][index, -1]
    psi = ref_profiles[output.PSI][index, 1:-1]

    # Override the dynamic runtime params with the loaded values.
    dynamic.profile_conditions.Ip = Ip_total
    dynamic.profile_conditions.T_e = T_e
    dynamic.profile_conditions.T_e_right_bc = T_e_bc
    dynamic.profile_conditions.T_i = T_i
    dynamic.profile_conditions.T_i_right_bc = T_i_bc
    dynamic.profile_conditions.n_e = n_e * constants.DENSITY_SCALING_FACTOR
    dynamic.profile_conditions.n_e_right_bc = (
        n_e_right_bc * constants.DENSITY_SCALING_FACTOR
    )
    dynamic.profile_conditions.psi = psi
    # When loading from file we want ne not to have transformations.
    # Both ne and the boundary condition are given in absolute values (not fGW).
    # Additionally we want to avoid normalizing to nbar.
    dynamic.profile_conditions.n_e_right_bc_is_fGW = False
    dynamic.profile_conditions.n_e_nbar_is_fGW = False
    static = dataclasses.replace(
        static,
        profile_conditions=dataclasses.replace(
            static.profile_conditions,
            n_e_right_bc_is_absolute=True,
            normalize_n_e_to_nbar=False,
        ),
    )

    result = initial_state._get_initial_state(static, dynamic, geo, step_fn)
    _verify_core_profiles(ref_profiles, index, result.core_profiles)


def _get_step_fn(torax_config):
  solver = mock.MagicMock()
  solver.source_models = source_models_lib.SourceModels(torax_config.sources)
  return mock.create_autospec(step_function.SimulationStepFn, solver=solver)


def _get_geo_and_runtime_params_providers(torax_config):
  static_runtime_params_slice = (
      build_runtime_params.build_static_params_from_config(torax_config)
  )
  dynamic_runtime_params_slice_provider = (
      build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
          torax_config
      )
  )
  return (
      static_runtime_params_slice,
      dynamic_runtime_params_slice_provider,
      torax_config.geometry.build_provider,
  )


def _get_geo_and_runtime_params_slice(torax_config):
  static, dynamic_provider, geo_provider = (
      _get_geo_and_runtime_params_providers(torax_config)
  )
  dynamic_runtime_params_slice_for_init, geo_for_init = (
      build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
          t=torax_config.numerics.t_initial,
          dynamic_runtime_params_slice_provider=dynamic_provider,
          geometry_provider=geo_provider,
      )
  )
  return (
      static,
      dynamic_runtime_params_slice_for_init,
      geo_for_init,
  )


def _verify_core_profiles(ref_profiles, index, core_profiles):
  """Verify core profiles matches a reference at given index."""
  np.testing.assert_allclose(
      core_profiles.T_e.value,
      ref_profiles[output.TEMPERATURE_ELECTRON][index, 1:-1],
  )
  np.testing.assert_allclose(
      core_profiles.T_i.value,
      ref_profiles[output.TEMPERATURE_ION][index, 1:-1],
  )
  np.testing.assert_allclose(
      core_profiles.n_e.value, ref_profiles[output.N_E][index, 1:-1]
  )
  np.testing.assert_allclose(
      core_profiles.n_e.right_face_constraint,
      ref_profiles[output.N_E][index, -1],
  )
  np.testing.assert_allclose(
      core_profiles.psi.value, ref_profiles[output.PSI][index, 1:-1]
  )
  np.testing.assert_allclose(
      core_profiles.psidot.value, ref_profiles[output.V_LOOP][index, 1:-1]
  )
  np.testing.assert_allclose(
      core_profiles.n_i.value, ref_profiles[output.N_I][index, 1:-1]
  )
  np.testing.assert_allclose(
      core_profiles.n_i.right_face_constraint,
      ref_profiles[output.N_I][index, -1],
  )

  np.testing.assert_allclose(
      core_profiles.q_face, ref_profiles[output.Q][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.s_face, ref_profiles[output.MAGNETIC_SHEAR][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_total, ref_profiles[output.J_TOTAL][index, 1:-1]
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_total_face[0],
      ref_profiles[output.J_TOTAL][index, 0],
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_total_face[-1],
      ref_profiles[output.J_TOTAL][index, -1],
  )
  np.testing.assert_allclose(
      core_profiles.currents.Ip_profile_face,
      ref_profiles[output.IP_PROFILE][index, :],
  )


if __name__ == '__main__':
  absltest.main()
