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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np
from torax import output
from torax.config import build_runtime_params
from torax.orchestration import initial_state
from torax.orchestration import step_function
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

    self.assertNotEqual(post_processed.E_cumulative_fusion, 0.0)
    self.assertNotEqual(post_processed.E_cumulative_external, 0.0)

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
    index = -1
    ref_profiles, ref_time = self._get_refs(test_config + '.nc', profiles)
    t = int(ref_time[index])

    config = self._get_config_dict(test_config + '.py')
    config['runtime_params']['numerics']['t_initial'] = t
    torax_config = model_config.ToraxConfig.from_dict(config)

    static, dynamic, geo = _get_geo_and_runtime_params_slice(torax_config)
    step_fn = _get_step_fn(torax_config)

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
    dynamic.profile_conditions.Ip_tot = Ip_total
    dynamic.profile_conditions.Te = temp_el
    dynamic.profile_conditions.Te_bound_right = temp_el_bc
    dynamic.profile_conditions.Ti = temp_ion
    dynamic.profile_conditions.Ti_bound_right = temp_ion_bc
    dynamic.profile_conditions.ne = ne
    dynamic.profile_conditions.ne_bound_right = (
        ne_bound_right
    )
    dynamic.profile_conditions.psi = psi
    # When loading from file we want ne not to have transformations.
    # Both ne and the boundary condition are given in absolute values (not fGW).
    dynamic.profile_conditions.ne_bound_right_is_fGW = (
        False
    )
    dynamic.profile_conditions.ne_is_fGW = False
    dynamic.profile_conditions.ne_bound_right_is_absolute = (
        True
    )
    # Additionally we want to avoid normalizing to nbar.
    dynamic.profile_conditions.normalize_to_nbar = False

    result = initial_state._get_initial_state(
        static, dynamic, geo, step_fn
    )
    _verify_core_profiles(ref_profiles, index, result.core_profiles)


def _get_step_fn(torax_config):
  stepper = mock.MagicMock()
  stepper.source_models = source_models_lib.SourceModels(
      torax_config.sources.source_model_config
  )
  return mock.create_autospec(step_function.SimulationStepFn, stepper=stepper)


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
