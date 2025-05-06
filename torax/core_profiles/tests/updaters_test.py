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
import numpy as np
from torax import jax_utils
from torax.config import build_runtime_params
from torax.core_profiles import updaters
from torax.fvm import cell_variable
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.tests.test_lib import default_configs
from torax.torax_pydantic import model_config


# pylint: disable=invalid-name
class UpdatersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)

  @parameterized.named_parameters(
      dict(
          testcase_name='Set from n_e',
          n_e_bound_right=None,
          normalize_to_nbar=False,
          n_e_is_fGW=False,
          n_e_bound_right_is_fGW=False,
          expected_n_e_bound_right=1.0,
      ),
      dict(
          testcase_name='Set and normalize from n_e',
          n_e_bound_right=None,
          normalize_to_nbar=True,
          n_e_is_fGW=False,
          n_e_bound_right_is_fGW=False,
          expected_n_e_bound_right=0.8050314,
      ),
      dict(
          testcase_name='Set and normalize from n_e in fGW',
          n_e_bound_right=None,
          normalize_to_nbar=True,
          n_e_is_fGW=True,
          n_e_bound_right_is_fGW=False,
          expected_n_e_bound_right=0.8050314,
      ),
      dict(
          testcase_name='Set from n_e_bound_right',
          n_e_bound_right=0.5,
          normalize_to_nbar=False,
          n_e_is_fGW=False,
          n_e_bound_right_is_fGW=False,
          expected_n_e_bound_right=0.5,
      ),
      dict(
          testcase_name='Set from n_e_bound_right absolute, ignore normalize',
          n_e_bound_right=0.5,
          normalize_to_nbar=True,
          n_e_is_fGW=False,
          n_e_bound_right_is_fGW=False,
          expected_n_e_bound_right=0.5,
      ),
      dict(
          testcase_name='Set from n_e in fGW',
          n_e_bound_right=None,
          normalize_to_nbar=False,
          n_e_is_fGW=True,
          n_e_bound_right_is_fGW=False,
          expected_n_e_bound_right=1,  # This will be scaled by fGW in test.
      ),
      dict(
          testcase_name='Set from n_e, ignore n_e_bound_right_is_fGW',
          n_e_bound_right=None,
          normalize_to_nbar=False,
          n_e_is_fGW=False,
          n_e_bound_right_is_fGW=True,
          expected_n_e_bound_right=1.0,
      ),
      dict(
          testcase_name='Set from n_e_bound_right, ignore n_e_is_fGW',
          n_e_bound_right=0.5,
          normalize_to_nbar=False,
          n_e_is_fGW=True,
          n_e_bound_right_is_fGW=False,
          expected_n_e_bound_right=0.5,
      ),
      dict(
          testcase_name=(
              'Set from n_e_bound_right, ignore n_e_is_fGW, ignore normalize'
          ),
          n_e_bound_right=0.5,
          normalize_to_nbar=True,
          n_e_is_fGW=True,
          n_e_bound_right_is_fGW=False,
          expected_n_e_bound_right=0.5,
      ),
  )
  def test_compute_boundary_conditions_n_e(
      self,
      n_e_bound_right,
      normalize_to_nbar,
      n_e_is_fGW,
      n_e_bound_right_is_fGW,
      expected_n_e_bound_right,
  ):
    """Tests that compute_boundary_conditions_t_plus_dt works."""
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'n_e': {0: {0: 1.5, 1: 1}},
        'n_e_is_fGW': n_e_is_fGW,
        'n_e_bound_right_is_fGW': n_e_bound_right_is_fGW,
        'nbar': 1,
        'normalize_to_nbar': normalize_to_nbar,
        'n_e_bound_right': n_e_bound_right,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice = provider(t=1.0)
    geo = torax_config.geometry.build_provider(t=1.0)

    boundary_conditions = updaters.compute_boundary_conditions_for_t_plus_dt(
        dt=torax_config.numerics.fixed_dt,
        static_runtime_params_slice=static_slice,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
        geo_t_plus_dt=geo,
        core_profiles_t=mock.ANY,  # Unused
    )

    if (n_e_is_fGW and n_e_bound_right is None) or (
        n_e_bound_right_is_fGW and n_e_bound_right is not None
    ):
      # Then we expect the boundary condition to be in fGW.
      nGW = (
          dynamic_runtime_params_slice.profile_conditions.I_total
          / (np.pi * geo.a_minor**2)
          * 1e20
          / dynamic_runtime_params_slice.numerics.nref
      )
      np.testing.assert_allclose(
          boundary_conditions['n_e']['right_face_constraint'],
          expected_n_e_bound_right * nGW,
      )
    else:
      np.testing.assert_allclose(
          boundary_conditions['n_e']['right_face_constraint'],
          expected_n_e_bound_right,
      )

  @parameterized.named_parameters(
      ('Set from Te', None, 1.0), ('Set from T_e_right_bc', 0.5, 0.5)
  )
  def test_compute_boundary_conditions_Te(
      self,
      T_e_right_bc,
      expected_T_e_right_bc,
  ):
    """Tests that compute_boundary_conditions_for_t_plus_dt works for Te."""
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'Te': {0: {0: 1.5, 1: 1}},
        'T_e_right_bc': T_e_right_bc,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice = provider(t=1.0)
    geo = torax_config.geometry.build_provider(t=1.0)

    boundary_conditions = updaters.compute_boundary_conditions_for_t_plus_dt(
        dt=torax_config.numerics.fixed_dt,
        static_runtime_params_slice=static_slice,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
        geo_t_plus_dt=geo,
        core_profiles_t=mock.ANY,  # Unused
    )

    self.assertEqual(
        boundary_conditions['temp_el']['right_face_constraint'],
        expected_T_e_right_bc,
    )

  @parameterized.named_parameters(
      ('Set from Ti', None, 1.0), ('Set from T_i_right_bc', 0.5, 0.5)
  )
  def test_compute_boundary_conditions_Ti(
      self,
      T_i_right_bc,
      expected_T_i_right_bc,
  ):
    """Tests that compute_boundary_conditions_for_t_plus_dt works for Ti."""
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'Ti': {0: {0: 1.5, 1: 1}},
        'T_i_right_bc': T_i_right_bc,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice = provider(t=1.0)
    geo = torax_config.geometry.build_provider(t=1.0)

    boundary_conditions = updaters.compute_boundary_conditions_for_t_plus_dt(
        dt=torax_config.numerics.fixed_dt,
        static_runtime_params_slice=static_slice,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
        geo_t_plus_dt=geo,
        core_profiles_t=mock.ANY,  # Unused
    )

    self.assertEqual(
        boundary_conditions['temp_ion']['right_face_constraint'],
        expected_T_i_right_bc,
    )

  def test_update_vloop_lcfs_from_psi(self):
    """Consistency check for _update_vloop_lcfs_from_psi.

    Check the the output inverts _calculate_psi_value_constraint_from_vloop
    as expected.
    """

    dt = 1.0
    theta = 1.0
    vloop_lcfs_t = 0.1
    vloop_lcfs_t_plus_dt_expected = 0.2
    psi_lcfs_t = 0.5
    psi_lcfs_t_plus_dt = updaters._calculate_psi_value_constraint_from_vloop(
        dt,
        theta,
        vloop_lcfs_t,
        vloop_lcfs_t_plus_dt_expected,
        psi_lcfs_t,
    )
    geo = geometry_pydantic_model.CircularConfig(n_rho=4).build_geometry()

    psi_t = cell_variable.CellVariable(
        value=np.ones_like(geo.rho) * 0.5,
        dr=geo.drho_norm,
        right_face_grad_constraint=0.0,
    )
    psi_t_plus_dt = cell_variable.CellVariable(
        value=np.ones_like(geo.rho) * psi_lcfs_t_plus_dt,
        dr=geo.drho_norm,
        right_face_grad_constraint=0.0,
    )

    vloop_lcfs_t_plus_dt = updaters._update_vloop_lcfs_from_psi(
        psi_t, psi_t_plus_dt, dt
    )

    np.testing.assert_allclose(
        vloop_lcfs_t_plus_dt, vloop_lcfs_t_plus_dt_expected
    )


if __name__ == '__main__':
  absltest.main()
