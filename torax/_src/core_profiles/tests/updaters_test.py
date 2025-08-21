# In file: torax/_src/core_profiles/tests/updaters_test.py

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
from jax import numpy as jnp
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.core_profiles import updaters
from torax._src.fvm import cell_variable
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


# pylint: disable=invalid-name
class UpdatersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Build a default geo object for convenience.
    self.geo = geometry_pydantic_model.CircularConfig(n_rho=4).build_geometry()

    T_e = cell_variable.CellVariable(
        value=jnp.ones_like(self.geo.rho_norm),
        dr=self.geo.drho_norm,
        right_face_constraint=1.0,
        right_face_grad_constraint=None,
    )
    n_e = cell_variable.CellVariable(
        value=jnp.ones_like(self.geo.rho_norm),
        dr=self.geo.drho_norm,
        right_face_constraint=1.0,
        right_face_grad_constraint=None,
    )

    self.core_profiles_t = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        T_e=T_e,
        n_e=n_e,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='Set from n_e',
          n_e_right_bc=None,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=1.0e20,
      ),
      dict(
          testcase_name='Set and normalize from n_e',
          n_e_right_bc=None,
          normalize_n_e_to_nbar=True,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=0.8050314e20,
      ),
      dict(
          testcase_name='Set and normalize from n_e in fGW',
          n_e_right_bc=None,
          normalize_n_e_to_nbar=True,
          n_e_nbar_is_fGW=True,
          n_e_right_bc_is_fGW=True,
          expected_n_e_right_bc=0.9609356e20,
      ),
      dict(
          testcase_name='Set from n_e_right_bc',
          n_e_right_bc=0.5e20,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=0.5e20,
      ),
      dict(
          testcase_name='Set from n_e_right_bc absolute, ignore normalize',
          n_e_right_bc=0.5e20,
          normalize_n_e_to_nbar=True,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=0.5e20,
      ),
      dict(
          testcase_name='Set from n_e in fGW',
          n_e_right_bc=None,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=True,
          n_e_right_bc_is_fGW=True,
          expected_n_e_right_bc=1.19366207319e20,
      ),
      dict(
          testcase_name='Set from n_e, ignore n_e_right_bc_is_fGW',
          n_e_right_bc=None,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=True,
          expected_n_e_right_bc=1.0e20,
      ),
      dict(
          testcase_name='Set from n_e_right_bc, which is in fGW',
          n_e_right_bc=0.5,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=True,
          expected_n_e_right_bc=0.59683103659e20,
      ),
      dict(
          testcase_name='Set from n_e_right_bc, ignore n_e_nbar_is_fGW',
          n_e_right_bc=0.5e20,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=True,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=0.5e20,
      ),
      dict(
          testcase_name=(
              'Set from n_e_right_bc, ignore n_e_nbar_is_fGW, ignore normalize'
          ),
          n_e_right_bc=0.5e20,
          normalize_n_e_to_nbar=True,
          n_e_nbar_is_fGW=True,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=0.5e20,
      ),
  )
  def test_compute_boundary_conditions_n_e(
      self,
      n_e_right_bc,
      normalize_n_e_to_nbar,
      n_e_nbar_is_fGW,
      n_e_right_bc_is_fGW,
      expected_n_e_right_bc,
  ):
    """Tests that compute_boundary_conditions_for_t_plus_dt works for n_e."""
    config = default_configs.get_default_config_dict()

    if n_e_nbar_is_fGW:
      nbar = 1.0
      n_e = {0: {0: 1.5, 1: 1}}
    else:
      nbar = 1.0e20
      n_e = {0: {0: 1.5e20, 1: 1e20}}

    config['profile_conditions'] = {
        'n_e': n_e,
        'n_e_nbar_is_fGW': n_e_nbar_is_fGW,
        'n_e_right_bc_is_fGW': n_e_right_bc_is_fGW,
        'nbar': nbar,
        'normalize_n_e_to_nbar': normalize_n_e_to_nbar,
        'n_e_right_bc': n_e_right_bc,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice = provider(t=1.0)

    boundary_conditions = updaters.compute_boundary_conditions_for_t_plus_dt(
        dt=torax_config.numerics.fixed_dt,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
        geo_t_plus_dt=self.geo,
        core_profiles_t=self.core_profiles_t,
    )

    np.testing.assert_allclose(
        boundary_conditions['n_e']['right_face_constraint'],
        expected_n_e_right_bc,
        rtol=1e-6,
    )

  @parameterized.named_parameters(
      ('Set from T_e', None, 1.0), ('Set from T_e_right_bc', 0.5, 0.5)
  )
  def test_compute_boundary_conditions_T_e(
      self,
      T_e_right_bc,
      expected_T_e_right_bc,
  ):
    """Tests that compute_boundary_conditions_for_t_plus_dt works for T_e."""
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'T_e': {0: {0: 1.5, 1: 1}},
        'T_e_right_bc': T_e_right_bc,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice = provider(t=1.0)

    boundary_conditions = updaters.compute_boundary_conditions_for_t_plus_dt(
        dt=torax_config.numerics.fixed_dt,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
        geo_t_plus_dt=self.geo,
        core_profiles_t=self.core_profiles_t,
    )

    self.assertEqual(
        boundary_conditions['T_e']['right_face_constraint'],
        expected_T_e_right_bc,
    )

  @parameterized.named_parameters(
      ('Set from T_i', None, 1.0), ('Set from T_i_right_bc', 0.5, 0.5)
  )
  def test_compute_boundary_conditions_T_i(
      self,
      T_i_right_bc,
      expected_T_i_right_bc,
  ):
    """Tests that compute_boundary_conditions_for_t_plus_dt works for T_i."""
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'T_i': {0: {0: 1.5, 1: 1}},
        'T_i_right_bc': T_i_right_bc,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice = provider(t=1.0)

    boundary_conditions = updaters.compute_boundary_conditions_for_t_plus_dt(
        dt=torax_config.numerics.fixed_dt,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
        geo_t_plus_dt=self.geo,
        core_profiles_t=self.core_profiles_t,
    )

    self.assertEqual(
        boundary_conditions['T_i']['right_face_constraint'],
        expected_T_i_right_bc,
    )

  def test_update_v_loop_lcfs_from_psi(self):
    """Consistency check for _update_v_loop_lcfs_from_psi.

    Check the output inverts _calculate_psi_value_constraint_from_v_loop
    as expected.
    """

    dt = 1.0
    theta = 1.0
    v_loop_lcfs_t = 0.1
    v_loop_lcfs_t_plus_dt_expected = 0.2
    psi_lcfs_t = 0.5
    psi_lcfs_t_plus_dt = updaters._calculate_psi_value_constraint_from_v_loop(
        dt,
        theta,
        v_loop_lcfs_t,
        v_loop_lcfs_t_plus_dt_expected,
        psi_lcfs_t,
    )

    psi_t = cell_variable.CellVariable(
        value=np.ones_like(self.geo.rho) * 0.5,
        dr=self.geo.drho_norm,
        right_face_grad_constraint=0.0,
    )
    psi_t_plus_dt = cell_variable.CellVariable(
        value=np.ones_like(self.geo.rho) * psi_lcfs_t_plus_dt,
        dr=self.geo.drho_norm,
        right_face_grad_constraint=0.0,
    )

    v_loop_lcfs_t_plus_dt = updaters._update_v_loop_lcfs_from_psi(
        psi_t, psi_t_plus_dt, dt
    )

    np.testing.assert_allclose(
        v_loop_lcfs_t_plus_dt, v_loop_lcfs_t_plus_dt_expected
    )


if __name__ == '__main__':
  absltest.main()
