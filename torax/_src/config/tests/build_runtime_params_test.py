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
import jax
import numpy as np
from torax._src import jax_utils
from torax._src.config import build_runtime_params
from torax._src.config import config_loader
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.pedestal_model import set_tped_nped
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic


class RuntimeParamsSliceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._torax_mesh = torax_pydantic.Grid1D(
        nx=4,
    )

  def test_time_dependent_provider_is_time_dependent(self):
    """Tests that the runtime_params provider is time dependent."""
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {'T_i_right_bc': {0.0: 2.0, 4.0: 4.0}}
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=1.0)
    np.testing.assert_allclose(
        runtime_params.profile_conditions.T_i_right_bc, 2.5
    )
    runtime_params = provider(t=2.0)
    np.testing.assert_allclose(
        runtime_params.profile_conditions.T_i_right_bc, 3.0
    )

  def test_boundary_conditions_are_time_dependent(self):
    """Tests that the boundary conditions are time dependent params."""
    # All of the following parameters are time-dependent fields, but they can
    # be initialized in different ways.
    profile_conditions = profile_conditions_lib.ProfileConditions(
        T_i_right_bc={0.0: 2.0, 4.0: 4.0},
        T_e_right_bc=4.5,  # not time-dependent.
        n_e_right_bc=({5.0: 6.0e20, 7.0: 8.0e20}, 'step'),
    )
    torax_pydantic.set_grid(profile_conditions, self._torax_mesh)
    np.testing.assert_allclose(
        profile_conditions.build_runtime_params(
            t=2.0,
        ).T_i_right_bc,
        3.0,
    )
    np.testing.assert_allclose(
        profile_conditions.build_runtime_params(
            t=4.0,
        ).T_e_right_bc,
        4.5,
    )
    np.testing.assert_allclose(
        profile_conditions.build_runtime_params(
            t=6.0,
        ).n_e_right_bc,
        6.0e20,
    )

  def test_pedestal_is_time_dependent(self):
    """Tests that the pedestal runtime params are time dependent."""
    pedestal = pedestal_pydantic_model.SetTpedNped.from_dict(
        dict(
            model_name='set_T_ped_n_ped',
            T_i_ped={0.0: 0.0, 1.0: 1.0},
            T_e_ped={0.0: 1.0, 1.0: 2.0},
            n_e_ped={0.0: 2.0e20, 1.0: 3.0e20},
            rho_norm_ped_top={0.0: 3.0, 1.0: 5.0},
            set_pedestal={0.0: True, 1.0: False},
        )
    )
    # Check at time 0.

    pedestal_params = pedestal.build_runtime_params(t=0.0)
    assert isinstance(pedestal_params, set_tped_nped.RuntimeParams)
    np.testing.assert_allclose(pedestal_params.set_pedestal, True)
    np.testing.assert_allclose(pedestal_params.T_i_ped, 0.0)
    np.testing.assert_allclose(pedestal_params.T_e_ped, 1.0)
    np.testing.assert_allclose(pedestal_params.n_e_ped, 2.0e20)
    np.testing.assert_allclose(pedestal_params.rho_norm_ped_top, 3.0)
    # And check after the time limit.
    pedestal_params = pedestal.build_runtime_params(t=1.0)
    assert isinstance(pedestal_params, set_tped_nped.RuntimeParams)
    np.testing.assert_allclose(pedestal_params.set_pedestal, False)
    np.testing.assert_allclose(pedestal_params.T_i_ped, 1.0)
    np.testing.assert_allclose(pedestal_params.T_e_ped, 2.0)
    np.testing.assert_allclose(pedestal_params.n_e_ped, 3.0e20)
    np.testing.assert_allclose(pedestal_params.rho_norm_ped_top, 5.0)

  @parameterized.parameters(
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          None,
          np.array([1.125, 1.375, 1.625, 1.875]),
          2.0,
          'T_i',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          3.0,
          np.array([1.125, 1.375, 1.625, 1.875]),
          3.0,
          'T_i',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          None,
          np.array([1.125, 1.375, 1.625, 1.875]),
          2.0,
          'T_e',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          3.0,
          np.array([1.125, 1.375, 1.625, 1.875]),
          3.0,
          'T_e',
      ),
      (
          {0: {0.0: 1.0e20, 1.0: 2.0e20}},
          None,
          np.array([1.125e20, 1.375e20, 1.625e20, 1.875e20]),
          2.0e20,
          'n_e',
      ),
      (
          {0: {0.0: 1.0e20, 1.0: 2.0e20}},
          3.0e20,
          np.array([1.125e20, 1.375e20, 1.625e20, 1.875e20]),
          3.0e20,
          'n_e',
      ),
  )
  def test_profile_conditions_set_electron_temperature_and_boundary_condition(
      self,
      var,
      var_boundary_condition,
      expected_var,
      expected_var_boundary_condition,
      var_name,
  ):
    """Tests that the profile conditions can set the electron temperature."""
    boundary_var_name = var_name + '_right_bc'

    temperatures = {
        var_name: var,
        boundary_var_name: var_boundary_condition,
    }
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict(
        temperatures
    )
    geo = geometry_pydantic_model.CircularConfig(n_rho=4).build_geometry()
    torax_pydantic.set_grid(profile_conditions, geo.torax_mesh)
    profile_condition_params = profile_conditions.build_runtime_params(t=0.0)

    np.testing.assert_allclose(
        getattr(profile_condition_params, var_name), expected_var
    )
    self.assertEqual(
        getattr(profile_condition_params, boundary_var_name),
        expected_var_boundary_condition,
    )

  @parameterized.product(
      n_e_right_bc=[
          None,
          1.0,
      ],
      n_e_right_bc_is_fGW=[
          True,
          False,
      ],
      n_e_nbar_is_fGW=[
          True,
          False,
      ],
  )
  def test_profile_conditions_set_electron_density_and_boundary_condition(
      self,
      n_e_right_bc,
      n_e_right_bc_is_fGW,  # pylint: disable=invalid-name
      n_e_nbar_is_fGW,  # pylint: disable=invalid-name
  ):
    """Tests that the profile conditions can set the electron density."""

    config = default_configs.get_default_config_dict()

    # Set correct order of magnitudes to pass Pydantic validation.
    if n_e_right_bc is not None:
      n_e_right_bc = 1.0 if n_e_right_bc_is_fGW else 1.0e20
    nbar = 1.0 if n_e_nbar_is_fGW else 1.0e20
    n_e = (
        {0.0: {0.0: 1.5, 1.0: 1.0}}
        if n_e_nbar_is_fGW
        else {0.0: {0.0: 1.5e20, 1.0: 1.0e20}}
    )

    config['profile_conditions'] = {
        'n_e_right_bc': n_e_right_bc,
        'n_e_right_bc_is_fGW': n_e_right_bc_is_fGW,
        'n_e_nbar_is_fGW': n_e_nbar_is_fGW,
        'nbar': nbar,
        'n_e': n_e,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    profile_condition_params = (
        build_runtime_params.RuntimeParamsProvider.from_config(torax_config)(
            t=0.0
        ).profile_conditions
    )

    if n_e_right_bc is None:
      # If the boundary condition was not set, it should inherit the fGW flag.
      self.assertEqual(
          profile_condition_params.n_e_right_bc_is_fGW,
          n_e_nbar_is_fGW,
      )
      # If the boundary condition was set check it is not absolute.
      self.assertFalse(profile_condition_params.n_e_right_bc_is_absolute)
    else:
      self.assertEqual(
          profile_condition_params.n_e_right_bc_is_fGW,
          n_e_right_bc_is_fGW,
      )
      self.assertTrue(profile_condition_params.n_e_right_bc_is_absolute)

  def test_runtime_params_provider_works_under_jit(self):
    torax_config = config_loader.build_torax_config_from_file(
        'tests/test_data/test_iterhybrid_rampup.py'
    )
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )

    @jax.jit
    def f(
        provider: build_runtime_params.RuntimeParamsProvider,
        t: float,
    ):
      return provider(t)

    with self.subTest('jit_compiles_and_returns_expected_value'):
      runtime_params = f(provider, t=0.1)
      # Check to make sure it's a valid object.
      self.assertIsInstance(
          runtime_params,
          runtime_params_slice.RuntimeParams,
      )
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    with self.subTest('jit_updates_value_without_recompile'):
      torax_config.update_fields({'profile_conditions.T_i_right_bc': 0.77})
      provider = build_runtime_params.RuntimeParamsProvider.from_config(
          torax_config
      )
      runtime_params = f(provider, t=0.1)
      self.assertEqual(runtime_params.profile_conditions.T_i_right_bc, 0.77)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)


if __name__ == '__main__':
  absltest.main()
