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
import numpy as np
from torax.config import build_runtime_params
from torax.config import profile_conditions as profile_conditions_lib
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.pedestal_model import set_tped_nped
from torax.sources import generic_current_source
from torax.sources import pydantic_model as sources_pydantic_model
from torax.tests.test_lib import default_configs
from torax.torax_pydantic import model_config
from torax.torax_pydantic import torax_pydantic


class RuntimeParamsSliceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._torax_mesh = torax_pydantic.Grid1D(nx=4, dx=0.25)

  def test_time_dependent_provider_is_time_dependent(self):
    """Tests that the runtime_params slice provider is time dependent."""
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {'Ti_bound_right': {0.0: 2.0, 4.0: 4.0}}
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice = provider(t=1.0)
    np.testing.assert_allclose(
        dynamic_runtime_params_slice.profile_conditions.Ti_bound_right, 2.5
    )
    dynamic_runtime_params_slice = provider(t=2.0)
    np.testing.assert_allclose(
        dynamic_runtime_params_slice.profile_conditions.Ti_bound_right, 3.0
    )

  def test_boundary_conditions_are_time_dependent(self):
    """Tests that the boundary conditions are time dependent params."""
    # All of the following parameters are time-dependent fields, but they can
    # be initialized in different ways.
    profile_conditions = profile_conditions_lib.ProfileConditions(
        Ti_bound_right={0.0: 2.0, 4.0: 4.0},
        Te_bound_right=4.5,  # not time-dependent.
        ne_bound_right=({5.0: 6.0, 7.0: 8.0}, 'step'),
    )
    torax_pydantic.set_grid(profile_conditions, self._torax_mesh)
    np.testing.assert_allclose(
        profile_conditions.build_dynamic_params(
            t=2.0,
        ).Ti_bound_right,
        3.0,
    )
    np.testing.assert_allclose(
        profile_conditions.build_dynamic_params(
            t=4.0,
        ).Te_bound_right,
        4.5,
    )
    np.testing.assert_allclose(
        profile_conditions.build_dynamic_params(
            t=6.0,
        ).ne_bound_right,
        6.0,
    )

  def test_pedestal_is_time_dependent(self):
    """Tests that the pedestal runtime params are time dependent."""
    pedestal = pedestal_pydantic_model.SetTpedNped.from_dict(
        dict(
            pedestal_model='set_tped_nped',
            Tiped={0.0: 0.0, 1.0: 1.0},
            Teped={0.0: 1.0, 1.0: 2.0},
            neped={0.0: 2.0, 1.0: 3.0},
            rho_norm_ped_top={0.0: 3.0, 1.0: 5.0},
            set_pedestal={0.0: True, 1.0: False},
        )
    )
    # Check at time 0.

    pedestal_params = pedestal.build_dynamic_params(t=0.0)
    assert isinstance(pedestal_params, set_tped_nped.DynamicRuntimeParams)
    np.testing.assert_allclose(pedestal_params.set_pedestal, True)
    np.testing.assert_allclose(pedestal_params.Tiped, 0.0)
    np.testing.assert_allclose(pedestal_params.Teped, 1.0)
    np.testing.assert_allclose(pedestal_params.neped, 2.0)
    np.testing.assert_allclose(pedestal_params.rho_norm_ped_top, 3.0)
    # And check after the time limit.
    pedestal_params = pedestal.build_dynamic_params(t=1.0)
    assert isinstance(pedestal_params, set_tped_nped.DynamicRuntimeParams)
    np.testing.assert_allclose(pedestal_params.set_pedestal, False)
    np.testing.assert_allclose(pedestal_params.Tiped, 1.0)
    np.testing.assert_allclose(pedestal_params.Teped, 2.0)
    np.testing.assert_allclose(pedestal_params.neped, 3.0)
    np.testing.assert_allclose(pedestal_params.rho_norm_ped_top, 5.0)

  def test_wext_in_dynamic_runtime_params_cannot_be_negative(self):
    sources = sources_pydantic_model.Sources.from_dict({
        generic_current_source.GenericCurrentSource.SOURCE_NAME: {
            'wext': {0.0: 1.0, 1.0: -1.0},
        },
    })
    torax_pydantic.set_grid(sources, self._torax_mesh)
    # While wext is positive, this should be fine.
    generic_current = sources.generic_current.build_dynamic_params(
        t=0.0,
    )
    np.testing.assert_allclose(generic_current.wext, 1.0)

    # Even 0 should be fine.
    generic_current = sources.generic_current.build_dynamic_params(
        t=0.5,
    )
    np.testing.assert_allclose(generic_current.wext, 0.0)
    # But negative values will cause an error.
    with self.assertRaises(RuntimeError):
      sources.generic_current.build_dynamic_params(
          t=1.0,
      )

  @parameterized.parameters(
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          None,
          np.array([1.125, 1.375, 1.625, 1.875]),
          2.0,
          'Ti',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          3.0,
          np.array([1.125, 1.375, 1.625, 1.875]),
          3.0,
          'Ti',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          None,
          np.array([1.125, 1.375, 1.625, 1.875]),
          2.0,
          'Te',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          3.0,
          np.array([1.125, 1.375, 1.625, 1.875]),
          3.0,
          'Te',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          None,
          np.array([1.125, 1.375, 1.625, 1.875]),
          2.0,
          'ne',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          3.0,
          np.array([1.125, 1.375, 1.625, 1.875]),
          3.0,
          'ne',
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

    boundary_var_name = var_name + '_bound_right'
    temperatures = {
        var_name: var,
        boundary_var_name: var_boundary_condition,
    }
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict(
        temperatures
    )
    geo = geometry_pydantic_model.CircularConfig(n_rho=4).build_geometry()
    torax_pydantic.set_grid(profile_conditions, geo.torax_mesh)
    dynamic_profile_conditions = profile_conditions.build_dynamic_params(
        t=0.0,
    )
    np.testing.assert_allclose(
        getattr(dynamic_profile_conditions, var_name), expected_var
    )
    self.assertEqual(
        getattr(dynamic_profile_conditions, boundary_var_name),
        expected_var_boundary_condition,
    )

  @parameterized.product(
      ne_bound_right=[
          None,
          1.0,
      ],
      ne_bound_right_is_fGW=[
          True,
          False,
      ],
      ne_is_fGW=[
          True,
          False,
      ],
  )
  def test_profile_conditions_set_electron_density_and_boundary_condition(
      self,
      ne_bound_right,
      ne_bound_right_is_fGW,  # pylint: disable=invalid-name
      ne_is_fGW,  # pylint: disable=invalid-name
  ):
    """Tests that the profile conditions can set the electron temperature."""
    profile_conditions = profile_conditions_lib.ProfileConditions(
        ne_bound_right=ne_bound_right,
        ne_bound_right_is_fGW=ne_bound_right_is_fGW,
        ne_is_fGW=ne_is_fGW,
    )
    geo = geometry_pydantic_model.CircularConfig(n_rho=4).build_geometry()
    torax_pydantic.set_grid(profile_conditions, geo.torax_mesh)
    dynamic_profile_conditions = profile_conditions.build_dynamic_params(
        t=0.0,
    )

    if ne_bound_right is None:
      # If the boundary condition was not set, it should inherit the fGW flag.
      self.assertEqual(
          dynamic_profile_conditions.ne_bound_right_is_fGW,
          ne_is_fGW,
      )
      # If the boundary condition was set check it is not absolute.
      self.assertFalse(dynamic_profile_conditions.ne_bound_right_is_absolute)
    else:
      self.assertEqual(
          dynamic_profile_conditions.ne_bound_right_is_fGW,
          ne_bound_right_is_fGW,
      )
      self.assertTrue(dynamic_profile_conditions.ne_bound_right_is_absolute)


if __name__ == '__main__':
  absltest.main()
