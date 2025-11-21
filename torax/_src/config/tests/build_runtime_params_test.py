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
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src.config import build_runtime_params
from torax._src.config import config_loader
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.edge import base as edge_base
from torax._src.edge import extended_lengyel_model
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.orchestration import run_simulation
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.pedestal_model import set_tped_nped
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic


class BuildRuntimeParamsTest(parameterized.TestCase):

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
          runtime_params_lib.RuntimeParams,
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


# pylint: disable=invalid-name
class RuntimeParamsProviderUpdateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torax_config = config_loader.build_torax_config_from_file(
        'tests/test_data/test_iterhybrid_rampup.py'
    )
    (
        self._params_provider,
        _,
        _,
        _,
    ) = run_simulation.prepare_simulation(torax_config)

  def test_update_runtime_params_provider(self):
    params_provider = self._params_provider
    # Create updates for a `TimeVaryingScalar`, `TimeVaryingArray` and `float`.
    ip_value = params_provider.profile_conditions.Ip.value
    ip_update = interpolated_param_1d.TimeVaryingScalarReplace(
        value=ip_value * 2.0,
    )
    T_e_cell_value = (
        params_provider.profile_conditions.T_e.get_cached_interpolated_param_cell.ys
    )
    # Needed for pylint.
    assert params_provider.profile_conditions.T_e.grid is not None
    T_e_update = interpolated_param_2d.TimeVaryingArrayReplace(
        value=T_e_cell_value * 3.0,
        rho_norm=params_provider.profile_conditions.T_e.grid.cell_centers,
    )
    qei_update = params_provider.sources.ei_exchange.Qei_multiplier * 4.0

    @jax.jit
    def f(ip_update, T_e_update, qei_update):
      provider_new = params_provider.update_provider(
          lambda x: (
              x.profile_conditions.Ip,
              x.profile_conditions.T_e,
              x.sources.ei_exchange.Qei_multiplier,
          ),
          (ip_update, T_e_update, qei_update),
      )
      t = jnp.array(1.0)
      return (
          provider_new.profile_conditions.Ip.get_value(t),
          provider_new.profile_conditions.T_e.get_value(t),
          provider_new.sources.ei_exchange.Qei_multiplier,
      )

    original_ip_value = params_provider.profile_conditions.Ip.get_value(1.0)
    original_T_e_value = params_provider.profile_conditions.T_e.get_value(1.0)
    original_qei_value = params_provider.sources.ei_exchange.Qei_multiplier
    ip_value_new, T_e_value_new, qei_value_new = f(
        ip_update, T_e_update, qei_update
    )
    num_compiles1 = jax_utils.get_number_of_compiles(f)
    ip_update_new = interpolated_param_1d.TimeVaryingScalarReplace(
        value=ip_value * 4.0,
    )
    ip_value_2, T_e_value_2, qei_value_2 = f(
        ip_update_new, T_e_update, qei_update
    )
    num_compiles2 = jax_utils.get_number_of_compiles(f)

    with self.subTest('jit_compiles_and_returns_expected_value'):
      self.assertEqual(num_compiles1, 1)
      np.testing.assert_allclose(ip_value_new, original_ip_value * 2.0)
      np.testing.assert_allclose(T_e_value_new, original_T_e_value * 3.0)
      np.testing.assert_allclose(qei_value_new, original_qei_value * 4.0)

    with self.subTest('jit_updates_value_without_recompile'):
      np.testing.assert_allclose(ip_value_2, original_ip_value * 4.0)
      np.testing.assert_allclose(T_e_value_2, original_T_e_value * 3.0)
      np.testing.assert_allclose(qei_value_2, original_qei_value * 4.0)
      # Check that the cache is reused.
      self.assertEqual(num_compiles2, 1)

  @parameterized.parameters(
      (
          lambda x: (x.profile_conditions.Ip,),
          (1.0,),
          'To replace a `TimeVaryingScalar` use a `TimeVaryingScalarReplace`',
      ),
      (
          lambda x: (x.profile_conditions.T_e,),
          (interpolated_param_1d.TimeVaryingScalarReplace(np.array([1.0])),),
          'To replace a `TimeVaryingArray` use a `TimeVaryingArrayReplace`',
      ),
      (
          lambda x: (x.sources.ei_exchange.Qei_multiplier,),
          (True,),
          'To replace a scalar or `Array` pass a scalar or `Array`',
      ),
  )
  def test_update_runtime_params_provider_raises_for_invalid_replacements(
      self, get_node, replacement, expected_error_message
  ):
    params_provider = self._params_provider
    with self.assertRaises(
        ValueError,
        msg=expected_error_message,
    ):
      params_provider.update_provider(
          get_node,
          replacement,
      )

  def test_update_runtime_params_provider_raises_for_invalid_node(self):
    params_provider = self._params_provider
    with self.assertRaises(ValueError):
      params_provider.update_provider(
          # Attempt to change a bool field.
          lambda x: (x.profile_conditions.n_e_nbar_is_fGW,),
          (True,),
      )

  def test_update_runtime_params_provider_result_can_be_updated_again(self):
    params_provider = self._params_provider
    ip_value = params_provider.profile_conditions.Ip.value
    ip_update = interpolated_param_1d.TimeVaryingScalarReplace(
        value=ip_value * 2.0,
    )
    ip_second_update = interpolated_param_1d.TimeVaryingScalarReplace(
        value=ip_value * 4.0,
    )
    updated_provider = params_provider.update_provider(
        lambda x: (x.profile_conditions.Ip,),
        (ip_update,),
    )
    re_updated_provider = updated_provider.update_provider(
        lambda x: (x.profile_conditions.Ip,),
        (ip_second_update,),
    )
    final_ip = re_updated_provider.profile_conditions.Ip.get_value(0.0)
    original_ip = params_provider.profile_conditions.Ip.get_value(0.0)
    np.testing.assert_allclose(final_ip, original_ip * 4.0)

  def test_update_runtime_params_provider_mapping(self):
    params_provider = self._params_provider
    ip_value = params_provider.profile_conditions.Ip.value
    ip_update = interpolated_param_1d.TimeVaryingScalarReplace(
        value=ip_value * 2.0,
    )
    T_e_value = (
        params_provider.profile_conditions.T_e.get_cached_interpolated_param_cell.ys
    )
    assert params_provider.profile_conditions.T_e.grid is not None
    T_e_update = interpolated_param_2d.TimeVaryingArrayReplace(
        value=T_e_value * 3.0,
        rho_norm=params_provider.profile_conditions.T_e.grid.cell_centers,
    )
    value_Qei_multiplier = params_provider.sources.ei_exchange.Qei_multiplier
    Qei_multiplier_update = value_Qei_multiplier * 4.0

    # when
    @jax.jit
    def f(ip_update, T_e_update, Qei_multiplier_update):
      # update the provider
      provider_new = self._params_provider.update_provider_from_mapping(
          {
              'profile_conditions.Ip': ip_update,
              'profile_conditions.T_e': T_e_update,
              'sources.ei_exchange.Qei_multiplier': Qei_multiplier_update,
          },
      )
      # this provider can then be used as overrides for the step function.
      t = jnp.array(0.0)
      return (
          provider_new.profile_conditions.Ip.get_value(t),
          provider_new.profile_conditions.T_e.get_value(t),
          provider_new.sources.ei_exchange.Qei_multiplier,
      )

    ip_value_new, T_e_value_new, Qei_multiplier_new = f(
        ip_update, T_e_update, Qei_multiplier_update
    )

    # then
    self.assertEqual(jax_utils.get_number_of_compiles(f), 1)
    original_ip_value = self._params_provider.profile_conditions.Ip.get_value(
        0.0
    )
    np.testing.assert_allclose(ip_value_new, original_ip_value * 2.0)
    original_T_e_value = self._params_provider.profile_conditions.T_e.get_value(
        0.0
    )
    np.testing.assert_allclose(T_e_value_new, original_T_e_value * 3.0)
    original_Qei_multiplier = (
        self._params_provider.sources.ei_exchange.Qei_multiplier
    )
    np.testing.assert_allclose(
        Qei_multiplier_new, original_Qei_multiplier * 4.0
    )

    ip_update_new = interpolated_param_1d.TimeVaryingScalarReplace(
        value=ip_value * 4.0,
    )
    ip_value_new, T_e_value_new, Qei_multiplier_new = f(
        ip_update_new, T_e_update, Qei_multiplier_update
    )
    np.testing.assert_allclose(ip_value_new, original_ip_value * 4.0)
    np.testing.assert_allclose(T_e_value_new, original_T_e_value * 3.0)
    np.testing.assert_allclose(
        Qei_multiplier_new, original_Qei_multiplier * 4.0
    )
    self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

  def test_update_runtime_params_provider_mapping_raises_for_invalid_key(self):
    value = self._params_provider.profile_conditions.Ip.value
    ip_update = interpolated_param_1d.TimeVaryingScalarReplace(
        value=value * 2.0,
    )

    @jax.jit
    def f(ip_update):
      # update the provider
      provider_new = self._params_provider.update_provider_from_mapping(
          # incorrectly spelt key for Ip.
          {'profile_conditions.IP': ip_update},
      )
      return provider_new

    with self.assertRaises(ValueError, msg='Attribute IP not found.'):
      f(ip_update)


class UpdateRuntimeParamsFromEdgeTest(parameterized.TestCase):

  def test_update_impurities_scales_profile(self):
    _ENRICHMENT_FACTOR = 2.0
    _OUTPUT_CONCENTRATION = 0.1
    _INITIAL_EDGE_RATIO = 0.02
    _INITIAL_AXIS_RATIO = 0.01
    config_dict = default_configs.get_default_config_dict()
    # Set impurity mode to n_e_ratios and define a profile
    config_dict['plasma_composition']['impurity'] = {
        'impurity_mode': 'n_e_ratios',
        'species': {'N': {0: _INITIAL_AXIS_RATIO, 1: _INITIAL_EDGE_RATIO}},
    }
    config_dict['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    # Set up edge model config
    config_dict['edge'] = {
        'model_name': 'extended_lengyel',
        'computation_mode': 'inverse',
        'update_impurities': True,
        'enrichment_factor': {'N': _ENRICHMENT_FACTOR},
        'seed_impurity_weights': {'N': 1.0},
        # Dummy values for other required fields.
        'target_electron_temp': 1.0,
        'parallel_connection_length': 1.0,
        'divertor_parallel_length': 1.0,
        'toroidal_flux_expansion': 1.0,
        'target_angle_of_incidence': 1.0,
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    edge_outputs = mock.MagicMock(spec=edge_base.EdgeModelOutputs)
    edge_outputs.seed_impurity_concentrations = {
        'N': jnp.array(_OUTPUT_CONCENTRATION)
    }

    initial_impurity_params = runtime_params.plasma_composition.impurity
    assert isinstance(runtime_params.edge, extended_lengyel_model.RuntimeParams)
    assert isinstance(
        initial_impurity_params, electron_density_ratios.RuntimeParams
    )
    initial_n_e_ratios = initial_impurity_params.n_e_ratios['N']

    updated_runtime_params = build_runtime_params._update_impurities(
        runtime_params, edge_outputs
    )

    updated_impurity_params = updated_runtime_params.plasma_composition.impurity
    assert isinstance(
        updated_impurity_params, electron_density_ratios.RuntimeParams
    )
    updated_n_e_ratios = updated_impurity_params.n_e_ratios['N']

    # Expected scaling logic:
    conc_lcfs = _OUTPUT_CONCENTRATION / _ENRICHMENT_FACTOR
    scaling_factor = conc_lcfs / _INITIAL_EDGE_RATIO

    initial_n_e_ratios_face = initial_impurity_params.n_e_ratios_face['N']
    updated_n_e_ratios_face = updated_impurity_params.n_e_ratios_face['N']

    np.testing.assert_allclose(
        updated_n_e_ratios, initial_n_e_ratios * scaling_factor, rtol=1e-5
    )
    np.testing.assert_allclose(
        updated_n_e_ratios_face,
        initial_n_e_ratios_face * scaling_factor,
        rtol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()
