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
from jax import numpy as jnp
import numpy as np
from torax import constants
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.sources import source_models as source_models_lib
from torax._src.torax_pydantic import model_config
from torax.tests.test_lib import default_configs

# pylint: disable=invalid-name


class SetTemperatureDensityPedestalModelTest(parameterized.TestCase):

  @parameterized.product(
      T_i_ped=[5, {0.0: 5.0, 1.0: 10.0}],
      T_e_ped=[4, {0.0: 4.0, 1.0: 8.0}],
      n_e_ped=[0.7e20, {0.0: 0.7e20, 1.0: 0.9e20}],
      rho_norm_ped_top=[{0.0: 0.5, 1.0: 0.7}],
      n_e_ped_is_fGW=[False, True],
      time=[0.0, 1.0],
  )
  def test_build_and_call_pedestal_model(
      self,
      T_i_ped,
      T_e_ped,
      n_e_ped,
      rho_norm_ped_top,
      n_e_ped_is_fGW,
      time,
  ):
    if n_e_ped_is_fGW:
      if isinstance(n_e_ped, dict):
        n_e_ped = {
            key: value / constants.DENSITY_SCALING_FACTOR
            for key, value in n_e_ped.items()
        }
      else:
        n_e_ped /= constants.DENSITY_SCALING_FACTOR
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'T_i_ped': T_i_ped,
        'T_e_ped': T_e_ped,
        'rho_norm_ped_top': rho_norm_ped_top,
        'n_e_ped': n_e_ped,
        'n_e_ped_is_fGW': n_e_ped_is_fGW,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    static_runtime_params_slice = (
        build_runtime_params.build_static_params_from_config(torax_config)
    )
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources, neoclassical=torax_config.neoclassical
    )

    geo = torax_config.geometry.build_provider(time)
    dynamic_runtime_params_slice = provider(t=time)
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    core_profiles = initialization.initial_core_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
    )
    pedestal_model_output = pedestal_model(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
    )

    if isinstance(T_i_ped, (float, int)):
      self.assertEqual(pedestal_model_output.T_i_ped, T_i_ped)
    else:
      self.assertEqual(pedestal_model_output.T_i_ped, T_i_ped[time])
    if isinstance(T_e_ped, (float, int)):
      self.assertEqual(pedestal_model_output.T_e_ped, T_e_ped)
    else:
      self.assertEqual(pedestal_model_output.T_e_ped, T_e_ped[time])
    if isinstance(rho_norm_ped_top, (float, int)):
      self.assertEqual(pedestal_model_output.rho_norm_ped_top, rho_norm_ped_top)
    else:
      self.assertEqual(
          pedestal_model_output.rho_norm_ped_top, rho_norm_ped_top[time]
      )

    if isinstance(n_e_ped, (float, int)):
      expected_n_e_ped = n_e_ped
    else:
      expected_n_e_ped = n_e_ped[time]
    if n_e_ped_is_fGW:
      nGW = (
          dynamic_runtime_params_slice.profile_conditions.Ip
          / 1e6  # Convert to MA.
          / (jnp.pi * geo.a_minor**2)
          * 1e20
      )
      expected_n_e_ped *= nGW / constants.DENSITY_SCALING_FACTOR
    else:
      expected_n_e_ped /= constants.DENSITY_SCALING_FACTOR
    np.testing.assert_allclose(pedestal_model_output.n_e_ped, expected_n_e_ped)


if __name__ == '__main__':
  absltest.main()
