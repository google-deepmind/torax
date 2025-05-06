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
from torax import jax_utils
from torax.config import build_runtime_params
from torax.core_profiles import initialization
from torax.sources import source_models as source_models_lib
from torax.tests.test_lib import default_configs
from torax.torax_pydantic import model_config
# pylint: disable=invalid-name


class SetPressureTemperatureRatioAndDensityPedestalModelTest(
    parameterized.TestCase
):

  @parameterized.product(
      neped=[0.7, {0.0: 0.7, 1.0: 0.9}],
      rho_norm_ped_top=[{0.0: 0.5, 1.0: 0.7}],
      neped_is_fGW=[False, True],
      time=[0.0, 1.0],
      ion_electron_temperature_ratio=[1.0, {0.0: 1.0, 1.0: 1.5}],
  )
  def test_build_and_call_pedestal_model(
      self,
      neped,
      rho_norm_ped_top,
      neped_is_fGW,
      time,
      ion_electron_temperature_ratio,
  ):
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'pedestal_model': 'set_pped_tpedratio_nped',
        'set_pedestal': True,
        'neped': neped,
        'neped_is_fGW': neped_is_fGW,
        'rho_norm_ped_top': rho_norm_ped_top,
        'ion_electron_temperature_ratio': ion_electron_temperature_ratio,
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
        sources=torax_config.sources.source_model_config
    )
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    jitted_pedestal_model = jax_utils.jit(pedestal_model)

    geo = torax_config.geometry.build_provider(time)
    dynamic_runtime_params_slice = provider(t=time)
    core_profiles = initialization.initial_core_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
    )
    pedestal_model_output = jitted_pedestal_model(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
    )

    if isinstance(rho_norm_ped_top, (float, int)):
      self.assertEqual(pedestal_model_output.rho_norm_ped_top, rho_norm_ped_top)
    else:
      self.assertEqual(
          pedestal_model_output.rho_norm_ped_top, rho_norm_ped_top[time]
      )

    if isinstance(neped, (float, int)):
      expected_neped = neped
    else:
      expected_neped = neped[time]
    if neped_is_fGW:
      nGW = (
          dynamic_runtime_params_slice.profile_conditions.I_total
          / (jnp.pi * geo.a_minor**2)
          * 1e20
          / dynamic_runtime_params_slice.numerics.nref
      )
      expected_neped *= nGW
    np.testing.assert_allclose(pedestal_model_output.neped, expected_neped)

    if isinstance(ion_electron_temperature_ratio, (float, int)):
      np.testing.assert_allclose(
          pedestal_model_output.Tiped / pedestal_model_output.Teped,
          ion_electron_temperature_ratio,
      )
    else:
      np.testing.assert_allclose(
          pedestal_model_output.Tiped / pedestal_model_output.Teped,
          ion_electron_temperature_ratio[time],
      )


if __name__ == '__main__':
  absltest.main()
