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
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.pedestal_model import (
    pedestal_transition_state as pedestal_transition_state_lib,
)
from torax._src.sources import source_profile_builders
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config

# pylint: disable=invalid-name


class SetPressureTemperatureRatioAndDensityPedestalModelTest(
    parameterized.TestCase
):

  @parameterized.product(
      n_e_ped=[0.7e20, {0.0: 0.7e20, 1.0: 0.9e20}],
      rho_norm_ped_top=[{0.0: 0.5, 1.0: 0.7}],
      n_e_ped_is_fGW=[False, True],
      time=[0.0, 1.0],
      T_i_T_e_ratio=[1.0, {0.0: 1.0, 1.0: 1.5}],
  )
  def test_build_and_call_pedestal_model(
      self,
      n_e_ped,
      rho_norm_ped_top,
      n_e_ped_is_fGW,
      time,
      T_i_T_e_ratio,
  ):
    if n_e_ped_is_fGW:
      if isinstance(n_e_ped, dict):
        n_e_ped = {key: value / 1e20 for key, value in n_e_ped.items()}
      else:
        n_e_ped /= 1e20
    config_dict = default_configs.get_default_config_dict()
    config_dict['pedestal'] = {
        'structure_model': {
            'model_name': 'set_P_ped_n_ped',
            'n_e_ped': n_e_ped,
            'n_e_ped_is_fGW': n_e_ped_is_fGW,
            'rho_norm_ped_top': rho_norm_ped_top,
            'T_i_T_e_ratio': T_i_T_e_ratio,
        }
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    jitted_pedestal_model = jax.jit(pedestal_model)

    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config, t_override=time)
    )
    source_states = source_models.initial_state(runtime_params)
    source_profiles, _ = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=source_states,
        explicit=True,
    )
    pedestal_state = pedestal_model.initial_state(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )
    pedestal_model_output, _ = jitted_pedestal_model(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_profiles=source_profiles,
        pedestal_transition_state=(
            pedestal_transition_state_lib.PedestalTransitionState.empty()
        ),
        pedestal_state=pedestal_state,
    )

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
          runtime_params.profile_conditions.I_p
          / 1e6  # Convert to MA.
          / (jnp.pi * geo.a_minor**2)
          * 1e20
      )
      expected_n_e_ped *= nGW

    np.testing.assert_allclose(pedestal_model_output.n_e_ped, expected_n_e_ped)

    if isinstance(T_i_T_e_ratio, (float, int)):
      np.testing.assert_allclose(
          pedestal_model_output.T_i_ped / pedestal_model_output.T_e_ped,
          T_i_T_e_ratio,
      )
    else:
      np.testing.assert_allclose(
          pedestal_model_output.T_i_ped / pedestal_model_output.T_e_ped,
          T_i_T_e_ratio[time],
      )

  def test_pedestal_pressure_multiplier(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['pedestal'] = {
        'structure_model': {
            'model_name': 'set_P_ped_n_ped',
            'n_e_ped': 0.7e20,
            'n_e_ped_is_fGW': False,
            'rho_norm_ped_top': 0.91,
            'T_i_T_e_ratio': 1.0,
            'P_ped': 1e5,
            'P_ped_multiplier': 1.0,
        }
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    pedestal_model = torax_config.pedestal.build_pedestal_model()

    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    source_states = source_models.initial_state(runtime_params)
    source_profiles, _ = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        source_states=source_states,
        explicit=True,
    )
    pedestal_state = pedestal_model.initial_state(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )
    output_1, _ = pedestal_model(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_profiles=source_profiles,
        pedestal_transition_state=(
            pedestal_transition_state_lib.PedestalTransitionState.empty()
        ),
        pedestal_state=pedestal_state,
    )

    # Modify multiplier to 1.5
    config_dict['pedestal']['structure_model']['P_ped_multiplier'] = 1.5
    torax_config_2 = config.ToraxConfig.from_dict(config_dict)
    provider_2 = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config_2
    )
    runtime_params_2 = provider_2(t=0.0)
    pedestal_state_2 = pedestal_model.initial_state(
        runtime_params=runtime_params_2,
        geo=geo,
        core_profiles=core_profiles,
    )
    output_2, _ = pedestal_model(
        runtime_params=runtime_params_2,
        geo=geo,
        core_profiles=core_profiles,
        source_profiles=source_profiles,
        pedestal_transition_state=(
            pedestal_transition_state_lib.PedestalTransitionState.empty()
        ),
        pedestal_state=pedestal_state_2,
    )

    np.testing.assert_allclose(output_2.T_e_ped, output_1.T_e_ped * 1.5)
    np.testing.assert_allclose(output_2.T_i_ped, output_1.T_i_ped * 1.5)
    np.testing.assert_allclose(output_2.n_e_ped, output_1.n_e_ped)


if __name__ == '__main__':
  absltest.main()
