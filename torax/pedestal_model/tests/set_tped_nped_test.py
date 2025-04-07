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
from torax.config import build_runtime_params
from torax.core_profiles import initialization
from torax.sources import source_models as source_models_lib
from torax.torax_pydantic import model_config


class SetTemperatureDensityPedestalModelTest(parameterized.TestCase):
  """Tests for the `torax.pedestal_model.set_tped_nped` module."""

  @parameterized.product(
      Tiped=[5, {0.0: 5.0, 1.0: 10.0}],
      Teped=[4, {0.0: 4.0, 1.0: 8.0}],
      neped=[0.7, {0.0: 0.7, 1.0: 0.9}],
      rho_norm_ped_top=[{0.0: 0.5, 1.0: 0.7}],
      neped_is_fGW=[False, True],
      time=[0.0, 1.0],
  )
  def test_build_and_call_pedestal_model(
      # pylint: disable=invalid-name
      self,
      Tiped,
      Teped,
      neped,
      rho_norm_ped_top,
      neped_is_fGW,
      time,
      # pylint: enable=invalid-name
  ):
    """Test we can build and call the pedestal model with expected outputs."""
    torax_config = model_config.ToraxConfig.from_dict(
        dict(
            runtime_params=dict(),
            geometry=dict(geometry_type='circular'),
            pedestal=dict(
                pedestal_model='set_tped_nped',
                Tiped=Tiped,
                Teped=Teped,
                rho_norm_ped_top=rho_norm_ped_top,
                neped=neped,
                neped_is_fGW=neped_is_fGW,
            ),
            sources=dict(),
            stepper=dict(),
            transport=dict(),
        )
    )
    provider = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        torax_config.runtime_params,
        transport=torax_config.transport,
        sources=torax_config.sources,
        torax_mesh=torax_config.geometry.build_provider.torax_mesh,
        pedestal=torax_config.pedestal,
    )
    static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            profile_conditions=torax_config.profile_conditions,
            numerics=torax_config.numerics,
            plasma_composition=torax_config.plasma_composition,
            sources=torax_config.sources,
            torax_mesh=torax_config.geometry.build_provider.torax_mesh,
            stepper=torax_config.stepper,
        )
    )
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
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

    if isinstance(Tiped, (float, int)):
      self.assertEqual(pedestal_model_output.Tiped, Tiped)
    else:
      self.assertEqual(pedestal_model_output.Tiped, Tiped[time])
    if isinstance(Teped, (float, int)):
      self.assertEqual(pedestal_model_output.Teped, Teped)
    else:
      self.assertEqual(pedestal_model_output.Teped, Teped[time])
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
      # pylint: disable=invalid-name
      nGW = (
          dynamic_runtime_params_slice.profile_conditions.Ip_tot
          / (jnp.pi * geo.Rmin**2)
          * 1e20
          / dynamic_runtime_params_slice.numerics.nref
      )
      # pylint: enable=invalid-name
      expected_neped *= nGW
    self.assertEqual(pedestal_model_output.neped, expected_neped)


if __name__ == '__main__':
  absltest.main()
