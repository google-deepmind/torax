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
"""Tests for basic pedestal model."""
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from torax import core_profile_setters
from torax import geometry
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.pedestal_model import set_tped_nped
from torax.sources import source_models as source_models_lib


class SetTemperatureDensityPedestalModelTest(parameterized.TestCase):
  """Tests for the `torax.pedestal_model.basic` module."""

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = set_tped_nped.RuntimeParams()
    geo = geometry.build_circular_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)

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
    pedestal_runtime_params = set_tped_nped.RuntimeParams(
        Tiped=Tiped,
        Teped=Teped,
        rho_norm_ped_top=rho_norm_ped_top,
        neped=neped,
        neped_is_fGW=neped_is_fGW,
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    source_models_builder = source_models_lib.SourceModelsBuilder()
    source_models = source_models_builder()
    geo = geometry.build_circular_geometry()
    provider = runtime_params_slice.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        sources=source_models_builder.runtime_params,
        torax_mesh=geo.torax_mesh,
        pedestal=pedestal_runtime_params,
    )
    geo = geometry.build_circular_geometry()
    builder = set_tped_nped.SetTemperatureDensityPedestalModelBuilder(
        runtime_params=pedestal_runtime_params
    )
    dynamic_runtime_params_slice = provider(t=time)
    pedestal_model = builder()
    core_profiles = core_profile_setters.initial_core_profiles(
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
