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
import numpy as np
from torax import core_profile_setters
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import set_pped_tpedratio_nped
from torax.sources import source_models as source_models_lib


class SetPressureTemperatureRatioAndDensityPedestalModelTest(
    parameterized.TestCase
):
  """Tests for the `torax.pedestal_model.set_pped_tpedratio_nped` module."""

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = set_pped_tpedratio_nped.RuntimeParams()
    geo = geometry.build_circular_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)

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
    # pylint: disable=invalid-name
    """Test we can build and call the pedestal model with expected outputs."""
    pedestal_runtime_params = set_pped_tpedratio_nped.RuntimeParams(
        neped=neped,
        neped_is_fGW=neped_is_fGW,
        rho_norm_ped_top=rho_norm_ped_top,
        ion_electron_temperature_ratio=ion_electron_temperature_ratio,
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
    builder = set_pped_tpedratio_nped.SetPressureTemperatureRatioAndDensityPedestalModelBuilder(
        runtime_params=pedestal_runtime_params
    )
    dynamic_runtime_params_slice = provider(t=time)
    pedestal_model = builder()
    static_runtime_params_slice = (
        runtime_params_slice.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            source_runtime_params=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
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
