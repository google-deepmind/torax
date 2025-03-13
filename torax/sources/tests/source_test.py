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
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import generic_current_source
from torax.sources import pydantic_model as source_pydantic_model
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source_models as source_models_lib


class SourceTest(parameterized.TestCase):
  """Tests for the base class Source."""

  def test_zero_profile_works_by_default(self):
    """The default source impl should support profiles with all zeros."""
    sources = source_pydantic_model.Sources.from_dict({
        generic_current_source.GenericCurrentSource.SOURCE_NAME: {
            'mode': runtime_params_lib.Mode.ZERO,
        }
    })
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    source = source_models.sources[
        generic_current_source.GenericCurrentSource.SOURCE_NAME
    ]
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            sources=sources,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
    )
    np.testing.assert_allclose(
        profile[0], np.zeros_like(geo.torax_mesh.cell_centers)
    )

  @parameterized.parameters(
      (runtime_params_lib.Mode.ZERO, np.array([0, 0, 0, 0])),
      (
          runtime_params_lib.Mode.MODEL_BASED,
          np.array([2.771899e-01, 9.061386e05, 4.113863e01, 2.593838e-14]),
      ),
      (runtime_params_lib.Mode.PRESCRIBED, np.array([3, 3, 3, 3])),
  )
  def test_correct_mode_called(
      self,
      mode,
      expected_profile,
  ):
    sources = source_pydantic_model.Sources.from_dict({
        generic_current_source.GenericCurrentSource.SOURCE_NAME: {
            'mode': mode,
            'prescribed_values': 3,
        }
    })
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    source = source_models.sources[
        generic_current_source.GenericCurrentSource.SOURCE_NAME
    ]
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry_pydantic_model.CircularConfig(n_rho=4).build_geometry()
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            sources=sources,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
    )
    np.testing.assert_allclose(
        profile[0],
        expected_profile,
        atol=1e-6,
        rtol=1e-6,
    )


if __name__ == '__main__':
  absltest.main()
