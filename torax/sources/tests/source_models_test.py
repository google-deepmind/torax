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
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax import core_profile_setters
from torax.config import runtime_params as runtime_params_lib
from torax.config import runtime_params_slice
from torax.geometry import circular_geometry
from torax.sources import generic_current_source
from torax.sources import runtime_params as source_runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib


class SourceModelsTest(parameterized.TestCase):

  @parameterized.parameters(
      (source_runtime_params_lib.Mode.ZERO, 0),
      (source_runtime_params_lib.Mode.PRESCRIBED, 1),
  )
  def test_external_current_source_sums_all_psi_sources(
      self, bar_source_mode, bar_source_value
  ):
    """Test that external current source sums all external psi sources."""
    # Create a custom bar source that affects psi.
    source_name = 'bar'

    @dataclasses.dataclass(frozen=True)
    class BarSource(source_lib.Source):
      """A test source."""

      @property
      def affected_core_profiles(self):
        return (source_lib.AffectedCoreProfile.PSI,)

      @property
      def source_name(self) -> str:
        return source_name

    bar_source_builder = source_lib.make_source_builder(BarSource)()

    source_models_builder = source_models_lib.SourceModelsBuilder(
        {source_name: bar_source_builder},
    )
    # Set the default generic current source to MODEL_BASED instead of ZERO.
    source_models_builder.runtime_params[
        generic_current_source.GenericCurrentSource.SOURCE_NAME
    ].mode = source_runtime_params_lib.Mode.MODEL_BASED
    # and set the bar source to the given mode.
    source_models_builder.runtime_params['bar'].mode = bar_source_mode
    source_models_builder.runtime_params['bar'].prescribed_values = 1
    source_models = source_models_builder()
    runtime_params = runtime_params_lib.GeneralRuntimeParams()
    geo = circular_geometry.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        runtime_params_slice.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            source_runtime_params=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
        static_runtime_params_slice=static_runtime_params_slice,
    )
    expected_generic_current_source = source_models.psi_sources[
        generic_current_source.GenericCurrentSource.SOURCE_NAME
    ].get_value(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        None,
    )[0]

    external_current_source = source_models.external_current_source(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
        static_runtime_params_slice=static_runtime_params_slice,
    )

    # check the external current source is the sum of the generic current
    # source and the foo source.
    np.testing.assert_allclose(
        external_current_source,
        expected_generic_current_source + bar_source_value,
    )


if __name__ == '__main__':
  absltest.main()
