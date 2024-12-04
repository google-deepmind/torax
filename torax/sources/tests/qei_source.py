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

"""Tests for qei_source."""

import dataclasses
from absl.testing import absltest
from torax import core_profile_setters
from torax import geometry
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import qei_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib


class QeiSourceTest(test_lib.SourceTestCase):
  """Tests for QeiSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=qei_source.QeiSource,
        runtime_params_class=qei_source.RuntimeParams,
        unsupported_modes=[
            runtime_params_lib.Mode.FORMULA_BASED,
        ],
    )

  def test_expected_mesh_states(self):
    # This function is reimplemented here as QeiSource does not appear in
    # source_models, which the parent class uses to build the source
    source = qei_source.QeiSource()
    self.assertSameElements(
        source.affected_core_profiles,
        self._expected_affected_core_profiles,
    )

  def test_source_value(self):
    """Checks that the default implementation from Sources gives values."""
    source_builder = self._source_class_builder()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {'qei_source': source_builder}
    )
    source_models = source_models_builder()
    source = source_models.sources['qei_source']
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
    )
    dynamic_slice = runtime_params_slice.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        sources=source_models_builder.runtime_params,
        torax_mesh=geo.torax_mesh,
    )(
        t=runtime_params.numerics.t_initial,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    assert isinstance(source, qei_source.QeiSource)  # required for pytype.
    qei = source.get_qei(
        static_slice,
        dynamic_slice,
        dynamic_slice.sources['qei_source'],
        geo,
        core_profiles,
    )
    self.assertIsNotNone(qei)

  def test_invalid_source_types_raise_errors(self):
    source_builder = self._source_class_builder()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {'qei_source': source_builder}
    )
    source_models = source_models_builder()
    source = source_models.sources['qei_source']
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
    )
    dynamic_slice = runtime_params_slice.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        sources=source_models_builder.runtime_params,
        torax_mesh=geo.torax_mesh,
    )(
        t=runtime_params.numerics.t_initial,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    for unsupported_mode in self._unsupported_modes:
      with self.subTest(unsupported_mode.name):
        with self.assertRaises(ValueError):
          static_slice = runtime_params_slice.build_static_runtime_params_slice(
              runtime_params,
              source_runtime_params={
                  'qei_source': dataclasses.replace(
                      source_builder.runtime_params,
                      mode=unsupported_mode,
                  )
              },
          )
          # Force pytype to recognize `source` has `get_qei`
          assert isinstance(source, qei_source.QeiSource)
          source.get_qei(
              static_slice,
              dynamic_slice,
              dynamic_slice.sources['qei_source'],
              geo,
              core_profiles,
          )


if __name__ == '__main__':
  absltest.main()
