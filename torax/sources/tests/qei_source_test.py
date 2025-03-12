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
from torax.config import build_runtime_params
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import qei_source
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib


class QeiSourceTest(test_lib.SourceTestCase):
  """Tests for QeiSource."""

  # pytype: disable=signature-mismatch
  def setUp(self):
    super().setUp(
        source_name=qei_source.QeiSource.SOURCE_NAME,
        source_config_class=qei_source.QeiSourceConfig,
        needs_source_models=False,
    )
  # pytype: enable=signature-mismatch

  def test_source_value(self):
    """Checks that the default implementation from Sources gives values."""
    sources = sources_pydantic_model.Sources.from_dict({
        qei_source.QeiSource.SOURCE_NAME: {},
    })
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    source = source_models.sources['qei_source']
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    dynamic_slice = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )(
        t=runtime_params.numerics.t_initial,
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    assert isinstance(source, qei_source.QeiSource)  # required for pytype.
    qei = source.get_qei(
        static_slice,
        dynamic_slice,
        geo,
        core_profiles,
    )
    self.assertIsNotNone(qei)


if __name__ == '__main__':
  absltest.main()
