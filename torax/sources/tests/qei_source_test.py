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
from torax.core_profiles import initialization
from torax.sources import qei_source
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib
from torax.torax_pydantic import model_config


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
    torax_config = model_config.ToraxConfig.from_dict({
        "runtime_params": {},
        "geometry": {"geometry_type": "circular"},
        "sources": {
            self._source_name: {},
        },
        "stepper": {},
        "transport": {},
        "pedestal": {},
    })
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
    )
    source = source_models.sources["qei_source"]
    self.assertIsInstance(source, qei_source.QeiSource)
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    dynamic_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )(
            t=torax_config.numerics.t_initial,
        )
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    qei = source.get_qei(
        static_slice,
        dynamic_slice,
        geo,
        core_profiles,
    )
    self.assertIsNotNone(qei)


if __name__ == "__main__":
  absltest.main()
