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

from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.sources import qei_source
from torax._src.sources.tests import test_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


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
    config = default_configs.get_default_config_dict()
    config["sources"] = {self._source_name: {}}
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models(
        neoclassical=torax_config.neoclassical
    )
    source = source_models.qei_source
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
