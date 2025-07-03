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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from torax._src.config import build_runtime_params
from torax._src.mhd import pydantic_model as mhd_pydantic_model
from torax._src.mhd import runtime_params as mhd_runtime_params
from torax._src.mhd.sawtooth import pydantic_model as sawtooth_pydantic_model
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.mhd.sawtooth import sawtooth_model
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import transport_model as transport_model_lib


class MHDPydanticModelTest(parameterized.TestCase):
  """Tests for the MHD Pydantic model and dynamic params construction."""

  def setUp(self):
    super().setUp()
    self.transport_model = mock.Mock(spec=transport_model_lib.TransportModel)
    self.source_models = mock.Mock(spec=source_models_lib.SourceModels)
    self.pedestal_model = mock.Mock(spec=pedestal_model_lib.PedestalModel)
    self.neoclassical_models = mock.Mock(
        spec=neoclassical_models_lib.NeoclassicalModels
    )

  def test_no_mhd_config(self):
    """Tests the case where the 'mhd' key is entirely absent."""
    torax_config = model_config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )

    self.assertIsInstance(torax_config.mhd, mhd_pydantic_model.MHD)
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_slice = provider(t=0.0)
    self.assertIsInstance(
        dynamic_slice.mhd, mhd_runtime_params.DynamicMHDParams
    )
    self.assertIs(dynamic_slice.mhd.sawtooth, None)

  def test_empty_mhd_config(self):
    """Tests the case where 'mhd' key exists but is an empty dict."""
    config = default_configs.get_default_config_dict()
    config['mhd'] = {}
    torax_config = model_config.ToraxConfig.from_dict(config)
    static_runtime_params_slice = (
        build_runtime_params.build_static_params_from_config(torax_config)
    )

    self.assertIsInstance(torax_config.mhd, mhd_pydantic_model.MHD)
    assert isinstance(torax_config.mhd, mhd_pydantic_model.MHD)
    mhd_models = torax_config.mhd.build_mhd_models(
        static_runtime_params_slice=static_runtime_params_slice,
        transport_model=self.transport_model,
        source_models=self.source_models,
        pedestal_model=self.pedestal_model,
        neoclassical_models=self.neoclassical_models,
    )
    self.assertIs(mhd_models.sawtooth, None)
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_slice = provider(t=0.0)
    self.assertIsInstance(
        dynamic_slice.mhd, mhd_runtime_params.DynamicMHDParams
    )
    self.assertIs(dynamic_slice.mhd.sawtooth, None)

  def test_mhd_config_with_sawtooth(self):
    """Tests the case with a valid sawtooth configuration."""
    config = default_configs.get_default_config_dict()
    config['mhd'] = {
        'sawtooth': {
            'trigger_model': {
                'model_name': 'simple',
                'minimum_radius': 0.06,
            },
            'redistribution_model': {'model_name': 'simple'},
        }
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    static_runtime_params_slice = (
        build_runtime_params.build_static_params_from_config(torax_config)
    )

    self.assertIsInstance(torax_config.mhd, mhd_pydantic_model.MHD)
    assert torax_config.mhd is not None
    self.assertIsInstance(
        torax_config.mhd.sawtooth, sawtooth_pydantic_model.SawtoothConfig
    )

    mhd_models = torax_config.mhd.build_mhd_models(
        static_runtime_params_slice=static_runtime_params_slice,
        transport_model=self.transport_model,
        source_models=self.source_models,
        pedestal_model=self.pedestal_model,
        neoclassical_models=self.neoclassical_models,
    )
    self.assertIsInstance(mhd_models.sawtooth, sawtooth_model.SawtoothModel)

    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_slice = provider(t=0.0)
    sawtooth_dynamic_params = dynamic_slice.mhd.sawtooth
    self.assertIsInstance(
        sawtooth_dynamic_params, sawtooth_runtime_params.DynamicRuntimeParams
    )
    self.assertEqual(
        sawtooth_dynamic_params.trigger_params.minimum_radius, 0.06
    )


if __name__ == '__main__':
  absltest.main()
