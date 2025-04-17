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
from torax.config import build_runtime_params
from torax.config import runtime_params as general_runtime_params
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.mhd import pydantic_model as mhd_pydantic_model
from torax.mhd import runtime_params as mhd_runtime_params
from torax.mhd.sawtooth import pydantic_model as sawtooth_pydantic_model
from torax.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax.mhd.sawtooth import sawtooth_model
from torax.tests.test_lib import default_sources
from torax.torax_pydantic import model_config


class MHDPydanticModelTest(parameterized.TestCase):
  """Tests for the MHD Pydantic model and dynamic params construction."""

  def setUp(self):
    super().setUp()
    self.geo = geometry_pydantic_model.CircularConfig().build_geometry()
    self.runtime_params = general_runtime_params.GeneralRuntimeParams()
    self.sources = default_sources.get_default_sources()

  def test_no_mhd_config(self):
    """Tests the case where the 'mhd' key is entirely absent."""
    config_dict = {
        'runtime_params': {},
        'geometry': {'geometry_type': 'circular'},
        'pedestal': {},
        'sources': {},
        'stepper': {},
        'time_step_calculator': {},
        'transport': {},
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)

    self.assertIsInstance(torax_config.mhd, mhd_pydantic_model.MHD)
    provider = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        runtime_params=self.runtime_params,
        sources=self.sources,
        torax_mesh=self.geo.torax_mesh,
        mhd=torax_config.mhd,
    )
    dynamic_slice = provider(t=0.0)
    self.assertIsInstance(
        dynamic_slice.mhd, mhd_runtime_params.DynamicMHDParams
    )
    self.assertIs(dynamic_slice.mhd.sawtooth, None)

  def test_empty_mhd_config(self):
    """Tests the case where 'mhd' key exists but is an empty dict."""
    config_dict = {
        'runtime_params': {},
        'geometry': {'geometry_type': 'circular'},
        'pedestal': {},
        'sources': {},
        'stepper': {},
        'time_step_calculator': {},
        'transport': {},
        'mhd': {},
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)

    self.assertIsInstance(torax_config.mhd, mhd_pydantic_model.MHD)
    assert isinstance(torax_config.mhd, mhd_pydantic_model.MHD)
    mhd_models = torax_config.mhd.build_mhd_models()
    self.assertIs(mhd_models.sawtooth, None)
    provider = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        runtime_params=self.runtime_params,
        sources=self.sources,
        torax_mesh=self.geo.torax_mesh,
        mhd=torax_config.mhd,
    )
    dynamic_slice = provider(t=0.0)
    self.assertIsInstance(
        dynamic_slice.mhd, mhd_runtime_params.DynamicMHDParams
    )
    self.assertIs(dynamic_slice.mhd.sawtooth, None)

  def test_mhd_config_with_sawtooth(self):
    """Tests the case with a valid sawtooth configuration."""
    config_dict = {
        'runtime_params': {},
        'geometry': {'geometry_type': 'circular'},
        'pedestal': {},
        'sources': {},
        'stepper': {},
        'time_step_calculator': {},
        'transport': {},
        'mhd': {
            'sawtooth': {
                'trigger_model_config': {
                    'trigger_model_type': 'simple',
                    'minimum_radius': 0.06,
                },
                'redistribution_model_config': {
                    'redistribution_model_type': 'simple'
                },
            }
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)

    self.assertIsInstance(torax_config.mhd, mhd_pydantic_model.MHD)
    assert torax_config.mhd is not None
    self.assertIsInstance(
        torax_config.mhd.sawtooth, sawtooth_pydantic_model.SawtoothConfig
    )

    mhd_models = torax_config.mhd.build_mhd_models()
    self.assertIn('sawtooth', mhd_models)
    self.assertIsInstance(mhd_models['sawtooth'], sawtooth_model.SawtoothModel)

    provider = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        runtime_params=self.runtime_params,
        sources=self.sources,
        torax_mesh=self.geo.torax_mesh,
        mhd=torax_config.mhd,
    )
    dynamic_slice = provider(t=0.0)
    self.assertIn('sawtooth', dynamic_slice.mhd)
    sawtooth_dynamic_params = dynamic_slice.mhd.sawtooth
    self.assertIsInstance(
        sawtooth_dynamic_params, sawtooth_runtime_params.DynamicRuntimeParams
    )
    self.assertEqual(
        sawtooth_dynamic_params.trigger_params.minimum_radius, 0.06
    )


if __name__ == '__main__':
  absltest.main()
