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
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from torax.sources import base
from torax.sources import bootstrap_current_source
from torax.sources import fusion_heat_source
from torax.sources import gas_puff_source
from torax.sources import generic_current_source
from torax.sources import pydantic_model
from torax.sources import qei_source
from torax.sources import runtime_params as source_runtime_params_lib
from torax.sources import source_models as source_models_lib
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_constant_fraction
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit


class PydanticModelTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          config={
              'gas_puff_source': {
                  'puff_decay_length': 0.3,
                  'S_puff_tot': 0.0,
              }
          },
          expected_sources_model=gas_puff_source.GasPuffSourceConfig,
      ),
      dict(
          config={
              'j_bootstrap': {
                  'bootstrap_mult': 0.3,
              }
          },
          expected_sources_model=bootstrap_current_source.BootstrapCurrentSourceConfig,
      ),
      dict(
          config={
              'fusion_heat_source': {},
          },
          expected_sources_model=fusion_heat_source.FusionHeatSourceConfig,
      ),
      dict(
          config={
              'impurity_radiation_heat_sink': {},
          },
          expected_sources_model=impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig,
      ),
      dict(
          config={
              'impurity_radiation_heat_sink': {
                  'model_function_name': 'radially_constant_fraction_of_Pin'
              },
          },
          expected_sources_model=impurity_radiation_constant_fraction.ImpurityRadiationHeatSinkConstantFractionConfig,
      ),
  )
  def test_correct_source_model(
      self,
      config: dict[str, Any],
      expected_sources_model: type[base.SourceModelBase],
  ):
    sources_model = pydantic_model.Sources.from_dict(config)
    self.assertIsInstance(
        sources_model.source_model_config[list(config.keys())[0]],
        expected_sources_model,
    )
    # Check that the 3 default sources are always present.
    for key in [
        bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME,
        qei_source.QeiSource.SOURCE_NAME,
        generic_current_source.GenericCurrentSource.SOURCE_NAME,
    ]:
      self.assertIn(key, sources_model.source_model_config.keys())

  def test_adding_standard_source_via_config(self):
    """Tests that a source can be added with overriding defaults."""
    sources = pydantic_model.Sources.from_dict({
        'gas_puff_source': {
            'puff_decay_length': 1.23,
        },
        'ohmic_heat_source': {
            'is_explicit': True,
            'mode': 'ZERO',  # turn it off.
        },
    })
    source_models = source_models_lib.SourceModels(sources.source_model_config)
    # The non-standard ones are still off.
    self.assertEqual(
        sources.source_model_config['j_bootstrap'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        sources.source_model_config['generic_current_source'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        sources.source_model_config['qei_source'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    # But these new sources have been added.
    self.assertLen(source_models.sources, 5)
    self.assertLen(source_models.standard_sources, 3)
    # With the overriding params.
    gas_puff_config = sources.source_model_config['gas_puff_source']
    self.assertIsInstance(gas_puff_config, gas_puff_source.GasPuffSourceConfig)
    self.assertEqual(
        gas_puff_config.puff_decay_length.get_value(0.0),
        1.23,
    )
    self.assertEqual(
        sources.source_model_config['gas_puff_source'].mode,
        source_runtime_params_lib.Mode.MODEL_BASED,  # On by default.
    )
    self.assertEqual(
        sources.source_model_config['ohmic_heat_source'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )


if __name__ == '__main__':
  absltest.main()
