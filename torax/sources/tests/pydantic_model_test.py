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
from torax.sources import bootstrap_current_source
from torax.sources import fusion_heat_source
from torax.sources import gas_puff_source
from torax.sources import pydantic_model
from torax.sources import runtime_params
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
                  'model_func': 'radially_constant_fraction_of_Pin'
              },
          },
          expected_sources_model=impurity_radiation_constant_fraction.ImpurityRadiationHeatSinkConstantFractionConfig,
      ),
  )
  def test_correct_source_model(
      self,
      config: dict[str, Any],
      expected_sources_model: type[runtime_params.SourceModelBase],
  ):
    sources_model = pydantic_model.Sources.from_dict(config)
    self.assertIsInstance(
        sources_model.source_model_config[list(config.keys())[0]],
        expected_sources_model,
    )


if __name__ == '__main__':
  absltest.main()
