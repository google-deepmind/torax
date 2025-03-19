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

"""Tests for the validation of source models."""

from absl.testing import absltest
import pydantic

from torax.sources import pydantic_model
from torax.sources import runtime_params


class SourceModelsValidationTest(absltest.TestCase):
  """Tests for validations in source models."""

  def test_bremsstrahlung_and_mavrin_active_check(self):
    """Tests that bremsstrahlung and Mavrin models cannot be active together."""
    # Test valid configuration: bremsstrahlung is ZERO
    valid_config_1 = {
        'bremsstrahlung_heat_sink': {'mode': 'ZERO'},
        'impurity_radiation_heat_sink': {
            'mode': 'PRESCRIBED',
            'model_function_name': 'impurity_radiation_mavrin_fit',
        },
    }
    sources = pydantic_model.Sources.model_validate(valid_config_1)
    self.assertEqual(sources.bremsstrahlung_heat_sink.mode, runtime_params.Mode.ZERO)
    self.assertEqual(
        sources.impurity_radiation_heat_sink.mode, runtime_params.Mode.PRESCRIBED
    )

    # Test valid configuration: Mavrin is ZERO
    valid_config_2 = {
        'bremsstrahlung_heat_sink': {'mode': 'PRESCRIBED'},
        'impurity_radiation_heat_sink': {
            'mode': 'ZERO',
            'model_function_name': 'impurity_radiation_mavrin_fit',
        },
    }
    sources = pydantic_model.Sources.model_validate(valid_config_2)
    self.assertEqual(sources.bremsstrahlung_heat_sink.mode, runtime_params.Mode.PRESCRIBED)
    self.assertEqual(sources.impurity_radiation_heat_sink.mode, runtime_params.Mode.ZERO)

    # Test valid configuration: impurity_radiation is using constant fraction model
    valid_config_3 = {
        'bremsstrahlung_heat_sink': {'mode': 'PRESCRIBED'},
        'impurity_radiation_heat_sink': {
            'mode': 'PRESCRIBED',
            'model_function_name': 'radially_constant_fraction_of_Pin',
        },
    }
    sources = pydantic_model.Sources.model_validate(valid_config_3)
    self.assertEqual(sources.bremsstrahlung_heat_sink.mode, runtime_params.Mode.PRESCRIBED)
    self.assertEqual(
        sources.impurity_radiation_heat_sink.mode, runtime_params.Mode.PRESCRIBED
    )

    # Test invalid configuration: both Mavrin and bremsstrahlung are active
    invalid_config = {
        'bremsstrahlung_heat_sink': {'mode': 'PRESCRIBED'},
        'impurity_radiation_heat_sink': {
            'mode': 'PRESCRIBED',
            'model_function_name': 'impurity_radiation_mavrin_fit',
        },
    }
    with self.assertRaisesRegex(
        ValueError, 'Both bremsstrahlung_heat_sink and impurity_radiation_heat_sink'
    ):
      pydantic_model.Sources.model_validate(invalid_config)


if __name__ == '__main__':
  absltest.main() 