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
import copy
import dataclasses
from typing import Literal

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import config_loader
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import base as source_base_pydantic_model
from torax._src.sources import gas_puff_source as gas_puff_source_lib
from torax._src.sources import runtime_params
from torax._src.sources import source as source_lib
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import register_config
from torax._src.torax_pydantic import torax_pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params.DynamicRuntimeParams):
  a: array_typing.FloatScalar
  b: bool


def double_gas_puff_source(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[chex.Array, ...]:
  """Calculates external source term for n from puffs."""
  output = gas_puff_source_lib.calc_puff_source(
      unused_static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      source_name,
      unused_state,
      unused_calculated_source_profiles,
      unused_conductivity,
  )
  return 2 * output


class NewGasPuffSourceModelConfig(source_base_pydantic_model.SourceModelBase):
  """New source model config."""

  model_name: Literal['test_model_function'] = 'test_model_function'
  a: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(1.0)
  b: bool = False

  @property
  def model_func(self) -> source_lib.SourceProfileFunction:
    return double_gas_puff_source

  def build_source(self) -> source_lib.Source:
    return gas_puff_source_lib.GasPuffSource(model_func=self.model_func)

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        a=self.a.get_value(t),
        b=self.b,
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
    )


class DuplicateGasPuffSourceModelConfig(
    source_base_pydantic_model.SourceModelBase
):
  # Name that is already registered.
  model_name: Literal['exponential'] = 'exponential'
  a: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(1.0)
  b: bool = False

  @property
  def model_func(self) -> source_lib.SourceProfileFunction:
    return double_gas_puff_source

  def build_source(self) -> source_lib.Source:
    return gas_puff_source_lib.GasPuffSource(model_func=self.model_func)

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        a=self.a.get_value(t),
        b=self.b,
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
    )


class RegisterConfigTest(parameterized.TestCase):

  def test_register_source_model_config(self):
    config_name = 'test_iterhybrid_rampup.py'
    test_config_path = 'tests/test_data/' + config_name
    config_module = config_loader.import_module(test_config_path)
    config = copy.deepcopy(config_module['CONFIG'])
    # Register the new source model config against the gas puff source.
    register_config.register_source_model_config(
        NewGasPuffSourceModelConfig, 'gas_puff'
    )

    # Load the original config and check the gas puff source is expected type.
    config_pydantic = model_config.ToraxConfig.from_dict(config)
    gas_puff_config = config_pydantic.sources.gas_puff
    self.assertIsInstance(
        gas_puff_config, gas_puff_source_lib.GasPuffSourceConfig
    )
    gas_puff = gas_puff_config.build_source()
    self.assertIsInstance(gas_puff, gas_puff_source_lib.GasPuffSource)
    dynamic_params = gas_puff_config.build_dynamic_params(t=0.0)
    self.assertIsInstance(
        dynamic_params, gas_puff_source_lib.DynamicGasPuffRuntimeParams
    )

    # Now modify the original config to use the new config.
    del config['sources']['gas_puff']
    config['sources']['gas_puff'] = {
        'model_name': 'test_model_function',  # new registered name.
        'a': 2.0,
    }
    config_pydantic = model_config.ToraxConfig.from_dict(config)
    # Check we build the correct config.
    new_gas_puff_config = config_pydantic.sources.gas_puff
    self.assertIsInstance(new_gas_puff_config, NewGasPuffSourceModelConfig)
    # Check the dynamic params are built correctly.
    new_dynamic_params = new_gas_puff_config.build_dynamic_params(t=0.0)
    self.assertIsInstance(new_dynamic_params, DynamicRuntimeParams)
    self.assertEqual(new_dynamic_params.a, 2.0)
    self.assertEqual(new_dynamic_params.b, False)

  def test_error_thrown_if_model_name_is_already_registered(self):
    with self.assertRaises(ValueError):
      register_config.register_source_model_config(
          DuplicateGasPuffSourceModelConfig, 'gas_puff'
      )

  @parameterized.parameters('qei', 'j_bootstrap')
  def test_error_thrown_if_using_special_source(self, special_source):
    with self.assertRaisesRegex(
        ValueError,
        'Cannot register a new source model config for the qei or j_bootstrap'
        ' sources.',
    ):
      register_config.register_source_model_config(
          NewGasPuffSourceModelConfig, special_source
      )

  def test_error_thrown_if_source_not_supported(self):
    with self.assertRaisesRegex(
        ValueError,
        'The source name foo_source is not supported.',
    ):
      register_config.register_source_model_config(
          NewGasPuffSourceModelConfig, 'foo_source'
      )

if __name__ == '__main__':
  absltest.main()
