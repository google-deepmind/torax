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
from typing import Annotated, Literal, cast

from absl.testing import absltest
from absl.testing import parameterized
import pydantic
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class ModelA(torax_pydantic.BaseModelFrozen):
  model_name: Literal['A'] = 'A'
  param_a: int = 1


class ModelB(torax_pydantic.BaseModelFrozen):
  model_name: Literal['B'] = 'B'
  param_b: str = 'foo'


# This is the pattern used in TORAX for discriminated unions with defaults.
ModelConfig = Annotated[
    ModelA | ModelB,
    pydantic.BeforeValidator(
        torax_pydantic.create_default_model_injector('model_name', 'A')
    ),
]


class Model(torax_pydantic.BaseModelFrozen):
  model: ModelConfig = pydantic.Field(discriminator='model_name')


class CreateDefaultModelInjectorTest(parameterized.TestCase):

  def test_injects_default_model_name_for_empty_dict(self):
    """Tests that the default model is used for an empty dict."""
    model = Model.from_dict({'model': {}})
    self.assertIsInstance(model.model, ModelA)
    # The cast is needed to placate the type checker.
    model_a = cast(ModelA, model.model)
    self.assertEqual(model_a.param_a, 1)

  def test_injects_default_and_preserves_other_params(self):
    """Tests that other parameters are preserved when injecting the default."""
    model = Model.from_dict({'model': {'param_a': 2}})
    self.assertIsInstance(model.model, ModelA)
    model_a = cast(ModelA, model.model)
    self.assertEqual(model_a.param_a, 2)

  def test_does_not_inject_if_model_name_present(self):
    """Tests that an explicit model_name is respected."""
    model = Model.from_dict({'model': {'model_name': 'B', 'param_b': 'bar'}})
    self.assertIsInstance(model.model, ModelB)
    model_b = cast(ModelB, model.model)
    self.assertEqual(model_b.param_b, 'bar')

  def test_does_not_mutate_input_dict(self):
    """Tests that the validator does not mutate the input dictionary."""
    input_dict = {'model': {'param_a': 5}}
    original_input_dict = copy.deepcopy(input_dict)
    Model.from_dict(input_dict)
    self.assertEqual(input_dict, original_input_dict)


if __name__ == '__main__':
  absltest.main()
