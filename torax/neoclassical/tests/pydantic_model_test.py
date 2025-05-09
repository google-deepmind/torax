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
from torax.neoclassical import pydantic_model


class PydanticModelTest(parameterized.TestCase):

  def test_default_model_from_dict(self):
    model = pydantic_model.Neoclassical.from_dict({"bootstrap_current": {}})
    self.assertEqual(model.bootstrap_current.model_name, "zeros")

  @parameterized.parameters("zeros", "sauter")
  def test_model_name(self, model_name):
    model = pydantic_model.Neoclassical.from_dict(
        {"bootstrap_current": {"model_name": model_name}}
    )
    self.assertEqual(model.bootstrap_current.model_name, model_name)


if __name__ == "__main__":
  absltest.main()
