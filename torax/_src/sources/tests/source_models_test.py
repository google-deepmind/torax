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

from torax._src.neoclassical import \
    pydantic_model as neoclassical_pydantic_model
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.test_utils import default_sources


class SourceModelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.default_sources_config = sources_pydantic_model.Sources.from_dict(
        default_sources.get_default_source_config()
    )
    self.default_neoclassical_config = (
        neoclassical_pydantic_model.Neoclassical.from_dict({})
    )
    self.default_source_model = self.default_sources_config.build_models()

  def test_equal_hash_and_eq_same_config(self):

    test_source_model = self.default_sources_config.build_models()

    self.assertEqual(self.default_source_model, test_source_model)
    self.assertEqual(hash(self.default_source_model), hash(test_source_model))

  def test_equal_hash_and_eq_same_sources_but_different_param(self):

    modified_sources_dict = default_sources.get_default_source_config()
    modified_sources_dict['generic_heat']['P_total'] = 10e6
    modified_sources_config = sources_pydantic_model.Sources.from_dict(
        modified_sources_dict
    )
    test_source_model = modified_sources_config.build_models()

    self.assertEqual(self.default_source_model, test_source_model)
    self.assertEqual(hash(self.default_source_model), hash(test_source_model))

  def test_hash_and_eq_different_with_different_standard_source(self):

    modified_sources_dict = default_sources.get_default_source_config().pop(
        'fusion'
    )
    modified_sources_config = sources_pydantic_model.Sources.from_dict(
        modified_sources_dict
    )
    test_source_model = modified_sources_config.build_models()

    self.assertNotEqual(self.default_source_model, test_source_model)
    self.assertNotEqual(
        hash(self.default_source_model), hash(test_source_model)
    )


if __name__ == '__main__':
  absltest.main()
