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

import dataclasses
from typing import ClassVar
from absl.testing import absltest
from absl.testing import parameterized
import immutabledict
from torax._src.neoclassical import config as neoclassical_config
from torax._src.sources import config as sources_config
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.test_utils import default_sources


class SourceModelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.default_sources_config = (
        sources_config.SourcesConfig.from_dict(
            default_sources.get_default_source_config()
        )
    )
    self.default_neoclassical_config = (
        neoclassical_config.AnalyticalNeoclassicalConfig.from_dict({})
    )
    self.default_source_model = self.default_sources_config.build_models()

  def test_equal_hash_and_eq_same_config(self):

    test_source_model = self.default_sources_config.build_models()

    self.assertEqual(self.default_source_model, test_source_model)
    self.assertEqual(hash(self.default_source_model), hash(test_source_model))

  def test_equal_hash_and_eq_same_sources_but_different_param(self):

    modified_sources_dict = default_sources.get_default_source_config()
    modified_sources_dict['generic_heat']['P_total'] = 10e6
    modified_sources_config = sources_config.SourcesConfig.from_dict(
        modified_sources_dict
    )
    test_source_model = modified_sources_config.build_models()

    self.assertEqual(self.default_source_model, test_source_model)
    self.assertEqual(hash(self.default_source_model), hash(test_source_model))

  def test_hash_and_eq_different_with_different_standard_source(self):

    modified_sources_dict = default_sources.get_default_source_config().pop(
        'fusion'
    )
    modified_sources_config = sources_config.SourcesConfig.from_dict(
        modified_sources_dict
    )
    test_source_model = modified_sources_config.build_models()

    self.assertNotEqual(self.default_source_model, test_source_model)
    self.assertNotEqual(
        hash(self.default_source_model), hash(test_source_model)
    )

  def test_basic_sources(self):
    sources_dict = default_sources.get_default_source_config()
    sources_dict['ecrh'] = {}
    sources_dict['icrh'] = {
        'model_name': 'scaled_profile',
        'minority_species': 'He3',
    }
    sources_dict['pellet'] = {}
    sources_dict['impurity_radiation'] = {
        'model_name': 'P_in_scaled_flat_profile',
    }
    configured_models = sources_config.SourcesConfig.from_dict(
        sources_dict
    ).build_models()

    basic = configured_models.basic_sources
    self.assertIn('generic_current', basic)
    self.assertIn('ecrh', basic)
    self.assertIn('icrh', basic)
    self.assertIn('fusion', basic)
    self.assertIn('pellet', basic)
    self.assertIn('gas_puff', basic)
    self.assertIn('bremsstrahlung', basic)

    # Core profiles and final sources should not be in basic_sources
    self.assertNotIn('ohmic', basic)
    self.assertNotIn('impurity_radiation', basic)

  def test_source_partitioning(self):
    sources_dict = default_sources.get_default_source_config()
    sources_dict['ecrh'] = {}
    sources_dict['icrh'] = {
        'model_name': 'scaled_profile',
        'minority_species': 'He3',
    }
    sources_dict['pellet'] = {}
    sources_dict['impurity_radiation'] = {
        'model_name': 'P_in_scaled_flat_profile',
    }
    configured_models = sources_config.SourcesConfig.from_dict(
        sources_dict
    ).build_models()

    basic = configured_models.basic_sources
    core_profiles_sources = configured_models.core_profiles_sources
    final_sources = configured_models.final_sources

    self.assertTrue(set(basic.keys()).isdisjoint(core_profiles_sources.keys()))
    self.assertTrue(set(basic.keys()).isdisjoint(final_sources.keys()))
    self.assertTrue(
        set(core_profiles_sources.keys()).isdisjoint(final_sources.keys())
    )

    self.assertEqual(
        set(basic.keys())
        | set(core_profiles_sources.keys())
        | set(final_sources.keys()),
        set(configured_models.standard_sources.keys()),
    )

    self.assertIn('ohmic', core_profiles_sources)
    self.assertIn('impurity_radiation', final_sources)
    self.assertNotIn('ohmic', basic)
    self.assertNotIn('impurity_radiation', basic)

  def test_mavrin_fit_partitioned_into_basic_sources(self):
    sources_dict = default_sources.get_default_source_config()
    sources_dict['impurity_radiation'] = {
        'model_name': 'mavrin_fit',
    }
    configured_models = sources_config.SourcesConfig.from_dict(
        sources_dict
    ).build_models()

    self.assertIn('impurity_radiation', configured_models.basic_sources)
    self.assertNotIn('impurity_radiation', configured_models.final_sources)

  def test_multiple_final_sources_raises_error(self):
    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class DummyFinalSource(source_lib.FinalSource):
      SOURCE_ID: ClassVar[str] = 'final1'
      AFFECTED_CORE_PROFILES: ClassVar[
          tuple[source_lib.AffectedCoreProfile, ...]
      ] = (source_lib.AffectedCoreProfile.TEMP_EL,)

      def _get_model_value(self, *args, **kwargs):
        raise NotImplementedError

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class DummyFinalSource2(source_lib.FinalSource):
      SOURCE_ID: ClassVar[str] = 'final2'
      AFFECTED_CORE_PROFILES: ClassVar[
          tuple[source_lib.AffectedCoreProfile, ...]
      ] = (source_lib.AffectedCoreProfile.TEMP_ION,)

      def _get_model_value(self, *args, **kwargs):
        raise NotImplementedError

    with self.assertRaisesRegex(
        ValueError, 'At most one FinalSource may be configured in SourceModels'
    ):
      source_models_lib.SourceModels(
          qei_source=qei_source_lib.QeiSource(),
          standard_sources=immutabledict.immutabledict({
              'f1': DummyFinalSource(),
              'f2': DummyFinalSource2(),
          }),
      )


if __name__ == '__main__':
  absltest.main()
