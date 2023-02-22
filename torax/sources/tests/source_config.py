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

"""Tests for source_config.py."""

from absl.testing import absltest
from absl.testing import parameterized
from torax import config as config_lib
from torax.sources import source_config
from torax.sources import source_profiles


class SourceConfigTest(parameterized.TestCase):
  """Tests for SourceConfig and related functions."""

  def test_source_config_keys_match_default_sources(self):
    """Makes sure the source configs always have the default sources."""
    config = config_lib.Config()
    sources = source_profiles.Sources()
    self.assertSameElements(config.sources.keys(), sources.all_sources.keys())

    # Try overriding some elements.
    config = config_lib.Config(
        sources=dict(
            gas_puff_source=source_config.SourceConfig(
                source_type=source_config.SourceType.ZERO
            ),
            nbi_particle_source=source_config.SourceConfig(
                source_type=source_config.SourceType.ZERO
            ),
        )
    )
    # Still should have all the same keys because Config should add back the
    # defaults.
    self.assertSameElements(config.sources.keys(), sources.all_sources.keys())


if __name__ == '__main__':
  absltest.main()
