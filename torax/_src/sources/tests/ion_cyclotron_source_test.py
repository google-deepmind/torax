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
"""Tests for global ion_cyclotron_source infrastructure (model-agnostic)."""

import dataclasses

from absl.testing import absltest
import jax
import numpy as np
from torax._src.sources import source as source_lib
from torax._src.sources import source_profiles
from torax._src.sources import source_state as source_state_lib
from torax._src.sources.ion_cyclotron_source import base as icrh_base
from torax._src.sources.tests import test_lib

# Most of the checks and computations in TORAX require float64.
jax.config.update('jax_enable_x64', True)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class _DummySource(icrh_base.IonCyclotronSource):

  def _get_model_value(self, *args, **kwargs):
    return (
        source_profiles.SourceOutput(profiles=()),
        source_state_lib.SourceState(),
    )


class _DummyConfig(icrh_base.IonCyclotronSourceConfig):

  def build_source(self) -> _DummySource:
    return _DummySource()


class IonCyclotronSourceBaseTest(test_lib.SourceTestCase):
  """Tests for model-agnostic IonCyclotronSource infrastructure."""

  source_name = icrh_base.IonCyclotronSource.SOURCE_ID
  source_config_class = _DummyConfig
  extra_source_config_kwargs = {'minority_species': 'He3'}

  def test_source_name(self):
    self.assertEqual(icrh_base.IonCyclotronSource.SOURCE_ID, 'icrh')

  def test_affected_core_profiles(self):
    expected = (
        source_lib.AffectedCoreProfile.TEMP_ION,
        source_lib.AffectedCoreProfile.TEMP_EL,
    )
    self.assertEqual(
        icrh_base.IonCyclotronSource.AFFECTED_CORE_PROFILES, expected
    )

  def test_build_source_returns_correct_type(self):
    """IonCyclotronSourceConfig.build_source returns IonCyclotronSource."""
    config = _DummyConfig(minority_species='He3')
    source = config.build_source()
    self.assertIsInstance(source, icrh_base.IonCyclotronSource)

  def test_config_default_values(self):
    """Verify default config values are correct."""
    config = _DummyConfig(minority_species='He3')
    np.testing.assert_allclose(config.P_total.get_value(0.0), 10e6)
    np.testing.assert_allclose(
        config.absorption_fraction.get_value(0.0), 1.0
    )


if __name__ == '__main__':
  absltest.main()
