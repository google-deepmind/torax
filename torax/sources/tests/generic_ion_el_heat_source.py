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

"""Tests for generic_ion_el_heat_source."""

from absl.testing import absltest
from torax.sources import generic_ion_el_heat_source
from torax.sources import source
from torax.sources import source_config
from torax.sources.tests import test_lib


class GenericIonElectronHeatSourceTest(test_lib.IonElSourceTestCase):
  """Tests for GenericIonElectronHeatSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=generic_ion_el_heat_source.GenericIonElectronHeatSource,
        unsupported_types=[
            source_config.SourceType.MODEL_BASED,
        ],
        expected_affected_core_profiles=(
            source.AffectedCoreProfile.TEMP_ION,
            source.AffectedCoreProfile.TEMP_EL,
        ),
    )


if __name__ == '__main__':
  absltest.main()
