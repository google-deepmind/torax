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

"""Tests for electron_cyclotron_source."""

from absl.testing import absltest

from torax import geometry
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import electron_cyclotron_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources.tests import test_lib


class ElectronCyclotronSourceTest(test_lib.SourceTestCase):
  """Tests for ElectronCyclotronSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=electron_cyclotron_source.ElectronCyclotronSource,
        source_class_builder=electron_cyclotron_source.ElectronCyclotronSourceBuilder,
        unsupported_modes=[],
        expected_affected_core_profiles=(
            source_lib.AffectedCoreProfile.TEMP_EL,
            source_lib.AffectedCoreProfile.PSI,),
    )


if __name__ == '__main__':
  absltest.main()
