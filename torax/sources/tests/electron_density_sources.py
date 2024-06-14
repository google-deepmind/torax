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

"""Tests for electron_density_sources."""

from absl.testing import absltest
from torax.sources import electron_density_sources as eds
from torax.sources import runtime_params
from torax.sources import source as source_lib
from torax.sources.tests import test_lib


class GasPuffSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for GasPuffSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=eds.GasPuffSource,
        source_class_builder=eds.GasPuffSourceBuilder,
        unsupported_modes=[
            runtime_params.Mode.MODEL_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.NE,),
    )


class PelletSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for PelletSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=eds.PelletSource,
        source_class_builder=eds.PelletSourceBuilder,
        unsupported_modes=[
            runtime_params.Mode.MODEL_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.NE,),
    )


class NBISourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for NBISource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=eds.NBIParticleSource,
        source_class_builder=eds.NBIParticleSourceBuilder,
        unsupported_modes=[
            runtime_params.Mode.MODEL_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.NE,),
    )


class RecombinationDensitySinkTest(test_lib.SingleProfileSourceTestCase):
  """Tests for RecombinationDensitySink."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=eds.RecombinationDensitySink,
        source_class_builder=eds.RecombinationDensitySinkBuilder,
        unsupported_modes=[
            runtime_params.Mode.MODEL_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.NE,),
    )


if __name__ == '__main__':
  absltest.main()
