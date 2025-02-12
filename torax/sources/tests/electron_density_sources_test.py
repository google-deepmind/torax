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
from torax.sources import electron_density_sources as eds
from torax.sources.tests import test_lib


class GasPuffSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for GasPuffSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=eds.GasPuffSource,
        runtime_params_class=eds.GasPuffRuntimeParams,
        source_name=eds.GasPuffSource.SOURCE_NAME,
        model_func=eds.calc_puff_source,
    )


class PelletSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for PelletSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=eds.PelletSource,
        runtime_params_class=eds.PelletRuntimeParams,
        source_name=eds.PelletSource.SOURCE_NAME,
        model_func=eds.calc_pellet_source,
    )


class SourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for GenericParticleSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=eds.GenericParticleSource,
        runtime_params_class=eds.GenericParticleSourceRuntimeParams,
        source_name=eds.GenericParticleSource.SOURCE_NAME,
        model_func=eds.calc_generic_particle_source,
    )


if __name__ == '__main__':
  absltest.main()
