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
from torax import geometry
from torax.sources import electron_density_sources as eds
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources.tests import test_lib


class GasPuffSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for GasPuffSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=eds.GasPuffSource,
        runtime_params_class=eds.GasPuffRuntimeParams,
        unsupported_modes=[
            runtime_params_lib.Mode.MODEL_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.NE,),
    )

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = eds.GasPuffRuntimeParams()
    geo = geometry.build_circular_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)


class PelletSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for PelletSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=eds.PelletSource,
        runtime_params_class=eds.PelletRuntimeParams,
        unsupported_modes=[
            runtime_params_lib.Mode.MODEL_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.NE,),
    )

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = eds.PelletRuntimeParams()
    geo = geometry.build_circular_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)


class SourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for GenericParticleSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=eds.GenericParticleSource,
        runtime_params_class=eds.GenericParticleSourceRuntimeParams,
        unsupported_modes=[
            runtime_params_lib.Mode.MODEL_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.NE,),
    )

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = eds.GenericParticleSourceRuntimeParams()
    geo = geometry.build_circular_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)


class RecombinationDensitySinkTest(test_lib.SingleProfileSourceTestCase):
  """Tests for RecombinationDensitySink."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=eds.RecombinationDensitySink,
        runtime_params_class=runtime_params_lib.RuntimeParams,
        unsupported_modes=[
            runtime_params_lib.Mode.MODEL_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.NE,),
    )


if __name__ == '__main__':
  absltest.main()
