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

"""Tests for external_current_source."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from torax import config as config_lib
from torax import config_slice
from torax import geometry
from torax.sources import external_current_source
from torax.sources import source as source_lib
from torax.sources import source_config
from torax.sources.tests import test_lib


class ExternalCurrentSourceTest(test_lib.SourceTestCase):
  """Tests for ExternalCurrentSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=external_current_source.ExternalCurrentSource,
        unsupported_types=[
            source_config.SourceType.MODEL_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.PSI,),
    )

  def test_source_value(self):
    """Tests that a formula-based source provides values."""
    source = external_current_source.ExternalCurrentSource()
    config = config_lib.Config()
    dynamic_slice = config_slice.build_dynamic_config_slice(config)
    self.assertIsInstance(source, external_current_source.ExternalCurrentSource)
    # Must be circular for jext_hires call.
    geo = geometry.build_circular_geometry(config)
    self.assertIsNotNone(
        source.get_value(
            source_type=dynamic_slice.sources[source.name].source_type,
            dynamic_config_slice=dynamic_slice,
            geo=geo,
        )
    )
    self.assertIsNotNone(
        source.jext_hires(
            source_type=dynamic_slice.sources[source.name].source_type,
            dynamic_config_slice=dynamic_slice,
            geo=geo,
        )
    )

  def test_invalid_source_types_raise_errors(self):
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    source = external_current_source.ExternalCurrentSource()
    dynamic_slice = config_slice.build_dynamic_config_slice(config)
    for unsupported_type in self._unsupported_types:
      with self.subTest(unsupported_type.name):
        with self.assertRaises(jax.interpreters.xla.xe.XlaRuntimeError):
          source.get_value(
              source_type=unsupported_type.value,
              dynamic_config_slice=dynamic_slice,
              geo=geo,
          )

  def test_extraction_of_relevant_profile_from_output(self):
    """Tests that the relevant profile is extracted from the output."""
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    source = external_current_source.ExternalCurrentSource()
    cell = source_lib.ProfileType.CELL.get_profile_shape(geo)
    fake_profile = (jnp.ones(cell), jnp.zeros(cell))
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.PSI.value,
            geo,
        ),
        jnp.ones(cell),
    )
    # For unrelated states, this should just return all 0s.
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.TEMP_ION.value,
            geo,
        ),
        jnp.zeros(cell),
    )


if __name__ == '__main__':
  absltest.main()
