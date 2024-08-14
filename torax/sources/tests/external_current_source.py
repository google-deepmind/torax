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
from torax import geometry
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import external_current_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources.tests import test_lib


class ExternalCurrentSourceTest(test_lib.SourceTestCase):
  """Tests for ExternalCurrentSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=external_current_source.ExternalCurrentSource,
        source_class_builder=external_current_source.ExternalCurrentSourceBuilder,
        unsupported_modes=[
            runtime_params_lib.Mode.MODEL_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.PSI,),
    )

  def test_source_value(self):
    """Tests that a formula-based source provides values."""
    source_builder = external_current_source.ExternalCurrentSourceBuilder()
    source = source_builder()
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    # Must be circular for jext_hires call.
    geo = geometry.build_circular_geometry()
    runtime_params_provider = runtime_params.make_provider(geo.torax_mesh)
    dynamic_slice = runtime_params_slice.build_dynamic_runtime_params_slice(
        runtime_params_provider,
        sources={
            'jext': source_builder.runtime_params,
        },
        geo=geo,
    )
    self.assertIsInstance(source, external_current_source.ExternalCurrentSource)

    self.assertIsNotNone(
        source.get_value(
            dynamic_runtime_params_slice=dynamic_slice,
            dynamic_source_runtime_params=dynamic_slice.sources['jext'],
            geo=geo,
        )
    )
    self.assertIsNotNone(
        source.jext_hires(
            dynamic_runtime_params_slice=dynamic_slice,
            dynamic_source_runtime_params=dynamic_slice.sources['jext'],
            geo=geo,
        )
    )

  def test_invalid_source_types_raise_errors(self):
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    source_builder = external_current_source.ExternalCurrentSourceBuilder()
    source = source_builder()
    for unsupported_mode in self._unsupported_modes:
      with self.subTest(unsupported_mode.name):
        with self.assertRaises(jax.lib.xla_client.XlaRuntimeError):
          source_builder.runtime_params.mode = unsupported_mode
          runtime_params_provider = runtime_params.make_provider(geo.torax_mesh)
          dynamic_slice = (
              runtime_params_slice.build_dynamic_runtime_params_slice(
                  runtime_params_provider,
                  sources={
                      'jext': source_builder.runtime_params,
                  },
                  geo=geo,
              )
          )
          source.get_value(
              dynamic_runtime_params_slice=dynamic_slice,
              dynamic_source_runtime_params=dynamic_slice.sources['jext'],
              geo=geo,
          )

  def test_profile_is_on_face_grid(self):
    """Tests that the profile is given on the face grid."""
    geo = geometry.build_circular_geometry()
    source_builder = external_current_source.ExternalCurrentSourceBuilder()
    source = source_builder()
    self.assertEqual(
        source.output_shape_getter(geo),
        source_lib.ProfileType.FACE.get_profile_shape(geo),
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    runtime_params_provider = runtime_params.make_provider(geo.torax_mesh)
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params_provider,
            sources={
                'jext': source_builder.runtime_params,
            },
            geo=geo,
        )
    )
    self.assertEqual(
        source.get_value(
            dynamic_runtime_params_slice,
            dynamic_runtime_params_slice.sources['jext'],
            geo,
            core_profiles=None,
        ).shape,
        source_lib.ProfileType.FACE.get_profile_shape(geo),
    )


if __name__ == '__main__':
  absltest.main()
