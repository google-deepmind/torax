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

"""Tests for generic_current_source."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np
from torax import geometry
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import generic_current_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources.tests import test_lib


class GenericCurrentSourceTest(test_lib.SourceTestCase):
  """Tests for GenericCurrentSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=generic_current_source.GenericCurrentSource,
        runtime_params_class=generic_current_source.RuntimeParams,
        unsupported_modes=[
            runtime_params_lib.Mode.MODEL_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.PSI,),
    )

  def test_generic_current_hires(self):
    """Tests that a formula-based source provides values when called."""
    source_builder = self._source_class_builder()
    source = source_builder()
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    # Must be circular for generic_current_hires call.
    geo = geometry.build_circular_geometry()
    dynamic_slice = runtime_params_slice.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        sources={
            generic_current_source.SOURCE_NAME: source_builder.runtime_params,
        },
        torax_mesh=geo.torax_mesh,
    )(
        t=runtime_params.numerics.t_initial,
    )
    self.assertIsInstance(source, generic_current_source.GenericCurrentSource)
    self.assertIsNotNone(
        source.generic_current_source_hires(
            dynamic_runtime_params_slice=dynamic_slice,
            dynamic_source_runtime_params=dynamic_slice.sources[
                generic_current_source.SOURCE_NAME
            ],
            geo=geo,
        )
    )

  def test_profile_is_on_face_grid(self):
    """Tests that the profile is given on the face grid."""
    geo = geometry.build_circular_geometry()
    source_builder = self._source_class_builder()
    source = source_builder()
    self.assertEqual(
        source.output_shape_getter(geo),
        source_lib.ProfileType.FACE.get_profile_shape(geo),
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    dynamic_runtime_params_slice = runtime_params_slice.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        sources={
            generic_current_source.SOURCE_NAME: source_builder.runtime_params,
        },
        torax_mesh=geo.torax_mesh,
    )(
        t=runtime_params.numerics.t_initial,
    )
    self.assertEqual(
        source.get_value(
            dynamic_runtime_params_slice,
            dynamic_runtime_params_slice.sources[
                generic_current_source.SOURCE_NAME
            ],
            geo,
            core_profiles=None,
        ).shape,
        source_lib.ProfileType.FACE.get_profile_shape(geo),
    )

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = generic_current_source.RuntimeParams()
    geo = geometry.build_circular_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)

  @parameterized.named_parameters(
      dict(
          testcase_name='psi_profile_yields_profile',
          affected_core_profile=source_lib.AffectedCoreProfile.PSI,
          expected_profile=np.array([1.5, 2.5]),
      ),
      dict(
          testcase_name='unaffected_profile_yields_zeros',
          affected_core_profile=source_lib.AffectedCoreProfile.TEMP_ION,
          expected_profile=np.array([0.0, 0.0]),
      ),
  )
  def test_get_source_profile_for_affected_core_profile_with(
      self,
      affected_core_profile: source_lib.AffectedCoreProfile,
      expected_profile: chex.Array,
  ):
    source_builder = self._source_class_builder()
    source = source_builder()

    # Build a face profile with 3 values on a 2-cell grid.
    geo = geometry.build_circular_geometry(n_rho=2)
    face_profile = np.array([1, 2, 3])

    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            face_profile,
            affected_core_profile.value,
            geo,
        ),
        expected_profile,
    )


if __name__ == '__main__':
  absltest.main()
