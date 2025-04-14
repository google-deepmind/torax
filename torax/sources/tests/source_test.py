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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import electron_cyclotron_source
from torax.sources import generic_current_source
from torax.sources import runtime_params as runtime_params_lib


class SourceTest(parameterized.TestCase):
  """Tests for the base class Source."""

  def test_zero_profile_works_by_default(self):
    """The default source impl should support profiles with all zeros."""
    source = generic_current_source.GenericCurrentSource()
    geo = mock.create_autospec(geometry.Geometry,
                               rho_norm=np.array([1, 1, 1, 1]))
    dynamic_source_params = {
        generic_current_source.GenericCurrentSource.SOURCE_NAME: (
            runtime_params_lib.DynamicRuntimeParams(
                prescribed_values=np.zeros_like(geo.rho_norm),
            )
        )
    }
    static_source_params = {
        generic_current_source.GenericCurrentSource.SOURCE_NAME: (
            runtime_params_lib.StaticRuntimeParams(
                mode=runtime_params_lib.Mode.ZERO.value,
                is_explicit=False,
            )
        )
    }
    static_slice = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources=static_source_params,
    )
    dynamic_slice = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        sources=dynamic_source_params,
    )
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=mock.ANY,
        calculated_source_profiles=None,
    )
    np.testing.assert_allclose(profile[0], np.zeros_like(geo.rho_norm))

  @parameterized.parameters(
      (runtime_params_lib.Mode.ZERO, np.array([0, 0, 0, 0])),
      (
          runtime_params_lib.Mode.MODEL_BASED,
          np.array([42, 42, 42, 42]),
      ),
      (runtime_params_lib.Mode.PRESCRIBED, np.array([3, 3, 3, 3])),
  )
  def test_correct_mode_called(
      self,
      mode,
      expected_profile,
  ):
    model_func = mock.MagicMock()
    model_func.return_value = np.full([4], 42.)
    source = generic_current_source.GenericCurrentSource(model_func=model_func)
    dynamic_source_params = {
        generic_current_source.GenericCurrentSource.SOURCE_NAME: (
            runtime_params_lib.DynamicRuntimeParams(
                prescribed_values=(np.full([4], 3.),),
            )
        )
    }
    static_source_params = {
        generic_current_source.GenericCurrentSource.SOURCE_NAME: (
            runtime_params_lib.StaticRuntimeParams(
                mode=mode.value,
                is_explicit=False,
            )
        )
    }
    static_slice = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources=static_source_params,
    )
    dynamic_slice = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        sources=dynamic_source_params,
    )
    # Make a geo with rho_norm as we need it for the zero profile shape.
    geo = mock.create_autospec(geometry.Geometry,
                               rho_norm=np.array([1, 1, 1, 1]))
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=mock.ANY,
        calculated_source_profiles=None,
    )
    np.testing.assert_allclose(
        profile[0],
        expected_profile,
        atol=1e-6,
        rtol=1e-6,
    )

  def test_prescribed_values_for_multiple_affected_profiles(self):
    source = electron_cyclotron_source.ElectronCyclotronSource()
    dynamic_source_params = {
        electron_cyclotron_source.ElectronCyclotronSource.SOURCE_NAME: (
            runtime_params_lib.DynamicRuntimeParams(
                prescribed_values=(np.full([4], 3.), np.full([4], 4.)),
            )
        )
    }
    static_source_params = {
        electron_cyclotron_source.ElectronCyclotronSource.SOURCE_NAME: (
            runtime_params_lib.StaticRuntimeParams(
                mode=runtime_params_lib.Mode.PRESCRIBED.value,
                is_explicit=False,
            )
        )
    }
    static_slice = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources=static_source_params,
    )
    dynamic_slice = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        sources=dynamic_source_params,
    )
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_slice,
        static_runtime_params_slice=static_slice,
        geo=mock.ANY,
        core_profiles=mock.ANY,
        calculated_source_profiles=None,
    )
    self.assertLen(profile, 2)
    np.testing.assert_allclose(
        profile[0],
        np.full([4], 3.),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        profile[1],
        np.full([4], 4.),
        atol=1e-6,
        rtol=1e-6,
    )

  def test_source_with_mismatched_prescribed_values_raises_error(self):
    source = electron_cyclotron_source.ElectronCyclotronSource()
    dynamic_source_params = {
        electron_cyclotron_source.ElectronCyclotronSource.SOURCE_NAME: (
            runtime_params_lib.DynamicRuntimeParams(
                prescribed_values=(np.full([4], 3.),),
            )
        )
    }
    static_source_params = {
        electron_cyclotron_source.ElectronCyclotronSource.SOURCE_NAME:
            runtime_params_lib.StaticRuntimeParams(
                mode=runtime_params_lib.Mode.PRESCRIBED.value,
                is_explicit=False,
            )
    }
    static_slice = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources=static_source_params,
    )
    dynamic_slice = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        sources=dynamic_source_params,
    )
    with self.assertRaisesRegex(
        ValueError,
        'the number of prescribed values must match the number of affected',
    ):
      source.get_value(
          dynamic_runtime_params_slice=dynamic_slice,
          static_runtime_params_slice=static_slice,
          geo=mock.ANY,
          core_profiles=mock.ANY,
          calculated_source_profiles=None,
      )

if __name__ == '__main__':
  absltest.main()
