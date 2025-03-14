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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
from torax.config import build_runtime_params
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice as runtime_params_slice_lib
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.tests.test_lib import default_sources


class RuntimeParamsSliceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._geo = geometry_pydantic_model.CircularConfig().build_geometry()

  def test_dynamic_slice_can_be_input_to_jitted_function(self):
    """Tests that the slice can be input to a jitted function."""

    def foo(
        runtime_params_slice: runtime_params_slice_lib.DynamicRuntimeParamsSlice,
    ):
      _ = runtime_params_slice  # do nothing.

    foo_jitted = jax.jit(foo)
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    dynamic_slice = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        torax_mesh=self._geo.torax_mesh,
    )(
        t=runtime_params.numerics.t_initial,
    )
    # Make sure you can call the function with dynamic_slice as an arg.
    foo_jitted(dynamic_slice)

  def test_static_runtime_params_slice_hash_same_for_same_params(self):
    """Tests that the hash is the same for the same static params."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    sources = default_sources.get_default_sources()
    static_slice1 = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=self._geo.torax_mesh,
    )
    static_slice2 = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=self._geo.torax_mesh,
    )
    self.assertEqual(hash(static_slice1), hash(static_slice2))

  def test_static_runtime_params_slice_hash_different_for_different_params(
      self,
  ):
    """Test that the hash changes when the static params change."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    sources = default_sources.get_default_sources()
    static_slice1 = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=self._geo.torax_mesh,
    )
    runtime_params_mod = dataclasses.replace(
        runtime_params,
        numerics=dataclasses.replace(
            runtime_params.numerics,
            ion_heat_eq=not runtime_params.numerics.ion_heat_eq,
        ),
    )
    static_slice2 = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params_mod,
        sources=sources,
        torax_mesh=self._geo.torax_mesh,
    )
    self.assertNotEqual(hash(static_slice1), hash(static_slice2))


if __name__ == '__main__':
  absltest.main()
