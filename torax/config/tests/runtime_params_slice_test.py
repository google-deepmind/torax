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
from absl.testing import parameterized
import jax
from torax.config import build_runtime_params
from torax.config import runtime_params_slice as runtime_params_slice_lib
from torax.tests.test_lib import default_configs
from torax.torax_pydantic import model_config


class RuntimeParamsSliceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._torax_config = model_config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict())
    self._torax_mesh = self._torax_config.geometry.build_provider.torax_mesh

  def test_dynamic_slice_can_be_input_to_jitted_function(self):
    """Tests that the slice can be input to a jitted function."""

    def foo(
        runtime_params_slice: runtime_params_slice_lib.DynamicRuntimeParamsSlice,
    ):
      _ = runtime_params_slice  # do nothing.

    foo_jitted = jax.jit(foo)
    dynamic_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            self._torax_config
        )(
            t=self._torax_config.numerics.t_initial,
        )
    )
    # Make sure you can call the function with dynamic_slice as an arg.
    foo_jitted(dynamic_slice)

  def test_static_runtime_params_slice_hash_same_for_same_params(self):
    """Tests that the hash is the same for the same static params."""
    static_slice1 = build_runtime_params.build_static_params_from_config(
        self._torax_config
    )
    static_slice2 = build_runtime_params.build_static_params_from_config(
        self._torax_config
    )
    self.assertEqual(hash(static_slice1), hash(static_slice2))

  def test_static_runtime_params_slice_hash_different_for_different_params(
      self,
  ):
    """Test that the hash changes when the static params change."""
    static_slice1 = build_runtime_params.build_static_params_from_config(
        self._torax_config
    )
    new_config = default_configs.get_default_config_dict()
    new_config['numerics']['evolve_ion_heat'] = (
        not self._torax_config.numerics.evolve_ion_heat
    )
    new_torax_config = model_config.ToraxConfig.from_dict(new_config)
    static_slice2 = build_runtime_params.build_static_params_from_config(
        new_torax_config
    )
    self.assertNotEqual(hash(static_slice1), hash(static_slice2))


if __name__ == '__main__':
  absltest.main()
