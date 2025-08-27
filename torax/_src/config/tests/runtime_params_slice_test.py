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
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params_slice as runtime_params_slice_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


class RuntimeParamsSliceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._torax_config = model_config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    self._torax_mesh = self._torax_config.geometry.build_provider.torax_mesh

  def test_dynamic_slice_can_be_input_to_jitted_function(self):
    """Tests that the slice can be input to a jitted function."""

    def foo(
        runtime_params_slice: runtime_params_slice_lib.RuntimeParams,
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


if __name__ == '__main__':
  absltest.main()
