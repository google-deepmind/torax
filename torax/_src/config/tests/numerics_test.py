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
import chex
import jax
from torax._src import jax_utils
from torax._src.config import numerics
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.torax_pydantic import torax_pydantic


class NumericsTest(parameterized.TestCase):

  def test_numerics_build_dynamic_params(self):
    nums = numerics.Numerics()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    torax_pydantic.set_grid(nums, geo.torax_mesh)
    nums.build_runtime_params(t=0.0)

  def test_numerics_under_jit(self):
    initial_resistivity_multiplier = 1.0
    updated_resistivity_multiplier = 2.0
    nums = numerics.Numerics(
        resistivity_multiplier=initial_resistivity_multiplier
    )
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    torax_pydantic.set_grid(nums, geo.torax_mesh)

    @jax.jit
    def f(numerics_model: numerics.Numerics):
      return numerics_model.build_runtime_params(t=0.0)

    with self.subTest('first_jit_compiles_and_returns_expected_value'):
      output = f(nums)
      chex.assert_trees_all_close(
          output.resistivity_multiplier, initial_resistivity_multiplier
      )
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    with self.subTest('second_jit_updates_value_without_recompile'):
      nums._update_fields(
          {'resistivity_multiplier': updated_resistivity_multiplier}
      )
      output = f(nums)
      chex.assert_trees_all_close(
          output.resistivity_multiplier, updated_resistivity_multiplier
      )
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)


if __name__ == '__main__':
  absltest.main()
