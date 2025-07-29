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
from torax._src import jax_utils
from torax._src.pedestal_model import pydantic_model


class NoPedestalTest(parameterized.TestCase):

  @parameterized.parameters(
      pydantic_model.SetPpedTpedRatioNped,
      pydantic_model.SetTpedNped,
      pydantic_model.NoPedestal,
  )
  def test_build_and_call_model(
      self, pydantic_model_class: type[pydantic_model.BasePedestal]
  ):
    pedestal_model = pydantic_model_class.from_dict({})

    @jax.jit
    def f(x: pydantic_model.BasePedestal):
      return x.build_dynamic_params(t=0.0)

    with self.subTest('first_jit_compiles_and_returns_expected_value'):
      output = f(pedestal_model)
      self.assertIsInstance(
          output, pydantic_model.runtime_params.DynamicRuntimeParams
      )
      self.assertFalse(output.set_pedestal)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    with self.subTest('second_jit_updates_value_without_recompile'):
      pedestal_model._update_fields({'set_pedestal': True})
      output = f(pedestal_model)
      self.assertTrue(output.set_pedestal)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)


if __name__ == '__main__':
  absltest.main()
