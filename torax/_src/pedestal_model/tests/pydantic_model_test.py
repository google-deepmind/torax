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
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib


class PedestalModelPydanticTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='set_pped_tpedratio_nped',
          pydantic_model_class=pydantic_model.SetPpedTpedRatioNped,
          unsupported_mode=pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT,
      ),
      dict(
          testcase_name='set_tped_nped',
          pydantic_model_class=pydantic_model.SetTpedNped,
          unsupported_mode=pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT,
      ),
      dict(
          testcase_name='dynamic_pedestal',
          pydantic_model_class=pydantic_model.DynamicPedestal,
          unsupported_mode=pedestal_runtime_params_lib.Mode.ADAPTIVE_SOURCE,
      ),
  )
  def test_unsupported_mode_raises_error(
      self,
      pydantic_model_class: type[pydantic_model.BasePedestal],
      unsupported_mode: pedestal_runtime_params_lib.Mode,
  ):
    with self.assertRaises(ValueError):
      pydantic_model_class.from_dict(
          {'mode': unsupported_mode, 'set_pedestal': True}
      )

  @parameterized.parameters(
      (
          pydantic_model.SetPpedTpedRatioNped,
          pedestal_runtime_params_lib.Mode.ADAPTIVE_SOURCE,
      ),
      (
          pydantic_model.SetTpedNped,
          pedestal_runtime_params_lib.Mode.ADAPTIVE_SOURCE,
      ),
      (
          pydantic_model.NoPedestal,
          pedestal_runtime_params_lib.Mode.ADAPTIVE_SOURCE,
      ),
      (
          pydantic_model.NoPedestal,
          pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT,
      ),
      (
          pydantic_model.DynamicPedestal,
          pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT,
      ),
  )
  def test_set_pedestal_does_not_trigger_recompile(
      self,
      pydantic_model_class: type[pydantic_model.BasePedestal],
      mode: pedestal_runtime_params_lib.Mode,
  ):
    pedestal_model = pydantic_model_class.from_dict({'mode': mode})

    @jax.jit
    def f(x: pydantic_model.BasePedestal):
      return x.build_runtime_params(t=0.0)

    with self.subTest('first_jit_compiles_and_returns_expected_value'):
      output = f(pedestal_model)
      self.assertIsInstance(output, pydantic_model.runtime_params.RuntimeParams)
      self.assertFalse(output.set_pedestal)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    with self.subTest('second_jit_updates_value_without_recompile'):
      pedestal_model._update_fields({'set_pedestal': True})
      output = f(pedestal_model)
      self.assertTrue(output.set_pedestal)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)


if __name__ == '__main__':
  absltest.main()
