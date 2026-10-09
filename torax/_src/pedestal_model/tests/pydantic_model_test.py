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
import jax
import pydantic
from torax._src import jax_utils
from torax._src.pedestal_model import pydantic_model
from torax._src.pedestal_model import runtime_params
from torax._src.pedestal_model.formation import prescribed_formation_model


class PedestalModelPydanticTest(parameterized.TestCase):

  @parameterized.parameters(
      pydantic_model.SetPpedTpedRatioNped,
      pydantic_model.SetTpedNped,
  )
  def test_build_and_call_model(
      self, pydantic_model_class: type[pydantic_model.BasePedestal]
  ):
    pedestal_model = pydantic_model_class.from_dict({})

    @jax.jit
    def f(x: pydantic_model.BasePedestal):
      return x.build_runtime_params(t=0.0)

    with self.subTest("first_jit_compiles_and_returns_expected_value"):
      output = f(pedestal_model)
      self.assertIsInstance(output, pydantic_model.runtime_params.RuntimeParams)
      self.assertFalse(output.set_pedestal)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    with self.subTest("second_jit_updates_value_without_recompile"):
      pedestal_model._update_fields({"set_pedestal": True})
      output = f(pedestal_model)
      self.assertTrue(output.set_pedestal)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

  def test_no_pedestal_always_inactive(self):
    pedestal_model = pydantic_model.NoPedestal.from_dict({})
    output = jax.jit(lambda x: x.build_runtime_params(t=0.0))(pedestal_model)
    self.assertFalse(output.set_pedestal)
    self.assertFalse(
        output.use_formation_model_with_internal_boundary_condition
    )
    self.assertIsInstance(
        output.formation,
        prescribed_formation_model.PrescribedFormationRuntimeParams,
    )
    self.assertFalse(output.formation.pedestal_active)

    with self.assertRaises(pydantic.ValidationError):
      pydantic_model.NoPedestal.from_dict({"set_pedestal": True})
    with self.assertRaises(pydantic.ValidationError):
      pydantic_model.NoPedestal.from_dict({
          "formation_model": {
              "model_name": "prescribed",
              "pedestal_active": True,
          },
      })

  def test_source_mode_validation(self):
    with self.subTest("allow_internal_boundary_condition"):
      pydantic_model.SetTpedNped.from_dict(
          {"use_formation_model_with_internal_boundary_condition": True}
      )

    with self.subTest("disallow_adaptive_transport"):
      with self.assertRaisesRegex(
          ValueError,
          "use_formation_model_with_internal_boundary_condition can only be"
          " True when mode is INTERNAL_BOUNDARY_CONDITION",
      ):
        pydantic_model.SetTpedNped.from_dict({
            "use_formation_model_with_internal_boundary_condition": True,
            "mode": runtime_params.Mode.ADAPTIVE_TRANSPORT,
        })

    with self.subTest("allow_prescribed_formation_with_adaptive_transport"):
      pedestal_model = pydantic_model.SetTpedNped.from_dict({
          "mode": runtime_params.Mode.ADAPTIVE_TRANSPORT,
          "formation_model": {
              "model_name": "prescribed",
              "pedestal_active": {0.0: False, 2.0: True},
              "base_multiplier": 1e-4,
          },
      })
      formation_model = pedestal_model.formation_model.build_formation_model()
      params_t0 = pedestal_model.build_runtime_params(t=1.0)
      params_t2 = pedestal_model.build_runtime_params(t=2.5)
      self.assertIsInstance(
          params_t0.formation,
          prescribed_formation_model.PrescribedFormationRuntimeParams,
      )
      self.assertIsInstance(
          params_t2.formation,
          prescribed_formation_model.PrescribedFormationRuntimeParams,
      )
      self.assertFalse(bool(params_t0.formation.pedestal_active))
      self.assertTrue(bool(params_t2.formation.pedestal_active))
      self.assertEqual(params_t0.formation.base_multiplier, 1e-4)

      # Verify PrescribedFormationModel.__call__ and transition conditions.
      dummy_rp_t0 = mock.Mock(pedestal=params_t0)
      dummy_rp_t2 = mock.Mock(pedestal=params_t2)
      mult_t0 = formation_model(
          dummy_rp_t0, mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
      )
      mult_t2 = formation_model(
          dummy_rp_t2, mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
      )
      self.assertEqual(float(mult_t0.chi_e_multiplier), 1.0)
      self.assertAlmostEqual(float(mult_t2.chi_e_multiplier), 1e-4)

      trigger_l_to_h_0, trigger_h_to_l_0 = (
          formation_model.evaluate_transition_conditions(
              dummy_rp_t0, mock.Mock(), mock.Mock(), mock.Mock()
          )
      )
      trigger_l_to_h_2, trigger_h_to_l_2 = (
          formation_model.evaluate_transition_conditions(
              dummy_rp_t2, mock.Mock(), mock.Mock(), mock.Mock()
          )
      )
      self.assertEqual(
          (bool(trigger_l_to_h_0), bool(trigger_h_to_l_0)), (False, True)
      )
      self.assertEqual(
          (bool(trigger_l_to_h_2), bool(trigger_h_to_l_2)), (True, False)
      )

  def test_transition_time_width_validation(self):
    with self.subTest("allow_positive_values"):
      pydantic_model.SetTpedNped.from_dict({"transition_time_width": 0.5})

    with self.subTest("allow_zero_values"):
      pydantic_model.SetTpedNped.from_dict({"transition_time_width": 0.0})

    with self.subTest("disallow_negative_values"):
      with self.assertRaises(ValueError):
        pydantic_model.SetTpedNped.from_dict({"transition_time_width": -1.0})

  def test_invalid_model_name_error_message(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"Input tag 'invalid_model' found using 'model_name' does not match any"
        r" of the expected tags",
    ):
      pydantic.TypeAdapter(pydantic_model.PedestalConfig).validate_python(
          {"model_name": "invalid_model"}
      )

  def test_missing_model_name_error_message(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"Unable to extract tag using discriminator 'model_name'",
    ):
      pydantic.TypeAdapter(pydantic_model.PedestalConfig).validate_python(
          {"n_e_ped": 0.7e20}
      )

  def test_invalid_formation_model_error_message(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"Input tag 'invalid_formation' found using 'model_name' does not match"
        r" any of the expected tags",
    ):
      pydantic_model.SetTpedNped.from_dict(
          {"formation_model": {"model_name": "invalid_formation"}}
      )


if __name__ == "__main__":
  absltest.main()
