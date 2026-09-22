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
import jax.numpy as jnp
from torax._src.test_utils import default_configs
from torax._src.transport_model import pydantic_model
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import tglfnn_ukaea_transport_model
from torax._src.transport_model.tests import tglf_based_transport_model_test


class TglfnnUkaeaTransportModelTest(parameterized.TestCase):

  def test_hash_and_eq_same(self):
    model1 = tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel(
        machine='multimachine'
    )
    model2 = tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel(
        machine='multimachine'
    )

    self.assertEqual(hash(model1), hash(model2))
    self.assertEqual(model1, model2)

  def test_canonical_physics_dict(self):
    _, (runtime_params, geo, core_profiles, _, two_point_mask) = (
        tglf_based_transport_model_test._get_config_and_model_inputs({
            'core_transport_models': {
                'bohm-gyrobohm': {'model_name': 'bohm-gyrobohm'},
            },
        })
    )
    tglfnn_config = pydantic_model.TGLFNNukaeaTransportModel(
        machine='multimachine'
    )
    model = tglfnn_config.build_transport_model()
    transport_params = tglfnn_config.build_runtime_params(t=0.0)
    tglf_inputs = model._prepare_tglf_inputs(
        transport=transport_params,
        geo=geo,
        core_profiles=core_profiles,
        poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier,
        two_point_mask=two_point_mask,
    )
    physics_dict = tglf_based_transport_model.get_canonical_physics_dict(
        tglf_inputs
    )
    self.assertIn('s_hat', physics_dict)
    self.assertIn('SHAT', physics_dict)
    self.assertIn('inv_aspect_ratio', physics_dict)
    self.assertIn('RLNS_1', physics_dict)
    self.assertIn('RLTS_1', physics_dict)
    self.assertIn('RLTS_2', physics_dict)
    self.assertIn('Q_LOC', physics_dict)
    self.assertEqual(physics_dict['SHAT'].shape, geo.rho_face_norm.shape)

  def test_dynamic_feature_binding_multimachine(self):
    _, (runtime_params, geo, core_profiles, _, two_point_mask) = (
        tglf_based_transport_model_test._get_config_and_model_inputs({
            'core_transport_models': {
                'bohm-gyrobohm': {'model_name': 'bohm-gyrobohm'},
            },
        })
    )
    tglfnn_config = pydantic_model.TGLFNNukaeaTransportModel(
        machine='multimachine'
    )
    model = tglfnn_config.build_transport_model()
    transport_params = tglfnn_config.build_runtime_params(t=0.0)
    tglf_inputs = model._prepare_tglf_inputs(
        transport=transport_params,
        geo=geo,
        core_profiles=core_profiles,
        poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier,
        two_point_mask=two_point_mask,
    )
    inputs_tensor = model._prepare_tglfnn_inputs(tglf_inputs)
    # multimachine has 13 inputs
    self.assertEqual(inputs_tensor.shape, (geo.rho_face_norm.shape[0], 13))

  def test_dynamic_feature_binding_missing_feature_raises(self):
    _, (runtime_params, geo, core_profiles, _, two_point_mask) = (
        tglf_based_transport_model_test._get_config_and_model_inputs({
            'core_transport_models': {
                'bohm-gyrobohm': {'model_name': 'bohm-gyrobohm'},
            },
        })
    )
    tglfnn_config = pydantic_model.TGLFNNukaeaTransportModel(
        machine='multimachine'
    )
    model = tglfnn_config.build_transport_model()
    transport_params = tglfnn_config.build_runtime_params(t=0.0)
    tglf_inputs = model._prepare_tglf_inputs(
        transport=transport_params,
        geo=geo,
        core_profiles=core_profiles,
        poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier,
        two_point_mask=two_point_mask,
    )
    dummy_model = mock.MagicMock()
    dummy_model.input_labels = ('RLNS_1', 'NON_EXISTENT_FEATURE_XYZ')
    object.__setattr__(model, 'model', dummy_model)
    with self.assertRaises(ValueError) as ctx:
      model._prepare_tglfnn_inputs(tglf_inputs)
    self.assertIn('NON_EXISTENT_FEATURE_XYZ', str(ctx.exception))

  def test_predict_with_uncertainty(self):
    _, (runtime_params, geo, core_profiles, _, two_point_mask) = (
        tglf_based_transport_model_test._get_config_and_model_inputs({
            'core_transport_models': {
                'bohm-gyrobohm': {'model_name': 'bohm-gyrobohm'},
            },
        })
    )
    tglfnn_config = pydantic_model.TGLFNNukaeaTransportModel(
        machine='multimachine'
    )
    model = tglfnn_config.build_transport_model()
    transport_params = tglfnn_config.build_runtime_params(t=0.0)
    tglf_inputs = model._prepare_tglf_inputs(
        transport=transport_params,
        geo=geo,
        core_profiles=core_profiles,
        poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier,
        two_point_mask=two_point_mask,
    )
    means, variances = model.predict_with_uncertainty(tglf_inputs)
    for channel in ('efi_gb', 'efe_gb', 'pfi_gb'):
      self.assertIn(channel, means)
      self.assertIn(channel, variances)
      self.assertEqual(means[channel].shape, geo.rho_face_norm.shape)
      self.assertEqual(variances[channel].shape, geo.rho_face_norm.shape)
      # Variances must be non-negative
      self.assertTrue(jnp.all(variances[channel] >= 0.0))

    rel_unc = model.compute_relative_uncertainty(tglf_inputs)
    self.assertEqual(rel_unc.shape, geo.rho_face_norm.shape)
    self.assertTrue(jnp.all(rel_unc >= 0.0))


if __name__ == '__main__':
  absltest.main()
